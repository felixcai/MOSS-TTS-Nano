# `app_onnx.py` 内存相关分析（启动 → 流式服务）

本文记录从启动到完成一次（或多次）音频输出服务过程中，**可能导致内存持续升高或无法回落**的代码路径与原因推断。流式服务调用链与细节可参考同目录下 `代码理解-explore.md` 第 11 章。

## 观察到的现象（供对照）

1. 从进程启动到服务端口打开：内存约 **4.5G → 5.8G**。
2. 提供完服务之后：内存约升到 **12G** 左右。
3. 多次语音服务之后：内存 **15G 往上**。

以下两点与上述阶梯式上涨较为吻合，且分别对应「模型多份常驻」与「流式提前断开时的线程/队列行为」。

**代码状态**：仓库中已对第 1、2 点做针对性修复（单一默认 runtime、流式 worker 取消与队列 drain）；细节与变更清单见同目录 [`task.md`](task.md)。下文仍保留原始问题分析，便于对照历史现象。

---

## 1. `cpu_threads` 缓存导致 ONNX 运行时多份加载（5.8G → 12G 量级）

> **已修复**：`OnnxRequestRuntimeManager._build_runtime_locked` 现始终复用 `default_runtime`，不再按请求线程数新建第二套 ONNX 会话；与启动 `--cpu-threads` 不一致时仅打 `WARNING`。

**位置**：`app_onnx.py` 中 `OnnxRequestRuntimeManager._build_runtime_locked`（以及 `_cpu_runtimes` 字典的缓存策略）。

**原因简述**：

- PyTorch 路径（`app.py`）里，`cpu_threads` 往往通过 `torch.set_num_threads` 等在同一套模型上调节，**模型通常只有一份**。
- ONNX 路径里，`OnnxNanoTTSServiceAdapter` 内部创建的 `OnnxTtsRuntime` / ORT `InferenceSession` 的线程数在会话创建时即确定；为兼容前端表单里不同的 `cpu_threads`，`OnnxRequestRuntimeManager` 对**每个不同的 `cpu_threads` 整数值**会**新建**一个 `OnnxNanoTTSServiceAdapter` 并**永久**放入 `self._cpu_runtimes`。

**与现象的对应**：

- **4.5G → 5.8G**：启动时按 `main()` 的 `--cpu-threads`（默认多为 `os.cpu_count()`）创建 `default_runtime`，完成 ONNX 资源加载，常驻内存明显上升。
- **5.8G → 12G**：首次实际请求若传入的 `cpu_threads` 与启动默认值不一致，会触发 `_build_runtime_locked` 再建一份完整 Adapter（相当于**再加载一套**大模型相关资源），内存接近**翻倍**并不意外。

**性质**：这更像是**按线程数维度缓存多份大模型**的设计结果，不一定是传统意义的「泄漏」，但若用户/UI 频繁切换不同线程数，字典会持续增长，表现为**常驻内存只增不减**。

---

## 2. 流式路径：消费端提前结束与后台 worker + 有界队列（12G → 15G+ 的风险）

> **已修复**：`synthesize_stream` 内增加 `_stop_event`、带超时的 `_safe_put`、消费端 `finally` 中 drain 队列，避免客户端断开后 worker 永久阻塞在满队 `put()` 上。

**位置**：`app_onnx.py` 中 `OnnxNanoTTSServiceAdapter.synthesize_stream`：后台线程 `_worker` + `queue.Queue(maxsize=128)` + 外层 `while True: item = event_queue.get(); yield item`。

**原因简述**：

- 流式合成在适配器内用**守护线程**跑 ONNX，通过 `event_queue` 向 `synthesize_stream` 的生成器 `yield` 事件。
- 队列 **`maxsize=128`**：若消费端（例如 FastAPI `StreamingResponse` 的生成器）因**客户端断开、刷新、关闭**等提前结束，则外层迭代停止，**不再 `get()` 出队**。
- 后台 `_worker` 若仍继续 `put()` 音频/结果事件，队列满后 **`put()` 会阻塞**；线程若一直阻塞在 `put()` 上，则该次请求相关的**大对象**（如 worker 内累积的波形列表、`all_waveforms` / `all_generated_frames` 等）在逻辑上可能长期挂在线程栈与引用链上，**难以被 GC**，表现为多次请求后内存**阶梯式累积**。

**与现象的对应**：

- 若存在**多次中途断开**的流式连接，每次可能遗留一个**卡在 `put()` 上的 worker 线程**及其持有数据，与「越做服务内存越高」一致。
- 注意：这与「正常跑完一整条流式且消费端一直读到 `None`」的路径不同；后者理论上应能结束 worker 并释放本次局部数据。

**性质**：在客户端/网络不稳定或主动中断较多的场景下，更接近**资源未随生成器生命周期可靠回收**（线程 + 有界队列组合的经典风险），需用「取消信号 / 非阻塞 put / drain 或增大配合关闭协议」等策略缓解。

---

## 3. 修复思路（实现时需在代码中落实，本文仅作备忘）

**缓解 1（多份 ONNX 常驻）**：

- 在部署场景下可评估：**始终使用单一 `OnnxNanoTTSServiceAdapter`**（忽略或规范化请求里的 `cpu_threads`），或限制允许的线程数档位并对字典做 LRU/上限，避免无界缓存多份完整会话。

**缓解 2（流式 worker 与队列）**：

- 在外层生成器收到 `GeneratorExit` 或显式关闭时，向 worker 发**取消事件**，worker 在循环中检查并尽快退出，避免无限 `put` 阻塞。
- 或将 `put` 改为带超时的非阻塞策略，并在取消时丢弃或 drain 队列；与 `app.py` 里 `_run_streaming_job` 的 `job.is_closed` 等语义对齐更佳。

### 3.1 已实现（与 `task.md` 对齐）

- **Fix 1**：始终复用启动时创建的 `OnnxNanoTTSServiceAdapter`，避免按 `cpu_threads` 多份加载 ONNX。
- **Fix 2**：流式生成器退出时置停止事件并 drain 内部 `event_queue`；worker 侧用可中断的入队逻辑，并在 chunk 边界检查停止信号。
- **观测**：`app_onnx.py` 中 `_log_memory` 同时输出 **`proc_rss`（当前进程工作集）** 与 **`sys_used`（整机已用物理内存，psutil）**，便于与任务管理器对照。

---

## 4. 相关代码锚点（便于跳转）

| 主题                      | 文件                | 说明                                                                                       |
| ----------------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| ONNX 按 `cpu_threads` 缓存 | `app_onnx.py`     | `OnnxRequestRuntimeManager._build_runtime_locked`、`self._cpu_runtimes`                   |
| 流式 worker + 有界队列        | `app_onnx.py`     | `OnnxNanoTTSServiceAdapter.synthesize_stream` 内 `event_queue`、`Thread(target=_worker)`   |
| HTTP 流式任务与队列            | `app.py`          | `StreamingJob`、`audio_queue`、`_run_streaming_job`（与 ONNX 共用路由，底层 `synthesize_stream` 不同） |
| 调用链说明                   | `代码理解-explore.md` | 第 11 章 `generate-stream`                                                                 |

---

## 5. 日志里 `proc_rss` 与 `sys_used` 为何常对不上

- **`proc_rss`**：当前 Python 进程的常驻物理内存（在 Windows 上更接近 Working Set）。主要反映进程地址空间里**已被 OS 挂到该进程上的页**；C++ 扩展（ONNX Runtime）里大量**已提交、但近期未频繁访问**的页，可能不全部体现在 RSS 里。
- **`sys_used`**：`psutil.virtual_memory().used`，**整机**已用物理内存，与任务管理器「已用内存」口径接近，包含其他进程、文件缓存、子进程等。

因此会出现：**一次推理里 `proc_rss` 只涨几百 MB，而 `sys_used` 涨 2GB+**。这往往来自 ONNX/allocator 的大块提交、内存映射、子进程（如文本规范化相关进程）或其它系统缓存，并不矛盾。做容量规划时，**更应盯 `sys_used`（或任务管理器）**。

---

## 6. 同一次 `POST /api/generate-stream/start` 里为何 `build_runtime` 打两次日志

`app.py` 在同一路由里对 `runtime_manager` 做了**两次**独立加锁取 runtime：

1. **`_resolve_voice_clone_text_chunks`**：在起后台线程**之前**，`call_with_runtime` → `split_voice_clone_text`，先把全文切成 `text_chunks` 写入 `StreamingJob`（供前端分句与进度）。
2. **后台 `_run_streaming_job`**：`iter_with_runtime` → `synthesize_stream`，真正流式合成。

两次都会进入 `OnnxRequestRuntimeManager._locked_runtime` → `_build_runtime_locked`，因此日志里会出现两条相邻的 `[MEM] build_runtime...`（时间差通常只有几毫秒）。**不是泄漏或重复建会话**（修复后两次都返回同一 `default_runtime`）。

---

## 7. 实测日志模式：短请求重复 vs 长文本 + 换音色

### 问题日志

这次做了如下操作：

1、用第一种音色，读了一段文字

2026-04-24 18:02:54,406 INFO root: [MEM] stream_worker: chunk 0 done | proc_rss=1425.7 MB | sys_used=8082.5 MB

2026-04-24 18:02:54,421 INFO root: Nano-TTS stream RTF | stream_id=stream-1777024961315-5aacdad9 | audio_chunks=90 | total_audio_s=15.680 | first_audio_latency_s=0.9795 | rtf_first=12.2410 | rtf_steady=0.7729

2026-04-24 18:02:54,421 INFO root: [MEM] stream_worker: finally exit | proc_rss=1437.3 MB | sys_used=8099.7 MB

2、相同的音色，又读了几次同样的文字：

2026-04-24 18:03:16,482 INFO root: [MEM] stream_worker: chunk 0 done | proc_rss=1467.8 MB | sys_used=8110.3 MB

2026-04-24 18:03:16,497 INFO root: Nano-TTS stream RTF | stream_id=stream-1777024984519-329be8e3 | audio_chunks=70 | total_audio_s=16.400 | first_audio_latency_s=0.4952 | rtf_first=6.1855 | rtf_steady=0.7002

2026-04-24 18:03:16,498 INFO root: [MEM] stream_worker: finally exit | proc_rss=1479.9 MB | sys_used=8126.4 MB

2026-04-24 18:03:34,596 INFO root: normalized text chars_before=103 chars_after=95 stage=robust_pre

2026-04-24 18:03:34,653 INFO root: [MEM] build_runtime: cpu_threads=4 | proc_rss=1469.0 MB | sys_used=8145.6 MB

2026-04-24 18:03:34,654 WARNING root: OnnxRequestRuntimeManager: ignoring cpu_threads=4 (default=8) to avoid loading a second ONNX session; reusing default runtime.

INFO: 127.0.0.1:44110 - "POST /api/generate-stream/start HTTP/1.1" 200 OK

2026-04-24 18:03:34,658 INFO root: [MEM] build_runtime: cpu_threads=4 | proc_rss=1469.0 MB | sys_used=8146.5 MB

2026-04-24 18:03:34,658 WARNING root: OnnxRequestRuntimeManager: ignoring cpu_threads=4 (default=8) to avoid loading a second ONNX session; reusing default runtime.

2026-04-24 18:03:34,660 INFO root: [MEM] stream_worker: start | proc_rss=1469.0 MB | sys_used=8146.5 MB

3、换了一种音色，去读了一段更长的语音

2026-04-24 18:04:45,352 INFO root: normalized text chars_before=579 chars_after=563 stage=robust_pre

2026-04-24 18:04:45,603 INFO root: [MEM] build_runtime: cpu_threads=4 | proc_rss=1507.9 MB | sys_used=8183.1 MB

2026-04-24 18:04:45,603 WARNING root: OnnxRequestRuntimeManager: ignoring cpu_threads=4 (default=8) to avoid loading a second ONNX session; reusing default runtime.

INFO: 127.0.0.1:56340 - "POST /api/generate-stream/start HTTP/1.1" 200 OK

2026-04-24 18:04:45,609 INFO root: [MEM] build_runtime: cpu_threads=4 | proc_rss=1507.9 MB | sys_used=8183.6 MB

2026-04-24 18:04:45,609 WARNING root: OnnxRequestRuntimeManager: ignoring cpu_threads=4 (default=8) to avoid loading a second ONNX session; reusing default runtime.

2026-04-24 18:04:45,610 INFO root: [MEM] stream_worker: start | proc_rss=1507.9 MB | sys_used=8183.6 MB

2026-04-24 18:05:01,251 INFO root: [MEM] stream_worker: chunk 0 done | proc_rss=1624.3 MB | sys_used=10856.9 MB

2026-04-24 18:05:21,599 INFO root: [MEM] stream_worker: chunk 1 done | proc_rss=1672.8 MB | sys_used=10921.1 MB

2026-04-24 18:05:41,821 INFO root: [MEM] stream_worker: chunk 2 done | proc_rss=1688.7 MB | sys_used=10970.9 MB

2026-04-24 18:05:59,819 INFO root: [MEM] stream_worker: chunk 3 done | proc_rss=1698.8 MB | sys_used=10941.1 MB

2026-04-24 18:06:18,303 INFO root: [MEM] stream_worker: chunk 4 done | proc_rss=1708.8 MB | sys_used=10967.6 MB

2026-04-24 18:06:30,754 INFO root: [MEM] stream_worker: chunk 5 done | proc_rss=1716.2 MB | sys_used=10985.7 MB

2026-04-24 18:06:30,853 INFO root: Nano-TTS stream RTF | stream_id=stream-1777025085606-7f5be609 | audio_chunks=294 | total_audio_s=152.560 | first_audio_latency_s=1.5067 | rtf_first=18.8294 | rtf_steady=0.6786

2026-04-24 18:06:30,854 INFO root: [MEM] stream_worker: finally exit | proc_rss=1772.2 MB | sys_used=11041.9 MB

分析一下

### 7.1 同音色、同短文本多次请求

`sys_used` 每次只有**几十 MB 量级**波动：说明修复后**无宏观泄漏**；小幅上涨可能来自 GC 尚未回收、队列中尚未消费的 PCM、或 ONNX arena 的碎片。

### 7.2 换音色 + 更长文本（多 chunk）

常见模式：

- **Chunk 0 结束**时 `sys_used` 出现**单次巨大跃升**（例如 +2GB 量级）：往往对应「本轮推理第一次触达当前配置下的**峰值工作集**」——更长上下文、更长参考音频编码后的 prompt token、以及 ONNX **Arena** 向系统一次性申请的大块内存。
- **后续 chunk**：`sys_used` 增量明显变小甚至略降：说明引擎在**复用已申请好的内存池**，不必再向 OS 要同样量级的新页。
- **`finally exit` 后 `sys_used` 未必回落**：ORT 等原生 allocator 常采用**高水位（high watermark）**策略，为下次请求保留池子，不立即把物理页还给操作系统；这是**性能与 RSS 报表之间的权衡**，不等于 Python 层又泄漏了一份对象图。

---

## 8. 这是「模型行为」吗？Python 侧能控制什么？

### 8.1 模型 / 运行时侧（本质）

- **参考音频（voice clone）**：换音色或换上传的 prompt，会走 codec 编码得到 `prompt_audio_codes`；更长参考音频 → **更长的前缀序列**，后续每一步 decode 都要带着这份上下文，**内存随有效序列长度上升**（具体关系由导出 ONNX 的图结构决定，常见为随长度近似线性或超线性）。
- **自回归 TTS decode**：生成长度增加时，中间激活与（若图中有）KV 类状态会占用更多内存；长文本还会被切成多个 chunk，但**每个 chunk 内**仍可能在该 chunk 的峰值上触顶 allocator。

以上主要由**导出模型 + ORT 执行计划 +  allocator** 决定，不全是 Python 里几行 list 能解释的。

### 8.2 Python / 配置侧可做的优化（可操作清单）

| 手段                                                          | 说明                                                                                                                                                        |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **减小 `voice_clone_max_text_tokens`**（请求或前端表单）               | 单 chunk 文本更短，降低单次 prefill/decode 的序列峰值，通常直接压低内存尖峰。                                                                                                        |
| **限制参考音频时长**                                                | 上传或 demo 的 prompt 控制在数秒级即可兼顾音色；过长 prompt 是 KV/前缀长度的主要推手之一。                                                                                                |
| **ORT SessionOptions（改 `ort_cpu_runtime.py` 的 `_session`）** | 例如 `add_session_config_entry("session.memory_arena_shrink_strategy", "cpu:0")` 鼓励 arena 在空闲时收缩；或 `enable_cpu_mem_arena = False` 换更低常驻、略损性能。需按版本文档验证键名与行为。 |
| **合理 `cpu_threads`**                                        | 线程本地 arena 可能放大常驻；内存紧张时可适当降低 intra-op 线程数（与启动参数一致，避免误以为请求里改线程会换会话——修复后请求侧线程数已被忽略）。                                                                        |
| **`max_new_frames` 等生成上限**                                  | 限制极端长语音生成的步数，避免最坏情况下的长时间自回归与中间张量堆积。                                                                                                                       |

---

*记录日期：基于对话整理；第 1、2 节问题已在代码中修复，第 5–8 节为日志解读与优化备忘。更细的变更列表见 [`task.md`](task.md)。*
