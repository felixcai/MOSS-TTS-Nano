# `app_onnx.py` 内存相关分析（启动 → 流式服务）

本文记录从启动到完成一次（或多次）音频输出服务过程中，**可能导致内存持续升高或无法回落**的代码路径与原因推断。流式服务调用链与细节可参考同目录下 `代码理解-explore.md` 第 11 章。

## 观察到的现象（供对照）

1. 从进程启动到服务端口打开：内存约 **4.5G → 5.8G**。
2. 提供完服务之后：内存约升到 **12G** 左右。
3. 多次语音服务之后：内存 **15G 往上**。

以下两点与上述阶梯式上涨较为吻合，且分别对应「模型多份常驻」与「流式提前断开时的线程/队列行为」。

---

## 1. `cpu_threads` 缓存导致 ONNX 运行时多份加载（5.8G → 12G 量级）

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

---

## 4. 相关代码锚点（便于跳转）

| 主题                      | 文件                | 说明                                                                                       |
| ----------------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| ONNX 按 `cpu_threads` 缓存 | `app_onnx.py`     | `OnnxRequestRuntimeManager._build_runtime_locked`、`self._cpu_runtimes`                   |
| 流式 worker + 有界队列        | `app_onnx.py`     | `OnnxNanoTTSServiceAdapter.synthesize_stream` 内 `event_queue`、`Thread(target=_worker)`   |
| HTTP 流式任务与队列            | `app.py`          | `StreamingJob`、`audio_queue`、`_run_streaming_job`（与 ONNX 共用路由，底层 `synthesize_stream` 不同） |
| 调用链说明                   | `代码理解-explore.md` | 第 11 章 `generate-stream`                                                                 |

---

*记录日期：基于对话整理；若后续代码已修复上述点，请在本文件追加「已修复版本」说明以免误导。*
