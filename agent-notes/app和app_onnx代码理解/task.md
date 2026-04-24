# Task：`app_onnx.py` 内存修复与进程内存日志

本文档记录「内存修复与日志」任务的**计划**与**最终实现**，便于与代码 diff 对照。背景分析见同目录 [`app_onnx内存泄露.md`](app_onnx内存泄露.md)；HTTP 流式调用链见 [`代码理解-explore.md`](代码理解-explore.md) 第 11 章。

---

## 一、涉及文件（计划）

| 文件                                           | 作用             |
| -------------------------------------------- | -------------- |
| [`app_onnx.py`](../../app_onnx.py)           | 全部逻辑改动         |
| [`requirements.txt`](../../requirements.txt) | 增加 `psutil` 依赖 |

---

## 二、计划原文摘要

### Fix 1：`OnnxRequestRuntimeManager` 多实例缓存

**问题**：`_build_runtime_locked` 对每个不同的 `cpu_threads` 整数值永久创建并缓存一个完整 `OnnxNanoTTSServiceAdapter`，导致重复加载 ONNX 大模型。

**计划改法**：不再按线程数另建新实例，改为**始终返回 `default_runtime`**。当请求的 `cpu_threads` 与 `default_runtime.thread_count` 不一致时，只打 `WARNING`，不新建 Adapter。`self._cpu_runtimes` 与 `_lock` 可保留以兼容现有结构，但不再向字典追加新 key。

### Fix 2：`synthesize_stream` worker 取消机制

**问题**：`event_queue(maxsize=128)` 在消费端提前断开时不再 `get()`，后台 `_worker` 在 `put()` 上满队阻塞，持有的大块 NumPy 等难以回收。

**计划改法**：

1. 在 `event_queue` 之后增加 `threading.Event()`（停止信号）。
2. 用 `_safe_put(item)` 替代裸 `put`：`put(..., timeout=0.5)` 循环，若停止事件已置位则丢弃并返回。
3. `_worker` 内所有入队（含 `_emit_waveform`、error、`None` 哨兵）均走 `_safe_put`。
4. 外层消费生成器用 `try/finally`：退出时置停止事件并 `get_nowait` drain 队列，使阻塞中的 `put` 有机会在超时轮询内感知停止。

### Memory Logging

**前提**：`requirements.txt` 增加 `psutil`；未安装时 `_log_memory` 静默跳过。

**辅助函数**：`_log_memory(label)` → `logging.info("[MEM] %s | rss=%.1f MB", ...)`。

**计划打点（6 处）**：

1. `main()`：`OnnxNanoTTSServiceAdapter` 创建之后 — `main: runtime created`
2. `main()`：`warmup_manager.start()` 之后 — `main: warmup started (port not open yet)`
3. `_build_runtime_locked` 进入时 — `build_runtime: cpu_threads=N`
4. `synthesize_stream._worker` 开头 — `stream_worker: start`
5. 每个 text chunk 处理完毕后 — `stream_worker: chunk N done`
6. `_worker` 的 `finally` 中哨兵前 — `stream_worker: finally exit`

---

## 三、最终实现记录（与仓库当前代码一致）

### 3.1 `requirements.txt`

- 新增一行：`psutil>=5.9.0`

### 3.2 `app_onnx.py`

#### `_log_memory`

- 位置：模块级，`_CpuDeviceInfo` 之前。
- 实现：`import psutil` 后取 `Process().memory_info().rss`，以 MB 打印；任意异常则 `pass`。

#### Fix 1：`_build_runtime_locked`

- 实现与计划一致：先 `_log_memory(f"build_runtime: cpu_threads={cpu_threads}")`，若 `cpu_threads != self.default_runtime.thread_count` 则 `logging.warning(...)`，**始终** `return self.default_runtime`。
- 说明：`OnnxRequestRuntimeManager.__init__` 仍初始化 `self._cpu_runtimes = {default_runtime.thread_count: default_runtime}`，但 `_build_runtime_locked` 不再向该字典写入新条目。

#### Fix 2：`OnnxNanoTTSServiceAdapter.synthesize_stream`

相对计划的**额外实现**（为更快响应取消）：

- 在每个 `chunk_index` 循环开始处：若 `_stop_event.is_set()` 则 `break`。
- 每个 chunk 结束后：再次检查 `_stop_event`，若已设置则 `break`，避免继续句间 pause。
- 仅当 `not _stop_event.is_set()` 时执行整段 `_concat_waveforms`、写 `app_onnx_stream_output.wav`、`_safe_put(result)`；取消路径下跳过，减少无效大数组拼接与写盘。
- `_worker` 的 `finally`：先 `_log_memory("stream_worker: finally exit")`，再 `_safe_put(None)`；若消费端已离开且 `_safe_put` 因停止丢弃 `None`，与外层 `finally` drain 的行为一致（注释中已说明）。

外层消费循环：

- `try: while True: get → yield`；`finally` 中 `_stop_event.set()` 并 drain `event_queue`。

#### Fix 2 类型注解

- `_safe_put` 参数类型为 `dict[str, object] | None`（与 `Queue` 泛型一致），与计划中 `dict | None` 等价。

#### 内存日志（与计划 6 处对应）

| 序号  | 位置                                                     | 日志 label                                   |
| --- | ------------------------------------------------------ | ------------------------------------------ |
| 1   | `main()`，`runtime = OnnxNanoTTSServiceAdapter(...)` 之后 | `main: runtime created`                    |
| 2   | `main()`，`warmup_manager.start()` 之后                   | `main: warmup started (port not open yet)` |
| 3   | `_build_runtime_locked` 开头                             | `build_runtime: cpu_threads={N}`           |
| 4   | `_worker` 内 `try` 块首行                                  | `stream_worker: start`                     |
| 5   | 每个 chunk 处理完成后                                         | `stream_worker: chunk {chunk_index} done`  |
| 6   | `_worker` `finally`，`_safe_put(None)` 之前               | `stream_worker: finally exit`              |

### 3.3 未改动的部分

- `app.py` 流式路由、`StreamingJob`、`iter_with_runtime` 等**未修改**；本任务仅改 ONNX 适配层与依赖。
- Cursor 计划文件（`.cursor/plans/...`）**未写入本仓库**；本 `task.md` 为仓库内归档。

### 3.4 使用与注意

- 安装依赖：`pip install -r requirements.txt`（需包含 `psutil` 才有 `[MEM]` 日志）。
- ONNX 侧请求参数中的 `cpu_threads` 若与进程启动时 `--cpu-threads` 不一致，将收到警告且**实际仍使用默认 Adapter 的 ORT 线程配置**；若未来需要按请求切换线程数，需另行设计（例如单会话 + 文档说明限制）。

---

## 四、相关文档索引

- [`app_onnx内存泄露.md`](app_onnx内存泄露.md) — 问题现象与原因分析
- [`代码理解-explore.md`](代码理解-explore.md) — `app.py` / `app_onnx.py` 调用链
