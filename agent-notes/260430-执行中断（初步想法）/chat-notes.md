# ONNX TTS 流式生成执行中断机制探讨

## 问题背景

在 `app_onnx.py` 的 `synthesize_stream` 接口中，当前如果客户端断开连接（触发 `GeneratorExit`），会设置 `_stop_event`。但是，这个中断信号只在遍历长文本切分出来的 `text_chunks` 循环的**开头和结尾**生效。

一旦进入了单个 chunk 的生成，底层会调用 `ort_cpu_runtime.py` 中的 `generate_audio_frames`。这是一个紧密的自回归推理循环，在这个过程中，即使 `_stop_event` 被设置，底层循环也会继续死磕算完当前 chunk，导致 CPU 资源被无效占用（虽然生成的音频会被 `_safe_put` 默默丢弃）。

## 核心生成流程拆解

在 `generate_audio_frames` 中，一个 chunk 的生成严格分为两个阶段：

1. **Prefill（预填充阶段）**：
   
   - 将参考音频的 token 和当前 chunk 的文本 token 拼接。
   - 一次性送入 `prefill` ONNX 模型。
   - 输出 KV Cache（历史缓存）和 global_hidden（全局隐状态，作为生成第一帧的条件）。

2. **Decode（自回归解码阶段）**：
   
   - 进入 `for step_index in range(max_new_frames):` 循环。
   - 逐帧（frame）生成音频，每帧包含多个声道（如 8 个 channel）的 token。
   - 每一帧的所有声道 token 生成完毕后，触发 `on_frame` 回调。
   - 带着新生成的 token 更新 KV Cache，进入下一帧循环。

## 中断机制的实现方案

### 方案一：Hack 做法（不改底层代码）

在 `app_onnx.py` 的 `_on_frame` 回调中检查 `_stop_event`，如果已设置则抛出异常（如 `InterruptedError`），强行打断底层的 `for` 循环。外层捕获该异常并静默处理。

- **缺点**：粗暴的异常打断可能会跳过底层循环结束后的清理步骤（如 Arena 收缩空跑），不利于内存安全。

### 方案二：优雅做法（改造底层代码）

将 `stop_event`（或一个返回 bool 的 `is_cancelled()` 回调）传入 `generate_audio_frames` 函数，并在两个关键节点进行检查：

1. **Prefill 之后检查**：
   - Prefill 是一个耗时较长的单次操作。完成后立刻检查中断信号。
   - 若已取消，直接 `return []` 退出，不进入 Decode 循环。
2. **Decode 循环内部检查**：
   - 在 `for step_index in range(...)` 循环内部，每一帧生成完毕后（触发 `on_frame` 之前或之后）检查中断信号。
   - 若已取消，直接 `break` 跳出循环。

**优势**：

- **控制流清晰**：没有粗暴的异常打断，函数能正常走到结尾。
- **内存安全**：循环正常 `break` 后，代码还能继续执行循环外面的 `Arena 收缩空跑`（Shrinkage），这对于 ONNX 释放 GPU/CPU 内存碎片非常重要。这是标准的流式推理中断（Early Stopping）实现思路。
