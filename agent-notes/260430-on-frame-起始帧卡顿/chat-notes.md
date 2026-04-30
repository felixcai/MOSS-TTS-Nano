# 1. Stream Generate 核心调用链路

在 MOSS-TTS-Nano 项目中使用 ONNX 执行流式生成（stream generate）的过程中，主要涉及以下 4 个核心文件，形成了一个从 Web API 路由到底层 ONNX 模型推理的清晰链路：

## 1.1 文件职责与调用关系

* **`app.py` (Web API 与任务调度层)**
  
  * **职责**：FastAPI 的服务端入口，负责处理 HTTP 请求和管理后台流式任务。
  * **动作**：接收 `/api/generate-stream/start` 请求，创建后台线程运行 `_run_streaming_job`。
  * **调用**：调用底层传入的 `runtime` 适配器的 `synthesize_stream` 方法，迭代获取音频数据块并返回给客户端。

* **`app_onnx.py` (ONNX 适配与流式控制层)**
  
  * **职责**：ONNX 版本的入口脚本，提供 `OnnxNanoTTSServiceAdapter`。
  * **动作**：实现 `synthesize_stream`，启动 `_worker` 线程。
  * **调用**：
    * 向上：通过 `yield` 和 `queue.Queue` 将音频块返回给 `app.py`。
    * 向下：调用 `onnx_tts_runtime.py` 准备数据。
    * 核心：调用 `generate_audio_frames` 开始生成，并提供 `_on_frame` 回调函数。在 `_on_frame` 中触发 Codec 解码。

* **`onnx_tts_runtime.py` (TTS 业务逻辑层)**
  
  * **职责**：定义 `OnnxTtsRuntime`，负责 TTS 业务逻辑和数据预处理。
  * **动作**：文本正则化、分词（Tokenization）、参考音频编码等。
  * **调用**：被 `app_onnx.py` 调用执行数据准备，将特征数据传递给父类 `OrtCpuRuntime` 进行推理。

* **`ort_cpu_runtime.py` (ONNX 核心推理层)**
  
  * **职责**：最底层的模型执行层，直接与 `onnxruntime` 交互。
  * **动作**：管理所有 ONNX InferenceSession。
  * **调用**：实现 `generate_audio_frames` 进行自回归生成，每生成一帧触发 `on_frame` 回调。提供 `CodecStreamingDecodeSession.run_frames` 供回调使用，将离散 token 解码为连续音频波形。

## 1.2 核心流程总结

1. **[用户请求]** ➡️ `app.py` 接收请求并启动流式任务。
2. **[发起流式生成]** ➡️ `app.py` 调用 `app_onnx.py` 的 `synthesize_stream`。
3. **[数据预处理]** ➡️ `app_onnx.py` 调用 `onnx_tts_runtime.py` 切分文本（Text Chunks）、编码 Prompt 音频。
4. **[启动推理循环]** ➡️ `app_onnx.py` 对每个 Chunk 调用 `ort_cpu_runtime.py` 的 `generate_audio_frames`，传入 `_on_frame` 回调。
5. **[逐帧生成与回调]** ➡️ `ort_cpu_runtime.py` 自回归生成**声学 Token**，每生成一帧触发一次 `_on_frame`。
6. **[实时解码音频]** ➡️ `app_onnx.py` 的 `_on_frame` 收集 Token 后，调用 `ort_cpu_runtime.py` 的 `CodecStreamingDecodeSession.run_frames` 将 Token **解码为真实的音频波形 (PCM)**。
7. **[返回数据]** ➡️ 解码后的音频数据入队，`yield` 给 `app.py`，流式响应给客户端。

---

# 2. 音频 Token 与 Neural Audio Codec 知识

在上述流程的第 6 步中，将“声学 Token”解码为“音频波形”的过程，并非使用传统的数学公式或库函数（如 MP3、AAC 解码），而是**调用了一个神经网络模型（Neural Audio Codec）的 Decoder 部分**。

## 2.1 为什么不能用传统 Codec？

大语言模型（LLM，如 GPT 架构）的强项是处理**“离散的符号（Discrete Tokens）”**（例如词表中的单词 ID），它们通过预测下一个 Token 的概率分布来生成内容。

传统的音频编解码器（如 MP3, Opus）基于信号处理和心理声学，输出的是**连续且高维的浮点数特征（频谱系数）**。如果让 LLM 去预测这些无限搜索空间的连续浮点数，模型是无法收敛学习的。

## 2.2 神经 Codec 的魔法：离散化与语义压缩

为了让 LLM 能够像生成文本一样生成音频，必须将音频转化为离散的 Token。这就是 Neural Audio Codec（如 EnCodec, SoundStream）的作用。

* **残差矢量量化（RVQ）**：神经 Codec 强制将提取的音频特征“对齐”到一个固定的密码本（Codebook）上。
* **声音的“字母表”**：它将连续的声波切片，并用有限的（例如 1024 个）固定“声音单词”来表示。一段音频就被压缩成了类似 `[12, 456, 89, 1022...]` 的整数 ID 序列。
* **适配 LLM**：将声音转化为整数 ID 序列后，TTS 模型就可以像训练文本 GPT 一样，进行自回归训练：“给定文本 Token，预测下一个声音 Token ID”。

## 2.3 为什么解码（Detokenize）必须用神经网络？

将丰富的声音强行压缩成有限的 Token ID，是一个**极度有损（Lossy）**的过程。

* **信息丢失**：一个 Token ID 只是一个高度抽象的语义符号，丢失了大量关于相位、细微音色、环境底噪等物理细节。
* **神经网络的“脑补”能力**：传统的数学逆变换无法从如此高度压缩的离散 ID 中还原出高保真的连续波形。**Decoder 神经网络的作用，就是根据这些高度压缩的“语义线索（Token IDs）”，利用其在海量音频数据上学到的先验知识，“脑补”出丢失的高频细节和相位信息**，最终渲染出听起来逼真、连续的物理声波（PCM）。

**总结**：流式 Codec 解码器执行的是一个高度抽象的语义符号（Token ID）到高维物理信号（波形）的非线性映射，这必须依赖神经网络强大的非线性拟合能力来完成。

---

# 3. 音频播放时机与前后端交互链路

## 3.1 播放时机与4步链路

每一帧生成完音频后，并不会立刻在浏览器中播放，而是经历以下 4 个步骤的流水线：

1. **后端：解码后立刻放入队列**。在 `app_onnx.py` 中，`_decode_pending` 将声学 Token 解码为 PCM 波形（`float32` 格式）后，会立刻通过 `_emit_waveform` 将其放入一个线程安全的队列（`event_queue`）中。
2. **后端：转码并推入流式通道**。`app.py` 中的后台线程不断从队列中取出波形数据，将其转换为 `int16` 的 PCM 字节流，并放入该请求专属的 `job.audio_queue` 中。
3. **前后端交互：HTTP Streaming 持续拉取**。FastAPI 提供的 `/api/generate-stream/{stream_id}/audio` 接口通过 `StreamingResponse` 保持一个长连接。只要 `job.audio_queue` 里有数据，就会立刻作为 HTTP Chunk 发送给前端。
4. **前端：接收即播放**。前端浏览器通过 Fetch API 持续读取这个数据流，并将接收到的 PCM 字节流喂给 Web Audio API 进行播放。

## 3.2 Float32 到 Int16 PCM 的转换

在上述第 2 步中，系统会调用 `_audio_to_pcm16le_bytes` 将 `float32` 格式的波形数据转换为 `int16` 的 PCM 字节流。

* **实现方式**：这个过程是由标准的数学库（`numpy`）完成的，而不是大模型。它只是简单的数值缩放（将 `[-1.0, 1.0]` 映射到 `[-32767, 32767]`）和类型转换。
* **作用**：大模型（Codec Decoder）输出的原始波形是高精度的 `float32` 格式，但大多数标准的音频硬件（如声卡）和网络音频传输协议，默认期望接收的是 `int16` 格式的 PCM 数据。转换是为了兼容标准播放设备并减少一半的网络传输带宽。

## 3.3 前后端流控与背压机制

在后端生成音频的过程中，**后端本身不会主动等待前端播放完成**。它会全速生成下一个 Text Chunk 和下一帧音频。
但是，为了防止后端生成过快导致内存撑爆，系统引入了**背压（Backpressure）机制**：

* `app_onnx.py` 中的 `event_queue` 设置了 `maxsize=128`。
* 如果前端网络拉取太慢，导致队列满了，后端的 `_safe_put` 函数就会阻塞等待（每次最多等 0.5 秒）。
* 这种机制确保了生成速度和消费速度的动态平衡。

---

# 4. 起始帧卡顿问题分析

## 4.1 问题现象与定位

在实际使用中，可能会遇到语音转文本的前一小段内容有卡顿感，但由于整体 RTF（实时率）小于 1，随后很快就能流畅播放。
这个问题通常出在**前端音频缓冲池耗尽（Starvation）**。由于初始阶段网络传输或前端处理的微小波动，如果后端一开始发送的音频数据包太小（例如只有 1 帧，即 0.04 秒的音频），前端播放器很快就会播完并陷入等待，从而产生卡顿感。

## 4.2 动态 Batch Size 策略 (`_resolve_stream_decode_frame_budget`)

导致初始数据包太小的根源，在于底层的一个动态调度函数：`_resolve_stream_decode_frame_budget`。
该函数的作用是**根据当前已发送音频相对实时播放的超前量（`lead_seconds`），动态决定每次 Codec 解码的帧批大小（Batch Size）**：

* 如果超前量很小（`< 0.20s`），默认策略是 `return 1`。即只要生成 1 帧，就立刻解码发送，以追求极致的**首字节延迟（TTFB）**。
* 随着超前量增加，它会逐渐将 Batch Size 放大到 2、4 甚至 8，以提高计算吞吐量。
  正是这个默认的 `return 1` 策略，导致了初始数据包过小，容易引发前端卡顿。

## 4.3 前端 Batch Size 参数的实际作用

前端页面上虽然有 `TTS Max Batch Size` 和 `Codec Max Batch Size` 两个选项，但在 **ONNX Stream Generate** 模式下：

* **这两个参数是无效的**：在 `app_onnx.py` 的入口处，前端传来的这两个参数会被立刻丢弃（`del`）。
* **TTS Batch Size 锁死为 1**：为了追求极低的首字延迟，并简化 ONNX KV Cache 的状态管理，文本到声学 Token 的生成阶段被硬编码为逐个 Chunk 串行处理（Batch Size = 1）。
* **Codec Batch Size 动态接管**：声学 Token 到音频的解码阶段，其 Batch Size 完全由上述的 `_resolve_stream_decode_frame_budget` 函数动态接管。

## 4.4 帧生成时间与卡顿的缓解

MOSS-TTS-Nano 采用的 Codec 帧率为 25 Hz，即 **1 帧 = 0.04 秒**。

* 如果按默认策略初始发送 1 帧，前端只有 0.04 秒的缓冲，极易卡顿。
* 如果将 `_resolve_stream_decode_frame_budget` 的初始返回值改为 `4`，意味着系统会攒够 4 帧（**0.16 秒**）的音频后再发送。虽然牺牲了微小的首字节延迟，但前端能一口气拿到 0.16 秒的缓冲，大大降低了初始播放卡顿的概率。
