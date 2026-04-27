# MOSS-TTS-Nano ONNX Stream Generate 调用链路分析

在 MOSS-TTS-Nano 项目中使用 ONNX 执行流式生成（stream generate）的过程中，这 4 个文件扮演着不同的角色，从上层的 Web API 路由一直到底层的 ONNX 模型推理，形成了一个清晰的调用链路。

它们的整体调用关系和各自的职责如下：

## 1. `app.py` (Web API 与任务调度层)
这是 FastAPI 的服务端入口，负责处理 HTTP 请求和管理后台流式任务。
* **主要动作**：当用户请求 `/api/generate-stream/start` 接口时，`app.py` 会创建一个后台线程来运行 `_run_streaming_job` 函数。
* **调用关系**：`_run_streaming_job` 会调用底层传入的 `runtime` 适配器（即 `app_onnx.py` 中的 `OnnxNanoTTSServiceAdapter`）的 `synthesize_stream` 方法，并不断迭代获取生成的音频数据块（chunks），然后通过网络流返回给客户端。

## 2. `app_onnx.py` (ONNX 适配与流式控制层)
这个文件是 ONNX 版本的入口脚本，它将原版的 `app.py` 包装起来，并提供了专门针对 ONNX 的服务适配器 `OnnxNanoTTSServiceAdapter`。
* **主要动作**：实现了 `synthesize_stream` 方法。该方法内部会启动一个 `_worker` 线程。
* **调用关系**：
  * **向上**：通过 Python 的 `yield` 和队列（`queue.Queue`），将生成的音频块源源不断地返回给 `app.py`。
  * **向下**：调用 `onnx_tts_runtime.py` 准备文本和音频 Prompt 数据。
  * **核心流式逻辑**：调用 `self.runtime.generate_audio_frames(..., on_frame=_on_frame)` 开始生成声学 token。同时定义了 `_on_frame` 回调函数，当底层每生成一帧 token 时，回调函数会被触发，并调用 `CodecStreamingDecodeSession.run_frames` 将 token 解码为实际的音频波形（Waveform），放入队列中。

## 3. `onnx_tts_runtime.py` (TTS 业务逻辑层)
定义了 `OnnxTtsRuntime` 类，它继承自底层的 `OrtCpuRuntime`。主要负责 TTS 领域相关的业务逻辑和数据预处理。
* **主要动作**：处理文本正则化、分词（Tokenization）、参考音频（Prompt Audio）的编码等。
* **调用关系**：
  * 被 `app_onnx.py` 调用，执行如 `split_voice_clone_text`（长文本切片）、`resolve_prompt_audio_codes`（提取参考音频的声学特征）、`encode_text`（文本转 ID）以及 `build_voice_clone_request_rows`（构建模型输入矩阵）等准备工作。
  * 作为桥梁，将处理好的特征数据传递给其父类 `OrtCpuRuntime` 进行实际的模型推理。

## 4. `ort_cpu_runtime.py` (ONNX 核心推理层)
这是最底层的模型执行层，直接与 `onnxruntime` 交互，负责运行神经网络。定义了 `OrtCpuRuntime` 基类和 `CodecStreamingDecodeSession`。
* **主要动作**：管理所有的 ONNX InferenceSession（如 prefill, decode, local_decoder, codec_decode_step 等）。
* **调用关系**：
  * **自回归生成**：被 `app_onnx.py` 调用的 `generate_audio_frames` 方法实际上是在这里实现的。它包含一个循环，不断运行 ONNX 模型生成下一个 token。每生成一帧，就会触发一次从 `app_onnx.py` 传进来的 `on_frame` 回调。
  * **流式解码**：其中的 `CodecStreamingDecodeSession` 类负责维护流式解码的状态（KV Cache 等），在 `app_onnx.py` 的回调中被调用（`run_frames`），将生成的离散声学 token 实时解码为连续的音频波形（PCM 数据）。

---

## 🔄 核心调用链路总结 (Stream Generate 流程)

1. **[用户请求]** ➡️ `app.py` (`generate_stream_start` -> `_run_streaming_job`)
2. **[发起流式生成]** ➡️ `app.py` 调用 `app_onnx.py` 的 `OnnxNanoTTSServiceAdapter.synthesize_stream`
3. **[数据预处理]** ➡️ `app_onnx.py` 调用 `onnx_tts_runtime.py` 的方法（切分文本、编码 Prompt 音频等）
4. **[启动推理循环]** ➡️ `app_onnx.py` 调用 `ort_cpu_runtime.py` 的 `generate_audio_frames`，并传入 `_on_frame` 回调
5. **[逐帧生成与回调]** ➡️ `ort_cpu_runtime.py` 在 ONNX 模型中自回归生成 token，每生成一帧触发一次 `_on_frame`
6. **[实时解码音频]** ➡️ `app_onnx.py` 的 `_on_frame` 收集到足够的 token 后，调用 `ort_cpu_runtime.py` 中的 `CodecStreamingDecodeSession.run_frames` 将 token 解码为音频数据
7. **[返回数据]** ➡️ 解码后的音频数据被放入队列，`app_onnx.py` `yield` 给 `app.py`，最终流式响应给客户端。