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

---

# `ort_cpu_runtime.py` 函数调用分类（仅 Stream Generate 调用链视角）

以下分类对应 `ort_cpu_runtime.py` 中各函数在 **ONNX 流式生成（stream generate）** 路径上的角色：入口函数由其他文件调用；内部函数仅在本文件内互相调用；非调用链函数不参与一次完整的「请求 → 流式产出音频」链路（例如预热、非流式解码、列表类工具接口）。

## 1 调用链入口函数（被其他文件调用）

| 函数                                             | 典型调用方                                                         | 说明                                                                               |
| ---------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `OrtCpuRuntime.__init__`                       | `onnx_tts_runtime.py`（`OnnxTtsRuntime` 通过 `super().__init__`） | 读取 manifest / meta、创建全部 ONNX `InferenceSession`、构造 `CodecStreamingDecodeSession` |
| `OrtCpuRuntime.resolve_manifest_relative_path` | `onnx_tts_runtime.py`                                         | 将 manifest 中的相对路径解析为绝对路径，并处理旧版目录名别名（`MODEL_DIR_ALIAS_MAP`）                       |
| `OrtCpuRuntime.build_voice_clone_request_rows` | `onnx_tts_runtime.py`                                         | 拼接参考音频 codes 与目标文本 token，得到 prefill 所需的 `inputIds` / `attentionMask`             |
| `OrtCpuRuntime.generate_audio_frames`          | `app_onnx.py`（`_worker` 线程）                                   | **流式生成核心入口**：prefill → 自回归 decode 循环；每帧结束后调用传入的 `on_frame`                       |
| `CodecStreamingDecodeSession.run_frames`       | `app_onnx.py`（`_on_frame` 回调内）                                | 将一批声学 token 帧送入 `codec_decode_step`，输出 PCM，并更新流式 KV Cache                        |
| `CodecStreamingDecodeSession.reset`            | `app_onnx.py`（新请求开始前）                                         | 清空流式解码状态，避免跨请求污染 KV Cache                                                        |
| `_normalize_sample_mode`                       | `OrtCpuRuntime.__init__` 及上层按需                                | 将外部 `sample_mode` 规范化为 `greedy` / `fixed` / `full`（含 `mixed3` 等别名处理）             |
| `_resolve_stream_decode_frame_budget`          | `app_onnx.py`（`_on_frame` 内）                                  | 根据已下发音频相对实时播放的「超前量」决定每次 `run_frames` 批处理的帧数，平衡首包延迟与吞吐                            |

## 2 调用链内部函数（仅本文件内部调用）

- **张量与隐状态**：`_flatten3d_int32`、`_flatten2d_int32`、`_extract_last_hidden`
- **采样与 logits 处理**：`_argmax`、`_softmax`、`_apply_repetition_penalty`、`_argmax_with_repetition_penalty`、`_sample_from_scores`、`_sample_assistant_text_token`、`_sample_audio_token`、`slice_audio_channel_logits`
- **请求行构造（供 `build_voice_clone_request_rows` 使用）**：`build_text_rows`、`build_audio_prefix_rows`
- **局部解码 / 整帧 ONNX 路径**：`run_local_decoder`、`create_empty_local_cached_past`、`run_local_cached_step`、`run_local_greedy_frame`、`run_local_fixed_sampled_frame`
- **会话与 manifest**：`_resolve_manifest_path`、`_create_sessions`、`_session`
- **流式预算辅助**：`_compute_stream_lead_seconds`（由 `_resolve_stream_decode_frame_budget` 调用）
- **流式 Codec 会话初始化**：`CodecStreamingDecodeSession.__post_init__`（dataclass 生命周期内调用 `reset`）

## 3 非调用链函数（不参与单次 Stream Generate 主路径）

| 函数                    | 说明                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| `warmup`              | 服务启动预热，触发各 ONNX 路径与 codec 流式会话的一次性执行，降低首请求冷启动延迟                                                              |
| `decode_full_audio`   | 使用 `codec_decode`（全量解码图）一次性解码；**流式路径**使用 `CodecStreamingDecodeSession.run_frames`（`codec_decode_step`），不走此函数 |
| `list_builtin_voices` | 返回 manifest 中内置音色列表，供 UI / 接口展示                                                                              |
| `list_text_samples`   | 返回 manifest 中文本示例列表，供 UI / 接口展示                                                                              |

## 4 与总体调用链路的对应关系

- **入口 + 内部**共同实现：文档第 4 节所述的 `prefill` / `decode` 自回归循环，以及 `on_frame` 与 `run_frames` 之间的衔接。
- **非调用链**函数仍属于 `ort_cpu_runtime.py` 的运行时能力，但在「一次 HTTP 流式生成请求」的因果链上位于旁路（初始化、非流式解码、元数据查询）。

## 5 内存阶梯式上涨且不回落的核心逻辑

在流式生成过程中，系统内存（或显存）经常表现出“阶梯式上涨且不回落”的现象。这并非 Python 脚本发生了内存泄漏，而是由以下三个核心机制共同决定的：

1. **切分处理（Text Chunking）**：长文本会被切分为多个独立的 `text_chunk` 逐个处理。
2. **完整推理循环**：每个 `text_chunk` 都要带着参考音频（音色），独立经历一次完整的、极其消耗内存的 `Prefill`（计算全局 Attention）和 `Decode`（自回归生成并累积 KV Cache）循环。
3. **只进不出（高水位策略）**：在 `Prefill` 或 `Decode` 过程中，只要当前需要的**单块连续内存**超过了 ORT Arena（内存池）中现有的最大空闲连续块，Arena 就会向操作系统申请一块**全新**的大内存。而之前申请的、可能已经碎片化的老内存，**绝对不会归还给操作系统**，而是留在 Arena 内部备用。

**总结**：

- 换了**更长的参考音频**（导致 `Prefill` 初始张量变大）会涨内存。
- 遇到了**更长的 `text_chunk`**（导致 `Decode` 步数变多，KV Cache 峰值变大）也会涨内存。
- 只要后续请求不打破历史的“最大连续内存需求纪录”，内存就几乎不再上涨。