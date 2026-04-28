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

在流式生成过程中，系统内存（或显存）经常表现出"阶梯式上涨且不回落"的现象。这并非 Python 脚本发生了内存泄漏，而是由以下三个核心机制共同决定的：

1. **切分处理（Text Chunking）**：长文本会被切分为多个独立的 `text_chunk` 逐个处理。
2. **完整推理循环**：每个 `text_chunk` 都要带着参考音频（音色），独立经历一次完整的、极其消耗内存的 `Prefill`（计算全局 Attention）和 `Decode`（自回归生成并累积 KV Cache）循环。
3. **只进不出（高水位策略）**：在 `Prefill` 或 `Decode` 过程中，只要当前需要的**单块连续内存**超过了 ORT Arena（内存池）中现有的最大空闲连续块，Arena 就会向操作系统申请一块**全新**的大内存。而之前申请的、可能已经碎片化的老内存，**绝对不会归还给操作系统**，而是留在 Arena 内部备用。

**总结**：

- 换了**更长的参考音频**（导致 `Prefill` 初始张量变大）会涨内存。
- 遇到了**更长的 `text_chunk`**（导致 `Decode` 步数变多，KV Cache 峰值变大）也会涨内存。
- 只要后续请求不打破历史的"最大连续内存需求纪录"，内存就几乎不再上涨。

---

# `onnx_tts_runtime.py` 函数调用分类（仅 Stream Generate 调用链视角）

以下分类对应 `onnx_tts_runtime.py` 中各函数在 **ONNX 流式生成（stream generate）** 路径上的角色。函数角色定义：

- **调用链入口**：被 `app_onnx.py` 直接调用或导入的函数
- **调用链内部**：仅在本文件内被其他函数调用，支撑入口函数完成工作
- **非调用链**：不参与单次 stream generate 请求路径的函数

## 1 调用链入口函数（被 app_onnx.py 直接调用/导入）

| 函数                                               | 调用方（stream generate 路径）                       | 说明                                             |
| ------------------------------------------------ | --------------------------------------------- | ---------------------------------------------- |
| `_merge_audio_channels`                          | `app_onnx.py` `_decode_pending` 闭包（直接 import） | 将 codec 解码输出的多声道数组堆叠为 `(samples, channels)` 波形 |
| `_concat_waveforms`                              | `app_onnx.py` `_worker`（直接 import）            | 把某 text_chunk 所有流式解码小段波形拼接为完整波形                |
| `_write_waveform_to_wav`                         | `app_onnx.py` `_worker` 末尾（直接 import）         | 所有 chunk 处理完毕后将最终波形写入磁盘                        |
| `OnnxTtsRuntime.__init__`                        | `OnnxNanoTTSServiceAdapter.__init__`          | 确认/下载模型、创建 ONNX Session、加载 tokenizer           |
| `resolve_prompt_audio_codes`                     | `_worker` 第一步                                 | 统一入口：自定义音频走编码路径，内置音色从 manifest 读取预编码 codes     |
| `split_voice_clone_text`                         | `_worker` 第二步                                 | 三层策略将长文本切分为不超过 token 预算的 text_chunk 列表         |
| `encode_text`                                    | `_worker` 循环内（每个 chunk 一次）                    | SentencePiece 分词，将文本转为 token ID 列表             |
| `estimate_voice_clone_inter_chunk_pause_seconds` | `_worker` 循环内（每两个 chunk 之间）                   | 根据词数决定相邻 chunk 间的静音时长（0.24s 或 0.40s）           |

## 2 调用链内部函数（仅本文件内部调用）

### 2.1 初始化支路（`__init__` → `ensure_browser_onnx_model_dir` 分支）

| 函数                                      | 调用方                                                            | 说明                                      |
| --------------------------------------- | -------------------------------------------------------------- | --------------------------------------- |
| `ensure_browser_onnx_model_dir`         | `OnnxTtsRuntime.__init__`                                      | 快速路径检查 manifest；缺失时自动下载 ONNX 资产         |
| `_resolve_model_dir_path`               | `ensure_browser_onnx_model_dir`、`_default_model_dir_requested` | 将 `model_dir` 参数规范化为绝对路径                |
| `_default_model_dir_requested`          | `ensure_browser_onnx_model_dir`                                | 判断是否使用默认目录，决定是否允许自动下载                   |
| `_find_manifest_path`                   | `ensure_browser_onnx_model_dir`                                | 遍历候选相对路径，找到 `browser_poc_manifest.json` |
| `_download_default_browser_onnx_assets` | `ensure_browser_onnx_model_dir`                                | 分别下载 TTS 和 Codec ONNX 资产并整理目录结构         |
| `_snapshot_download_repo`               | `_download_default_browser_onnx_assets`                        | 封装 `huggingface_hub.snapshot_download`  |
| `_normalize_download_layout`            | `_download_default_browser_onnx_assets`                        | 将 HuggingFace 下载产生的嵌套目录铺平到目标目录          |
| `_find_directory_with_required_names`   | `_normalize_download_layout`                                   | 在目录树中找到包含所有必要文件的子目录                     |
| `_promote_directory_contents`           | `_normalize_download_layout`                                   | 将子目录内容提升（move）到目标目录                     |
| `_directory_contains_all`               | `_find_directory_with_required_names`                          | 检查目录是否包含所有 required 文件名                 |

### 2.2 文本切分支路（`split_voice_clone_text` → 各辅助函数）

| 函数                                    | 调用方                                                          | 说明                           |
| ------------------------------------- | ------------------------------------------------------------ | ---------------------------- |
| `_prepare_text_for_sentence_chunking` | `split_voice_clone_text`                                     | 清理文本、补全末尾标点、对短英文补前导空格        |
| `_split_text_by_punctuation`          | `split_voice_clone_text`                                     | 按指定标点集切分句子/子句，保留右括号等关闭标点     |
| `_join_sentence_parts`                | `split_voice_clone_text`                                     | 合并两段文本时根据 CJK 判断是否插入空格       |
| `_contains_cjk`                       | `_prepare_text_for_sentence_chunking`、`_join_sentence_parts` | 检测文本是否含 CJK 字符，以选择不同处理规则     |
| `count_text_tokens`                   | `split_voice_clone_text`、`split_text_by_token_budget`        | 实际 encode 并返回 token 数，用于预算判断 |
| `split_text_by_token_budget`          | `split_voice_clone_text`                                     | 二分查找最长合法前缀并在自然边界处截断（第三层兜底）   |

### 2.3 音频编码支路（`resolve_prompt_audio_codes` → 各辅助函数）

| 函数                       | 调用方                          | 说明                                              |
| ------------------------ | ---------------------------- | ----------------------------------------------- |
| `encode_reference_audio` | `resolve_prompt_audio_codes` | 调用 ONNX `codec_encode` Session 将参考音频编码为离散 token |
| `_load_reference_audio`  | `encode_reference_audio`     | 加载音频文件并重采样/转换声道数，输出符合 codec 要求的 numpy 张量        |

## 3 非调用链函数（不参与单次 Stream Generate 主路径）

| 函数                        | 说明                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `_ensure_text_normalizer` | 仅被 `prepare_synthesis_text` 调用，用于文本正则化，stream 路径跳过文本正则化                                                                                                |
| `prepare_synthesis_text`  | 仅被非 stream 路径的 `synthesize` 调用                                                                                                                         |
| `decode_full_audio_safe`  | 全量 codec 解码；stream 路径使用 `CodecStreamingDecodeSession.run_frames` 逐帧解码                                                                                  |
| `synthesize_single_chunk` | 仅被 `synthesize` 调用；stream generate 路径中 `app_onnx.py` 的 `_worker` 绕过此函数，直接调用 `encode_text` / `build_voice_clone_request_rows` / `generate_audio_frames` |
| `synthesize`              | 被 `app_onnx.py` 的非 stream `OnnxNanoTTSServiceAdapter.synthesize` 调用，不在 stream 路径上                                                                      |

## 4 stream generate 路径在本文件的完整调用树

```
app_onnx.py  OnnxNanoTTSServiceAdapter.__init__
  └─ OnnxTtsRuntime.__init__                         [入口]
       ├─ ensure_browser_onnx_model_dir              [内部]
       │    ├─ _resolve_model_dir_path               [内部]
       │    ├─ _default_model_dir_requested          [内部]
       │    ├─ _find_manifest_path                   [内部]
       │    └─ _download_default_browser_onnx_assets [内部]
       │         ├─ _snapshot_download_repo          [内部]
       │         └─ _normalize_download_layout       [内部]
       │              ├─ _find_directory_with_required_names [内部]
       │              │    └─ _directory_contains_all        [内部]
       │              └─ _promote_directory_contents         [内部]
       └─ (OrtCpuRuntime.__init__ 创建所有 ONNX Session)

app_onnx.py  synthesize_stream._worker
  ├─ runtime.resolve_prompt_audio_codes              [入口]
  │    └─ encode_reference_audio                     [内部]
  │         └─ _load_reference_audio                 [内部]
  ├─ runtime.split_voice_clone_text                  [入口]
  │    ├─ _prepare_text_for_sentence_chunking        [内部]
  │    │    └─ _contains_cjk                         [内部]
  │    ├─ _split_text_by_punctuation                 [内部]
  │    ├─ _join_sentence_parts                       [内部]
  │    │    └─ _contains_cjk                         [内部]
  │    ├─ count_text_tokens                          [内部]
  │    └─ split_text_by_token_budget                 [内部]
  │         └─ count_text_tokens                     [内部]
  ├─ runtime.encode_text            (每 chunk)       [入口]
  ├─ (runtime.build_voice_clone_request_rows → ort_cpu_runtime.py)
  ├─ (runtime.generate_audio_frames → ort_cpu_runtime.py)
  ├─ runtime.estimate_voice_clone_inter_chunk_pause_seconds [入口]
  ├─ _merge_audio_channels          (直接 import)    [入口]
  ├─ _concat_waveforms              (直接 import)    [入口]
  └─ _write_waveform_to_wav         (直接 import)    [入口]
```

---

## 5 stream generate 路径的内存积累分析（`_worker` 视角）

### 5.1 `_worker` 的调用粒度

`_worker` 是**一次 HTTP 请求启动一次**的后台线程（由 `synthesize_stream` 创建）。所有 text chunk 均在**同一个 `_worker` 调用**内的 `for chunk_index, chunk_text in enumerate(text_chunks)` 循环中串行处理，不存在"每个 chunk 启动一个 worker"的情况。因此"`_worker` 生命周期"与"一次完整请求的生命周期"等价。

### 5.2 路径上三个内存积累点

以下三处在 `_worker` 内持续占用内存，但**均随 `_worker` 退出而变为可回收状态**，不跨请求泄漏。

#### 积累点 1：`all_generated_frames`（死代码，完全无用）

```python
all_generated_frames: list[list[int]] = []          # _worker 开头声明

for chunk_index, chunk_text in enumerate(text_chunks):
    generated_frames = self.runtime.generate_audio_frames(...)
    all_generated_frames.extend(generated_frames)   # 每个 chunk 追加

# 末尾 _format_result_payload 根本不接收 all_generated_frames
_safe_put({"type": "result", **self._format_result_payload(waveform=waveform, ...)})
```

`all_generated_frames` 随总音频帧数线性增长，但最终 result payload 完全不使用它，是纯粹的死代码积累。

**可回收时机**：`_worker` 函数退出（所有 chunk 处理完毕 + sentinel 入队后返回）。整个请求期间持续存活。

#### 积累点 2：`_emit_waveform` 内的双份 PCM 拷贝

```python
def _emit_waveform(waveform, *, is_pause):
    emitted_chunks.append(np.asarray(waveform, dtype=np.float32))   # 拷贝 1
    _safe_put({
        "waveform_numpy": np.asarray(waveform, dtype=np.float32),   # 拷贝 2
        ...
    })
```

每次发出一段音频，同一份 PCM 数据会产生两个独立 numpy 对象。

**可回收时机**：

- 拷贝 1（存入 `emitted_chunks`）：对 chunk 0…N-2，下一轮循环体执行 `emitted_chunks = []` 时旧列表引用归零，随即可回收；对最后一个 chunk，要等 `_worker` 退出。
- 拷贝 2（存入队列 dict）：消费端（`synthesize_stream` 的 `while True: event_queue.get()` 循环）处理下一个 item 时，前一个 dict 引用归零，随即可回收。正常情况下释放较及时；若队列积压满（`maxsize=128`），则最多 128 份并存。

#### 积累点 3：`all_waveforms` + 最终拼接的 2× 峰值

```python
all_waveforms: list[np.ndarray] = []          # _worker 开头声明，持续追加

for ...:
    chunk_waveform = _concat_waveforms(emitted_chunks)
    all_waveforms.append(chunk_waveform)       # 每 chunk 完成后追加

# 所有 chunk 结束后：
waveform = _concat_waveforms(all_waveforms)    # ← 与 all_waveforms 同时存活，达到 2× 峰值
output_path = _write_waveform_to_wav(...)      # 内部还产生 clipped + pcm16 中间数组
```

在 `_worker` 末尾，`all_waveforms`（各 chunk 波形列表）和 `waveform`（最终拼接大数组）同时存活，内存中同时持有约 **2× 总音频 PCM 数据**。这对应日志中 `chunk N done` → `finally exit` 阶段的 proc_rss 跳升（约 +56 MB）。

**可回收时机**：`_worker` 函数退出时。`_format_result_payload` 里的 `"waveform_numpy"` 会再持有一个引用进队列，该引用在消费端处理 result item 后才释放。

### 5.3 生命周期汇总

| 内存对象                                 | 可回收时机                                  | 存活跨度                   |
| ------------------------------------ | -------------------------------------- | ---------------------- |
| `all_generated_frames`（全部帧，死代码）      | `_worker` 函数退出                         | 整个请求                   |
| `emitted_chunks` 小段数组（chunk 0…N-2）   | 下一 chunk 循环体执行 `emitted_chunks = []` 时 | 相邻两个 chunk 之间          |
| `emitted_chunks` 小段数组（最后 chunk）      | `_worker` 函数退出                         | 请求尾段                   |
| 队列 dict 的 `waveform_numpy`（audio 事件） | 消费端处理下一个 item 时                        | 队列持有期间（正常情况下很短）        |
| `all_waveforms`（各 chunk 拼接后的波形）      | `_worker` 函数退出                         | 整个请求                   |
| `waveform`（最终大数组）+ result dict 引用    | `_worker` 退出后，消费端处理 result item 时      | 整个请求 + result item 消费前 |

### 5.4 与跨请求内存增长的关系

**这三个积累点不是跨请求内存增长的直接原因。** 它们的生命周期均被限制在单次 `_worker` 内，`_worker` 退出后 Python 层引用归零，GC 可以回收。

跨请求内存只涨不降（日志里多次请求后 `proc_rss` / `sys_used` 持续走高）的真正原因是 **ORT Arena 高水位策略**（C++ 层行为，详见同目录 `app_onnx内存泄露.md` §5 / §7.3）：每次请求触达新的内存需求峰值时，Arena 向 OS 申请的内存块不会归还，永久保留为高水位。

这三个积累点的实际影响是：**抬高单次请求内的 Python heap 峰值**，可能间接促使 ORT Arena 在更高的并发内存压力下申请更大块，从而间接推高高水位；但它们本身不构成跨请求泄漏。

---

# `app_onnx.py` 函数调用分类（仅初始化和 Stream Generate 调用链视角）

以下分类对应 `app_onnx.py` 中各函数在 **服务初始化** 和 **ONNX 流式生成（stream generate）** 两条路径上的角色：

- **调用链入口**：被其他文件（`app.py`）直接调用，进入本文件的函数
- **调用链内部**：仅在本文件内部（或闭包内部）被其他函数调用，支撑入口函数完成工作
- **非调用链-初始化**：仅在服务启动 / 初始化 / 预热阶段调用，不在任何 per-request 路径上
- **非调用链**：存在于代码中但不在 stream generate 路径上（专属非流式路径、UI 渲染或纯接口兼容）

---

## 1 调用链入口函数（被其他文件调用）

| 函数                                            | 调用方                                                                   | 说明                                                                                                                                                              |
| --------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OnnxNanoTTSServiceAdapter.__init__`          | `main()`（本文件启动时）                                                      | 创建唯一的 ONNX 适配器实例；内部创建 `OnnxTtsRuntime`，进而创建全部 ONNX InferenceSession 和 `CodecStreamingDecodeSession`                                                             |
| `OnnxRequestRuntimeManager.iter_with_runtime` | `app.py` 的 `_run_streaming_job`（每次 stream 请求）                         | **app.py 进入本文件的真正入口**；通过 `_locked_runtime` 获取运行时实例和排他执行锁，在内部执行 `factory(runtime)` 触发 `synthesize_stream`，并将每个 yield item 逐一转发给调用方                               |
| `OnnxNanoTTSServiceAdapter.synthesize_stream` | `iter_with_runtime` 内部的 `factory(runtime)`（`factory` 闭包由 `app.py` 传入） | **stream generate 推理入口**；虽然机械调用方是同文件的 `iter_with_runtime`，但触发它的 `factory` 闭包定义在 `app.py` 中，实质上仍属于跨文件驱动；使用 `threading.Thread + queue.Queue` 将推理线程与上层 yield 生成器解耦 |

`app.py` 并不直接调用 `synthesize_stream`，而是将 `factory` 闭包传入 `iter_with_runtime`，完整调用路径如下：

```
app.py  _run_streaming_job
    ↓ 调用
OnnxRequestRuntimeManager.iter_with_runtime(factory=<闭包>)
    ↓ 在内部执行 factory(runtime)，factory 是 app.py 传进来的闭包，
      闭包内容是 runtime.synthesize_stream(...)
OnnxNanoTTSServiceAdapter.synthesize_stream(...)
```

---

## 2 调用链内部函数（仅文件内部调用）

### 2.1 `OnnxNanoTTSServiceAdapter` 成员方法

| 函数                          | 调用方                                   | 说明                                                                                                               |
| --------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `_apply_generation_options` | `synthesize_stream._worker`           | 将本次请求的推理超参数（`sample_mode`、温度、`top_p/k` 等）写入 `runtime.manifest["generation_defaults"]`，供 `ort_cpu_runtime` 在推理时读取 |
| `_resolve_sample_mode`      | `_apply_generation_options`、`_worker` | 将外部传入的 `sample_mode / do_sample` 组合规范化为 `"greedy"` / `"fixed"` / `"full"` 三种枚举值                                  |
| `_format_result_payload`    | `synthesize_stream._worker` 末尾        | 将完整生成结果打包为 result event dict（wav 路径、完整波形、耗时等元数据），发送给 `app.py` 消费                                                 |

### 2.2 `synthesize_stream` 内部闭包（按调用顺序）

| 闭包函数              | 调用方                                                      | 说明                                                                                                                                       |
| ----------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `_safe_put`       | `_worker`、`_emit_waveform`                               | 线程安全地将 item 放入 `event_queue`；队列满时自旋等待（最多 0.5 s/次），收到停止信号后静默丢弃并返回 `False`                                                                 |
| `_worker`         | `threading.Thread`（在 `synthesize_stream` 中启动）            | 在独立线程中执行全部推理工作：解析超参数 → 编码 Prompt → 切分文本 → 逐 chunk prefill/decode → codec 解码 → 音频入队 → 写 WAV → 发送 result                                   |
| `_emit_waveform`  | `_decode_pending`（正常音频）、`_worker`（停顿静音）                  | 将一段已解码的 PCM 波形发送给上层消费者；追踪 `emitted_samples_total` 以计算 `lead_seconds`（音频超前播放量）                                                            |
| `_decode_pending` | `_on_frame`（force=False）、`_worker` chunk 结束时（force=True） | 将 `pending_decode_frames` 中积累的声学 token 批量送入 `codec_streaming_session.run_frames` 解码为 PCM；`force=False` 时按动态预算批处理，`force=True` 时强制处理所有剩余帧 |
| `_on_frame`       | `ort_cpu_runtime.generate_audio_frames`（每生成一帧 token 触发）  | 将新帧追加到 `pending_decode_frames` 缓冲，并以非强制模式尝试批量解码                                                                                          |

### 2.3 `OnnxRequestRuntimeManager` 成员方法

| 函数                      | 调用方                                     | 说明                                                                                           |
| ----------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------- |
| `_locked_runtime`       | `iter_with_runtime`、`call_with_runtime` | 解析 `cpu_threads` → 构建/获取运行时 → 持有 `_execution_lock`（保证同一时刻只有一个推理任务在运行）                        |
| `_resolve_cpu_threads`  | `_locked_runtime`                       | 将外部传入的 `cpu_threads` 参数规范化为有效正整数                                                             |
| `_build_runtime_locked` | `_locked_runtime`                       | 在已持有 `_lock` 的情况下返回目标运行时；当前实现始终复用 `default_runtime`，忽略 `cpu_threads` 差异以避免加载第二个 ONNX Session |

### 2.4 模块级辅助函数

| 函数            | 调用方                                                                       | 说明                                                |
| ------------- | ------------------------------------------------------------------------- | ------------------------------------------------- |
| `_log_memory` | `_worker`（start / chunk N done / finally exit 三处）、`_build_runtime_locked` | 记录进程 RSS 和系统内存用量，用于追踪 stream generate 过程中的内存阶梯式变化 |

---

## 3 非调用链-初始化函数（服务启动 / 预热阶段调用）

| 函数/类                               | 调用方                                  | 说明                                                                                                                              |
| ---------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `_CpuDeviceInfo`                   | `OnnxNanoTTSServiceAdapter.__init__` | 在初始化时实例化，填充 `self.device` 属性以兼容 `app.py` 对 `runtime.device.type` 的访问                                                            |
| `OnnxNanoTTSServiceAdapter.warmup` | `app.py` 的 `WarmupManager`（服务启动后触发）  | 触发一次完整的非流式推理，目的是让 ONNX Session 完成首次 JIT 编译 / Arena 预分配，降低首请求冷启动延迟                                                               |
| `_render_index_html_onnx`          | `app.py`（用户访问首页 `GET /` 时）           | 将 ONNX 版本的 UI 差异注入前端页面（标题、采样模式下拉框、disabled 样式等）；函数本身在 `main()` 中通过 `legacy_app._render_index_html = _render_index_html_onnx` 注入 |
| `parse_args`                       | `main()`（启动时调用一次）                    | 解析 CLI 启动参数（`--model-dir`、`--host`、`--port`、`--cpu-threads` 等）                                                                  |
| `main`                             | `__main__`（命令行直接运行）                  | 程序入口：解析参数 → 创建运行时 → 启动预热 → 注入全局引用 → 启动 uvicorn 服务                                                                               |

---

## 4 非调用链函数（不在 stream generate 路径上）

| 函数                                                               | 说明                                                                                                                                     |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `OnnxNanoTTSServiceAdapter.get_model`                            | 兼容接口，返回 `self`；stream generate / 非流式推理均不经过此方法                                                                                          |
| `OnnxNanoTTSServiceAdapter.split_voice_clone_text`（适配器包装）        | 对 `runtime.split_voice_clone_text` 的公开包装，供 `app.py` 直接调用；stream generate 路径中 `_worker` 绕过此方法直接调用 `self.runtime.split_voice_clone_text` |
| `OnnxNanoTTSServiceAdapter.synthesize`                           | 同步（非流式）合成路径，被 `app.py` 的非流式 `/api/generate` 接口调用；stream generate 使用 `synthesize_stream`                                                |
| `OnnxRequestRuntimeManager.normalize_requested_execution_device` | 兼容接口，始终返回 `"cpu"`；`app.py` 在设备类型解析时可能调用                                                                                                |
| `OnnxRequestRuntimeManager.is_dedicated_cpu_request`             | 兼容接口，始终返回 `False`；`app.py` 在设备路由判断时可能调用                                                                                                |
| `OnnxRequestRuntimeManager.is_cpu_runtime_loaded`                | 状态查询接口，供 `app.py` 在初始化或健康检查时使用                                                                                                         |
| `OnnxRequestRuntimeManager.resolve_runtime`                      | 兼容接口，始终返回 `(default_runtime, "cpu")`；stream generate 路径使用 `iter_with_runtime`                                                          |
| `OnnxRequestRuntimeManager.call_with_runtime`                    | 同步（非流式）请求路径使用；stream generate 路径使用 `iter_with_runtime`                                                                                 |

---

## 5 stream generate 路径在本文件的完整调用树

```
app.py  _run_streaming_job
  └─ OnnxRequestRuntimeManager.iter_with_runtime          [入口]
       └─ _locked_runtime                                  [内部]
            ├─ _resolve_cpu_threads                        [内部]
            └─ _build_runtime_locked                       [内部]
                 └─ _log_memory                            [内部]

app.py  iter_with_runtime → factory(runtime)
  └─ OnnxNanoTTSServiceAdapter.synthesize_stream          [入口]
       ├─ _safe_put                                        [内部·闭包]
       └─ _worker（threading.Thread）                      [内部·闭包]
            ├─ _log_memory                                 [内部]
            ├─ _resolve_sample_mode                        [内部]
            ├─ _apply_generation_options                   [内部]
            ├─ runtime.resolve_prompt_audio_codes          → onnx_tts_runtime.py
            ├─ runtime.split_voice_clone_text              → onnx_tts_runtime.py
            ├─ for chunk_index, chunk_text in text_chunks:
            │    ├─ runtime.encode_text                    → onnx_tts_runtime.py
            │    ├─ runtime.build_voice_clone_request_rows → ort_cpu_runtime.py
            │    ├─ codec_streaming_session.reset          → ort_cpu_runtime.py
            │    ├─ _emit_waveform                         [内部·闭包]
            │    │    └─ _safe_put
            │    ├─ _decode_pending                        [内部·闭包]
            │    │    ├─ _resolve_stream_decode_frame_budget → ort_cpu_runtime.py
            │    │    ├─ codec_streaming_session.run_frames  → ort_cpu_runtime.py
            │    │    └─ _emit_waveform
            │    ├─ _on_frame（回调，由 generate_audio_frames 触发）[内部·闭包]
            │    │    └─ _decode_pending(force=False)
            │    ├─ runtime.generate_audio_frames          → ort_cpu_runtime.py
            │    ├─ _decode_pending(force=True)
            │    ├─ codec_streaming_session.reset          → ort_cpu_runtime.py
            │    ├─ _concat_waveforms                      → onnx_tts_runtime.py
            │    ├─ _log_memory
            │    └─ _emit_waveform（chunk 间静音停顿）
            ├─ _concat_waveforms                           → onnx_tts_runtime.py
            ├─ _write_waveform_to_wav                      → onnx_tts_runtime.py
            ├─ _format_result_payload                      [内部]
            └─ _safe_put（result event / error event / sentinel None）
```

---

## 6 warmup 覆盖范围的遗漏点

`OnnxNanoTTSServiceAdapter.warmup` 调用的是 `self.synthesize()`（非流式路径），而非 `self.runtime.warmup()`（`OrtCpuRuntime` 自带的底层预热）。两者对 ONNX Session 的覆盖范围不同：

| ONNX Session                            | 路径              | `OnnxNanoTTSServiceAdapter.warmup` | `OrtCpuRuntime.warmup` |
| --------------------------------------- | --------------- | ---------------------------------- | ---------------------- |
| `prefill`                               | 两条路径共用          | ✅ 已预热                              | ✅ 已预热                  |
| `local_cached_step` / `local_decoder` 等 | 两条路径共用          | ✅ 已预热                              | ✅ 已预热                  |
| `codec_encode`                          | 两条路径共用          | ✅ 已预热                              | ✅ 已预热                  |
| `codec_decode`（全量解码）                    | **仅非流式路径**      | ✅ 已预热                              | ✅ 已预热                  |
| `codec_decode_step`（逐帧流式解码）             | **仅 stream 路径** | ❌ 未预热                              | ✅ 已预热                  |

`OrtCpuRuntime.warmup` 在末尾显式执行了 `codec_streaming_session.run_frames`（第 519 行），覆盖了 `codec_decode_step` Session：

```python
# ort_cpu_runtime.py  warmup()
empty_frames = [([0] * int(self.manifest["tts_config"]["n_vq"]))]
self.decode_full_audio(empty_frames)            # 预热 codec_decode
self.codec_streaming_session.reset()
self.codec_streaming_session.run_frames(empty_frames)  # 预热 codec_decode_step
self.codec_streaming_session.reset()
```

但 `OnnxNanoTTSServiceAdapter.warmup` 没有调用 `self.runtime.warmup()`，导致 **`codec_decode_step` Session 在服务启动后仍然是冷的**。（且这个代码，没有任何地方被调用到，是个死代码）

**影响：** 第一次 `synthesize_stream` 请求的 codec 流式解码阶段（`_decode_pending` → `codec_streaming_session.run_frames`）会触发 `codec_decode_step` 的首次 JIT 编译 / Arena 预分配，产生额外的首帧延迟。后续请求不受影响。

所有 Session 共用同一个 `OrtCpuRuntime` 实例内的 ORT Arena 内存池，因此非流式 warmup 对 TTS prefill/decode 部分的 Arena 预分配是有效的，遗漏的只有 `codec_decode_step` 这一个 Session。

---

## 7 单实例、串行处理设计

`OnnxRequestRuntimeManager` 在整个服务生命周期内只维护一个 `OnnxNanoTTSServiceAdapter` 实例（`default_runtime`）。`_build_runtime_locked` 无论请求传入何种 `cpu_threads` 参数，始终返回同一个实例：

```python
def _build_runtime_locked(self, cpu_threads: int) -> OnnxNanoTTSServiceAdapter:
    if cpu_threads != self.default_runtime.thread_count:
        logging.warning(
            "OnnxRequestRuntimeManager: ignoring cpu_threads=%d (default=%d) "
            "to avoid loading a second ONNX session; reusing default runtime.",
            ...
        )
    return self.default_runtime   # 始终返回同一实例
```

`_execution_lock` 在 `_locked_runtime` 的 `with` 块中持有，生命周期覆盖整个 `iter_with_runtime` 迭代过程（即一次完整的 stream generate 请求从开始到最后一帧 yield 完毕）。多个并发请求的实际执行时序如下：

```
用户A 请求 ──→ 获得 _execution_lock ──→ 推理中（数秒至数十秒）
                                              ↓ 完成后释放锁
用户B 请求 ──→ 阻塞等待 _execution_lock ──────→ 才能开始推理
用户C 请求 ──→ 阻塞等待 _execution_lock ────────────────────→ 再等
```

**这是有意为之的设计，而非缺陷。** 原因是当前 runtime 对象内存在以下共享可变状态，无法安全地被多个并发请求同时使用：

| 共享状态                                | 位置              | 并发时的问题                                       |
| ----------------------------------- | --------------- | -------------------------------------------- |
| `codec_streaming_session`（KV Cache） | `OrtCpuRuntime` | 不同请求的逐帧解码会互相覆盖 KV Cache                      |
| `manifest["generation_defaults"]`   | `OrtCpuRuntime` | `_apply_generation_options` 写入的温度、采样模式等会互相覆盖 |
| `self.runtime.rng`                  | `OrtCpuRuntime` | 随机数种子被后来请求覆盖，影响采样结果                          |

若要支持真正的并发，需要为每个并发槽位创建独立的 runtime 实例（或至少独立的 `codec_streaming_session` + 独立的配置副本），当前代码的 `_cpu_runtimes: dict[int, OnnxNanoTTSServiceAdapter]` 字典结构为此预留了扩展空间，但实际未实现。
