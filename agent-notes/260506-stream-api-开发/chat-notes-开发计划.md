# simple_moss 剥离与本地五接口协议 — 存档

参考笔记：`chat-notes-开发思路.md` 中章节「保留原前端 stream generate 多接口协议」。

---

## 一、原始需求（用户要求，原样整理）

### 目标

把 `app_onnx.py`、`app.py`、`onnx_tts_runtime.py`、`ort_cpu_runtime.py` 进行简化，并把代码放到类似于 `simple_moss` 的目录里，单独维护。

### 涉及的业务

1. 以 `app_onnx.py` 为入口的初始化、warmup 流程；
2. 以 `app_onnx.py` / `app.py` 为入口的 stream generate 流程；
3. 只使用内置（preset）voice；
4. 使用 ONNX 模型接口，且 ONNX 内部已改为 GPU 运行方式（需要保留）。

### 约束

1. 只保留所涉及业务的代码、函数，去掉其他无关的流程代码；
2. 只筛选代码，不修改代码（也就是不要去改原有的逻辑）；
3. 原先的 app 的 5 个 HTTP 接口，改为本地接口，但接口含义保持不变，包括：
   - **start**：创建任务  
   - **audio**：消费音频流  
   - **status**：异步轮询进度  
   - **result**：获取最终结果  
   - **close**：取消或清理任务  

### 当时要求的输出结构

1. 每个文件，需要保留哪些函数、类不变；
2. 每个文件，需要完全去掉哪些函数、类；
3. 每个文件，需要修改哪些函数（修改指的是，去掉里面某些用不到的分支逻辑，但不修改仍然需要的逻辑）；
4. 每个文件，需要新增哪些函数。

---

## 二、修改计划（Agent 整理，按文件归档）

### 结论目录建议

建议新目录保持四层：

- `simple_moss/simple_app_onnx.py`：入口、初始化、warmup、ONNX stream adapter。
- `simple_moss/api_facade.py`：本地 `start` / `audio` / `status` / `result` / `close` 五接口语义。
- `simple_moss/simple_onnx_tts_runtime.py`：TTS 业务预处理。
- `simple_moss/simple_ort_cpu_runtime.py`：ONNX 推理和 codec stream decode，保留当前 CUDA provider。

---

### `app_onnx.py`

**需要保留不变：**

- `_log_memory`
- `_CpuDeviceInfo`
- `OnnxNanoTTSServiceAdapter.__init__`
- `OnnxNanoTTSServiceAdapter.warmup`
- `OnnxNanoTTSServiceAdapter._apply_generation_options`
- `OnnxNanoTTSServiceAdapter._resolve_sample_mode`
- `OnnxNanoTTSServiceAdapter.synthesize_stream`
- `FIXED_BUILTIN_VOICE = "Lingyu"` 这类固定 preset voice 机制

**需要完全去掉：**

- `get_model`（若新的 warmup 不再复用 `app.py.WarmupManager` 兼容接口）—— 条件已确认，可以去掉。
- `split_voice_clone_text` 包装方法（除非 facade 仍想通过 adapter 预先算 chunks）
- `synthesize` 非流式接口
- `_render_index_html_onnx`
- `parse_args`、`main`
- `uvicorn`、`legacy_app` 注入、HTML 替换、VSCODE root path、`share` 参数相关逻辑
- `OnnxRequestRuntimeManager` 中的多 runtime / cpu_threads 缓存逻辑可整体不迁移；新 facade 用单实例锁即可

**需要修改 / 收窄：**

- `OnnxNanoTTSServiceAdapter.__init__`：去掉仅 Web demo 兼容的可选项可以斟酌，但不改 `OnnxTtsRuntime(...)` 初始化逻辑
- `warmup`：保留当前调用 `self.runtime.warmup(voice_name=voice_name)` 的流程，去掉历史注释块与无用依赖
- `synthesize_stream`：固定走 `FIXED_BUILTIN_VOICE`，去掉 `voice` / `prompt_audio_path` 动态分支；保留 `_worker`、`event_queue`、`_stop_event`、`_decode_pending`、`_on_frame`、`generate_audio_frames`、`codec_streaming_session.run_frames`；可去掉 `all_generated_frames`（死代码）；若新 `result()` 不需要 WAV/base64，可去掉 `_write_waveform_to_wav` 与磁盘路径结果；否则保留最终 waveform 汇总
- `OnnxRequestRuntimeManager`：不建议整体迁移；将 `_execution_lock` 思想迁到 `MossStreamFacade` 或 `MossStreamApiFacade`

**需要新增：**

- `create_default_adapter(config)` 或 `build_runtime(config)`：本地初始化入口，替代 `main()`
- `warmup_runtime(adapter)`：本地 warmup 包装，替代 `WarmupManager` 依赖
- 可选 `stream_events(...)`：薄包装 `adapter.synthesize_stream(...)`，供 facade 调用

---

### `app.py`

**需要保留不变：**

- `StreamingJob`
- `StreamingJobManager`
- `_format_run_status`
- `_format_stream_status`
- `_normalize_stream_chunk_index`
- `_audio_to_pcm16le_bytes`
- `_coerce_bool`
- `_run_streaming_job` 的核心逻辑
- `_put_stream_audio`

**需要完全去掉：**

- `DemoEntry`、`_load_demo_entries`
- `_resolve_vscode_root_path`
- `_render_index_html`、`_build_app`
- 所有 FastAPI route：`index`、`health`、`warmup_status`、`text_normalization_status`、`demo_prompt_audio`、`generate_stream_*`、`generate`
- `_resolve_demo_entry`、`_resolve_prompt_audio_request`
- `_persist_uploaded_prompt_audio`、`_sanitize_uploaded_prompt_filename`、`_format_uploaded_prompt_display_name`
- `_audio_to_wav_bytes`
- `_read_audio_file_base64`（若 `result()` 不再返回 WAV base64）
- `_maybe_delete_file`（若新版本不写临时上传与最终 WAV）
- `RequestRuntimeManager` 原 PyTorch 设备路由
- `WarmupManager` / `WarmupSnapshot`（若 warmup 在 `simple_app_onnx.py` 同步执行）

**需要修改 / 收窄：**

- `_run_streaming_job`：去掉 `prompt_audio_path` / `prompt_audio_display_path` / `prompt_audio_cleanup_path`；固定 preset 后 `_stream_factory` 传 `voice=None, prompt_audio_path=None`；去掉 `tts_max_batch_size` / `codec_max_batch_size`；去掉 `requested_execution_device` / `cpu_threads` / `_resolve_attn_for_runtime`；保留 event 消费、PCM 转换、状态更新、audio queue、result 写入、异常、sentinel
- `StreamingJob.final_result`：可保留旧含义；若不写 WAV/base64，结果可只含 `run_status`、`text_chunks`、`sample_rate`、`channels`、`audio_chunk_ranges`、`emitted_audio_seconds` 等
- 五接口不迁移为 HTTP，而迁移为本地方法

**需要新增：**

- `class MossStreamApiFacade`：`start(...)`、`audio(stream_id)`、`status(stream_id)`、`result(stream_id)`、`close(stream_id)`
- `class MossStreamFacade`：单次底层 stream generate，内部 `threading.Lock` 串行
- 可选：`StreamGenerateRequest` / `StreamStartResponse` 等 dataclass
- 可选：`run_local_stream_generate_test(...)` 走完整五接口协议

---

### `onnx_tts_runtime.py`

**需要保留不变：**

- `_resolve_model_dir_path`、`_default_model_dir_requested`、`_find_manifest_path`
- `_directory_contains_all`、`_find_directory_with_required_names`
- `_promote_directory_contents`、`_normalize_download_layout`
- `_snapshot_download_repo`、`_download_default_browser_onnx_assets`
- `ensure_browser_onnx_model_dir`
- `_contains_cjk`、`_prepare_text_for_sentence_chunking`、`_split_text_by_punctuation`、`_join_sentence_parts`
- `_merge_audio_channels`、`_concat_waveforms`
- `OnnxTtsRuntime.__init__`
- `encode_text`、`count_text_tokens`、`split_text_by_token_budget`、`split_voice_clone_text`
- `estimate_voice_clone_inter_chunk_pause_seconds`
- `resolve_prompt_audio_codes`

**需要完全去掉：**

- `_ensure_text_normalizer`、`prepare_synthesis_text`
- `_load_reference_audio`、`encode_reference_audio`
- `decode_full_audio_safe`、`synthesize_single_chunk`、`synthesize`
- `_write_waveform_to_wav`（若新 `result()` 不再返回最终 WAV/base64）
- 随上述删除：`torch`、`torchaudio`、`prepare_tts_request_texts`、`WeTextProcessingManager` 等不再需要的依赖

**需要修改 / 收窄：**

- `OnnxTtsRuntime.__init__`：去掉 `_text_normalizer_manager` 字段
- `resolve_prompt_audio_codes`：只保留内置 voice 分支，删除 `prompt_audio_path -> encode_reference_audio` 分支
- imports：随删除函数清理，不改核心算法
- 可选新增 `resolve_fixed_prompt_audio_codes()`，内部仍用 `list_builtin_voices()` 查 manifest

**需要新增：**

- `resolve_builtin_voice_prompt_audio_codes(voice: str | None) -> list[list[int]]`（收窄后的清晰入口，可选）
- 可选 `get_codec_audio_format() -> tuple[int, int]`（`sample_rate` / `channels`）

---

### `ort_cpu_runtime.py`

**需要保留不变：**

- `SAMPLE_MODE_GREEDY` / `FIXED` / `FULL`
- `_log_memory`、`_flatten3d_int32`、`_flatten2d_int32`、`_extract_last_hidden`
- `_normalize_sample_mode`、`_compute_stream_lead_seconds`、`_resolve_stream_decode_frame_budget`
- `CodecStreamingDecodeSession` 及 `__post_init__`、`reset`、`run_frames`
- `OrtCpuRuntime.__init__`、`_resolve_manifest_path`、`resolve_manifest_relative_path`
- `_session`、`_create_sessions`
- `list_builtin_voices`、`list_text_samples`（若 warmup 仍用 text sample）
- `OrtCpuRuntime.warmup`
- `build_text_rows`、`build_audio_prefix_rows`、`build_voice_clone_request_rows`
- `run_local_fixed_sampled_frame`、`generate_audio_frames`

**按实际采样模式决定是否保留：**

- 若只保留当前默认 fixed ONNX 路径：可去掉 `_argmax`、`_apply_repetition_penalty`、`_argmax_with_repetition_penalty`、`_softmax`、`_sample_assistant_text_token`、`_sample_audio_token` 及 `run_local_decoder`、`create_empty_local_cached_past`、`run_local_cached_step`、`run_local_greedy_frame`、`slice_audio_channel_logits`
- 若仍保留 full/greedy 含义：上述须保留

**需要完全去掉：**

- `_slice_channel_major_audio`、`decode_full_audio`
- 已注释的 `codec_encode` / `codec_decode_full` 相关 session 创建逻辑（保持文件瘦身时可不再携带大块注释）

**需要修改 / 收窄：**

- `_session`：保留 `CUDAExecutionProvider` 与 `ORT_ENABLE_BASIC`，不要退回 CPU
- `_create_sessions`：只加载 stream 必需：`prefill`、`decode`、`local_fixed_sampled_frame`、`codec_decode_step`；若保留 greedy/full 再加载对应 optional session
- `generate_audio_frames`：若只保留 fixed，可删 greedy/full/local_decoder fallback；fixed 分支、decode KV 更新、`on_frame`、GPU arena shrinkage 保持原逻辑
- `warmup`：保留 builtin voice + text sample + `generate_audio_frames` + `codec_streaming_session.run_frames`

---

### 最小新增本地接口语义（`api_facade.py`）

- **start(text, …)**：创建 `StreamingJob`，后台线程启动 `_run_streaming_job`，返回 `stream_id` 与初始状态。
- **audio(stream_id)**：阻塞消费 `job.audio_queue`，yield PCM-s16le bytes。
- **status(stream_id)**：返回 `job.snapshot()` 及 `status_text` / `stream_metrics`。
- **result(stream_id)**：未完成对应 202 语义；完成后返回最终结果、`text_chunks`、`audio_chunk_ranges` 等。
- **close(stream_id)**：设置 `is_closed`，塞 sentinel，删除 job。

**分工：** `app_onnx.py` 负责「怎么生成」；`onnx_tts_runtime.py` + `ort_cpu_runtime.py` 负责「模型推理」；`api_facade.py` 负责「本地模拟原五接口协议」；从 `app.py` 只筛出 job 管理与 stream job 消费逻辑，不保留 HTTP/UI/Demo 层。

---

*存档日期：2026-05-06*
