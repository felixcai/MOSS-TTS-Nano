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
- `simple_moss/simple_ort_cpu_runtime.py`：ONNX 推理和 codec stream decode，保留当前 CUDA provider。 —— 修改：这里的文件名，应该改为 simple_ort_gpu_runtime.py。

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
- `split_voice_clone_text` 包装方法（除非 facade 仍想通过 adapter 预先算 chunks） —— 保留原先的 /start 逻辑，这个方法需要保留。
- `synthesize` 非流式接口
- `_render_index_html_onnx`
- `parse_args`、`main`
- `uvicorn`、`legacy_app` 注入、HTML 替换、VSCODE root path、`share` 参数相关逻辑
- `OnnxRequestRuntimeManager` 中的多 runtime / cpu_threads 缓存逻辑可整体不迁移；新 facade 用单实例锁即可

#### `OnnxRequestRuntimeManager` 补充说明：`cpu_threads` 与单实例锁

当前 ONNX 版本里的 `cpu_threads` 不是严格意义上的代码内写死，但默认行为接近于固定为 1：

- `app_onnx.py` 的启动参数 `--cpu-threads` 默认值是 `1`。
- 启动时该值会传给 `OnnxNanoTTSServiceAdapter`，再传到 `OnnxTtsRuntime` / `OrtCpuRuntime`，最终用于创建 ONNX `InferenceSession` 时设置 `intra_op_num_threads`。
- ONNX Session 创建完成后，线程配置基本就随这个 runtime 固定下来。
- 原前端或 `/api/generate-stream/start` 请求里虽然仍可传 `cpu_threads`，但在当前 ONNX 版本中，请求级 `cpu_threads` 实际不会改变底层 runtime。

原因是 `OnnxRequestRuntimeManager._build_runtime_locked(...)` 当前为了避免重复加载第二套 ONNX Session，会忽略与默认 runtime 不同的 `cpu_threads` 请求，并继续复用 `default_runtime`。也就是说，类里看起来有 `_cpu_runtimes` 这样的 runtime 缓存字典，但当前实际行为并不是“按不同线程数创建多个 runtime”，而是“始终复用同一个默认 ONNX runtime”。

因此迁移到 `simple_moss` 时，不需要保留这套多 runtime / cpu_threads 缓存结构。新 facade 可以只创建一个 ONNX runtime 实例，并用一把执行锁保护它：

```text
MossStreamApiFacade.start(...)
  -> 创建 StreamingJob
  -> 启动后台线程
  -> 后台线程进入 MossStreamFacade.stream_generate(...)
  -> 获取 execution_lock
  -> 独占使用同一个 ONNX runtime 直到本次生成结束
```

这里的“单实例锁”不是指第二个请求不处理，而是指第二个请求排队等待：

- 第一个 `start` 创建任务 A，后台线程 A 拿到执行锁，开始真实 ONNX 推理并持续产出音频。
- 第二个 `start` 创建任务 B，接口仍可以立即返回新的 `stream_id`。
- 后台线程 B 启动后会阻塞在同一把执行锁上，等待任务 A 完成。
- 在等待期间，任务 B 的 `status` 可以返回已创建 / 等待中 / running 之类的状态，但 `audio` 暂时不会产出真实 PCM。
- 任务 A 完成释放锁后，任务 B 才进入真实推理，随后开始向自己的 audio queue 写入音频。

这与原代码行为一致：原来的 `OnnxRequestRuntimeManager.iter_with_runtime(...)` 内部也是通过 `_execution_lock` 保证同一时刻只有一个推理任务使用 ONNX runtime。区别只是：旧代码为了兼容 PyTorch 版 `app.py` 的设备路由和 `cpu_threads` 表单参数，保留了一层较复杂的 manager；新 `simple_moss` 可以把它收窄成“单 runtime + 单 execution lock + job queue/线程”的简单模型。

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
- `StreamingJob.final_result`：可保留旧含义；若不写 WAV/base64，结果可只含 `run_status`、`text_chunks`、`sample_rate`、`channels`、`audio_chunk_ranges`、`emitted_audio_seconds` 等 —— 已确认，不需要WAV/base64。
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
- `_write_waveform_to_wav`（若新 `result()` 不再返回最终 WAV/base64） —— 已经确认，不返回最终 WAV/base64。
- 随上述删除：`torch`、`torchaudio`、`prepare_tts_request_texts`、`WeTextProcessingManager` 等不再需要的依赖

**需要修改 / 收窄：**

- `OnnxTtsRuntime.__init__`：去掉 `_text_normalizer_manager` 字段
- `resolve_prompt_audio_codes`：只保留内置 voice 分支，删除 `prompt_audio_path -> encode_reference_audio` 分支
- imports：随删除函数清理，不改核心算法
- 可选新增 `resolve_fixed_prompt_audio_codes()`，内部仍用 `list_builtin_voices()` 查 manifest

**需要新增：**

- `resolve_builtin_voice_prompt_audio_codes(voice: str | None) -> list[list[int]]`（收窄后的清晰入口，可选）
- 可选 `get_codec_audio_format() -> tuple[int, int]`（`sample_rate` / `channels`）

#### 三个新增项的作用说明

- `resolve_fixed_prompt_audio_codes()`：固定 preset voice 专用入口。简化版只使用固定内置音色时，上层不再需要传 `voice` / `prompt_audio_path` 或保留动态分支；该方法内部可直接使用固定音色常量，再通过 `list_builtin_voices()` 从 manifest 查到对应 prompt audio codes。
- `resolve_builtin_voice_prompt_audio_codes(voice: str | None) -> list[list[int]]`：只支持内置 voice 的收窄入口，用来替代原来更宽的 `resolve_prompt_audio_codes(voice, prompt_audio_path)`。它不再处理 `prompt_audio_path -> encode_reference_audio` 的上传音频 / 克隆音色路径；如果 `voice is None`，可以回退到默认内置音色。
- `get_codec_audio_format() -> tuple[int, int]`：向 facade 暴露底层 codec 的 PCM 格式元数据，例如 `sample_rate` 和 `channels`。这样 `MossStreamApiFacade.start()` / `status()` / `result()` 不需要硬编码音频格式；如果底层 ONNX codec 输出格式变化，上层只需要读取 runtime 暴露的格式。

这三个新增项都属于“收窄后更清晰的薄封装”，不改变核心推理算法；主要目的是让 `api_facade.py` 与 runtime 的调用关系更直观。

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

- 若只保留当前默认 fixed ONNX 路径：可去掉 `_argmax`、`_apply_repetition_penalty`、`_argmax_with_repetition_penalty`、`_softmax`、`_sample_assistant_text_token`、`_sample_audio_token` 及 `run_local_decoder`、`create_empty_local_cached_past`、`run_local_cached_step`、`run_local_greedy_frame`、`slice_audio_channel_logits` —— 确认，只跑fixed ONNX。
- 若仍保留 full/greedy 含义：上述须保留 —— 不保留 full/greedy。

**需要完全去掉：**

- `_slice_channel_major_audio`、`decode_full_audio`
- 已注释的 `codec_encode` / `codec_decode_full` 相关 session 创建逻辑（保持文件瘦身时可不再携带大块注释）

**需要修改 / 收窄：**

- `_session`：保留 `CUDAExecutionProvider` 与 `ORT_ENABLE_BASIC`，不要退回 CPU
- `_create_sessions`：只加载 stream 必需：`prefill`、`decode`、`local_fixed_sampled_frame`、`codec_decode_step`；若保留 greedy/full 再加载对应 optional session —— 不保留 full/greedy。
- `generate_audio_frames`：若只保留 fixed，可删 greedy/full/local_decoder fallback；fixed 分支、decode KV 更新、`on_frame`、GPU arena shrinkage 保持原逻辑 —— 确认，只跑fixed ONNX。
- `warmup`：保留 builtin voice + text sample + `generate_audio_frames` + `codec_streaming_session.run_frames`

---

### 最小新增本地接口语义（`api_facade.py`）

- **start(text, …)**：创建 `StreamingJob`，后台线程启动 `_run_streaming_job`，返回 `stream_id` 与初始状态。
- **audio(stream_id)**：阻塞消费 `job.audio_queue`，yield PCM-s16le bytes。
- **status(stream_id)**：返回 `job.snapshot()` 及 `status_text` / `stream_metrics`。
- **result(stream_id)**：未完成对应 202 语义；完成后返回最终结果、`text_chunks`、`audio_chunk_ranges` 等。
- **close(stream_id)**：设置 `is_closed`，塞 sentinel，删除 job。

**分工：** `app_onnx.py` 负责「怎么生成」；`onnx_tts_runtime.py` + `ort_cpu_runtime.py` 负责「模型推理」；`api_facade.py` 负责「本地模拟原五接口协议」；从 `app.py` 只筛出 job 管理与 stream job 消费逻辑，不保留 HTTP/UI/Demo 层。

### `simple_moss` 流式类关系说明

当前 `simple_moss` 中与本地流式调用相关的核心类可以分成三层：

```text
最外层本地 API 层
MossStreamApiFacade
  ├─ 持有 StreamingJobManager
  └─ 持有 MossStreamFacade

任务状态层
StreamingJobManager
  └─ 管理多个 StreamingJob

推理执行层
MossStreamFacade
  └─ 持有 OnnxNanoTTSServiceAdapter
        └─ 持有 OnnxTtsRuntime
              └─ 持有 OrtCpuRuntime / ONNX sessions
```

- `OnnxNanoTTSServiceAdapter`：真正执行 TTS 推理适配的类，内部持有 `OnnxTtsRuntime`。它负责初始化 ONNX runtime、执行 warmup、拆分文本 chunk、构造 request rows、调用 `generate_audio_frames`、调用 codec streaming decode，并通过 `synthesize_stream(...)` 持续产出 `audio` / `result` 事件。
- `MossStreamFacade`：推理串行锁包装层，持有 `OnnxNanoTTSServiceAdapter` 和 `_execution_lock`。它的职责不是生成音频，而是保证同一时刻只有一个 stream generation 真正进入底层 ONNX runtime。
- `StreamingJob`：一次流式请求的任务状态对象，保存 `stream_id`、`audio_queue`、`state`、`run_status`、`error`、音频进度、`text_chunks`、`final_result` 等信息。它本身不执行推理，只记录一次请求的状态和待消费音频队列。
- `StreamingJobManager`：`StreamingJob` 的线程安全注册表，维护 `stream_id -> StreamingJob` 的映射，负责 `create()`、`get()`、`close()`、`delete()`。
- `MossStreamApiFacade`：对外暴露的本地五接口门面，提供 `start(text)`、`audio(stream_id)`、`status(stream_id)`、`result(stream_id)`、`close(stream_id)`。它一边通过 `StreamingJobManager` 管理任务，一边通过 `MossStreamFacade` 把生成请求交给底层 adapter。

一次完整调用流程：

```text
1. 调用 MossStreamApiFacade.start(text)
2. StreamingJobManager.create() 创建 StreamingJob，返回 stream_id
3. MossStreamApiFacade 启动后台线程 _run_streaming_job(job, stream_facade, params)
4. 后台线程调用 MossStreamFacade.stream_generate(...)
5. MossStreamFacade 获取 _execution_lock，独占底层 ONNX runtime
6. MossStreamFacade 调用 OnnxNanoTTSServiceAdapter.synthesize_stream(...)
7. adapter 生成 audio/result 事件
8. _run_streaming_job 将 audio 事件转为 PCM bytes，写入 job.audio_queue
9. 调用方通过 MossStreamApiFacade.audio(stream_id) 消费 PCM bytes
10. result 事件到达后，_run_streaming_job 写入 job.final_result 并向 audio_queue 放入 None 结束标记
11. 调用方通过 result(stream_id) 获取最终结果，通过 close(stream_id) 清理任务
```

一句话总结：`OnnxNanoTTSServiceAdapter` 负责“生成”；`MossStreamFacade` 负责“串行化生成”；`StreamingJob` 负责“保存一次任务的状态和音频队列”；`StreamingJobManager` 负责“按 `stream_id` 管理任务”；`MossStreamApiFacade` 负责“把这些能力包装成本地五接口”。

#### 当前 `simple_moss` 会创建的 ONNX sessions

当前 `simple_moss/simple_ort_gpu_runtime.py` 的 `_create_sessions()` 只会创建 4 个 `onnxruntime.InferenceSession`：

```text
OrtCpuRuntime._create_sessions()
  ├─ sessions["prefill"]              -> moss_tts_prefill.onnx
  ├─ sessions["decode"]               -> moss_tts_decode_step.onnx
  ├─ sessions["local_fixed_sampled_frame"]
  │                                    -> moss_tts_local_fixed_sampled_frame.onnx
  └─ sessions["codec_decode_step"]     -> codec 目录中的 decode_step onnx
```

这 4 个 session 的职责：

- `prefill`：第一次把完整 prompt / text / reference audio codes 输入 TTS global transformer，生成初始 `global_hidden` 和 KV cache。
- `decode`：每生成一帧声学 token 后，更新 global transformer 的 `global_hidden` 和 KV cache，为下一帧生成推进上下文。
- `local_fixed_sampled_frame`：当前 fixed 模式下的单帧声学 token 生成器，一次生成一整帧 audio codebook token。
- `codec_decode_step`：将声学 token 帧流式解码为真实音频 waveform。

当前 `simple_moss` 不会创建以下 session：

- `local_decoder`
- `local_cached_step`
- `local_greedy_frame`
- `codec_encode`
- `codec_decode_full`

虽然 `tts_browser_onnx_meta.json` 中仍记录了 `local_decoder`、`local_cached_step`、`local_fixed_sampled_frame`，但简化后的 `_create_sessions()` 只读取并加载 `local_fixed_sampled_frame`，因此初始化阶段主要内存增长对应的就是上述 4 个 session。

#### `prefill` 与 `decode` 是否可以省略

在当前 ONNX 流式生成架构中，`prefill` 和 `decode` 基本都不能省略。

`prefill` 是开头的一次性上下文初始化：

```text
完整 prompt / text / reference audio codes
  -> prefill ONNX
  -> 初始 global_hidden
  -> 初始 KV cache
```

如果没有 `prefill`，模型不知道当前要合成什么文本、使用什么参考音色，也没有第一帧生成所需的 `global_hidden` 和 KV cache。

`decode` 是每帧后的上下文推进：

```text
上一帧 audio token + past KV cache
  -> decode ONNX
  -> 下一步 global_hidden
  -> 更新后的 KV cache
```

整体循环可以理解为：

```text
prefill 得到 global_hidden_0
  -> local_fixed_sampled_frame 生成 frame_0
  -> decode(frame_0) 得到 global_hidden_1
  -> local_fixed_sampled_frame 生成 frame_1
  -> decode(frame_1) 得到 global_hidden_2
  -> ...
```

如果没有 `decode`，`global_hidden` 不会随着已生成音频更新，后续每一帧都会基于同一个旧上下文生成，无法形成正常连续语音。因此当前架构里：

- `prefill` 负责“初始化上下文”；
- `decode` 负责“每帧后推进上下文”；
- `local_fixed_sampled_frame` 负责“根据当前上下文生成一帧声学 token”；
- `codec_decode_step` 负责“把声学 token 解码成 waveform”。

#### `test_local_api.py` 日志中的 RTF 与内存观测

一次 `python -m simple_moss.test_local_api` 测试日志中，生成文本为：

```text
你好，这是一段来自 simple_moss 的本地合成测试语音。
```

生成结果核心指标：

```text
audio_chunks=18
total_audio_s=6.880
first_audio_latency_s=0.5643
rtf_first=1.7580
rtf_steady=0.6755
elapsed=5.01s
```

RTF 结论：

- 首帧延迟约 `0.564s`，从 `start` 到第一段 PCM 到达约半秒。
- `rtf_first=1.7580`，首段生成慢于实时，主要包含 prefill、首次 decode、codec 启动等冷启动开销。
- `rtf_steady=0.6755`，稳定生成阶段快于实时（RTF < 1）。
- 整体 RTF 约为 `5.01 / 6.88 = 0.73`，即生成 6.88 秒音频实际耗时约 5.01 秒，整体仍快于实时。

按 `sys_used` 观察系统内存消耗（不是单进程精确 RSS，也不是 GPU 显存）：

```text
初始化开始:        11561 MB
初始化完成:        12885 MB
warmup 开始:       12885 MB
warmup 完成:       13567 MB
真实生成完成:      13869 MB
```

分阶段估算：

- 初始化阶段增长约 `12885 - 11561 = 1324 MB`，主要发生在创建 ONNX sessions：`prefill`、`decode`、`local_fixed_sampled_frame`、`codec_decode_step`。
- warmup 阶段增长约 `13567 - 12885 = 682 MB`，主要来自第一次 prefill / decode / codec run 后的运行时缓存、Arena、临时张量等。
- 真实 stream 生成阶段增长约 `13869 - 13567 = 302 MB`。
- 从进程开始到本次日志峰值，系统已用内存总增长约 `13869 - 11561 = 2308 MB`，约 2.3 GB。

补充说明：`sys_used` 是整机系统内存已用量，会受到其他进程、系统缓存和内存回收影响。若看当前 Python 进程 RSS，本次日志从约 `50.8 MB` 增长到约 `801.7 MB`，进程 RSS 增长约 `751 MB`。

#### PyTorch 与 ONNX 模型大小、内存和拆图原理

从静态文件大小看，当前模型资产大致呈现：

```text
PyTorch:
  pytorch_model.bin ≈ 229 MB

ONNX:
  moss_tts_global_shared.data ≈ 430 MB
  moss_tts_local_shared.data  ≈ 224 MB
  另有多个 .onnx 小图
```

因此存在一种合理可能：**PyTorch 版本在常驻权重文件层面更小，ONNX 版本因为多图拆分和外部权重数据而更占内存**。但不能只通过磁盘文件大小直接判断运行时内存，因为运行时还包含：

- ONNX Runtime / PyTorch 的图结构与优化缓存；
- CUDA provider / CUDA allocator 的内部缓存；
- memory arena / 临时张量 / workspace；
- KV cache；
- 是否 FP32 / FP16；
- 多个 ONNX session 是否重复持有或映射部分权重；
- warmup 后运行时缓存是否常驻。

在当前 `simple_moss` 中，ONNX 不是加载一个大模型，而是加载多个 session：

```text
prefill
decode
local_fixed_sampled_frame
codec_decode_step
```

每个 session 都可能有自己的图优化结构、内存 arena 和执行缓存。因此在这个项目当前形态下，ONNX 版本更像是“为推理速度和部署便利做了多图拆分”，不一定追求最小内存占用。

ONNX 拆成多个 session 的原理是：原始流式 TTS 推理流程中存在多个“输入输出形态不同、调用频率不同、优化目标不同”的阶段，拆开后可以分别优化和调度：

```text
text / prompt
  -> prefill
  -> 循环 N 次：
       local_fixed_sampled_frame
       decode
  -> 每积累若干帧：
       codec_decode_step
  -> waveform
```

- `prefill`：处理完整上下文，输入是长序列，输出初始 `global_hidden` 和 KV cache。
- `decode`：后续每步只处理一个新 frame row，输入包含 `past_key/value`，输出下一步 `global_hidden` 和更新后的 KV cache。
- `local_fixed_sampled_frame`：根据当前 `global_hidden` 一次生成一整帧声学 token，并把 top-k / top-p / repetition penalty / 随机采样逻辑固化到 ONNX 图内。
- `codec_decode_step`：属于 codec 子系统，将声学 token 流式解码为 waveform，并维护自己的 streaming cache。

这种拆分适合流式输出：外层 Python 可以控制何时解码 codec、每次解码多少帧、如何平衡首帧延迟与吞吐、如何中途停止，以及如何把音频块放入 queue。但代价是 session 数变多、图优化结构和内存 arena 变多，模型资产和运行时内存可能增大。

PyTorch 也支持 stream 生成，但它通常不需要把磁盘模型文件拆成多个文件。原因是：**PyTorch 的“拆分”发生在 Python 动态控制流里，ONNX 的“拆分”发生在导出的静态图文件里。**

PyTorch 侧通常是：

```text
pytorch_model.bin          权重
modeling_moss_tts_nano.py  模型结构和生成逻辑
```

代码运行时可以灵活调用不同子模块：

```text
self.transformer(...)          -> global / prefill / decode 逻辑
self.local_transformer(...)    -> local frame generation
self.audio_lm_heads[...]       -> audio token logits
sample                         -> 采样
codec decode                   -> waveform
yield audio event              -> 流式输出
```

因此 PyTorch 文件看起来是一个权重文件，但运行逻辑仍然是分阶段的；ONNX 为了让静态图可部署、可重复调用、便于流式调度，通常会把这些阶段导出成多个 ONNX graph / session。

严谨比较 PyTorch 与 ONNX 的内存占用，需要分别测量：

```text
PyTorch 初始化后 RSS / GPU 显存
PyTorch warmup 后 RSS / GPU 显存
PyTorch 一次生成后 RSS / GPU 显存

ONNX 初始化后 RSS / GPU 显存
ONNX warmup 后 RSS / GPU 显存
ONNX 一次生成后 RSS / GPU 显存
```

仅看文件大小只能说明“当前 ONNX 资产更大”，不能直接等价为“ONNX 运行时一定更耗内存”。但在当前多 session、双 `.data` 外部权重文件的形态下，ONNX 版本更耗内存是合理且需要重点观察的风险点。

---

# `simple_moss` 外部参数与实际生效情况

本章节整理当前 `simple_moss/test_local_api.py`、`simple_moss/simple_app_onnx.py`、`simple_moss/api_facade.py` 中可以由外部调用方传入的参数，并标注这些参数是否真正影响到底层 ONNX 执行。

## 当前 CLI 暴露的参数

`python -m simple_moss.test_local_api` 当前通过 argparse 直接暴露 4 个参数：

| 参数                 | 默认值                                | 用途                                          | 传到哪里                                                     | 是否影响底层                 |
| ------------------ | ----------------------------------:| ------------------------------------------- | -------------------------------------------------------- | ---------------------- |
| `--model-dir`      | `None`                             | ONNX 模型目录；`None` 时由 runtime 默认解析到 `models/` | `create_default_adapter(model_dir=...)`                  | 是                      |
| `--text`           | `你好，这是一段来自 simple_moss 的本地合成测试语音。` | 要合成的文本                                      | `facade.start(text)`                                     | 是                      |
| `--output`         | `simple_moss_test_output.wav`      | 输出 WAV 文件路径                                 | `_write_pcm_to_wav(...)`                                 | 否，只影响测试脚本保存路径          |
| `--max-new-frames` | `375`                              | 每个 text chunk 最多生成多少声学 token 帧              | `create_default_adapter(...)`、`MossStreamApiFacade(...)` | 是，但 warmup 阶段会被临时改成 16 |

当前 `test_local_api.py` 中还有一项重要参数没有通过 CLI 暴露，而是写死在代码里：

```python
facade = MossStreamApiFacade(
    adapter,
    max_new_frames=max_new_frames,
    voice_clone_max_text_tokens=16,
)
```

因此 `voice_clone_max_text_tokens` 当前对 stream generate 有效，但不能通过命令行传入。

## 初始化阶段参数

测试脚本初始化阶段调用：

```python
adapter = create_default_adapter(
    model_dir=model_dir,
    cpu_threads=1,
    max_new_frames=max_new_frames,
)
```

`create_default_adapter(...)` 实际支持：

| 参数               | 默认值    | 当前 test 是否暴露 | 是否影响底层  | 说明                                                                     |
| ---------------- | ------:| ------------ | ------- | ---------------------------------------------------------------------- |
| `model_dir`      | `None` | 是            | 是       | 决定加载哪个 ONNX 模型目录                                                       |
| `output_dir`     | `None` | 否            | 影响有限    | 决定 adapter 输出目录；当前 result 不写 WAV/base64，主要是路径元数据/目录准备                  |
| `cpu_threads`    | `1`    | 否，写死为 1      | 有效但影响有限 | 写入 ORT `intra_op_num_threads`，主要影响 CPU ops / ORT 内部线程，CUDA 主计算不一定明显受影响 |
| `max_new_frames` | `375`  | 是            | 是       | 写入 manifest 默认生成帧上限；stream 阶段会使用，warmup 阶段会临时覆盖为 16                    |

## Warmup 阶段参数

测试脚本 warmup 阶段调用：

```python
warmup_result = warmup_runtime(adapter)
```

当前 `warmup_runtime(adapter)` 只接受已创建的 `adapter`，没有额外参数。继续往下，`adapter.warmup()` 当前也没有外部可传参数。

warmup 当前固定行为：

- 固定使用 `FIXED_BUILTIN_VOICE`，当前为 `"Lingyu"`。
- 调用 `OrtCpuRuntime.warmup(voice_name=voice_name)`。
- 使用 manifest 中第一条 `text_samples` 的 `text_token_ids`。
- 不走 `split_voice_clone_text(...)`，因此没有 `voice_clone_max_text_tokens` / `max_text_chunk_size` 概念。
- 临时将 `manifest["generation_defaults"]["max_new_frames"]` 改成 `16`，warmup 结束后恢复原值。

因此，**warmup 阶段当前没有可以由 test 外部直接传入的参数**。外部传入的 `max_new_frames` 不决定 warmup 生成帧数；warmup 内部固定最多生成 16 帧。

## Stream Generate 阶段参数

### `MossStreamApiFacade.__init__` 默认参数

`MossStreamApiFacade(adapter, ...)` 会保存一组默认生成参数，供后续 `start(...)` 在未显式传参时使用：

| 参数                            | 默认值               | 当前 test 是否暴露     | 当前实际效果                                                         |
| ----------------------------- | -----------------:| ---------------- | -------------------------------------------------------------- |
| `adapter`                     | 必填                | 内部创建             | 有效，底层推理 adapter                                                |
| `max_new_frames`              | `375`             | 是                | 有效，控制每个 text chunk 生成帧上限                                       |
| `voice_clone_max_text_tokens` | `75`              | 否，当前 test 写死为 16 | 有效，控制文本切 chunk                                                 |
| `attn_implementation`         | `"model_default"` | 否                | 不建议改；当前用于解析 sample_mode                                        |
| `do_sample`                   | `True`            | 否                | 不建议改；`False` 可能映射到 greedy，当前 fixed-only 路径可能失效                 |
| `text_temperature`            | `1.0`             | 否                | 当前 fixed ONNX 路径基本不生效                                          |
| `text_top_p`                  | `1.0`             | 否                | 当前 fixed ONNX 路径基本不生效                                          |
| `text_top_k`                  | `50`              | 否                | 当前 fixed ONNX 路径基本不生效                                          |
| `audio_temperature`           | `0.8`             | 否                | 当前 fixed ONNX 路径基本不生效                                          |
| `audio_top_p`                 | `0.95`            | 否                | 当前 fixed ONNX 路径基本不生效                                          |
| `audio_top_k`                 | `25`              | 否                | 当前 fixed ONNX 路径基本不生效                                          |
| `audio_repetition_penalty`    | `1.2`             | 否                | 当前 fixed ONNX 路径基本不生效；只传 `repetition_seen_mask`，不传动态 penalty 值 |
| `seed`                        | `None`            | 否                | 有效，重置 Python RNG，影响传给 `local_fixed_sampled_frame.onnx` 的随机数    |

### `MossStreamApiFacade.start(...)` 单次请求参数

`start(...)` 支持在单次请求级别覆盖 facade 默认参数：

```python
facade.start(
    text,
    max_new_frames=...,
    voice_clone_max_text_tokens=...,
    attn_implementation=...,
    do_sample=...,
    text_temperature=...,
    text_top_p=...,
    text_top_k=...,
    audio_temperature=...,
    audio_top_p=...,
    audio_top_k=...,
    audio_repetition_penalty=...,
    seed=...,
)
```

当前测试脚本只调用：

```python
start_resp = facade.start(text)
```

因此实际使用的是 `MossStreamApiFacade.__init__` 中保存的默认值。

## 参数实际生效分类

### 实际有效，值得外部暴露

这些参数当前确实影响底层执行或输出：

| 参数                            | 生效原因                                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| `model_dir`                   | 决定加载哪个 ONNX 模型资产                                                                        |
| `cpu_threads`                 | 设置 ORT `intra_op_num_threads`，对 CPU ops / ORT 内部线程可能有影响                                 |
| `max_new_frames`              | `generate_audio_frames` 循环上限，控制每个 text chunk 最多生成多少声学 token 帧                           |
| `voice_clone_max_text_tokens` | 传入 `split_voice_clone_text(..., max_tokens=...)`，影响文本切 chunk、prefill 长度、chunk 数、首帧延迟和内存 |
| `seed`                        | 重置 `self.runtime.rng`，影响传入 fixed ONNX 的 `assistant_random_u` / `audio_random_u`         |
| `text`                        | 输入文本，直接决定合成内容                                                                           |
| `output` / `output_wav`       | 只影响测试脚本最终 WAV 保存路径，不影响模型推理                                                              |
| `output_dir`                  | 影响 adapter 输出目录，当前生成结果不落 WAV/base64，推理影响有限                                              |

### 形式上可传，但当前 fixed-only 路径基本不生效

这些参数会被 `_apply_generation_options(...)` 写入 `runtime.manifest["generation_defaults"]`，但当前 `simple_ort_gpu_runtime.py` 只保留 fixed ONNX 路径。

实际调用 `local_fixed_sampled_frame.onnx` 时只传入：

```python
{
    "global_hidden": global_hidden,
    "repetition_seen_mask": repetition_seen_mask,
    "assistant_random_u": assistant_random_u,
    "audio_random_u": audio_random_u,
}
```

没有把 temperature / top-k / top-p / repetition penalty 作为动态输入传给 ONNX。因此下列参数当前基本不影响底层：

- `text_temperature`
- `text_top_p`
- `text_top_k`
- `audio_temperature`
- `audio_top_p`
- `audio_top_k`
- `audio_repetition_penalty`

这些参数更像原始 full / greedy / Python 采样路径的遗留参数。当前 fixed 模式下，采样策略大概率已经固化在 `moss_tts_local_fixed_sampled_frame.onnx` 中。

### 可能影响流程，但当前不建议外部改动

| 参数                    | 当前行为                                                                       | 风险                                                       |
| --------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------- |
| `attn_implementation` | 被当作 `sample_mode` 解析；默认 `"model_default"` + `do_sample=True` 会落到 `"fixed"` | 如果传 `"full"` / `"greedy"`，当前 simple_moss 未保留对应路径，可能不生成音频 |
| `do_sample`           | 默认 `True` 时走 fixed；`False` 会映射到 greedy                                     | 当前没有 greedy session，可能导致 fixed-only 生成路径失效               |

## 建议

当前最值得暴露为稳定外部参数的是：

```text
--model-dir
--text
--output
--max-new-frames
--voice-clone-max-text-tokens
--seed
--cpu-threads
--output-dir
```

当前不建议暴露或不建议用户随意调整的是：

```text
attn_implementation
do_sample
text_temperature
text_top_p
text_top_k
audio_temperature
audio_top_p
audio_top_k
audio_repetition_penalty
```

原因是：这些参数要么当前 fixed ONNX 路径不读取，要么改动后可能让 fixed-only 生成路径失效。

---

*存档日期：2026-05-06*
