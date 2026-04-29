# warmup 内存使用记录（优化后）

## 原始 MEM 日志整理

按 `python app_onnx.py` 启动日志中的 `[MEM]` 记录顺序整理如下：

```text
[MEM] main: startup baseline | proc_rss=359.9 MB | sys_used=4265.5 MB
[MEM] main: before runtime init | proc_rss=359.9 MB | sys_used=4265.5 MB
[MEM] runtime_init: OnnxNanoTTSServiceAdapter __init__ entry | proc_rss=359.9 MB | sys_used=4265.5 MB
[MEM] runtime_init: OnnxNanoTTSServiceAdapter output_dir ready, before OnnxTtsRuntime | proc_rss=359.9 MB | sys_used=4265.5 MB
[MEM] OnnxTtsRuntime.__init__: before ensure_browser_onnx_model_dir | proc_rss=359.9 MB | sys_used=4265.5 MB
[MEM] OnnxTtsRuntime.__init__: after ensure_browser_onnx_model_dir, before OrtCpuRuntime | proc_rss=359.9 MB | sys_used=4265.5 MB
[MEM] runtime_init: OrtCpuRuntime __init__ entry (model_dir+thread_count) | proc_rss=359.9 MB | sys_used=4265.5 MB
[MEM] runtime_init: OrtCpuRuntime manifest+tts_meta+codec_meta loaded | proc_rss=373.3 MB | sys_used=4269.5 MB
[MEM] runtime_init: OrtCpuRuntime before _create_sessions | proc_rss=373.3 MB | sys_used=4269.5 MB
[MEM] runtime_init: _create_sessions TTS part done (prefill/decode/local_*) | proc_rss=641.4 MB | sys_used=4455.5 MB
[MEM] runtime_init: _create_sessions codec part done (encode/decode_full/decode_step) | proc_rss=671.2 MB | sys_used=4483.2 MB
[MEM] runtime_init: OrtCpuRuntime after _create_sessions (all InferenceSession) | proc_rss=671.2 MB | sys_used=4483.2 MB
[MEM] runtime_init: OrtCpuRuntime CodecStreamingDecodeSession ready | proc_rss=674.3 MB | sys_used=4486.2 MB
[MEM] OnnxTtsRuntime.__init__: after OrtCpuRuntime super().__init__ | proc_rss=674.3 MB | sys_used=4486.2 MB
[MEM] OnnxTtsRuntime.__init__: after SentencePiece tokenizer load | proc_rss=677.0 MB | sys_used=4488.6 MB
[MEM] runtime_init: OnnxNanoTTSServiceAdapter after OnnxTtsRuntime, wiring paths | proc_rss=677.0 MB | sys_used=4488.6 MB
[MEM] runtime_init: OnnxNanoTTSServiceAdapter __init__ complete | proc_rss=677.0 MB | sys_used=4488.6 MB
[MEM] main: runtime created | proc_rss=677.0 MB | sys_used=4488.6 MB
[MEM] warmup: start | proc_rss=677.0 MB | sys_used=4488.6 MB
[MEM] main: warmup started (port not open yet) | proc_rss=677.0 MB | sys_used=4488.6 MB
[MEM] warmup: OrtCpuRuntime.warmup start | proc_rss=677.0 MB | sys_used=4488.6 MB
[MEM] warmup: before generate_audio_frames (prefill/local_decoder/decode) | proc_rss=677.3 MB | sys_used=4488.6 MB
[MEM] warmup: after generate_audio_frames | proc_rss=1062.5 MB | sys_used=5356.5 MB
[MEM] warmup: before codec_streaming_session.run_frames | proc_rss=1062.5 MB | sys_used=5356.5 MB
[MEM] warmup: after codec_streaming_session.run_frames | proc_rss=1140.1 MB | sys_used=5462.3 MB
[MEM] warmup: complete (codec_decode_step session done) | proc_rss=1140.1 MB | sys_used=5462.3 MB
```

## 明显内存跳变阶段

### 1. 启动基线到 metadata 加载

- 起点：`main: startup baseline`
  - `proc_rss=359.9 MB`
  - `sys_used=4265.5 MB`
- 到达：`OrtCpuRuntime manifest+tts_meta+codec_meta loaded`
  - `proc_rss=373.3 MB`
  - `sys_used=4269.5 MB`
- 增量：
  - 进程 RSS：约 `+13.4 MB`
  - 系统已用内存：约 `+4.0 MB`

这一段增长较小，主要对应 manifest、TTS meta、codec meta 等元信息加载。

### 2. 创建 TTS 相关 ONNX sessions

- 起点：`OrtCpuRuntime before _create_sessions`
  - `proc_rss=373.3 MB`
  - `sys_used=4269.5 MB`
- 到达：`_create_sessions TTS part done (prefill/decode/local_*)`
  - `proc_rss=641.4 MB`
  - `sys_used=4455.5 MB`
- 增量：
  - 进程 RSS：约 `+268.1 MB`
  - 系统已用内存：约 `+186.0 MB`

这是初始化阶段第一次明显跳变，主要发生在创建 TTS 侧的 `prefill`、`decode`、`local_*` 等 `InferenceSession`。

### 3. 创建 codec 相关 ONNX sessions

- 起点：`_create_sessions TTS part done (prefill/decode/local_*)`
  - `proc_rss=641.4 MB`
  - `sys_used=4455.5 MB`
- 到达：`_create_sessions codec part done (encode/decode_full/decode_step)`
  - `proc_rss=671.2 MB`
  - `sys_used=4483.2 MB`
- 增量：
  - 进程 RSS：约 `+29.8 MB`
  - 系统已用内存：约 `+27.7 MB`

codec 侧 session 创建也有增长，但明显小于 TTS 侧 session 创建。

### 4. warmup 执行 generate_audio_frames

- 起点：`warmup: before generate_audio_frames (prefill/local_decoder/decode)`
  - `proc_rss=677.3 MB`
  - `sys_used=4488.6 MB`
- 到达：`warmup: after generate_audio_frames`
  - `proc_rss=1062.5 MB`
  - `sys_used=5356.5 MB`
- 增量：
  - 进程 RSS：约 `+385.2 MB`
  - 系统已用内存：约 `+867.9 MB`

这是整段日志中最明显的跳变。它发生在 warmup 首次执行 TTS 推理路径时，可能包含 ONNX Runtime / CUDAExecutionProvider 的首次运行开销、缓存、workspace、kernel 初始化，以及中间张量分配等。

### 5. warmup 执行 codec_streaming_session.run_frames

- 起点：`warmup: before codec_streaming_session.run_frames`
  - `proc_rss=1062.5 MB`
  - `sys_used=5356.5 MB`
- 到达：`warmup: after codec_streaming_session.run_frames`
  - `proc_rss=1140.1 MB`
  - `sys_used=5462.3 MB`
- 增量：
  - 进程 RSS：约 `+77.6 MB`
  - 系统已用内存：约 `+105.8 MB`

这一段对应 codec streaming decode step 的首次运行，增长幅度中等，明显低于 `generate_audio_frames` 阶段。

## 总体结论

- 进程 RSS 从启动基线 `359.9 MB` 增长到 warmup 完成后的 `1140.1 MB`，总增量约 `+780.2 MB`。
- 系统已用内存从 `4265.5 MB` 增长到 `5462.3 MB`，总增量约 `+1196.8 MB`。
- 最大跳变发生在 `warmup: before generate_audio_frames` 到 `warmup: after generate_audio_frames`，进程 RSS 增加约 `+385.2 MB`。
- 初始化阶段最大跳变发生在 `_create_sessions` 的 TTS session 创建阶段，进程 RSS 增加约 `+268.1 MB`。

# warmup 内存使用记录（优化前）

## 原始 MEM 日志整理

按优化前 `python app_onnx.py` 启动日志中的 `[MEM]` 记录顺序整理如下：

```text
[MEM] main: startup baseline | proc_rss=358.1 MB | sys_used=4261.3 MB
[MEM] main: before runtime init | proc_rss=358.1 MB | sys_used=4261.3 MB
[MEM] runtime_init: OnnxNanoTTSServiceAdapter __init__ entry | proc_rss=358.1 MB | sys_used=4261.3 MB
[MEM] runtime_init: OnnxNanoTTSServiceAdapter output_dir ready, before OnnxTtsRuntime | proc_rss=358.1 MB | sys_used=4261.3 MB
[MEM] OnnxTtsRuntime.__init__: before ensure_browser_onnx_model_dir | proc_rss=358.1 MB | sys_used=4261.3 MB
[MEM] OnnxTtsRuntime.__init__: after ensure_browser_onnx_model_dir, before OrtCpuRuntime | proc_rss=358.1 MB | sys_used=4261.3 MB
[MEM] runtime_init: OrtCpuRuntime __init__ entry (model_dir+thread_count) | proc_rss=358.1 MB | sys_used=4261.3 MB
[MEM] runtime_init: OrtCpuRuntime manifest+tts_meta+codec_meta loaded | proc_rss=371.6 MB | sys_used=4265.3 MB
[MEM] runtime_init: OrtCpuRuntime before _create_sessions | proc_rss=371.6 MB | sys_used=4265.3 MB
[MEM] runtime_init: _create_sessions TTS part done (prefill/decode/local_*) | proc_rss=657.3 MB | sys_used=4549.2 MB
[MEM] runtime_init: _create_sessions codec part done (encode/decode_full/decode_step) | proc_rss=698.1 MB | sys_used=4816.4 MB
[MEM] runtime_init: OrtCpuRuntime after _create_sessions (all InferenceSession) | proc_rss=698.1 MB | sys_used=4816.4 MB
[MEM] runtime_init: OrtCpuRuntime CodecStreamingDecodeSession ready | proc_rss=701.2 MB | sys_used=4819.8 MB
[MEM] OnnxTtsRuntime.__init__: after OrtCpuRuntime super().__init__ | proc_rss=701.2 MB | sys_used=4819.8 MB
[MEM] OnnxTtsRuntime.__init__: after SentencePiece tokenizer load | proc_rss=704.2 MB | sys_used=4822.3 MB
[MEM] runtime_init: OnnxNanoTTSServiceAdapter after OnnxTtsRuntime, wiring paths | proc_rss=704.2 MB | sys_used=4822.3 MB
[MEM] runtime_init: OnnxNanoTTSServiceAdapter __init__ complete | proc_rss=704.2 MB | sys_used=4822.3 MB
[MEM] main: runtime created | proc_rss=704.2 MB | sys_used=4822.3 MB
[MEM] warmup: start | proc_rss=704.2 MB | sys_used=4822.3 MB
[MEM] main: warmup started (port not open yet) | proc_rss=704.2 MB | sys_used=4822.3 MB
[MEM] synthesize: resolve_prompt_audio_codes done (codec_encode session if custom audio, else builtin voice lookup) | proc_rss=704.2 MB | sys_used=4822.3 MB
[MEM] synthesize_single_chunk: generate_audio_frames done (prefill + local_decode sessions) | proc_rss=1012.8 MB | sys_used=5689.1 MB
[MEM] synthesize_single_chunk: decode_full_audio_safe done (codec_decode session) | proc_rss=1092.8 MB | sys_used=5736.1 MB
[MEM] warmup: synthesize done (prefill/local_decode/codec_decode sessions) | proc_rss=1093.6 MB | sys_used=5736.6 MB
[MEM] warmup: complete (codec_decode_step session done) | proc_rss=1141.3 MB | sys_used=5781.9 MB
```

## 明显内存跳变阶段

### 1. 启动基线到 metadata 加载

- 起点：`main: startup baseline`
  - `proc_rss=358.1 MB`
  - `sys_used=4261.3 MB`
- 到达：`OrtCpuRuntime manifest+tts_meta+codec_meta loaded`
  - `proc_rss=371.6 MB`
  - `sys_used=4265.3 MB`
- 增量：
  - 进程 RSS：约 `+13.5 MB`
  - 系统已用内存：约 `+4.0 MB`

这一段增长较小，主要对应 manifest、TTS meta、codec meta 等元信息加载。

### 2. 创建 TTS 相关 ONNX sessions

- 起点：`OrtCpuRuntime before _create_sessions`
  - `proc_rss=371.6 MB`
  - `sys_used=4265.3 MB`
- 到达：`_create_sessions TTS part done (prefill/decode/local_*)`
  - `proc_rss=657.3 MB`
  - `sys_used=4549.2 MB`
- 增量：
  - 进程 RSS：约 `+285.7 MB`
  - 系统已用内存：约 `+283.9 MB`

这是优化前初始化阶段第一次明显跳变，主要发生在创建 TTS 侧 `prefill`、`decode`、`local_*` 等 `InferenceSession`。

### 3. 创建 codec 相关 ONNX sessions

- 起点：`_create_sessions TTS part done (prefill/decode/local_*)`
  - `proc_rss=657.3 MB`
  - `sys_used=4549.2 MB`
- 到达：`_create_sessions codec part done (encode/decode_full/decode_step)`
  - `proc_rss=698.1 MB`
  - `sys_used=4816.4 MB`
- 增量：
  - 进程 RSS：约 `+40.8 MB`
  - 系统已用内存：约 `+267.2 MB`

codec 侧 session 创建的进程 RSS 增长中等，但系统已用内存增长较明显。

### 4. warmup 执行 generate_audio_frames

- 起点：`synthesize: resolve_prompt_audio_codes done`
  - `proc_rss=704.2 MB`
  - `sys_used=4822.3 MB`
- 到达：`synthesize_single_chunk: generate_audio_frames done (prefill + local_decode sessions)`
  - `proc_rss=1012.8 MB`
  - `sys_used=5689.1 MB`
- 增量：
  - 进程 RSS：约 `+308.6 MB`
  - 系统已用内存：约 `+866.8 MB`

这是优化前 warmup 阶段最明显的跳变，发生在首次执行 TTS 推理路径时，可能包含 ONNX Runtime / CUDAExecutionProvider 的首次运行开销、缓存、workspace、kernel 初始化，以及中间张量分配等。

### 5. warmup 执行 codec_decode full audio

- 起点：`synthesize_single_chunk: generate_audio_frames done (prefill + local_decode sessions)`
  - `proc_rss=1012.8 MB`
  - `sys_used=5689.1 MB`
- 到达：`synthesize_single_chunk: decode_full_audio_safe done (codec_decode session)`
  - `proc_rss=1092.8 MB`
  - `sys_used=5736.1 MB`
- 增量：
  - 进程 RSS：约 `+80.0 MB`
  - 系统已用内存：约 `+47.0 MB`

这一段对应 `codec_decode` 完整音频解码路径的首次运行，进程 RSS 增长较明显。

### 6. warmup 完成 codec_decode_step

- 起点：`warmup: synthesize done (prefill/local_decode/codec_decode sessions)`
  - `proc_rss=1093.6 MB`
  - `sys_used=5736.6 MB`
- 到达：`warmup: complete (codec_decode_step session done)`
  - `proc_rss=1141.3 MB`
  - `sys_used=5781.9 MB`
- 增量：
  - 进程 RSS：约 `+47.7 MB`
  - 系统已用内存：约 `+45.3 MB`

这一段对应 warmup 末尾的 `codec_decode_step` 首次运行，增长幅度小于 `generate_audio_frames` 和 `decode_full_audio_safe`。

## 总体结论

- 进程 RSS 从启动基线 `358.1 MB` 增长到 warmup 完成后的 `1141.3 MB`，总增量约 `+783.2 MB`。
- 系统已用内存从 `4261.3 MB` 增长到 `5781.9 MB`，总增量约 `+1520.6 MB`。
- 最大进程 RSS 跳变发生在 `resolve_prompt_audio_codes done` 到 `generate_audio_frames done`，进程 RSS 增加约 `+308.6 MB`。
- 最大系统已用内存跳变也发生在 `resolve_prompt_audio_codes done` 到 `generate_audio_frames done`，系统已用内存增加约 `+866.8 MB`。
- 初始化阶段最大跳变发生在 `_create_sessions` 的 TTS session 创建阶段，进程 RSS 增加约 `+285.7 MB`。

# 优化前后主要跳变阶段对比

## 总体内存变化对比

- 优化前：
  - 启动基线：`proc_rss=358.1 MB`，`sys_used=4261.3 MB`
  - warmup 完成：`proc_rss=1141.3 MB`，`sys_used=5781.9 MB`
  - 总增量：RSS 约 `+783.2 MB`，系统已用内存约 `+1520.6 MB`
- 优化后：
  - 启动基线：`proc_rss=359.9 MB`，`sys_used=4265.5 MB`
  - warmup 完成：`proc_rss=1140.1 MB`，`sys_used=5462.3 MB`
  - 总增量：RSS 约 `+780.2 MB`，系统已用内存约 `+1196.8 MB`
- 对比结论：
  - RSS 总增量基本持平，优化后少约 `3.0 MB`。
  - 系统已用内存总增量明显下降，优化后少约 `323.8 MB`。

## metadata 加载阶段

- 优化前：RSS 约 `+13.5 MB`，系统已用内存约 `+4.0 MB`
- 优化后：RSS 约 `+13.4 MB`，系统已用内存约 `+4.0 MB`
- 对比结论：两轮几乎一致，metadata 加载不是差异来源。

## TTS sessions 创建阶段

- 优化前：`OrtCpuRuntime before _create_sessions` 到 `_create_sessions TTS part done`
  - RSS 约 `+285.7 MB`
  - 系统已用内存约 `+283.9 MB`
- 优化后：`OrtCpuRuntime before _create_sessions` 到 `_create_sessions TTS part done`
  - RSS 约 `+268.1 MB`
  - 系统已用内存约 `+186.0 MB`
- 对比结论：
  - RSS 增量减少约 `17.6 MB`。
  - 系统已用内存增量减少约 `97.9 MB`。
  - 优化后 TTS session 初始化阶段的系统内存压力明显降低。

## codec sessions 创建阶段

- 优化前：`_create_sessions TTS part done` 到 `_create_sessions codec part done`
  - RSS 约 `+40.8 MB`
  - 系统已用内存约 `+267.2 MB`
- 优化后：`_create_sessions TTS part done` 到 `_create_sessions codec part done`
  - RSS 约 `+29.8 MB`
  - 系统已用内存约 `+27.7 MB`
- 对比结论：
  - RSS 增量减少约 `11.0 MB`。
  - 系统已用内存增量减少约 `239.5 MB`。
  - 这是优化前后系统内存差异最明显的初始化阶段。

## warmup TTS 推理阶段

- 优化前：`resolve_prompt_audio_codes done` 到 `generate_audio_frames done`
  - RSS 约 `+308.6 MB`
  - 系统已用内存约 `+866.8 MB`
- 优化后：`before generate_audio_frames` 到 `after generate_audio_frames`
  - RSS 约 `+385.2 MB`
  - 系统已用内存约 `+867.9 MB`
- 对比结论：
  - RSS 增量优化后反而增加约 `76.6 MB`。
  - 系统已用内存增量基本持平，优化后多约 `1.1 MB`。
  - 该阶段仍然是 warmup 期间最大的内存跳变来源，优化前后都集中在首次执行 TTS 推理路径。

## warmup codec 解码阶段

- 优化前：
  - `generate_audio_frames done` 到 `decode_full_audio_safe done`：RSS 约 `+80.0 MB`，系统已用内存约 `+47.0 MB`
  - `synthesize done` 到 `warmup complete`：RSS 约 `+47.7 MB`，系统已用内存约 `+45.3 MB`
  - 两段合计：RSS 约 `+127.7 MB`，系统已用内存约 `+92.3 MB`
- 优化后：
  - `before codec_streaming_session.run_frames` 到 `after codec_streaming_session.run_frames`：RSS 约 `+77.6 MB`，系统已用内存约 `+105.8 MB`
- 对比结论：
  - codec 解码相关 RSS 增量优化后减少约 `50.1 MB`。
  - codec 解码相关系统已用内存增量优化后增加约 `13.5 MB`。
  - 优化后 codec warmup 路径更集中，日志中不再拆成 `decode_full_audio_safe` 和最后 `codec_decode_step` 两段。

## 对比结论

| 对比项                     | 优化前          | 优化后          | 变化          | 结论                       |
| ----------------------- | ------------ | ------------ | ----------- | ------------------------ |
| 最终系统已用内存相对启动增量          | `+1520.6 MB` | `+1196.8 MB` | `-323.8 MB` | 优化后系统内存增量明显下降            |
| TTS sessions 创建系统内存增量   | `+283.9 MB`  | `+186.0 MB`  | `-97.9 MB`  | TTS session 初始化的系统内存压力降低 |
| codec sessions 创建系统内存增量 | `+267.2 MB`  | `+27.7 MB`   | `-239.5 MB` | 优化收益最明显的初始化阶段            |
| warmup TTS 推理系统内存增量     | `+866.8 MB`  | `+867.9 MB`  | `+1.1 MB`   | 最大系统内存跳变仍在该阶段，基本无变化      |
| warmup codec 解码系统内存增量   | `+92.3 MB`   | `+105.8 MB`  | `+13.5 MB`  | codec 解码相关系统内存增量略升       |

# generate_audio_frames 细粒度 MEM 记录与分析

本节记录一次在 `OrtCpuRuntime.warmup()` 中开启 `mem_trace_label="warmup: generate_audio_frames"` 后的细粒度 `[MEM]` 输出，并对 `sys_used`（系统已用内存）做相邻打点差分，用于拆分 `warmup TTS 推理系统内存增量` 的内部结构。

## 运行背景

- 进程：`python app_onnx.py`
- 时间：`2026-04-29 16:08:16` 起（日志片段）
- 说明：`sys_used` 来自 `psutil.virtual_memory().used`，是整机维度；若同一时间段还有其它线程/进程分配（例如文本正则化组件初始化、Uvicorn 启动），会把增量叠加进来，导致“阶段归因”不完全纯净。

## 进入 generate_audio_frames 之前的边界打点（便于对齐总量）

```text
[MEM] main: startup baseline | proc_rss=357.9 MB | sys_used=4270.6 MB
[MEM] warmup: before generate_audio_frames (prefill/local_decoder/decode) | proc_rss=676.2 MB | sys_used=4486.4 MB
```

从启动基线到进入 `generate_audio_frames` 前：

- `sys_used` 相对启动基线：`4486.4 - 4270.6 = +215.8 MB`（runtime init / session 创建等）

## generate_audio_frames 内部 MEM 日志（完整提取）

```text
[MEM] warmup: generate_audio_frames: entry | proc_rss=696.6 MB | sys_used=4504.8 MB
[MEM] warmup: generate_audio_frames: after build prefill inputs | proc_rss=696.6 MB | sys_used=4504.8 MB
[MEM] warmup: generate_audio_frames: before prefill.run | proc_rss=696.6 MB | sys_used=4504.8 MB
[MEM] warmup: generate_audio_frames: after prefill.run | proc_rss=933.8 MB | sys_used=5180.9 MB
[MEM] warmup: generate_audio_frames: after build prefill outputs and past cache | proc_rss=933.8 MB | sys_used=5180.9 MB
[MEM] warmup: generate_audio_frames: decode loop first frame start | proc_rss=934.0 MB | sys_used=5180.9 MB
[MEM] warmup: generate_audio_frames: before first local_fixed_sampled_frame.run | proc_rss=957.7 MB | sys_used=5210.1 MB
[MEM] warmup: generate_audio_frames: after first local_fixed_sampled_frame.run | proc_rss=998.6 MB | sys_used=5263.2 MB
[MEM] warmup: generate_audio_frames: before first decode.run | proc_rss=998.6 MB | sys_used=5263.2 MB
[MEM] warmup: generate_audio_frames: after first decode.run | proc_rss=1022.1 MB | sys_used=5305.8 MB
[MEM] warmup: generate_audio_frames: decode loop done, generated_frames=16 | proc_rss=1071.4 MB | sys_used=5869.1 MB
[MEM] warmup: generate_audio_frames: before decode arena shrinkage run | proc_rss=1071.4 MB | sys_used=5869.1 MB
[MEM] warmup: generate_audio_frames: after decode arena shrinkage run | proc_rss=1075.2 MB | sys_used=5877.0 MB
[MEM] warmup: generate_audio_frames: return | proc_rss=1075.2 MB | sys_used=5877.0 MB
```

## 相邻打点差分（重点看 sys_used）

以下按“上一条 `[MEM]` → 下一条 `[MEM]`”计算增量：

| 阶段（起点 → 终点）                                                                                | Δ `sys_used`    | Δ `proc_rss`    | 解释（对应代码语义）                                                                                                 |
| ------------------------------------------------------------------------------------------ | --------------- | --------------- | ---------------------------------------------------------------------------------------------------------- |
| `entry` → `before prefill.run`                                                             | `0.0 MB`        | `0.0 MB`        | 进入函数后构建 `prefill` 输入张量；此段未显示系统内存增长                                                                         |
| `before prefill.run` → `after prefill.run`                                                 | **`+676.1 MB`** | **`+237.2 MB`** | `prefill` 首次执行：产出 `global_hidden` 与 `past_*`（KV cache）并触发 ORT/CUDA 首次执行相关分配                                |
| `after prefill.run` → `before first local_fixed_sampled_frame.run`                         | `+29.2 MB`      | `+23.9 MB`      | 进入 decode 循环首帧前；日志时间上与英文 TN 初始化等输出交错，增量可能混入非 TTS 推理分配                                                      |
| `before first local_fixed_sampled_frame.run` → `after first local_fixed_sampled_frame.run` | `+53.1 MB`      | `+40.9 MB`      | 首帧 `local_fixed_sampled_frame`：一次性生成整帧 VQ tokens（fixed 采样路径）                                               |
| `after first local_fixed_sampled_frame.run` → `after first decode.run`                     | `+42.6 MB`      | `+23.5 MB`      | 首帧后 `decode`：把本帧 token 写回跨帧 KV，并更新 `global_hidden`                                                         |
| `after first decode.run` → `decode loop done`                                              | **`+563.3 MB`** | **`+49.3 MB`**  | 后续帧（本日志显示 `generated_frames=16`）持续 `local_fixed_sampled_frame` + `decode` 的累积效应；通常包含多帧 KV 更新与 allocator 行为 |
| `decode loop done` → `after decode arena shrinkage run`                                    | `+7.9 MB`       | `+3.8 MB`       | chunk 结束后触发 `decode` arena shrinkage 空跑；对 `sys_used` 影响较小                                                  |

从 `warmup: before generate_audio_frames` 到 `generate_audio_frames: return`：

- `sys_used`：`5877.0 - 4486.4 = +1390.6 MB`
- `proc_rss`：`1075.2 - 676.2 = +399.0 MB`

## 结论（面向“是否浪费”的问题对齐）

- 这三段最大的 `sys_used` 跳变分别来自 **`prefill.run`**、**首帧后的多帧 `decode` 累积**，以及 **`local_fixed_sampled_frame` 首帧推理**；它们对应 `generate_audio_frames` 的主干路径，并不是“多加载了无关 session 就一定不会出现”的那类成本。
- 若目标是 **降低常驻**：继续删除“永远不会走到的分支 session”（例如某些 local fallback、非 stream 的 `codec_decode` full）仍然有价值，但它主要影响 **初始化常驻**，对 **`prefill/decode` 执行期峰值** 的拆分帮助有限。
- 若目标是 **降低峰值或更准确归因**：需要控制并发（避免 warmup 与文本组件初始化、web server 启动重叠），或补充更细粒度到“每一帧 decode 前后”的打点（代价是日志量与开销显著增加）。

## generate_audio_frames 之后的边界打点（对齐 codec warmup）

```text
[MEM] warmup: after generate_audio_frames | proc_rss=1075.2 MB | sys_used=5877.0 MB
[MEM] warmup: before codec_streaming_session.run_frames | proc_rss=1075.2 MB | sys_used=5877.0 MB
[MEM] warmup: after codec_streaming_session.run_frames | proc_rss=1140.7 MB | sys_used=5969.1 MB
[MEM] warmup: complete (codec_decode_step session done) | proc_rss=1140.7 MB | sys_used=5969.1 MB
```

codec 流式解码段（同一次 run）：

- `before codec_streaming_session.run_frames` → `after`：`sys_used` `5969.1 - 5877.0 = +92.1 MB`
