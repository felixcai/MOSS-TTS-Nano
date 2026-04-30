# simple_moss stream generate 服务开发思路

## 1. 总体目标

目标是把 MOSS-TTS-Nano 当前偏 Demo/UI 的调用链，剥离成一个更适合外部服务调用的 stream generate 核心模块。

第一阶段不直接改 `app.py` / `app_onnx.py`，而是新建一个独立目录：

```text
simple_moss/
  __init__.py
  simple_ort_cpu_runtime.py
  simple_onnx_tts_runtime.py
  api_facade.py
  simple_types.py        # 可选
```

核心原则：

- 先保留完整的 stream generate 推理能力。
- 先去掉 UI、状态轮询、result/base64、临时 WAV 输出等 Demo 逻辑。
- 先做本地可调用的 facade，再在 facade 外面包 FastAPI。
- 第一版保持单实例串行推理，避免共享状态并发污染。

## 2. 当前调用链中真正需要保留的部分

现有 stream generate 主链路：

```text
app.py _run_streaming_job
  -> OnnxRequestRuntimeManager.iter_with_runtime
  -> OnnxNanoTTSServiceAdapter.synthesize_stream
  -> OnnxTtsRuntime.resolve_prompt_audio_codes
  -> OnnxTtsRuntime.split_voice_clone_text
  -> OnnxTtsRuntime.encode_text
  -> OrtCpuRuntime.build_voice_clone_request_rows
  -> OrtCpuRuntime.generate_audio_frames(on_frame=...)
  -> CodecStreamingDecodeSession.run_frames
  -> yield audio chunk
```

新模块要保留的是下层生成链，而不是旧的 Web 任务管理模型。

## 3. `simple_ort_cpu_runtime.py` 的职责

从 `ort_cpu_runtime.py` 中剥离 ONNX 核心推理能力。

必须保留：

- manifest / meta 加载
- ONNX `InferenceSession` 创建
- `CodecStreamingDecodeSession`
- `CodecStreamingDecodeSession.reset`
- `CodecStreamingDecodeSession.run_frames`
- `build_voice_clone_request_rows`
- `generate_audio_frames`
- 生成过程用到的采样、logits、KV cache、local decoder 辅助函数
- `_resolve_stream_decode_frame_budget`

第一版可以不迁移或延后迁移：

- 非流式 `decode_full_audio`
- `list_builtin_voices` / `list_text_samples` 等展示接口
- 过多兼容逻辑
- 与 Demo 或调试输出强绑定的逻辑

注意：`generate_audio_frames` 内部依赖较多，第一步建议以“行为不变”为主，不要一边剥离一边重写算法。

## 4. `simple_onnx_tts_runtime.py` 的职责

从 `onnx_tts_runtime.py` 中剥离 TTS 业务预处理层。

必须保留：

- `OnnxTtsRuntime.__init__` 中必要的模型目录、manifest、tokenizer 初始化
- `resolve_prompt_audio_codes`
- `split_voice_clone_text`
- `encode_text`
- `estimate_voice_clone_inter_chunk_pause_seconds`
- `_merge_audio_channels`
- `_concat_waveforms`（如果 facade 内部仍需要拼接）

是否保留自定义 prompt audio 取决于第一版范围：

- （确定：固定内置voice）如果第一版固定内置 voice，可以先不支持上传音频，减少 `codec_encode` 和音频加载依赖。
- （确定：不支持voice clone）如果第一版要支持 voice clone，则需要保留 `encode_reference_audio` 和 `_load_reference_audio`。

建议第一版先固定一种内置音色或固定 prompt audio，跑通外部 stream 服务后再扩展动态音色。

## 5. `api_facade.py` 的职责

`api_facade.py` 不直接等同于 HTTP API，而是一个可被 HTTP 层、CLI、测试函数共同调用的 headless facade。

建议核心形态：

```python
class MossStreamFacade:
    def __init__(self, config: MossStreamConfig): ...

    def stream_generate(self, text: str) -> Iterator[AudioChunk]:
        ...
```

它应该封装以下调用：

```text
resolve_prompt_audio_codes
  -> split_voice_clone_text
  -> encode_text
  -> build_voice_clone_request_rows
  -> generate_audio_frames(on_frame)
  -> codec_streaming_session.run_frames
  -> yield PCM/audio chunk
```

第一版 facade 输出建议尽量简单：

- `pcm_bytes`
- `sample_rate`
- `channels`
- `chunk_index`
- `is_pause`
- 可选：`emitted_audio_seconds`、`lead_seconds`

不建议第一版输出：

- 完整 `waveform_numpy`
- 完整 WAV base64
- 最终 result payload
- Demo 前端所需的 chunk range/status 文案

## 6. 本地 test 函数

`api_facade.py` 中可以提供一个本地测试函数，用来不经过 FastAPI 完整调用 stream generate。

建议命名：

- `smoke_test_stream_generate`
- 或 `run_local_stream_generate_test`

测试函数职责：

```text
构造 MossStreamFacade
  -> 调用 stream_generate(text)
  -> 逐块消费 PCM
  -> 可选写出 .pcm 或 .wav
  -> 打印 chunk 数、总音频时长、首包耗时等信息
```

这个函数的意义是让 `simple_moss` 在接入 Web 服务前，就能本地验证完整底层链路。

## 7. 不建议迁移的旧逻辑

`app.py` / `app_onnx.py` 中以下内容暂时不进入 `simple_moss` 第一版：

- HTML UI 渲染
- demo entries
- `/api/generate-stream/start`
- `/api/generate-stream/{stream_id}/audio`
- `/api/generate-stream/{stream_id}/status`
- `/api/generate-stream/{stream_id}/result`
- `/api/generate-stream/{stream_id}/close`
- `StreamingJob` / `StreamingJobManager`
- WAV base64 result
- 临时文件清理流程
- RTF 前端展示文案
- 多设备路由
- per-request `cpu_threads`
- 非流式 `/api/generate`

这些都是 Demo 服务层能力，不是 stream generate 核心能力。

## 8. 并发策略

第一版保持单实例串行。

原因是当前 runtime 内存在共享可变状态：

- `codec_streaming_session` 维护流式解码 KV cache
- `manifest["generation_defaults"]` 会被请求级参数修改
- `rng` 会被 seed / 采样逻辑影响

因此 `api_facade.py` 内部建议放一个 `threading.Lock`：

```text
stream_generate 进入时加锁
  -> 完整生成结束后释放锁
```

后续如果要支持并发，应该做 runtime 池，每个并发槽位持有独立 runtime 或至少独立 codec streaming session 和 generation config。

## 9. 推荐实施顺序

### 第一步：剥离 `simple_ort_cpu_runtime.py`

先搬核心 ONNX 推理层，尽量保持行为不变。

验收标准：

- 能创建 ONNX sessions。
- 能构造 codec streaming session。
- `generate_audio_frames` 和 `run_frames` 可被上层调用。

### 第二步：剥离 `simple_onnx_tts_runtime.py`

搬 TTS 预处理层。

验收标准：

- 能加载 tokenizer。
- 能解析内置 prompt audio codes。
- 能切分文本。
- 能 encode text。
- 能生成 `request_rows`。

### 第三步：实现 `api_facade.py`

把 app/app_onnx 中调用下层的核心逻辑封装进 facade。

验收标准：

- `stream_generate(text)` 可以持续 yield 音频 chunk。
- 本地 test 函数可以跑完整 stream generate。
- 不依赖 FastAPI、StreamingJob、UI、result/base64。

### 第四步：再包外部 HTTP 服务

等 facade 稳定后，再加一个很薄的 FastAPI 层。

建议第一版 HTTP 形态参考 Qwen TTS 的简单接口：

```text
POST /v1/audio/speech
```

请求体第一版只保留：

```json
{
  "input": "要合成的文本"
}
```

响应固定：

```text
audio/pcm; rate=24000
```

参数如 `model_dir`、`voice`、`max_new_frames`、采样参数等先放启动配置，不放到每次请求里。

## 10. 第一版边界总结

第一版要做的是：

- 新建 `simple_moss/`
- 剥离底层 ONNX runtime
- 剥离 TTS 预处理 runtime
- 用 `api_facade.py` 封装完整 stream generate
- 提供本地 test 函数跑通全链路
- 后续再加极简 HTTP stream endpoint

第一版不要做的是：

- 不迁移旧 Demo UI
- 不迁移多端点 stream job 模型
- 不做并发 runtime 池
- 不做复杂 per-request 参数
- 不保留最终 WAV/base64 结果链路

# stream generate 调用编排归属

## 1. 问题

对这条 facade 计划保留的调用链：

```text
resolve_prompt_audio_codes
  -> split_voice_clone_text
  -> encode_text
  -> build_voice_clone_request_rows
  -> generate_audio_frames(on_frame)
  -> codec_streaming_session.run_frames
  -> yield PCM/audio chunk
```

需要明确：这些东西现在是不是都在 `app` 层处理，还是在 `onnx_tts_runtime.py` 层处理。

## 2. 结论

不是都在 `app` 层。

更准确地说：这些函数的实现分布在 `onnx_tts_runtime.py` 和 `ort_cpu_runtime.py`，但“把它们串起来执行一次 stream generate”的编排逻辑现在主要在 `app_onnx.py` 的 `OnnxNanoTTSServiceAdapter.synthesize_stream()` 里。

分层关系：

```text
app.py
  HTTP/job 层：
  start/audio/status/result/close、StreamingJob、audio_queue、状态管理

app_onnx.py
  stream generate 编排层：
  调 resolve_prompt_audio_codes
  调 split_voice_clone_text
  调 encode_text
  调 build_voice_clone_request_rows
  调 generate_audio_frames
  在 on_frame 里调 codec_streaming_session.run_frames
  yield audio/result event

onnx_tts_runtime.py
  TTS 业务方法实现：
  resolve_prompt_audio_codes
  split_voice_clone_text
  encode_text
  estimate_voice_clone_inter_chunk_pause_seconds
  _merge_audio_channels 等

ort_cpu_runtime.py
  ONNX 核心推理实现：
  build_voice_clone_request_rows
  generate_audio_frames
  CodecStreamingDecodeSession.run_frames/reset
```

## 3. 对 simple_moss 拆分的影响

`api_facade.py` 要替代的不是 `onnx_tts_runtime.py`，而是替代 `app_onnx.py` 里“如何调用下层完成一次 stream generate”的编排代码。

因此：

- `simple_onnx_tts_runtime.py` 继续负责提供 `resolve_prompt_audio_codes`、`split_voice_clone_text`、`encode_text` 等 TTS 预处理能力。
- `simple_ort_cpu_runtime.py` 继续负责提供 `build_voice_clone_request_rows`、`generate_audio_frames`、`CodecStreamingDecodeSession.run_frames` 等 ONNX 推理能力。
- `api_facade.py` 负责把这些能力串起来，对外暴露一个更干净的 `stream_generate()`。

补充细节：`app.py` 里也会提前调用一次文本切分，主要是为了给前端展示 `text_chunks`，不是 stream generate 的核心推理路径。

# AudioChunk 输出字段与现有 app / app_onnx 返回关系

## 1. facade 建议输出字段

第一版 facade 可以把现有内部 event 和对外 PCM stream 合并成一个更清晰的 `AudioChunk` 输出。

建议字段：

- `pcm_bytes`
- `sample_rate`
- `channels`
- `text_chunk_index`
- `audio_chunk_index`
- `is_pause`
- 可选：`emitted_audio_seconds`
- 可选：`lead_seconds`

## 2. 各字段含义

`pcm_bytes` 是真正要给外部播放或传输的音频数据。它通常是 `float32 waveform` 转成的 `PCM signed 16-bit little-endian` 裸字节流，也就是当前 `app.py` 里 `_audio_to_pcm16le_bytes()` 产出的格式。外部 HTTP stream 最终主要就是连续返回这个字段。

`sample_rate` 是采样率，用来告诉播放器每秒多少个采样点。例如如果是 `24000`，播放器就要按 24kHz 播放这些 PCM bytes。

`channels` 是声道数。`1` 是单声道，`2` 是双声道。解码出来的 waveform 如果形状是 `(samples, channels)`，那 `channels` 就是第二维；如果是一维 waveform，通常视为 `1`。

`text_chunk_index` 表示当前音频小片段属于第几个文本分块。长文本会先被 `split_voice_clone_text` 切成多个 text chunk，每个 text chunk 独立走一次 prefill/decode。

`audio_chunk_index` 表示当前是 stream 输出的第几个音频小片段，按 yield 次数递增。它和 `text_chunk_index` 不是一回事。

`is_pause` 表示这个音频 chunk 是不是人为插入的静音停顿。当前逻辑在两个 text chunk 之间会插入一小段 silence，让句间衔接更自然。真实语音是 `False`，chunk 间静音是 `True`。

`emitted_audio_seconds` 是到目前为止已经向外发出了多少秒音频。它是累计值，不是当前 chunk 的长度。可以用于进度、监控、RTF 统计。

`lead_seconds` 是“已生成并发出的音频时长”相对于真实播放时间的超前量。例如 `lead_seconds=0.8` 表示生成端目前大约比实时播放多缓存了 0.8 秒音频。这个值主要用于流式解码调度：超前量小就少批量解码、降低首包延迟；超前量大就可以批量多解码一些、提高吞吐。

## 3. 当前 app_onnx 内部 event 已经返回什么

在 `app_onnx.py` 的 `synthesize_stream()` 内部，`audio` event 里已经返回了这些字段的大部分：

```python
{
    "type": "audio",
    "waveform_numpy": ...,
    "sample_rate": sample_rate,
    "channels": channels,
    "chunk_index": chunk_index,
    "emitted_audio_seconds": ...,
    "lead_seconds": lead_seconds,
    "is_pause": bool(is_pause),
}
```

所以在 `app_onnx.py -> app.py` 的内部调用层，有：

- `waveform_numpy`
- `sample_rate`
- `channels`
- `chunk_index`
- `emitted_audio_seconds`
- `lead_seconds`
- `is_pause`

这里的 `chunk_index` 更准确地说是 `text_chunk_index`，表示当前音频小片段属于第几个文本切片。

## 4. 当前 app.py 对外 audio stream 返回什么

`app_onnx.py` 内部 event 没有直接返回 `pcm_bytes`。

`pcm_bytes` 是在 `app.py` 的 `_run_streaming_job()` 里由 `waveform_numpy` 转出来的：

```python
waveform_numpy = np.asarray(event["waveform_numpy"], dtype=np.float32)
pcm_bytes = _audio_to_pcm16le_bytes(waveform_numpy)
```

然后 `pcm_bytes` 被放进 `job.audio_queue`，最终由：

```text
GET /api/generate-stream/{stream_id}/audio
```

以裸流形式返回给客户端。

当前对外 HTTP 客户端看到的是：

- `/audio` 响应体里只有连续的 `pcm_bytes`
- `sample_rate` / `channels` 在 HTTP header 里返回
- `chunk_index` / `is_pause` / `emitted_audio_seconds` / `lead_seconds` 不在音频流 body 里
- 这些状态类字段主要进入 `StreamingJob`，可通过 `/status` 间接看到一部分，比如当前 chunk、已发音频秒数、lead 等

## 5. AudioChunk 不等于 text chunk

`AudioChunk` 更建议表示“一次流式发出的音频小片段”，不等于一个 `text chunk` 的完整音频。

关系大概是：

```text
一个长文本
  -> split_voice_clone_text
  -> 多个 text chunk

一个 text chunk
  -> generate_audio_frames 逐帧生成声学 token
  -> codec_streaming_session.run_frames 分批解码
  -> 多个 AudioChunk
```

示意：

```text
text_chunk[0]
  -> AudioChunk 0
  -> AudioChunk 1
  -> AudioChunk 2

text_chunk[1]
  -> AudioChunk 3
  -> AudioChunk 4
  -> pause AudioChunk
```

所以 `AudioChunk.text_chunk_index` 的意思是：这个音频小片段属于第几个 text chunk，而不是“这个 AudioChunk 是第几个音频 chunk”。

为了避免误解，不建议在 facade 里继续使用模糊的 `chunk_index` 字段名，建议拆成：

```python
@dataclass
class AudioChunk:
    pcm_bytes: bytes
    sample_rate: int
    channels: int
    text_chunk_index: int
    audio_chunk_index: int
    is_pause: bool
    emitted_audio_seconds: float
    lead_seconds: float
```

第一版最小可以只保留 `pcm_bytes`、`sample_rate`、`channels`、`text_chunk_index`、`audio_chunk_index`、`is_pause`。

`emitted_audio_seconds` 和 `lead_seconds` 更偏调试、监控和后续 UI/状态扩展，可以保留为可选字段。
