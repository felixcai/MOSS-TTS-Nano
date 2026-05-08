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

## 8.1 共享状态、独占粒度与锁所在层

“共享状态”指的是：同一个 runtime 实例里，有一些会在生成过程中被读取和修改的对象；如果两个请求同时使用这个 runtime，它们会读写同一份对象，彼此影响。

当前服务不是每个请求都新建一个 `OnnxTtsRuntime` / `OrtCpuRuntime`，而是服务启动时创建一个 runtime，然后所有请求复用它：

```text
request A ┐
          ├─ 使用同一个 runtime 实例
request B ┘
```

这些共享状态包括：

- `codec_streaming_session`：维护流式 codec 解码的 KV cache。A 请求解码第 1、2、3 帧时，session 中保存的是 A 的历史状态；如果 B 请求同时调用 `reset()` 或 `run_frames()`，就可能覆盖 A 的状态，导致音频上下文串扰。
- `manifest["generation_defaults"]`：当前 `app_onnx.py` 会在每次请求开始时把采样参数写入 runtime 的 manifest，例如 `sample_mode`、temperature、top_p、top_k、repetition penalty、seed 等。如果 A 和 B 同时运行，A 后续生成时可能读到 B 写入的参数。
- `rng`：随机数生成器有内部状态。采样每抽一次 token，rng 状态都会前进；如果 A 和 B 同时使用同一个 rng，随机序列会交错。如果 B 重新设置 seed，也会影响 A 的采样结果。

因此当前代码用 `_execution_lock` 保证同一时刻只有一个请求进入生成流程：

```text
A 请求生成中：独占 runtime
B 请求等待：不能同时改 runtime 状态
```

这个独占粒度是“一次完整 stream generate 请求”，不是“一个 text chunk”。

也就是说：

```text
A 请求拿到 _execution_lock
  -> 处理 A 的所有 text chunks
  -> 每个 text chunk 都生成完音频
  -> chunk 间 pause 也发完
  -> 最终 result event 产生
  -> synthesize_stream 迭代结束
A 释放 _execution_lock

B 才能开始真正推理
```

原因是当前 `OnnxRequestRuntimeManager.iter_with_runtime()` 的结构是：

```python
with self._locked_runtime(...) as (...):
    for item in factory(runtime):
        yield item, execution_device, resolved_cpu_threads
```

`_locked_runtime()` 里的 `_execution_lock` 包住了整个 `for item in factory(runtime)`。而 `factory(runtime)` 对应一次完整的 `synthesize_stream(...)` 生成器；这个生成器内部会循环处理所有 `text_chunks`，不是一个 chunk 结束后就释放锁。

锁的位置也需要注意：不是 `app.py` 直接上锁，而是 `app.py` 调用 `runtime_manager.iter_with_runtime()`，锁在 `app_onnx.py` 里的 `OnnxRequestRuntimeManager._locked_runtime()` 中加上。

当前链路：

```text
app.py _run_streaming_job
  -> runtime_manager.iter_with_runtime(...)
     -> OnnxRequestRuntimeManager.iter_with_runtime(...)
        -> with self._locked_runtime(...)
           -> with self._execution_lock
              -> for item in factory(runtime):
                     yield item
```

拆到 `simple_moss` 后，这把锁更适合放在 `api_facade.py` 或 facade 内部的 runtime manager 中，而不是放在未来的 HTTP 路由层。这样无论是本地 test 调用还是 FastAPI 调用，都会遵守同一套单实例串行规则。

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

# ONNX stream generate warmup 放置位置

## 1. 当前旧代码中的 warmup 位置

当前旧代码里，warmup 是由 `app_onnx.py` 的启动流程触发，但管理逻辑在 `app.py` 的 `WarmupManager` 中。

调用关系大致是：

```text
app_onnx.py main()
  -> 创建 OnnxNanoTTSServiceAdapter
  -> 创建 app.py 的 WarmupManager
  -> warmup_manager.start()
  -> WarmupManager 内部调用 runtime.warmup()
```

也就是说，现有 warmup 仍然是 Demo/Web 启动框架的一部分，和 `app.py` / `app_onnx.py` 的服务启动流程绑定较深。

## 2. simple_moss 中建议放置的位置

按照 `simple_moss` 的开发思路，warmup 不应该放在未来的 HTTP 路由层，也不应该继续依赖 `app.py` 的 `WarmupManager`。

更合适的放置方式是：

```text
api_facade.py
  MossStreamFacade.__init__()
    -> 创建 simple_onnx_tts_runtime / simple_ort_cpu_runtime
    -> 可选执行 warmup()

或：

api_facade.py
  MossStreamFacade.warmup()
    -> 显式预热 stream generate 所需路径
```

更推荐第一版提供显式方法：

```python
facade = MossStreamFacade(config)
facade.warmup()

for chunk in facade.stream_generate(text):
    ...
```

这样本地 test 和未来 FastAPI 都能复用同一个 warmup 入口。HTTP 层只负责在启动时调用一次 `facade.warmup()`，不需要知道底层有哪些 ONNX session 要预热。

## 3. 各模块职责

建议拆分后职责如下：

- `simple_ort_cpu_runtime.py`：保留底层 `warmup()` 能力，负责真正预热 ONNX session。
- `simple_onnx_tts_runtime.py`：提供 TTS 层 warmup 所需的内置 voice、prompt codes、测试文本等准备能力。
- `api_facade.py`：提供对外统一 `warmup()`，并决定服务启动时是否调用。
- 未来 FastAPI 层：只调用 `facade.warmup()`，不管理 warmup 细节。

## 4. stream generate warmup 的覆盖重点

如果目标是 ONNX stream generate 服务，warmup 应该覆盖 stream 路径里的 `codec_decode_step`，也就是：

```text
CodecStreamingDecodeSession.run_frames()
```

旧的非流式 warmup 不够准确，可能只覆盖全量解码或非流式合成路径，容易漏掉第一次 stream 解码时 `codec_decode_step` 的冷启动成本。

因此 `simple_moss` 的 warmup 应该面向真正的 stream generate 链路，而不是仅复用旧的非流式 warmup。

# OnnxTtsRuntime 与 OrtCpuRuntime 的分工

## 1. runtime 分层

前面讨论中提到的 “runtime” 是一个泛称。当前 ONNX 路径里实际可以拆成两层：

```text
OnnxTtsRuntime
  -> TTS 业务运行时 / 输入准备层

OrtCpuRuntime
  -> ONNX 推理运行时 / 模型执行层
```

代码关系上，`OnnxTtsRuntime` 继承 `OrtCpuRuntime`。因此一个 `OnnxTtsRuntime` 实例同时拥有 TTS 输入准备能力和父类 `OrtCpuRuntime` 初始化出来的 ONNX session 执行能力。

## 2. OrtCpuRuntime 的职责

`OrtCpuRuntime` 负责底层 ONNX 执行。它关心的是模型 session、张量、KV cache、采样、token 生成和 codec 解码。

典型职责：

- 读取 manifest / meta。
- 创建 ONNX `InferenceSession`。
- 管理 `prefill`、`decode`、`local_decoder`、`codec_decode_step` 等模型 session。
- `build_voice_clone_request_rows`。
- `generate_audio_frames`。
- `CodecStreamingDecodeSession.reset` / `CodecStreamingDecodeSession.run_frames`。
- 采样逻辑：greedy、fixed、full、temperature、top_p、top_k、repetition penalty。
- 维护 codec streaming decode 的 KV cache。

ONNX 输出的 audio token 解码成音频 waveform，也在 `OrtCpuRuntime` 这一层完成，核心位置是：

```text
CodecStreamingDecodeSession.run_frames()
```

stream generate 中的解码链路：

```text
generate_audio_frames()
  -> 产出 audio token frame
  -> on_frame 回调
  -> codec_streaming_session.run_frames(frame_chunk)
  -> codec_decode_step ONNX session
  -> waveform float32
  -> app.py / facade 再转 pcm_s16le bytes
```

所以 `CodecStreamingDecodeSession.run_frames()` 输出的通常还不是最终 `pcm_bytes`，而是 numpy waveform。当前最终 `pcm_s16le bytes` 是在 `app.py` 中通过 `_audio_to_pcm16le_bytes()` 转出来的；拆到 `simple_moss` 后，这一步可以放到 `api_facade.py`。

## 3. OnnxTtsRuntime 的职责

`OnnxTtsRuntime` 负责 TTS 业务输入准备。它关心的是“用户给的文本、音色、参考音频”如何变成 `OrtCpuRuntime` 能吃的输入。

典型职责：

- 确认 / 下载 / 解析 ONNX 模型目录。
- 加载 SentencePiece tokenizer。
- `resolve_prompt_audio_codes`。
- 读取 manifest 中的内置音色 codes。
- 自定义参考音频编码：`encode_reference_audio`。
- 加载音频、重采样、声道处理。
- `split_voice_clone_text`。
- `encode_text`。
- `estimate_voice_clone_inter_chunk_pause_seconds`。
- `_merge_audio_channels`、`_concat_waveforms`、`_write_waveform_to_wav` 等音频包装辅助。

一句话区分：

```text
OnnxTtsRuntime：把 TTS 请求准备成模型输入。
OrtCpuRuntime：拿模型输入跑 ONNX，生成声学 token，再解码成音频。
```

## 4. OnnxTtsRuntime 输入准备过程中会用到哪些 session

需要分清初始化阶段和每次请求的输入准备阶段。

`OnnxTtsRuntime` 初始化时，因为继承 `OrtCpuRuntime`，会通过 `super().__init__()` 创建所有 ONNX sessions，包括 TTS 和 codec 相关 session：

```text
prefill
decode / local decoder 相关 session
codec_encode
codec_decode
codec_decode_step
```

但在每次请求中，`onnx_tts_runtime.py` 做“文本、音色、参考音频等输入准备”时，通常不会跑 TTS 生成模型。

输入准备阶段主要用到：

- `SentencePiece tokenizer`：`encode_text()`、`split_voice_clone_text()`、`count_text_tokens()` 会使用它。这不是 ONNX session。
- manifest 内置音色 codes：如果使用 builtin voice，`resolve_prompt_audio_codes()` 直接读取预编码 audio codes，不跑模型。
- `codec_encode` ONNX session：如果传入自定义 `prompt_audio_path`，`encode_reference_audio()` 会加载参考音频，然后调用 `codec_encode`，把参考音频编码成离散 audio token codes。
- 音频加载 / 重采样库：`_load_reference_audio()` 负责把 wav / mp3 等参考音频转成 codec encoder 需要的 waveform。

因此：

```text
内置 voice:
  onnx_tts_runtime.py 基本只读 manifest + tokenizer，不跑 ONNX session

自定义参考音频:
  onnx_tts_runtime.py 会用 codec_encode session
  把 prompt audio -> prompt audio codes

真正 TTS 生成:
  在 ort_cpu_runtime.py 里用 prefill / decode 等 session

audio token 解码成音频:
  在 ort_cpu_runtime.py 里用 codec_decode_step session
```

`onnx_tts_runtime.py` 这层每次请求中比较核心的 ONNX session 使用点，是自定义参考音频编码时的 `codec_encode`。文本分词和内置音色解析都不是模型推理。

## 5. 是否会创建两套 sessions

不会。继承不是组合创建两个实例。

当前关系类似：

```python
class OnnxTtsRuntime(OrtCpuRuntime):
    def __init__(...):
        ...
        super().__init__(...)
```

当创建：

```python
runtime = OnnxTtsRuntime(...)
```

实际只有一个对象实例。这个对象同时拥有：

```text
OnnxTtsRuntime 自己定义的方法 / 属性
+ OrtCpuRuntime 继承来的方法 / 属性
```

`super().__init__()` 只是执行父类初始化逻辑，在同一个对象上创建 session 属性，例如：

```text
runtime._sessions
runtime.codec_streaming_session
runtime.manifest
runtime.codec_meta
```

所以不是：

```text
OnnxTtsRuntime 实例一套 sessions
OrtCpuRuntime 实例一套 sessions
```

而是：

```text
一个 OnnxTtsRuntime 实例
  里面有父类 OrtCpuRuntime 初始化出来的一套 sessions
```

只有显式创建两个 runtime 对象时，才会创建两套 sessions，例如：

```python
r1 = OnnxTtsRuntime(...)
r2 = OnnxTtsRuntime(...)
```

或者：

```python
r1 = OrtCpuRuntime(...)
r2 = OnnxTtsRuntime(...)
```

这种情况下才会重复加载 ONNX sessions，占用两份内存。

## 6. 当前原逻辑最终创建和使用的对象

原来的 ONNX stream generate 主逻辑里，最终真正创建和使用的是一个 `OnnxTtsRuntime` 对象。

启动时先创建：

```python
runtime = OnnxNanoTTSServiceAdapter(...)
```

然后在 `OnnxNanoTTSServiceAdapter.__init__()` 里面创建：

```python
self.runtime = OnnxTtsRuntime(...)
```

对象关系：

```text
OnnxNanoTTSServiceAdapter
  -> self.runtime: OnnxTtsRuntime
       -> 继承 OrtCpuRuntime 的 ONNX session 能力
```

后续 stream generate 时：

```text
app.py
  -> OnnxRequestRuntimeManager
  -> OnnxNanoTTSServiceAdapter.synthesize_stream()
  -> self.runtime.resolve_prompt_audio_codes()
  -> self.runtime.generate_audio_frames()
  -> self.runtime.codec_streaming_session.run_frames()
```

这里的 `self.runtime` 都是同一个 `OnnxTtsRuntime` 实例。

`OrtCpuRuntime` 不是另外被创建出来的第二个 runtime 对象；它是 `OnnxTtsRuntime` 的父类初始化逻辑，负责在同一个对象里创建 ONNX sessions 和 codec streaming session。

最终可以理解为：

```text
服务适配器：OnnxNanoTTSServiceAdapter
真正 runtime：OnnxTtsRuntime（继承 OrtCpuRuntime）
```

拆 `simple_moss` 时也要注意：如果继续让 `SimpleOnnxTtsRuntime(SimpleOrtCpuRuntime)` 继承，就只创建一个 `SimpleOnnxTtsRuntime` 实例，不要再额外创建一个 `SimpleOrtCpuRuntime` 实例。

# 保留原前端 stream generate 多接口协议

## 1. 新的设计要求

原先计划里，`api_facade.py` 更偏向提供一个简单的：

```text
stream_generate(text) -> Iterator[AudioChunk]
```

但如果希望保留原前端一次 stream generate 的调用模式，那么不能只做单一生成器 facade，还需要保留一层“stream job API facade”，模拟原来的多接口异步协议。

也就是说：底层可以瘦身，但外部调用协议要尽量保持原来的 `/start`、`/audio`、`/status`、`/result`、`/close` 形态。

## 2. 原前端一次 stream generate 调用的后端接口

原前端一次 realtime stream generate 主要调用以下接口：

```text
1. POST /api/generate-stream/start
   -> 创建 stream job
   -> 后台开始生成
   -> 立即返回 stream_id、audio_url、status_url、result_url

2. GET /api/generate-stream/{stream_id}/status
   -> 前端轮询状态
   -> 获取 state、run_status、当前 chunk、emitted_audio_seconds、lead_seconds 等

3. GET /api/generate-stream/{stream_id}/audio
   -> 前端建立音频流
   -> 持续读取 PCM-s16le bytes

4. GET /api/generate-stream/{stream_id}/result
   -> 生成完成后获取最终结果
   -> 当前旧逻辑会返回 audio_base64、text_chunks、audio_chunk_ranges 等

5. POST /api/generate-stream/{stream_id}/close
   -> 前端结束或取消时关闭 stream
   -> 设置 is_closed，清理资源
```

这些接口共同构成的是一个异步 job 协议：

```text
start 创建任务
audio 消费音频流
status 异步轮询进度
result 获取最终结果
close 取消或清理任务
```

## 3. simple_moss 中建议保留两层 facade

为了既保留原接口形式，又让底层生成逻辑保持简单，`simple_moss` 可以拆成两层 facade。

底层生成 facade：

```text
MossStreamFacade.stream_generate(...)
  -> 负责 on_frame
  -> 负责 codec streaming decode
  -> 产出 AudioChunk
```

上层 job/api facade：

```text
MossStreamApiFacade.start(...)
MossStreamApiFacade.status(stream_id)
MossStreamApiFacade.audio(stream_id)
MossStreamApiFacade.result(stream_id)
MossStreamApiFacade.close(stream_id)
```

上层 job/api facade 负责模拟原来的 `/start`、`/status`、`/audio`、`/result`、`/close` 协议，包括：

- 创建和保存 stream job。
- 后台线程启动生成。
- 把底层 `AudioChunk.pcm_bytes` 写入 job 的 audio queue。
- 保存状态字段供 `status()` 查询。
- 保存最终结果供 `result()` 查询。
- 支持 `close()` 取消和清理。

## 4. 本地调用也按原接口语义执行

即便第一版不通过 HTTP，本地 test 也可以按原接口形式调用：

```python
api = MossStreamApiFacade(config)

start = api.start(text="...")
stream_id = start["stream_id"]

for pcm in api.audio(stream_id):
    status = api.status(stream_id)
    ...

result = api.result(stream_id)
api.close(stream_id)
```

这样本地测试覆盖的不只是底层生成链路，还覆盖了原前端依赖的异步 job 语义。

## 5. 未来 FastAPI 层的映射方式

未来如果要包 HTTP 服务，FastAPI 层可以非常薄，只做路由到 `MossStreamApiFacade` 方法的映射：

```text
POST /api/generate-stream/start
  -> api.start(...)

GET /api/generate-stream/{stream_id}/status
  -> api.status(stream_id)

GET /api/generate-stream/{stream_id}/audio
  -> StreamingResponse(api.audio(stream_id))

GET /api/generate-stream/{stream_id}/result
  -> api.result(stream_id)

POST /api/generate-stream/{stream_id}/close
  -> api.close(stream_id)
```

这样可以做到：

- 保留原前端和外部调用习惯。
- 保留异步查询和获取结果的流程。
- 避免把 FastAPI、UI、Demo 逻辑混进底层 runtime。
- 让 `api_facade.py` 成为可本地测试、可 HTTP 包装的稳定中间层。

## 6. 对原计划的修正

原计划中“只提供 `stream_generate(text)`”适合极简 TTS stream 服务，但不完全满足保留原前端协议的目标。

修正后的方向是：

```text
simple_ort_cpu_runtime.py
  -> ONNX 核心推理

simple_onnx_tts_runtime.py
  -> TTS 输入准备

MossStreamFacade
  -> 单次底层 stream generate

MossStreamApiFacade
  -> 原多接口异步 job 协议

FastAPI app
  -> 薄路由层，映射到 MossStreamApiFacade
```

## 7. 五个接口的调用时序

这五个接口不是按顺序一次性调用完，而是并行配合完成一次 stream generate。

整体时序：

```text
用户点击 Generate
  -> POST /api/generate-stream/start 创建任务
  -> 前端拿到 stream_id / audio_url / status_url / result_url

随后同时做两件事：
  A. GET /api/generate-stream/{stream_id}/audio 打开音频流，持续播放 PCM
  B. GET /api/generate-stream/{stream_id}/status 定时轮询状态，更新 UI / 进度

/audio 读完后：
  -> GET /api/generate-stream/{stream_id}/result 获取最终结果

页面离开、用户取消、重新生成或结果获取后：
  -> POST /api/generate-stream/{stream_id}/close 清理任务
```

更简化的示意：

```text
POST /start
   |
   |-- GET /audio   ====================> 持续读取 PCM，直到结束
   |
   |-- GET /status  -> -> -> -> -> -> ->  定时轮询
   |
   |-- GET /result  生成结束后获取最终结果
   |
   '-- POST /close  取消或最终清理
```

各接口调用时机：

- `POST /api/generate-stream/start`：用户点击生成后第一个调用。提交文本、prompt audio、采样参数等表单数据；后端创建 `StreamingJob`，启动后台线程生成音频，并立即返回 `stream_id`、`audio_url`、`status_url`、`result_url`。
- `GET /api/generate-stream/{stream_id}/audio`：拿到 `audio_url` 后调用。这个请求会一直挂着，作为 `StreamingResponse` 持续返回 PCM-s16le bytes。它不是轮询，而是持续读取的音频流。
- `GET /api/generate-stream/{stream_id}/status`：在音频播放期间定时调用，例如每隔几百毫秒或 1 秒一次。用于更新 UI：当前状态、已生成多少秒、lead、当前播放 / 生成到哪个 text chunk、是否 failed / done 等。
- `GET /api/generate-stream/{stream_id}/result`：通常在 `/audio` 流结束后调用，或者发现 `/status` 里 `ready=true` 后调用。旧前端里会在 audio 读取结束后最多重试多次 `/result`，因为后台 job 可能刚结束但 result 还没完全 ready。
- `POST /api/generate-stream/{stream_id}/close`：用于取消或清理。用户切换 demo、重新生成、页面关闭、出错时会调用；正常 result 获取完成后也可以调用 close 做最终清理。

因此 `simple_moss` 的本地 test 如果要保留原协议语义，也应该模拟这个过程：先 `start()`，然后一边消费 `audio()`，一边轮询 `status()`，最后 `result()`，最后 `close()`。

## 8. `/result` 接口的作用与瘦身策略

如果目标只是外部实时播放 PCM 流，`/result` 不是必须的。

原来的 `/result` 更多是为了 Demo 前端的“最终产物”和“回放展示”服务，不是流式播放本身必需。

它当前主要有几个作用：

- 获取完整 WAV / base64：`/audio` 返回的是裸 `pcm_s16le` 流，适合实时播放，但不是一个带 header 的完整音频文件。`/result` 会返回 `audio_base64`，旧逻辑里来自最终写出的 WAV 文件，方便前端生成可保存 / 可下载 / 可回放的完整音频。
- 获取最终元数据：例如 `text_chunks`、`audio_chunk_ranges`、`run_status`、`stream_metrics`、`prompt_audio_path`，用于展示“生成完成”和每段文本对应哪段音频时间。
- 确认后台任务最终完成：`/audio` 流结束说明音频队列没有更多 bytes 了，但旧架构里最终 result payload、WAV 写盘、base64 懒加载等逻辑是另一条状态链。前端通过 `/result` 确认 job 的最终状态和最终产物。

瘦身后的 `simple_moss` 可以保留 `/result` 的接口形式，但降级为最终状态接口，不一定返回 `audio_base64`。

第一版可以返回：

```json
{
  "stream_id": "...",
  "ready": true,
  "state": "done",
  "run_status": "Stream complete.",
  "text_chunks": [],
  "audio_chunk_ranges": []
}
```

如果不需要回放映射，也可以进一步简化为：

```json
{
  "stream_id": "...",
  "ready": true,
  "state": "done"
}
```

结论：

- `/audio`：实时音频数据，播放用。
- `/status`：生成过程中查状态，UI / 监控用。
- `/result`：生成结束后的最终确认和附加结果；旧 Demo 还用它拿完整 WAV / base64。
- 瘦身服务中建议保留 `/result` 的接口，但去掉 `audio_base64` 和写 WAV 逻辑。

## 9. `/close` 与 stop_event / 后端中断语义

当前没有 `/stop` 接口，对应的是：

```text
POST /api/generate-stream/{stream_id}/close
```

当前 `/close` 会中断一部分流程，但不是强制立即打断底层 ONNX 推理。

它主要做的是：

```text
job.is_closed = True
job.audio_queue.put_nowait(None)
```

因此它能立即影响：

- 前端 `/audio` 流会结束。
- `StreamingJob` 状态会变成 closed。
- `_run_streaming_job` 后续消费 `synthesize_stream()` 产出的 event 时，如果看到 `job.is_closed`，会 break。
- `_run_streaming_job` break 后，可能间接触发 `app_onnx.py` 的 `synthesize_stream()` generator 的 `finally`，然后设置内部 `_stop_event`。

但关键问题是：这个 `_stop_event` 没有传进 `generate_audio_frames()` 的自回归循环里。

所以如果后台已经进入某个 text chunk 的 ONNX 生成循环，它不会因为 `/close` 立刻停下来。它通常会继续跑到：

```text
当前 text chunk 生成结束
或 generate_audio_frames 返回
或后续 _worker 在 chunk 边界检查到 _stop_event
```

因此准确结论是：

```text
/close 会停止前端流和上层消费，
但不会强制立即中断当前底层 ONNX 推理。
```

### 9.1 当前存在的两个结束信号

当前代码里有两层 `None` sentinel，容易和普通 event 混淆。

第一层是 `app_onnx.py` 内部的 `event_queue`：

```text
_worker 结束时 _safe_put(None)
```

作用是告诉 `synthesize_stream()` 外层生成器：“底层 worker 完成了，可以退出”。

第二层是 `app.py` 的 `job.audio_queue`：

```text
_run_streaming_job 结束时 job.audio_queue.put_nowait(None)
```

作用是告诉 `/audio` 的 `StreamingResponse`：“没有更多音频了，可以结束响应”。

`/close` 会往 `job.audio_queue` 放一个 `None`，让前端 `/audio` 流结束；但它不会直接往 `app_onnx.py` 的内部 `event_queue` 放 `None`，也不会直接生成一个 `"result"` 或 `"end"` event。

### 9.2 `stop_event` 的现状

`agent-notes/260430-执行中断（初步想法）/chat-notes.md` 中提到的 `stop_event`，对应的是 `app_onnx.py` 的 `synthesize_stream()` 内部 stop 信号。

它不是 `/close` 直接创建的 HTTP event，而是 `synthesize_stream()` 内部用来通知 `_worker` 停止的 `threading.Event`：

```text
synthesize_stream()
  -> 创建 _stop_event
  -> _worker 线程生成音频
  -> 外层 generator 消费 event_queue
  -> generator 退出 / GeneratorExit 时，在 finally 里 _stop_event.set()
```

当前 `_stop_event` 的检查粒度较粗：

```text
text_chunks 循环开始检查 _stop_event
  -> 进入某个 text chunk
  -> generate_audio_frames(...) 跑完整个 chunk
text_chunks 循环结束附近再检查 _stop_event
```

所以如果 `/close` 发生在某个 chunk 正在 `generate_audio_frames()` 里自回归生成时，即使 `_stop_event` 被设置，也很难立刻打断当前 chunk。通常要等当前 chunk 生成结束。

### 9.3 simple_moss 中建议的中断设计

如果 `simple_moss` 要真正支持执行中断，建议把 `stop_event` 提升成 `StreamJob` 的一部分，而不是只藏在 `synthesize_stream()` 的局部变量里。

目标链路：

```text
POST /close
  -> job.is_closed = True
  -> job.stop_event.set()
  -> audio_queue 放 None，让前端音频流结束
  -> generate_audio_frames 内部下一次检查 stop_event 时退出
```

同时需要改造底层接口，让取消信号传进 `generate_audio_frames()`：

```text
generate_audio_frames(..., stop_event=...)
```

或：

```text
generate_audio_frames(..., is_cancelled=...)
```

建议在两个关键点检查：

```text
1. prefill 完成后检查
   如果取消，直接 return []，不进入 decode 循环

2. decode 每帧循环中检查
   如果取消，break
```

这样 `/close` 才能不只停止前端流和上层消费，还能让底层 ONNX decode 循环尽快、可控地退出。
