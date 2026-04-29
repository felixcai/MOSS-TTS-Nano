# warmup 修复分析笔记（2026-04-29）

## 1 问题描述

`OnnxNanoTTSServiceAdapter.warmup`（被 `app.py` 的 `WarmupManager` 在服务启动时调用）只走非流式路径（`self.synthesize()`），导致 **`codec_decode_step` Session 在服务启动后仍是冷的**。

第一次 `synthesize_stream` 请求的 codec 流式解码阶段（`_decode_pending` → `codec_streaming_session.run_frames`）会触发 `codec_decode_step` 的首次 JIT 编译 / ORT Arena 预分配，产生额外的首帧延迟。

---

## 2 各 ONNX Session 的预热覆盖现状

| ONNX Session                            | stream generate 是否需要 | 当前 `OnnxNanoTTSServiceAdapter.warmup` | `OrtCpuRuntime.warmup` |
| --------------------------------------- | -------------------- | ------------------------------------- | ---------------------- |
| `prefill`                               | ✅                    | ✅ 已预热                                 | ✅ 已预热                  |
| `local_cached_step` / `local_decoder` 等 | ✅                    | ✅ 已预热                                 | ✅ 已预热                  |
| `codec_encode`                          | 仅自定义音频路径             | ❌（内置音色跳过）                             | ❌（同样没有）                |
| `codec_decode`（全量解码）                    | ❌（非流式路径专用）           | ✅ 顺带覆盖                                | ✅ 顺带覆盖                 |
| `codec_decode_step`（逐帧流式解码）             | ✅                    | ❌ **遗漏**                              | ✅ 已预热                  |

- `codec_decode` 对 stream generate **不需要**，是 warmup 顺带覆盖的，无害但多余。
- `codec_encode` 两种 warmup 方案都没有覆盖；只有用户上传自定义参考音频时才走该 Session，内置音色读 manifest 预编码数据跳过，暂不处理。

---

## 3 `OrtCpuRuntime.warmup` 的调用现状

`OrtCpuRuntime.warmup` **是死代码，无任何调用方**。

- `app.py` 的 `WarmupManager` 调用的是 `self.runtime.warmup()`，这里的 `runtime` 是 `OnnxNanoTTSServiceAdapter`，走的是 `app_onnx.py` 里重写的 `warmup`，不会继续调到 `OrtCpuRuntime.warmup`。
- `onnx_tts_runtime.py` 里完全没有调用 warmup。

---

## 4 修复方案结论

### 方案 A：在现有 `self.synthesize()` 后追加 3 行（最小改动）

在 `app_onnx.py` 的 `OnnxNanoTTSServiceAdapter.warmup` 末尾追加：

```python
def warmup(self) -> dict[str, object]:
    voice_name = (...)
    result = self.synthesize(text="Warmup.", ...)   # 原有逻辑，改为先存 result
    # 补充预热 codec_decode_step（stream 路径专用，self.synthesize() 不覆盖）
    n_vq = int(self.runtime.manifest["tts_config"]["n_vq"])
    empty_frames = [([0] * n_vq)]
    self.runtime.codec_streaming_session.reset()
    self.runtime.codec_streaming_session.run_frames(empty_frames)
    self.runtime.codec_streaming_session.reset()
    return result
```

优点：改动最小，`FIXED_BUILTIN_VOICE` 逻辑完全保留。
缺点：需要在两处改（app_onnx.py），且 `codec_decode`（全量）仍被多余地预热。

---

### 方案 B：给 `OrtCpuRuntime.warmup` 加 `voice_name` 参数，整体替换（推荐）

**第一步**：修改 `ort_cpu_runtime.py`，给 `warmup` 加 keyword-only 参数 `voice_name`，默认 `None`（兼容原逻辑）：

```python
def warmup(self, *, voice_name: str | None = None) -> None:
    voices = self.list_builtin_voices()
    if voice_name is not None:
        voice = next((v for v in voices if v["voice"] == voice_name), voices[0])
    else:
        voice = voices[0]
    text_sample = self.list_text_samples()[0]
    # ... 后续 prefill / local_decoder / codec_decode / codec_decode_step 不变 ...
```

**第二步**：修改 `app_onnx.py`，`OnnxNanoTTSServiceAdapter.warmup` 整体替换为调用 `self.runtime.warmup`：

```python
def warmup(self) -> dict[str, object]:
    self.runtime.warmup(voice_name=FIXED_BUILTIN_VOICE)
    return {}   # app.py WarmupManager 期望拿到 dict
```

优点：一次覆盖全部所需 Session（包含 `codec_decode_step`），`FIXED_BUILTIN_VOICE` 得到传递，代码更简洁。
缺点：需修改 `ort_cpu_runtime.py` 和 `app_onnx.py` 两个文件。

---

### 方案对比

| 对比点                                   | 方案 A（追加 3 行）        | 方案 B（整体替换）                          |
| ------------------------------------- | ------------------- | ----------------------------------- |
| `prefill` 预热                          | ✅                   | ✅                                   |
| `local_cached_step` / `local_decoder` | ✅                   | ✅                                   |
| `codec_decode`（全量，非流式用）               | ✅（多余但无害）            | ✅（多余但无害）                            |
| `codec_decode_step`（流式）               | ✅ 修复后覆盖             | ✅ 修复后覆盖                             |
| `codec_encode`                        | ❌（两方案一致）            | ❌（两方案一致）                            |
| 使用 FIXED_BUILTIN_VOICE                | ✅                   | ✅（传入 voice_name）                    |
| 改动文件数                                 | 1（仅 app_onnx.py）    | 2（ort_cpu_runtime.py + app_onnx.py） |
| 代码整洁度                                 | 一般（synthesize + 补丁） | 好（单一调用入口）                           |

---

## 5 未解决的盲点

`codec_encode` Session 在两种方案下均未预热。

- **触发路径**：只有用户上传自定义参考音频时，`resolve_prompt_audio_codes` 才会调用 `encode_reference_audio` → `codec_encode`。
- **内置音色路径**：直接读 manifest 预编码数据，跳过 `codec_encode`。
- **处理建议**：如需覆盖，可在 warmup 里额外调用一次 `self.runtime.encode_reference_audio(某个内置音频文件路径)`，但需要准备一个实际音频文件，改动略复杂，可按需决定是否处理。
