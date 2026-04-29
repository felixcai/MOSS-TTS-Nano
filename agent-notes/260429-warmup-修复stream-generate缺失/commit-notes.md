# commit e1187b5 — 采用方案 B 整体替换 warmup 逻辑，并省去 codec_decode 预热

## 改动文件

- `app_onnx.py`
- `ort_cpu_runtime.py`

## 问题背景

`OnnxNanoTTSServiceAdapter.warmup` 原实现只调用 `self.synthesize()`（非流式路径），
该路径走 `codec_decode`（全量解码），不会触发 `codec_decode_step`（逐帧流式解码）Session。
同时，对于仅使用 stream generate 的场景，`codec_decode` Session 的加载和预热是多余的，白白占用了大量系统内存和显存。

## 修复内容 (方案 B)

### 1. `ort_cpu_runtime.py`：`OrtCpuRuntime.warmup`

**修改前**

```python
def warmup(self) -> None:
    voice = self.list_builtin_voices()[0]
    ...
    empty_frames = [([0] * int(self.manifest["tts_config"]["n_vq"]))]
    self.decode_full_audio(empty_frames)
    ...
```

**修改后**

```python
def warmup(self, *, voice_name: str | None = None) -> None:
    voices = self.list_builtin_voices()
    if voice_name is not None:
        voice = next((v for v in voices if v["voice"] == voice_name), voices[0])
    else:
        voice = voices[0]
    ...
    empty_frames = [([0] * int(self.manifest["tts_config"]["n_vq"]))]
    # self.decode_full_audio(empty_frames)  # 注释掉全量解码预热
    ...
```

**改动要点**

1. 增加 `voice_name` 参数，允许指定预热音色。
2. 注释掉 `self.decode_full_audio(empty_frames)`，彻底移除对 `codec_decode` Session 的预热依赖，使得后续可以在 `_create_sessions` 中安全地不加载该模型以节省内存。

### 2. `app_onnx.py`：`OnnxNanoTTSServiceAdapter.warmup`

**修改前**

```python
def warmup(self) -> dict[str, object]:
    voice_name = (...)
    return self.synthesize(...)  # 走非流式完整推理
```

**修改后**

```python
def warmup(self) -> dict[str, object]:
    _log_memory("warmup: start")
    voice_name = (...)

    t0 = time.perf_counter()
    self.runtime.warmup(voice_name=voice_name)
    t1 = time.perf_counter()

    _log_memory("warmup: complete (codec_decode_step session done)")

    return {
        "elapsed_seconds": t1 - t0,
        "audio_path": None,  # 兼容 app.py 中的 _maybe_delete_file 逻辑
    }
```

**改动要点**

1. 废弃原有的 `self.synthesize()` 预热方式，改为直接调用底层 `self.runtime.warmup()`。
2. 构造兼容的返回字典，包含 `elapsed_seconds` 和 `audio_path: None`，以满足 `app.py` 中 `WarmupManager` 的断言和文件清理逻辑。

## 预热覆盖情况（方案 B 修复后）

| ONNX Session                            | stream generate 需要 | 修复后覆盖                    |
| --------------------------------------- | ------------------ | ------------------------ |
| `prefill`                               | ✅                  | ✅（`runtime.warmup()` 路径） |
| `local_cached_step` / `local_decoder` 等 | ✅                  | ✅（`runtime.warmup()` 路径） |
| `codec_decode`（全量解码）                    | ❌ 非流式专用            | ❌ **已移除预热，可安全不加载**       |
| `codec_decode_step`（逐帧流式解码）             | ✅                  | ✅（`runtime.warmup()` 路径） |
| `codec_encode`                          | 仅自定义音频路径           | ❌ 仍未覆盖（内置音色跳过，暂不处理）      |

---

# commit d0a2ddf — 在 app_onnx 的 warmup 里，补上了 stream generate 的预热

## 改动文件

- `app_onnx.py`（唯一改动文件）

## 问题背景

`OnnxNanoTTSServiceAdapter.warmup` 原实现只调用 `self.synthesize()`（非流式路径），
该路径走 `codec_decode`（全量解码），不会触发 `codec_decode_step`（逐帧流式解码）Session。

结果：服务启动后 `codec_decode_step` Session 仍是冷的，第一次 `synthesize_stream` 请求
的 codec 流式解码阶段（`_decode_pending` → `codec_streaming_session.run_frames`）会触发
该 Session 的首次 JIT 编译 / ORT Arena 预分配，产生额外的首帧延迟。

## 修复内容

### `app_onnx.py`：`OnnxNanoTTSServiceAdapter.warmup`

**修改前**

```python
def warmup(self) -> dict[str, object]:
    voice_name = (
        FIXED_BUILTIN_VOICE
        if FIXED_BUILTIN_VOICE is not None
        else str(self.runtime.list_builtin_voices()[0]["voice"])
    )
    return self.synthesize(
        text="Warmup.",
        ...
    )
```

**修改后**

```python
def warmup(self) -> dict[str, object]:
    voice_name = (...)
    result = self.synthesize(       # 改为先存 result，不直接 return
        text="Warmup.",
        ...
    )
    # 补充预热 codec_decode_step
    n_vq = int(self.runtime.manifest["tts_config"]["n_vq"])
    empty_frames = [([0] * n_vq)]
    self.runtime.codec_streaming_session.reset()
    self.runtime.codec_streaming_session.run_frames(empty_frames)
    self.runtime.codec_streaming_session.reset()
    return result
```

**改动要点**

1. `return self.synthesize(...)` → 改为 `result = self.synthesize(...)`，之后再 `return result`，接口行为不变
2. 在 `synthesize()` 完成后追加 3 行：`reset` → `run_frames(empty_frames)` → `reset`，触发 `codec_decode_step` Session 的首次执行完成预热
3. 两次 `reset()` 的作用：第一次清除 Session 初始化时可能残留的状态，第二次确保预热产生的 KV Cache 不污染第一次真实请求

## 预热覆盖情况（修复后）

| ONNX Session                            | stream generate 需要 | 修复后覆盖                      |
| --------------------------------------- | ------------------ | -------------------------- |
| `prefill`                               | ✅                  | ✅（`synthesize()` 路径）       |
| `local_cached_step` / `local_decoder` 等 | ✅                  | ✅（`synthesize()` 路径）       |
| `codec_decode`（全量解码）                    | ❌ 非流式专用            | ✅ 顺带（`synthesize()` 路径，无害） |
| `codec_decode_step`（逐帧流式解码）             | ✅                  | ✅ **本次修复新增**               |
| `codec_encode`                          | 仅自定义音频路径           | ❌ 仍未覆盖（内置音色跳过，暂不处理）        |
