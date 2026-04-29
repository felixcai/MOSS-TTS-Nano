# commit 6e689f6 — 把方案B的预热逻辑，改成和方案A同样 （但仍然去掉了全量解码）

## 改动文件

- `ort_cpu_runtime.py`

## 问题背景

在采用方案 B（直接调用底层 `runtime.warmup()`）后，虽然省去了多余的 `codec_decode`（全量解码）预热，但发现底层的原生预热逻辑过于简陋：它只执行了 `prefill` 和 1 次局部解码器，**完全没有预热 `decode` Session**（自回归核心模型）。这会导致第一次真实请求时，自回归循环的第一步产生冷启动延迟。

为了达到和原方案 A（模拟真实请求生成 16 帧）一样的预热深度，我们需要修改底层的预热逻辑。

## 修复内容

### `ort_cpu_runtime.py`：`OrtCpuRuntime.warmup`

**修改前**

```python
def warmup(self, *, voice_name: str | None = None) -> None:
    ...
    # 冗长的手动 prefill 和 1 次 local_decoder 调用
    outputs = self.sessions["prefill"].run(...)
    ...
    if "local_cached_step" in self.sessions:
        self.run_local_cached_step(...)
    ...

    empty_frames = [([0] * int(self.manifest["tts_config"]["n_vq"]))]
    # self.decode_full_audio(empty_frames)
    self.codec_streaming_session.reset()
    self.codec_streaming_session.run_frames(empty_frames)
    self.codec_streaming_session.reset()
```

**修改后**

```python
def warmup(self, *, voice_name: str | None = None) -> None:
    ...
    # 强制设置 max_new_frames 为 16 (与方案 A 保持一致)，以触发完整的 generate_audio_frames 循环（包含 decode Session）
    original_max_new_frames = self.manifest["generation_defaults"]["max_new_frames"]
    self.manifest["generation_defaults"]["max_new_frames"] = 16

    try:
        # 直接调用 generate_audio_frames，这会预热 prefill, local_decoder, 以及 decode
        generated_frames = self.generate_audio_frames(request_rows)
    finally:
        # 恢复原始配置
        self.manifest["generation_defaults"]["max_new_frames"] = original_max_new_frames

    # ... 原有的冗长预热逻辑被注释保留 ...

    # 预热流式解码器
    self.codec_streaming_session.reset()
    # 传入刚才生成的真实帧（16帧）进行流式解码预热
    self.codec_streaming_session.run_frames(generated_frames)
    self.codec_streaming_session.reset()
```

**改动要点**

1. 废弃了手动拼接 `prefill` 和 `local_decoder` 的冗长代码（改为注释保留）。
2. 临时将 `max_new_frames` 覆盖为 `16`，然后直接调用核心的 `generate_audio_frames` 函数。这强制模型走完 16 次自回归循环，完美预热了 `prefill`、局部解码器以及 **`decode` Session**。
3. 预热流式解码器（`codec_decode_step`）时，不再使用全 0 的 `empty_frames`，而是传入刚才生成的 16 帧真实数据 `generated_frames`，更贴近真实场景。
4. 依然保持 `decode_full_audio` 被注释，确保全量解码不被加载。

## 预热覆盖情况（最终版）

| ONNX Session                            | stream generate 需要 | 最终覆盖情况                        |
| --------------------------------------- | ------------------ | ----------------------------- |
| `prefill`                               | ✅                  | ✅（`generate_audio_frames` 路径） |
| `local_cached_step` / `local_decoder` 等 | ✅                  | ✅（`generate_audio_frames` 路径） |
| **`decode`**                            | ✅                  | ✅ **（本次修复新增，自回归核心）**          |
| `codec_decode`（全量解码）                    | ❌ 非流式专用            | ❌ **已移除预热，可安全不加载**            |
| `codec_decode_step`（逐帧流式解码）             | ✅                  | ✅（传入真实 16 帧预热）                |
| `codec_encode`                          | 仅自定义音频路径           | ❌ 仍未覆盖（内置音色跳过，暂不处理）           |

---

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
