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
