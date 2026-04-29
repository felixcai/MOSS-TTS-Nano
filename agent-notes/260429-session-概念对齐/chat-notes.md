# ONNX Runtime `InferenceSession`（本仓库里说的 session）概念对齐

本文从“`session` 到底是什么”这个问题开始，把概念、与本仓库代码的对应关系、以及当前实现里 **实际会加载哪些 session**、**在什么路径调用** 做一次对齐整理。

## `session` 是什么（通用概念）

在本项目语境里，`session` 基本等同于 **`onnxruntime.InferenceSession`**：把某个 `.onnx` 模型（计算图 + 权重引用）加载进进程，并准备好反复执行 `run()` 的运行时对象。

它通常包含这些“重资源”：

- **模型权重与图结构**：往往对应磁盘上的 `*.onnx`，大权重可能外置到 `*.data`（外部权重）。
- **Execution Provider（EP）**：例如 `CUDAExecutionProvider` / `CPUExecutionProvider`，决定算子跑在 GPU 还是 CPU。
- **运行时缓存与分配器行为**：例如 workspace、arena、首次 kernel 初始化等；这会让“第一次推理”看起来特别吃内存/显存。

直观类比：

- `.onnx`：电路原理图 + 对外接口（输入输出张量名）
- `.data`（如有）：大仓库里的元器件本体（权重 tensor 的外部存储）
- `InferenceSession`：把图和权重接好并上电，之后可以反复 `run`

## 本仓库里 `self.sessions[...]` 是什么

`OrtCpuRuntime` 在初始化时创建 `self.sessions: dict[str, ort.InferenceSession]`，用字符串 key 管理多个 ONNX 模型，例如 `prefill`、`decode`、`codec_encode` 等。

创建入口：

```465:486:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\ort_cpu_runtime.py
    def _create_sessions(self) -> dict[str, ort.InferenceSession]:
        tts_dir = self.tts_meta_path.parent
        codec_dir = self.codec_meta_path.parent
        sessions: dict[str, ort.InferenceSession] = {
            "prefill": self._session(tts_dir / self.tts_meta["files"]["prefill"]),
            "decode": self._session(tts_dir / self.tts_meta["files"]["decode_step"]),
            # "local_decoder": self._session(tts_dir / self.tts_meta["files"]["local_decoder"]),
        }
        # if self.tts_meta["files"].get("local_greedy_frame"):
        #     sessions["local_greedy_frame"] = self._session(tts_dir / self.tts_meta["files"]["local_greedy_frame"])
        if self.tts_meta["files"].get("local_fixed_sampled_frame"):
            sessions["local_fixed_sampled_frame"] = self._session(
                tts_dir / self.tts_meta["files"]["local_fixed_sampled_frame"]
            )
        # if self.tts_meta["files"].get("local_cached_step"):
        #     sessions["local_cached_step"] = self._session(tts_dir / self.tts_meta["files"]["local_cached_step"])
        _log_memory("runtime_init: _create_sessions TTS part done (prefill/decode/local_*)")
        sessions["codec_encode"] = self._session(codec_dir / self.codec_meta["files"]["encode"])
        # sessions["codec_decode"] = self._session(codec_dir / self.codec_meta["files"]["decode_full"])
        sessions["codec_decode_step"] = self._session(codec_dir / self.codec_meta["files"]["decode_step"])
        _log_memory("runtime_init: _create_sessions codec part done (encode/decode_full/decode_step)")
        return sessions
```

### 磁盘路径怎么拼出来

- TTS 模型目录：`tts_dir = dirname(tts_meta.json)`
- Codec 模型目录：`codec_dir = dirname(codec_meta.json)`
- 具体文件名：来自 `tts_meta["files"][...]` / `codec_meta["files"][...]`

仓库示例（注意：字段名 `decode_step` 映射到 session key `decode`）：

```4:10:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\model_files\tts_browser_onnx_meta.json
  "files": {
    "prefill": "moss_tts_prefill.onnx",
    "decode_step": "moss_tts_decode_step.onnx",
    "local_decoder": "moss_tts_local_decoder.onnx",
    "local_cached_step": "moss_tts_local_cached_step.onnx",
    "local_fixed_sampled_frame": "moss_tts_local_fixed_sampled_frame.onnx"
  },
```

外部权重映射（解释为什么目录里会有 `*_shared.data`）：

```11:26:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\model_files\tts_browser_onnx_meta.json
  "external_data_files": {
    "moss_tts_prefill.onnx": [
      "moss_tts_global_shared.data"
    ],
    "moss_tts_decode_step.onnx": [
      "moss_tts_global_shared.data"
    ],
    "moss_tts_local_decoder.onnx": [
      "moss_tts_local_shared.data"
    ],
    "moss_tts_local_cached_step.onnx": [
      "moss_tts_local_shared.data"
    ],
    "moss_tts_local_fixed_sampled_frame.onnx": [
      "moss_tts_local_shared.data"
    ]
  },
```

## `global` / `local` 在本仓库里通常指什么

这不是 Python 的“全局变量”，而是 **模型拆分后的工程命名**：

- **global（`prefill` + `decode_step` + `moss_tts_global_shared.data`）**：更偏 **跨时间步** 的序列状态（长序列 KV、`global_hidden` 等），负责“沿着 token/帧序列推进”的主干。
- **local（`local_*` + `moss_tts_local_shared.data`）**：更偏 **帧内/局部** 的生成细节（多 VQ 声道如何一步步确定），常见是整帧一次推理（`local_fixed_sampled_frame`）或逐声道缓存（`local_cached_step`）或 fallback（`local_decoder`）。

`tts_browser_onnx_meta.json` 里也能看到 `global_layers` 与 `local_layers` 的配置并列存在（这是命名来源之一）。

## `codec_streaming_session` 和 `codec_decode_step` 的关系

`codec_decode_step` 是一个 `InferenceSession`，但它还被包进 `CodecStreamingDecodeSession`，用于维护流式解码 KV 状态：

```409:415:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\ort_cpu_runtime.py
        self.codec_streaming_session = CodecStreamingDecodeSession(
            codec_meta=self.codec_meta,
            session=self.sessions["codec_decode_step"],
        )
```

真正 `run()` 的位置在 `CodecStreamingDecodeSession.run_frames()`：

```331:347:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\ort_cpu_runtime.py
    def run_frames(self, frame_rows: list[list[int]]) -> tuple[np.ndarray, int] | None:
        ...
        feeds.update(self.state_feeds)
        outputs = self.session.run(None, feeds)
```

## 当前代码“实际会加载”的 session 清单（以 `_create_sessions` 为准）

结合当前实现，通常会创建：

- `prefill`
- `decode`（注意：来自 `tts_meta["files"]["decode_step"]`）
- `local_fixed_sampled_frame`（条件：`tts_meta["files"]` 中存在该键）
- `codec_encode`
- `codec_decode_step`

当前实现里 **显式注释掉** 的创建项（因此默认不应存在对应 `self.sessions[...]`）：

- `local_decoder`
- `local_cached_step`
- `local_greedy_frame`（且当前 `tts_browser_onnx_meta.json` 也未声明该文件）
- `codec_decode`（full decode）

风险提醒：`ort_cpu_runtime.py` 里仍保留了对 `local_decoder` / `local_cached_step` / `local_greedy_frame` / `codec_decode` 的 `run()` 代码路径；如果运行时分支命中但 session 未创建，会出现运行时错误。是否安全取决于 `generate_audio_frames` 的分支条件与实际部署参数。

## 这些 session 分别在什么业务路径里使用

### TTS：`generate_audio_frames`

核心使用：

- `sessions["prefill"].run(...)`：chunk 开始时的 prefill
- `sessions["decode"].run(...)`：每生成一帧后更新跨帧 KV / `global_hidden`
- 可能的 local 分支：`sessions["local_fixed_sampled_frame"]` /（若启用）`local_cached_step` / `local_greedy_frame` / fallback `local_decoder`

典型调用入口：

- **warmup**：`OrtCpuRuntime.warmup()` → `generate_audio_frames(...)`（不带 `on_frame`）

```513:575:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\ort_cpu_runtime.py
            generated_frames = self.generate_audio_frames(
                request_rows,
                mem_trace_label="warmup: generate_audio_frames",
            )
        ...
        self.codec_streaming_session.run_frames(generated_frames)
```

- **stream generate**：`app_onnx.py` → `generate_audio_frames(..., on_frame=...)`，`on_frame` 触发 codec 流式解码与 PCM 缓冲

```470:481:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\app_onnx.py
                    def _on_frame(...):
                        pending_decode_frames.append(list(frame))
                        _decode_pending(False)

                    try:
                        generated_frames = self.runtime.generate_audio_frames(request_rows, on_frame=_on_frame)
                        _decode_pending(True)
```

补充：warmup 的 `generate_audio_frames` **不传 `on_frame`**，因此它主要覆盖 **TTS token 生成 + TTS decode-step**；codec 流式解码通常在同一次 warmup 的 **`codec_streaming_session.run_frames`** 里单独发生（与 stream 的“交错发生”不完全相同）。

### Codec：`codec_encode`

仅在“用户提供自定义参考音频路径、需要实时编码 prompt”路径中使用：

```614:626:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\onnx_tts_runtime.py
        if prompt_audio_path:
            return self.encode_reference_audio(prompt_audio_path)
```

```587:597:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\onnx_tts_runtime.py
        outputs = self.sessions["codec_encode"].run(
            None,
            {
                "waveform": waveform,
                "input_lengths": np.asarray([waveform_length], dtype=np.int32),
            },
        )
```

说明：`codec_encode` session 可能在初始化就加载，但 **只有走自定义 prompt 音频才会频繁 `run()`**；内置音色读取预编码 codes 时可能用不到。

### Codec：`codec_decode_step`（通过 `codec_streaming_session`）

- **stream**：`app_onnx.py` 的 `_decode_pending` → `codec_streaming_session.run_frames`

```457:458:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\app_onnx.py
                        decoded = self.runtime.codec_streaming_session.run_frames(frame_chunk)
```

- **warmup**：`OrtCpuRuntime.warmup()` 末尾 `run_frames(generated_frames)`

```571:575:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\ort_cpu_runtime.py
        self.codec_streaming_session.run_frames(generated_frames)
```

## 前端 “greedy / do_sample” 与 `local_greedy_frame` 不是一回事

前端 `do_sample` 表单字段：

```1735:1749:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\app.py
      formData.append("do_sample", document.getElementById("do-sample").checked ? "1" : "0");
```

`app_onnx.py` 会把 `sample_mode` 与 `do_sample` 写回 `manifest["generation_defaults"]`，其中 **greedy 会强制 `do_sample=false`**：

```169:188:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\app_onnx.py
        resolved_sample_mode = self._resolve_sample_mode(sample_mode, do_sample=do_sample)
        ...
        generation_defaults["sample_mode"] = resolved_sample_mode
        generation_defaults["do_sample"] = resolved_sample_mode != "greedy"
```

而 `local_greedy_frame` 是 **可选 ONNX 实现**（“整帧 greedy 一次 `run`”的快路径），只有在 `self.sessions` 里真的创建了该 session 才会走：

```856:867:c:\Caiyifeng\02Codebase\code_opensource\MOSS-TTS-Nano\ort_cpu_runtime.py
            if "local_greedy_frame" in self.sessions and not bool(generation_defaults["do_sample"]):
                ...
                should_continue, frame = self.run_local_greedy_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                    repetition_penalty=float(generation_defaults["audio_repetition_penalty"]),
                )
```

因此会出现：

- UI 上 **greedy（关采样）** 仍然成立（语义层）
- 但磁盘/元数据里 **未必存在** `local_greedy_frame.onnx`（工程实现层）

## “删掉 session 能省内存”到底省的是什么

通常分两类：

- **初始化常驻**：不创建 `InferenceSession` 就不会加载对应 ONNX 权重与 ORT 侧常驻结构；这对“用不到的模型文件”很有效。
- **推理峰值**：`prefill.run` / 多帧 `decode.run` 等执行期分配，主要来自 **张量、KV、workspace/allocator**；它不一定随“少加载别的 session”同比例下降，因为主干推理仍在跑。

## 与本对话相关的 warmup 观测提醒（可选背景）

若需要把 warmup 的 `sys_used` 增量拆成更细子阶段，需要意识到：

- `sys_used` 是整机维度，可能混入其它线程/进程分配（例如服务启动并发初始化）。
- warmup 的 `generate_audio_frames` 默认不传 `on_frame`，与 stream 的“边生成边 codec”时序不同；codec 往往在 warmup 末尾单独 `run_frames`。
