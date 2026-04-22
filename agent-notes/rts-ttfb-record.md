## 总结（核对与汇总）

### 日志核对结论

- 文档中八段场景里，**标题、`python` 命令与关键指标日志**一致：Web 端为 `Nano-TTS stream RTF` / `Nano-TTS generate RTF`；**`infer_onnx.py` 命令行**为 `Nano-TTS infer_onnx stream RTF` / `Nano-TTS infer_onnx generate RTF`。未发现「宣称的模式与日志行不符」类错误。
- **Stream**（含 Web 与 `infer_onnx` 流式）：日志含多块 `audio_chunks`、`rtf_first` 与 `rtf_steady`（Web 另有 `stream_id`）。**Generate / 非流式**：`audio_chunks=1` 且 `rtf_steady=n/a`（或日志中写作 `n/a`），与语义一致。
- 文末 **`infer_onnx` 两节**：`--voice Lingyu` 与 `using built-in voice preset: Lingyu` 一致；`--realtime-streaming-decode 1` 对 `stream RTF` + `streaming=True`，`0` 对 `generate RTF` + `streaming=False`。章节标题写「infer」实为 **`infer_onnx.py`** CLI，非 `infer.py`。
- 同一 Web 应用（如 app_onnx 的 stream 与 generate）下，前文启动日志时间戳与进程号重复，属于同一次服务进程上的两次请求摘录，不视为数据错误。
- 若预期某几组实验使用完全同一段文本，时长或参数差异建议在记录中注明；例如 **Web app_onnx** 的 stream/generate 与 **`infer_onnx`+Lingyu** 的文本、音色路径不同，**`total_audio_s` 不可强行横向等同**。

### 指标汇总表

| 应用 | 模式 | TTFB（first_audio_latency_s） | RTF（首次） | RTF（稳定） | 合成音频时长（s） | chunks |
|------|------|-------------------------------|-------------|-------------|-------------------|--------|
| app_onnx | stream | 1.8866 | 23.5795 | 1.1769 | 22.960 | 283 |
| app_onnx | generate | 21.9405 | 1.0388 | — | 21.120 | 1 |
| app（CPU） | stream | 2.2932 | 28.6614 | 4.1260 | 25.360 | 313 |
| app（CPU） | generate | 96.6070 | 3.8215 | — | 25.280 | 1 |
| app_gpu（CUDA） | stream | 0.5219 | 6.5202 | 3.2361 | 23.520 | 290 |
| app_gpu（CUDA） | generate | 56.6833 | 2.2281 | — | 25.440 | 1 |
| infer_onnx（内置 Lingyu） | stream | 0.8470 | 10.2284 | 1.1857 | 31.680 | 391 |
| infer_onnx（内置 Lingyu） | generate | 33.0552 | 1.0434 | — | 31.680 | 1 |

### 简要分析（条列）

- **TTFB**：流式场景下 GPU Web 首包延迟最低（约 0.52s），**`infer_onnx`+Lingyu 流式**首包约 **0.85s**（同机 ONNX CPU），Web app_onnx 流式约 1.89s，CPU PyTorch Web 流式约 2.29s；非流式 generate 的「首包」在实现上往往接近整段生成完成，故 CPU/GPU Web generate 与 **`infer_onnx` generate** 的 TTFB 可达数十秒量级，不宜与流式首包语义混比。
- **流式 RTF（首次）**：反映「首段可听音频前」的生成压力；CPU PyTorch Web 流式 `rtf_first` 最高（约 28.66），Web app_onnx 流式也很高（约 23.58），**`infer_onnx`+Lingyu 流式**约 **10.23**（文本与 Web 实验不同，仅作同列参考），GPU Web 流式约 6.52。
- **流式 RTF（稳定）**：Web 侧 ONNX 流式约 1.18，**`infer_onnx`+Lingyu 流式**稳定段约 **1.19**，GPU Web 流式约 3.24，CPU PyTorch Web 流式约 4.13。
- **Generate 的 RTF（rtf_first）**：`audio_chunks=1` 时更接近「整段 wall / 总音频时长」；Web ONNX generate 约 1.04，**`infer_onnx`+Lingyu generate** 约 **1.04**，CPU Web 约 3.82，GPU Web 约 2.23；与流式稳定 RTF 不是同一测量窗口。
- **音频时长**：**`infer_onnx` 流式 / 非流式**两次均为 **31.680s**（同一段约 116 字中文、同一内置音色），自洽；CPU/GPU Web 各组内部 stream 与 generate 接近的样本可对照；Web app_onnx 两段与 Lingyu CLI 实验文本不同，时长差异预期内。

---

# app_onnx, stream模式运行

root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python app_onnx.py
2026-04-22 08:12:08.319369989 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
2026-04-22 08:12:15,196 INFO root: root_path=None
INFO:     Started server process [987]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
2026-04-22 08:12:15,876 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:12:15,876 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:12:15,876 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:12:15,876 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:12:15,876 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 08:12:15,876 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
INFO:     Uvicorn running on http://localhost:18083 (Press CTRL+C to quit)
2026-04-22 08:12:16,564 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:12:16,564 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:12:16,564 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:12:16,564 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:12:16,565 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 08:12:16,565 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
INFO:     127.0.0.1:40696 - "GET /api/generate-stream/stream-1776845684036-20dedb9f/status HTTP/1.1" 200 OK
2026-04-22 08:15:12,975 INFO root: Nano-TTS stream RTF | stream_id=stream-1776845684036-20dedb9f | audio_chunks=283 | total_audio_s=22.960 | first_audio_latency_s=1.8866 | rtf_first=23.5795 | rtf_steady=1.1769

---

# app_onnx, generate模式运行

root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python app_onnx.py
2026-04-22 08:12:08.319369989 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
2026-04-22 08:12:15,196 INFO root: root_path=None
INFO:     Started server process [987]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
2026-04-22 08:12:15,876 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:12:15,876 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:12:15,876 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:12:15,876 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:12:15,876 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 08:12:15,876 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
INFO:     Uvicorn running on http://localhost:18083 (Press CTRL+C to quit)
2026-04-22 08:12:16,564 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:12:16,564 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:12:16,564 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:12:16,564 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:12:16,565 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 08:12:16,565 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
2026-04-22 08:16:40,276 INFO root: Nano-TTS generate RTF | audio_chunks=1 | total_audio_s=21.120 | first_audio_latency_s=21.9405 | rtf_first=1.0388 | rtf_steady=n/a
INFO:     127.0.0.1:40696 - "POST /api/generate HTTP/1.1" 200 OK

---

# app, stream模式运行

root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python app.py
2026-04-22 08:18:23,448 INFO root: root_path=None
2026-04-22 08:18:23,449 INFO root: loading Nano-TTS checkpoint=OpenMOSS-Team/MOSS-TTS-Nano audio_tokenizer=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cpu dtype=torch.float32 attn=model_default
2026-04-22 08:18:24,351 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:18:24,351 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:18:24,352 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:18:24,352 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:18:24,352 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 08:18:24,352 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
INFO:     Started server process [1150]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:18083 (Press CTRL+C to quit)
2026-04-22 08:18:25,011 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:18:25,011 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:18:25,012 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:18:25,012 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:18:25,012 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 08:18:25,012 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
INFO:     127.0.0.1:45654 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45660 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
2026-04-22 08:18:32,442 INFO root: loading Nano-TTS audio tokenizer checkpoint=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cpu attn=sdpa compute_dtype=fp32
INFO:     127.0.0.1:45660 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:36540 - "GET /api/generate-stream/stream-1776846021079-cfca9825/status HTTP/1.1" 200 OK
2026-04-22 08:22:07,896 INFO root: Nano-TTS stream RTF | stream_id=stream-1776846021079-cfca9825 | audio_chunks=313 | total_audio_s=25.360 | first_audio_latency_s=2.2932 | rtf_first=28.6614 | rtf_steady=4.1260
INFO:     127.0.0.1:45660 - "GET /api/generate-stream/stream-1776846021079-cfca9825/status HTTP/1.1" 200 OK

---

# app, generate模式运行

root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python app.py
2026-04-22 08:18:23,448 INFO root: root_path=None
2026-04-22 08:18:23,449 INFO root: loading Nano-TTS checkpoint=OpenMOSS-Team/MOSS-TTS-Nano audio_tokenizer=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cpu dtype=torch.float32 attn=model_default
2026-04-22 08:18:24,351 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:18:24,351 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:18:24,352 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:18:24,352 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:18:24,352 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 08:18:24,352 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
INFO:     Started server process [1150]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:18083 (Press CTRL+C to quit)
2026-04-22 08:18:25,011 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:18:25,011 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:18:25,012 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:18:25,012 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:18:25,012 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 08:18:25,012 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
INFO:     127.0.0.1:45654 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45660 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
2026-04-22 08:18:32,442 INFO root: loading Nano-TTS audio tokenizer checkpoint=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cpu attn=sdpa compute_dtype=fp32
INFO:     127.0.0.1:45660 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:36540 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
2026-04-22 08:24:21,356 INFO root: Nano-TTS generate RTF | audio_chunks=1 | total_audio_s=25.280 | first_audio_latency_s=96.6070 | rtf_first=3.8215 | rtf_steady=n/a
INFO:     127.0.0.1:36540 - "POST /api/generate HTTP/1.1" 200 OK

---

# app_gpu, stream模式运行

root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python app_gpu.py 
2026-04-22 08:25:24,518 INFO root: root_path=None
2026-04-22 08:25:24,518 INFO root: loading Nano-TTS checkpoint=OpenMOSS-Team/MOSS-TTS-Nano audio_tokenizer=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cuda dtype=torch.bfloat16 attn=model_default
2026-04-22 08:25:25,258 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:25:25,258 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:25:25,258 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:25:25,258 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:25:25,258 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 08:25:25,258 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
INFO:     Started server process [1220]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:18083 (Press CTRL+C to quit)
2026-04-22 08:25:25,921 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:25:25,921 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:25:25,922 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:25:25,922 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:25:25,922 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 08:25:25,922 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
INFO:     127.0.0.1:45744 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45760 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45760 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45744 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
2026-04-22 08:25:35,716 INFO root: installed Nano-TTS CUDA streaming decode budget patch
2026-04-22 08:25:35,721 INFO root: loading Nano-TTS audio tokenizer checkpoint=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cuda attn=sdpa compute_dtype=fp32
INFO:     127.0.0.1:45744 - "GET /api/warmup-status HTTP/1.1" 200 OK
2026-04-22 08:28:54,325 INFO root: Nano-TTS stream RTF | stream_id=stream-1776846457766-907deff1 | audio_chunks=290 | total_audio_s=23.520 | first_audio_latency_s=0.5219 | rtf_first=6.5202 | rtf_steady=3.2361
INFO:     127.0.0.1:45744 - "GET /api/generate-stream/stream-1776846457766-907deff1/status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45744 - "GET /api/generate-stream/stream-1776846457766-907deff1/result HTTP/1.1" 200 OK
INFO:     127.0.0.1:45744 - "POST /api/generate-stream/stream-1776846457766-907deff1/close HTTP/1.1" 200 OK

---

# app_gpu, generate模式运行

root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python app_gpu.py 
2026-04-22 08:25:24,518 INFO root: root_path=None
2026-04-22 08:25:24,518 INFO root: loading Nano-TTS checkpoint=OpenMOSS-Team/MOSS-TTS-Nano audio_tokenizer=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cuda dtype=torch.bfloat16 attn=model_default
2026-04-22 08:25:25,258 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:25:25,258 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 08:25:25,258 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:25:25,258 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 08:25:25,258 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 08:25:25,258 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
INFO:     Started server process [1220]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:18083 (Press CTRL+C to quit)
2026-04-22 08:25:25,921 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:25:25,921 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 08:25:25,922 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:25:25,922 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 08:25:25,922 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 08:25:25,922 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
INFO:     127.0.0.1:45744 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45760 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45760 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:45744 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
2026-04-22 08:25:35,716 INFO root: installed Nano-TTS CUDA streaming decode budget patch
2026-04-22 08:25:35,721 INFO root: loading Nano-TTS audio tokenizer checkpoint=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cuda attn=sdpa compute_dtype=fp32
INFO:     127.0.0.1:45744 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:36076 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
2026-04-22 08:29:40,897 INFO root: voice_clone chunk batching device=cuda free_gb=10.33 max_memory_per_sample_gb=1.00 resolved_batch_size=2 chunk_count=2
2026-04-22 08:30:37,347 INFO root: Nano-TTS generate RTF | audio_chunks=1 | total_audio_s=25.440 | first_audio_latency_s=56.6833 | rtf_first=2.2281 | rtf_steady=n/a
INFO:     127.0.0.1:36076 - "POST /api/generate HTTP/1.1" 200 OK

---

# infer_onnx，内置音色，streaming

root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python infer_onnx.py --voice Lingyu --text "华灯初上，路口的红绿灯把斑马线切成一段段暖色。写字楼玻璃里还映着没下班的人影，外卖骑手从身边掠过，带起一阵冬夜里微凉的风。她握着温热的咖啡站在街边，忽然听见楼上有人开窗喊了一句晚安——这座城市很吵，却也在某个瞬间，让人觉得没那么孤单。" --realtime-streaming-decode  1
2026-04-22 10:06:26.326255896 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
2026-04-22 10:06:33,998 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 10:06:33,998 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 10:06:33,999 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 10:06:33,999 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 10:06:33,999 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 10:06:33,999 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
2026-04-22 10:06:34,655 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 10:06:34,655 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 10:06:34,655 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 10:06:34,655 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 10:06:34,655 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 10:06:34,655 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
2026-04-22 10:06:36,572 INFO root: normalized text chars_before=117 chars_after=116 stage=robust_pre
2026-04-22 10:06:36,612 INFO root: text normalization method=robust_pre+wetext:zh+robust_post language=zh text_chars=116
2026-04-22 10:06:36,612 INFO root: using built-in voice preset: Lingyu
2026-04-22 10:06:36,613 INFO root: normalized text chars_before=117 chars_after=116 stage=robust_pre
2026-04-22 10:07:14,575 INFO root: Nano-TTS infer_onnx stream RTF | audio_chunks=391 | total_audio_s=31.680 | first_audio_latency_s=0.8470 | rtf_first=10.2284 | rtf_steady=1.1857
2026-04-22 10:07:14,576 INFO root: saved generated audio to /workspace/MOSS-TTS-Nano/generated_audio/infer_onnx_output.wav sample_rate=48000 frames=391 sample_mode=fixed streaming=True

---

# infer_onnx，内置音色，no streaming

root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python infer_onnx.py --voice Lingyu --text "华灯初上，路口的红绿灯把斑马线切成一段段暖色。写字楼玻璃里还映着没下班的人影，外卖骑手从身边掠过，带起一阵冬夜里微凉的风。她握着温热的咖啡站在街边，忽然听见楼上有人开窗喊了一句晚安——这座城市很吵，却也在某个瞬间，让人觉得没那么孤单。" --realtime-streaming-decode  0
2026-04-22 10:10:21.572789659 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
2026-04-22 10:10:29,151 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 10:10:29,151 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 10:10:29,152 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 10:10:29,152 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 10:10:29,152 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 10:10:29,152 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
2026-04-22 10:10:29,804 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 10:10:29,804 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 10:10:29,804 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 10:10:29,804 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 10:10:29,804 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 10:10:29,804 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
2026-04-22 10:10:31,728 INFO root: normalized text chars_before=117 chars_after=116 stage=robust_pre
2026-04-22 10:10:31,769 INFO root: text normalization method=robust_pre+wetext:zh+robust_post language=zh text_chars=116
2026-04-22 10:10:31,769 INFO root: using built-in voice preset: Lingyu
2026-04-22 10:10:31,770 INFO root: normalized text chars_before=117 chars_after=116 stage=robust_pre
2026-04-22 10:11:04,825 INFO root: Nano-TTS infer_onnx generate RTF | audio_chunks=1 | total_audio_s=31.680 | first_audio_latency_s=33.0552 | rtf_first=1.0434 | rtf_steady=n/a
2026-04-22 10:11:04,825 INFO root: saved generated audio to /workspace/MOSS-TTS-Nano/generated_audio/infer_onnx_output.wav sample_rate=48000 frames=391 sample_mode=fixed streaming=False
