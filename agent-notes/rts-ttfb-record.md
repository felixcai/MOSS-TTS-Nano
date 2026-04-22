## 总结（核对与汇总）

### 日志核对结论

- 六种区块的标题、`python` 启动命令与关键日志行（`Nano-TTS stream RTF` / `Nano-TTS generate RTF`）一致，未发现「模式与数据串台」类错误。
- Stream：均含 `stream_id`、多块 `audio_chunks`、`rtf_first` 与 `rtf_steady`；Generate：均为 `audio_chunks=1` 且 `rtf_steady=n/a`，与语义一致。
- 同一应用（如 onnx 的 stream 与 generate）下，前文启动日志时间戳与进程号重复，属于同一次服务进程上的两次请求摘录，不视为数据错误。
- 若预期 ONNX 的 stream 与 generate 使用完全同一段文本，二者 `total_audio_s`（22.960 vs 21.120）差异建议在实验记录中注明是否同请求；单凭日志无法断定为贴错。

### 指标汇总表

| 应用 | 模式 | TTFB（first_audio_latency_s） | RTF（首次） | RTF（稳定） | 合成音频时长（s） | chunks |
|------|------|-------------------------------|-------------|-------------|-------------------|--------|
| app_onnx | stream | 1.8866 | 23.5795 | 1.1769 | 22.960 | 283 |
| app_onnx | generate | 21.9405 | 1.0388 | — | 21.120 | 1 |
| app（CPU） | stream | 2.2932 | 28.6614 | 4.1260 | 25.360 | 313 |
| app（CPU） | generate | 96.6070 | 3.8215 | — | 25.280 | 1 |
| app_gpu（CUDA） | stream | 0.5219 | 6.5202 | 3.2361 | 23.520 | 290 |
| app_gpu（CUDA） | generate | 56.6833 | 2.2281 | — | 25.440 | 1 |

### 简要分析（条列）

- **TTFB**：流式场景下 GPU 首包延迟最低（约 0.52s），ONNX 流式次之（约 1.89s），CPU 流式约 2.29s；非流式 generate 的「首包」在实现上往往接近整段生成完成，故 CPU/GPU generate 的 TTFB 可达数十秒，不宜与流式 TTFB 直接横向对比语义。
- **流式 RTF（首次）**：数值反映「从请求到首块音频」相对音频时长的比值，受冷启动与首块策略影响大；CPU PyTorch 流式 `rtf_first` 最高（约 28.66），ONNX 流式也很高（约 23.58），GPU 流式明显较低（约 6.52）。
- **流式 RTF（稳定）**：稳定阶段 ONNX 流式最优（约 1.18），GPU 流式次之（约 3.24），CPU 流式最高（约 4.13），与推理后端与设备预期相符。
- **Generate 的 RTF（rtf_first）**：在 `audio_chunks=1` 时更接近「整段生成的实时率」；ONNX generate 约 1.04（相对实时略慢），CPU 约 3.82，GPU 约 2.23；与流式稳定 RTF 不是同一测量窗口，仅可作同列参考。
- **音频时长**：CPU 的 stream/generate（25.36 / 25.28）与 GPU 的 stream/generate（23.52 / 25.44）各自接近，更像同一批实验下的可比样本；ONNX 两段时长差距相对大，若需严格对比应固定输入文本与参数。

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
