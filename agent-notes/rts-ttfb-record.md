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
