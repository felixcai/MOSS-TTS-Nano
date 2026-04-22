
stream模式运行
root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python app.py
2026-04-22 07:26:02,452 INFO root: root_path=None
2026-04-22 07:26:02,453 INFO root: loading Nano-TTS checkpoint=OpenMOSS-Team/MOSS-TTS-Nano audio_tokenizer=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cpu dtype=torch.float32 attn=model_default
2026-04-22 07:26:03,327 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 07:26:03,327 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 07:26:03,328 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 07:26:03,328 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 07:26:03,328 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 07:26:03,328 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
INFO:     Started server process [890]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:18083 (Press CTRL+C to quit)
2026-04-22 07:26:03,997 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 07:26:03,997 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 07:26:03,997 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 07:26:03,997 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 07:26:03,997 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 07:26:03,997 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
2026-04-22 07:28:03,573 INFO root: Nano-TTS stream RTF | stream_id=stream-1776842805308-baaaaab4 | audio_chunks=226 | total_audio_s=18.080 | first_audio_latency_s=2.2696 | rtf_first=28.3665 | rtf_steady=4.2111

generate模式运行
root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python app.py
2026-04-22 07:46:22,203 INFO root: root_path=None
2026-04-22 07:46:22,203 INFO root: loading Nano-TTS checkpoint=OpenMOSS-Team/MOSS-TTS-Nano audio_tokenizer=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cpu dtype=torch.float32 attn=model_default
2026-04-22 07:46:23,090 WETEXT INFO found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 07:46:23,090 INFO wetext-zh_normalizer: found existing fst: /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_tagger.fst
2026-04-22 07:46:23,090 WETEXT INFO                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 07:46:23,090 INFO wetext-zh_normalizer:                     /workspace/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/zh_tn_verbalizer.fst
2026-04-22 07:46:23,090 WETEXT INFO skip building fst for zh_normalizer ...
2026-04-22 07:46:23,090 INFO wetext-zh_normalizer: skip building fst for zh_normalizer ...
INFO:     Started server process [938]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:18083 (Press CTRL+C to quit)
2026-04-22 07:46:23,751 WETEXT INFO found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 07:46:23,751 INFO wetext-en_normalizer: found existing fst: /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-04-22 07:46:23,751 WETEXT INFO                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 07:46:23,751 INFO wetext-en_normalizer:                     /opt/miniforge/envs/moss/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-04-22 07:46:23,751 WETEXT INFO skip building fst for en_normalizer ...
2026-04-22 07:46:23,751 INFO wetext-en_normalizer: skip building fst for en_normalizer ...
INFO:     127.0.0.1:58348 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:58356 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:58356 - "GET /api/warmup-status HTTP/1.1" 200 OK
INFO:     127.0.0.1:58348 - "GET /api/text-normalization-status HTTP/1.1" 200 OK
2026-04-22 07:46:31,198 INFO root: loading Nano-TTS audio tokenizer checkpoint=OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano device=cpu attn=sdpa compute_dtype=fp32
2026-04-22 07:48:38,435 INFO root: Nano-TTS generate RTF | audio_chunks=1 | total_audio_s=16.560 | first_audio_latency_s=64.4373 | rtf_first=3.8911 | rtf_steady=n/a
INFO:     127.0.0.1:33680 - "POST /api/generate HTTP/1.1" 200 OK