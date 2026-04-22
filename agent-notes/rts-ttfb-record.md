

root@nvidia-desktop:/workspace/MOSS-TTS-Nano# python app.py
stream模式运行
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
