# simple_moss 中 stream generate 的 1 帧对应多少播放时间

问题：现在 stream 生成一帧，对应到播放端，是多少秒？这里只考虑 `simple_moss` 中的 stream generate。

## 结论

在 `simple_moss` 这条 stream generate 链路里，**模型生成的 1 帧声学 token，基本对应约 `0.08` 秒，也就是 `80ms` 的可播放音频**。

更具体地说：

- 采样率是 `48000 Hz`
- 1 帧大约对应 `3840` 个 sample
- 时长约为 `3840 / 48000 = 0.08s`

## 推导依据

### 代码链路

`simple_moss` 中的流式链路大致如下：

```text
simple_app_onnx.py::_on_frame(...)
  -> pending_decode_frames.append(frame)
  -> _decode_pending(...)
  -> codec_streaming_session.run_frames(frame_chunk)
  -> 返回 (audio, audio_length)
  -> _emit_waveform(...)
```

其中：

- `_on_frame(...)` 表示“每生成 1 帧声学 token 后”的回调；
- 这些 frame 会先积累到 `pending_decode_frames`；
- 然后 `_decode_pending(...)` 批量把若干帧送进 `codec_streaming_session.run_frames(frame_chunk)` 解码；
- `run_frames()` 返回的 `audio_length` 说明最终输出音频长度与输入 frame 数存在稳定对应关系。

### 仓库内已有实测日志反推

仓库内已有日志记录可直接反推出单帧时长：

- `391` 帧对应 `31.680` 秒音频
- `421` 帧对应 `34.080` 秒音频

换算：

- `31.680 / 391 ≈ 0.0810s`
- `34.080 / 421 ≈ 0.08095s`

两组结果都非常接近 `0.08s`，因此可以认为当前 `simple_moss` 的 **1 frame ≈ 80ms 音频**。

## 需要区分的两个概念

### 模型生成 1 帧

这是模型侧时间分辨率：

- **1 帧 ≈ 80ms**

### 播放端每次实际收到的音频块

这不是“每生成 1 帧就立刻发 1 次音频包”。

在 `simple_moss` 中，`_decode_pending(...)` 会按批处理若干帧再送去 codec 解码。当前默认配置里：

- `short_frame_num = 6`
- 当缓冲更充足时，会放大到 `1.5x`，也就是 `9`

因此播放端更常见收到的是：

- `6` 帧一包：`6 * 80ms = 480ms`
- `9` 帧一包：`9 * 80ms = 720ms`

## 最终结论

如果问题是：

> stream 生成 1 帧，对应多少秒？

答案是：

- **模型层面：约 `80ms / frame`**

如果问题是：

> 播放端每次通常收到多长的音频块？

答案更接近：

- **通常约 `480ms` 或 `720ms` 一块**，取决于 `decode_pending` 当前按 `6` 帧还是 `9` 帧批量解码。

## 谁决定 1 帧对应多少 sample

补充问题：实际上，大模型会在训练的时候确定，一帧对应的 sample 数么？

结论：**本质上会被训练体系固定下来，但更准确地说，不是 TTS 大模型自己临时决定，而是 audio tokenizer / codec 先定义了时间分辨率；TTS 大模型是在这个固定分辨率上学习生成 frame 序列。**

可以拆成两层理解：

- **先由 audio tokenizer / codec 决定时间网格**：例如每帧对应多少个 sample，或者等价地定义 `downsample_rate`、`hop_length`、`frame_rate`。
- **再由 TTS 大模型学习按帧生成**：模型训练时面对的不是原始 waveform sample，而是按固定时间粒度离散化后的 audio token frame。

因此：

- “1 frame 对应多少 sample”主要是 **tokenizer / codec 侧的属性**
- TTS 大模型是在这个固定粒度上学习生成
- 推理时一般不会动态变成“这一帧 40ms、下一帧 90ms”

### 代码中的对应关系

仓库中的 `model_files/modeling_moss_tts_nano.py` 明确体现了：系统会从 audio tokenizer 侧解析时间分辨率相关字段：

```1546:1558:model_files/modeling_moss_tts_nano.py
    def _resolve_audio_tokenizer_downsample_rate(audio_tokenizer) -> int:
        for holder in (audio_tokenizer, getattr(audio_tokenizer, "config", None)):
            if holder is None:
                continue
            for attr_name in ("downsample_rate", "hop_length", "frame_size"):
                value = getattr(holder, attr_name, None)
                if value is not None:
                    return int(value)
            sampling_rate = getattr(holder, "sampling_rate", None)
            frame_rate = getattr(holder, "frame_rate", None)
            if sampling_rate is not None and frame_rate not in (None, 0):
                return int(round(float(sampling_rate) / float(frame_rate)))
```

这说明在当前体系里，`downsample_rate / hop_length / frame_rate` 这些量，本来就是 tokenizer / codec 定义好的元信息。

## `codec_browser_onnx_meta.json` 给出的直接证据

补充问题：`model_files/codec_browser_onnx_meta.json` 这个从 tokenizer 中拿出来的文件，里面有相关信息吗？

结论：**有，而且就是关键证据。**

该文件的 `codec_config` 直接写出了：

```20:24:model_files/codec_browser_onnx_meta.json
  "codec_config": {
    "sample_rate": 48000,
    "channels": 2,
    "downsample_rate": 3840,
    "num_quantizers": 16
  },
```

这里最关键的是：

- `sample_rate = 48000`
- `downsample_rate = 3840`

据此可以直接得到：

- **1 frame = 3840 samples**
- **时长 = `3840 / 48000 = 0.08s = 80ms`**

这意味着前面“`1 frame ≈ 80ms`”不再只是根据日志做的经验反推，而是有 tokenizer / codec 元数据直接支持。

## 更准确的最终表述

综合后续讨论后，这个问题更严谨的表达应为：

- `simple_moss` 当前 stream generate 中，**1 frame 对应 3840 个 sample**
- 这个数字不是播放端随便估出来的，而是来自 `model_files/codec_browser_onnx_meta.json` 中的 `codec_config.downsample_rate`
- 在 `sample_rate = 48000` 下，对应 **`80ms / frame`**
- TTS 大模型是在这个由 tokenizer / codec 预先确定的时间分辨率上学习和生成的
