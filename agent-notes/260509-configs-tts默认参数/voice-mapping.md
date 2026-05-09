# Voice Mapping

按 `simple_moss/_voice_list.py` 中的模型预设音色，与 `assets/demo.jsonl` 中的 `role` 字段（参考音频文件）做对照。

说明：

- `是否在 demo 中` 是按 `audio_file` 是否出现在 `demo.jsonl` 里判断。
- `音频是否存在` 是按当前 `assets/audio` 目录中是否能看到对应文件来判断。
- `demo 名称` 来自 `demo.jsonl` 的 `name` 字段。
- 某些 `audio_file` 可能对应多个 demo 名称，例如 `en_4.wav`。

| voice     | audio_file  | 是否在 demo 中 | 音频是否存在 | demo 名称                                                                                                                                                                                                                                                                                                                         |
| --------- | ----------- | ---------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Junhao`  | `zh_1.wav`  | 是          | 是      | `🇨🇳 欢迎关注模思智能`                                                                                                                                                                                                                                                                                                                 |
| `Zhiming` | `zh_3.wav`  | 是          | 是      | `🇨🇳 京味胡同闲聊`                                                                                                                                                                                                                                                                                                                   |
| `Weiguo`  | `zh_10.wav` | 是          | 是      | `🇨🇳 中国人的时间观念与文化逻辑`                                                                                                                                                                                                                                                                                                            |
| `Xiaoyu`  | `zh_11.wav` | 是          | 是      | `🇨🇳 杨幂 - 与自己同行`                                                                                                                                                                                                                                                                                                               |
| `Yuewen`  | `zh_4.wav`  | 是          | 是      | `🇨🇳 台湾腔`                                                                                                                                                                                                                                                                                                                      |
| `Lingyu`  | `zh_6.wav`  | 是          | 是      | `🇨🇳 深夜温柔晚安`                                                                                                                                                                                                                                                                                                                   |
| `Trump`   | `en_1.wav`  | 否          | 否      | -                                                                                                                                                                                                                                                                                                                               |
| `Ava`     | `en_2.wav`  | 是          | 是      | `🇺🇸 The Bitter Lesson`                                                                                                                                                                                                                                                                                                        |
| `Bella`   | `en_3.wav`  | 是          | 是      | `🇺🇸 A Gentle Reminder`                                                                                                                                                                                                                                                                                                        |
| `Adam`    | `en_4.wav`  | 是          | 是      | `🇺🇸 English News` / `🇰🇷 뉴스` / `🇪🇸 Noticiero` / `🇫🇷 Journal` / `🇩🇪 Nachrichten` / `🇮🇹 Telegiornale` / `🇭🇺 Híradó` / `🇷🇺 Новости` / `🇮🇷 اخبار` / `🇸🇦 النشرة الإخبارية` / `🇵🇱 Wiadomości` / `🇵🇹 Noticiário` / `🇨🇿 Zprávy` / `🇩🇰 Nyhederne` / `🇸🇪 Nyheterna` / `🇬🇷 Ειδήσεις` / `🇹🇷 Haber Bülteni` |
| `Nathan`  | `en_8.wav`  | 是          | 是      | `🇺🇸 The Quiet Motion of the World`                                                                                                                                                                                                                                                                                            |
| `Soyo`    | `jp_1.wav`  | 否          | 否      | -                                                                                                                                                                                                                                                                                                                               |
| `Saki`    | `jp_2.wav`  | 是          | 是      | `🇯🇵 ニュース`                                                                                                                                                                                                                                                                                                                     |
| `Mortis`  | `jp_3.wav`  | 否          | 否      | -                                                                                                                                                                                                                                                                                                                               |
| `Umiri`   | `jp_4.wav`  | 否          | 否      | -                                                                                                                                                                                                                                                                                                                               |
| `Mei`     | `jp_5.wav`  | 否          | 否      | -                                                                                                                                                                                                                                                                                                                               |
| `Anon`    | `jp_6.wav`  | 否          | 否      | -                                                                                                                                                                                                                                                                                                                               |
| `Arisa`   | `jp_7.wav`  | 否          | 否      | -                                                                                                                                                                                                                                                                                                                               |

## 小结

- 模型预设音色总数：`18`
- 在 `demo.jsonl` 中有映射的模型音色：`11`
- 在 `demo.jsonl` 中没有映射的模型音色：`7`
- 在 `assets/audio` 中能看到对应音频文件的模型音色：`11`
- 在 `assets/audio` 中看不到对应音频文件的模型音色：`7`

未出现在 `demo.jsonl` 中的模型音色：

- `Trump`
- `Soyo`
- `Mortis`
- `Umiri`
- `Mei`
- `Anon`
- `Arisa`

在 `assets/audio` 中看不到对应音频文件的模型音色：

- `Trump`
- `Soyo`
- `Mortis`
- `Umiri`
- `Mei`
- `Anon`
- `Arisa`

额外说明：

- `demo.jsonl` 里还存在一些并不在当前模型预设音色表中的参考音频，例如 `en_6.wav`、`en_7.wav`。
- 因此 `demo.jsonl` 更适合看作“页面演示素材表”，而不是“模型支持音色全集”。

## voice 所属语言识别

当前代码库里，确实存在“根据 `voice` 判断语言”的逻辑，但它主要出现在文本正则化链路里，而不是 `simple_moss` 的主推理链路里。

现状逻辑：

- `text_normalization_pipeline.py` 中维护了一份硬编码的 `ENGLISH_VOICES` 集合：`Trump`、`Ava`、`Bella`、`Adam`、`Nathan`。
- `resolve_text_normalization_language(text, voice)` 的判定顺序是：
  - 文本里有中文字符，则判定为 `zh`
  - 文本里有英文字母，则判定为 `en`
  - 否则如果 `voice` 在 `ENGLISH_VOICES` 中，则判定为 `en`
  - 否则默认回退到 `zh`
- `simple_moss/simple_app_onnx.py` 的 `normalize_text()` 会把 `voice` 传入 `prepare_tts_request_texts()`，因此会间接触发上述语言判定逻辑。

结论：

- 原代码里有“按 `voice` 推断语言”的逻辑，但实现方式是**硬编码英文 voice 名单**。
- 当前主推理链路和 warmup 逻辑，并没有直接使用 `group` 字段来做语种判断。
- 现有模型配置中的 `group` 字段已经足够支撑语言识别，例如：
  - `Chinese Male` / `Chinese Female` -> 可映射为 `zh`
  - `English Male` / `English Female` -> 可映射为 `en`
  - `Japanese Female` -> 可映射为 `ja`

对后续配置设计的意义：

- 如果后面要做“根据 voice 自动选择 warmup 文本”“按语言展示音色”“按语言走不同规范化策略”，现有配置字段已经够用。
- 相比继续维护 `ENGLISH_VOICES` 这类硬编码名单，更推荐后续统一基于 `group` 做语言归类。
