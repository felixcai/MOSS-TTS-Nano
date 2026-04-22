# ONNX 内置音色（`builtin_voices`）

数据摘自仓库内 `model_files/browser_poc_manifest.json`。程序里使用的 ID 为 **`voice`** 字段（例如 `infer_onnx.py --voice <name>` 须完全一致）。**性别**由 manifest 中的 **`group`**（语种 + Male/Female）推断。

| `voice` | `display_name` | `group`（原文） | 性别 |
|---------|----------------|-----------------|------|
| Junhao | CN 欢迎关注模思智能 | Chinese Male | 男 |
| Zhiming | CN 京味胡同闲聊 | Chinese Male | 男 |
| Weiguo | CN 说书 | Chinese Male | 男 |
| Xiaoyu | CN 明星 | Chinese Female | 女 |
| Yuewen | CN 机车 | Chinese Female | 女 |
| Lingyu | CN 深夜电台 | Chinese Female | 女 |
| Trump | EN Trump | English Male | 男 |
| Ava | EN The Bitter Lesson | English Female | 女 |
| Bella | EN A Gentle Reminder | English Female | 女 |
| Adam | EN English News | English Male | 男 |
| Nathan | EN The Quiet Motion of the World | English Male | 男 |
| Soyo | JP Soyo | Japanese Female | 女 |
| Saki | JP Saki | Japanese Female | 女 |
| Mortis | JP Mortis | Japanese Female | 女 |
| Umiri | JP Umiri | Japanese Female | 女 |
| Mei | JP Togawa | Japanese Female | 女 |
| Anon | JP Anon | Japanese Female | 女 |
| Arisa | JP Arisa | Japanese Female | 女 |

**汇总**：男声 6 个（Junhao, Zhiming, Weiguo, Trump, Adam, Nathan）；女声 12 个。日语条目在 manifest 中均为 `Japanese Female`。
