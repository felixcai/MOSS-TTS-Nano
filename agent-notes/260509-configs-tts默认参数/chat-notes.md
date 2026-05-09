# simple_moss TTS 核心参数梳理

基于 `simple_moss` 下的核心代码 (`api_facade.py`, `simple_app_onnx.py`, `simple_onnx_tts_runtime.py`)，目前 TTS 过程（初始化、预热、流式生成）涉及的核心参数如下：

## 1. 初始化阶段参数 (Initialization)

这些参数通常在服务启动、创建 TTS Adapter 时配置。

| 参数名              | 类型            | 默认值                                         | 说明                     | 修改方式                             |
| ---------------- | ------------- | ------------------------------------------- | ---------------------- | -------------------------------- |
| `model_dir`      | string / Path | `None` (内部自动推断或下载)                          | ONNX 模型及分词器的所在根目录。     | 抽取到 \_config.py中。不需要启动传参         |
| `output_dir`     | string / Path | `None` (默认 fallback 到根目录 `generated_audio`) | 日志或临时文件保存的输出目录。        | 抽取到 \_config.py中。不需要启动传参         |
| `cpu_threads`    | int           | `1` 或 `4`                                   | ONNX 运行时绑定的 CPU 线程数。   | 抽取到 \_config.py中。不需要启动传参。默认值统一为1 |
| `max_new_frames` | int           | `375`                                       | 全局的默认音频 token 推理最大帧数。  | 抽取到 \_config.py中。不需要启动传参         |
| `enable_wetext`  | bool          | `True`                                      | 是否在生成前执行 WeText 文本正则化。 | 抽取到 \_config.py中，不需要启动传参         |

## 2. 预热阶段参数 (Warmup)

目前的 `warmup_runtime` 未暴露外部显式参数，但其底层执行逻辑强关联以下参数配置：

| 参数名                    | 类型     | 默认值        | 说明                                                          | 修改方式 |
| ---------------------- | ------ | ---------- | ----------------------------------------------------------- | ---- |
| `voice_name` / `voice` | string | `"Lingyu"` | 用于预热的音色标识（代码内当前使用 `FIXED_BUILTIN_VOICE` 常量固定）。后续需考虑对外开放此配置。 | 抽取到 \_config.py中。需要启动传参     |

## 3. 流式生成阶段参数 (Stream Generate)

这些参数可在每次合成时传入覆盖。控制着文本的切割逻辑、推理时的采样算法及生成风格。

| 参数名                           | 类型          | 默认值               | 说明                                                             | 修改方式 |
| ----------------------------- | ----------- | ----------------- | -------------------------------------------------------------- | ---- |
| `text`                        | string      | (无，调用时必填)         | 要合成的目标文本内容。                                                    | |
| `voice`                       | string      | `"Lingyu"`        | 发音人音色标识（可随每次请求指定。当前代码中受限于 `FIXED_BUILTIN_VOICE` 常量，未来需完全放开配置）。 | 抽取到 \_config.py中。需要启动传参     |
| `max_new_frames`              | int         | `375`             | 覆盖初始化设置的单次推断最大生成帧数。                                            | 抽取到 \_config.py中。不需要启动传参         |
| `voice_clone_max_text_tokens` | int         | `75`              | 智能文本分句边界配置：单个 Chunk 最大允许的 Token 预算长度。                          | 抽取到 \_config.py中。需要启动传参         |
| `attn_implementation`         | string      | `"model_default"` | 解析为 `fixed`、`full` 或 `greedy` 的采样策略标识。                         | 抽取到 \_config.py中。不需要启动传参         |
| `do_sample`                   | bool        | `True`            | 是否启用采样，若为 `False`，将强制 fallback 为 Greedy 贪心模式。                  | 抽取到 \_config.py中。不需要启动传参         |
| `text_temperature`            | float       | `1.0`             | Text/Semantic 生成时的温度系数，控制采样多样性。                                | 抽取到 \_config.py中。不需要启动传参         |
| `text_top_p`                  | float       | `1.0`             | Text/Semantic 生成时的 Top-P（核采样）截断概率。                             | 抽取到 \_config.py中。不需要启动传参         |
| `text_top_k`                  | int         | `50`              | Text/Semantic 生成时的 Top-K 截断个数。                                 | 抽取到 \_config.py中。不需要启动传参         |
| `audio_temperature`           | float       | `0.8`             | Audio/Acoustic 生成时的温度系数。                                       | 抽取到 \_config.py中。不需要启动传参         |
| `audio_top_p`                 | float       | `0.95`            | Audio/Acoustic 生成时的 Top-P 截断概率。                                | 抽取到 \_config.py中。不需要启动传参         |
| `audio_top_k`                 | int         | `25`              | Audio/Acoustic 生成时的 Top-K 截断个数。                                | 抽取到 \_config.py中。不需要启动传参         |
| `audio_repetition_penalty`    | float       | `1.2`             | 重复惩罚系数，防止循环生成同一音频片段。                                           | 抽取到 \_config.py中。不需要启动传参         |
| `seed`                        | int \| None | `None`            | 生成器的随机种子，传入具体整数时可复现确定性的推理结果。                                   | 抽取到 \_config.py中。不需要启动传参         |
| `chunk_pause_seconds`         | float       | `2.0`             | Chunk(长分句)之间插入的静音时长；若短分句内部判断小于等于8个混合词，会按此值减半(1.0s)插入静音。        | 抽取到 \_config.py中。需要启动传参         |
| `short_frame_num`             | int         | `6`               | 流式 Codec 解码时的基础批处理帧数。生成超前量较小时使用此值，超前量充足时使用该值的 1.5 倍（即 9）以提升吞吐。 | 抽取到 \_config.py中。需要启动传参         |

---

**接下来的建议(对应目标的2和3):**
可以把上述参数中的**默认值**统一抽离至单独的配置文件（例如新建 `simple_moss/config.py` 或独立的 `settings.yaml/json`）。再在应用层(CLI 或 HTTP Request)封装参数接收逻辑以允许灵活覆写。