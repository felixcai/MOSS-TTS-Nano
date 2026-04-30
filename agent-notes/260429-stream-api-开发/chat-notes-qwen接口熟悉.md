# Qwen TTS 接口熟悉

`stream-api-ref` 目录下的这两个文件共同构成了一个基于 Qwen3 的本地文本转语音（TTS）服务模块。它们的分工非常明确，一个负责底层模型推理，另一个负责上层 API 服务封装。具体内容如下：

### 1. `_local_qwen_tts_model.py` (底层模型推理层)

这个文件是 TTS 模型的核心实现，主要负责与 AI 模型进行交互和音频数据处理。

* **核心类**：实现了 `LocalQwenTTSModel` 类。
* **底层依赖**：依赖 `faster-qwen3-tts` 库，支持在 NVIDIA GPU (CUDA) 或 CPU 上运行。
* **模型加载**：支持从 ModelScope（魔搭社区）或 Hugging Face 下载并缓存模型权重。
* **生成模式**：支持三种语音生成方式：
  * **声音克隆 (Voice cloning)**：通过参考音频和参考文本克隆音色。
  * **自定义音色 (Custom voice)**：使用模型预设的说话人音色。
  * **声音设计 (Voice design)**：通过提示词（Instruct）来控制生成声音的特征。
* **流式输出**：包含复杂的音频处理逻辑（如去除静音、重采样到 24000Hz、转换为 int16 PCM 格式），并以异步流（AsyncGenerator）的形式分块（chunk）返回音频数据，从而实现极低的延迟。
* **独立运行能力**：文件末尾包含了一段 `if __name__ == "__main__":` 代码，允许通过命令行直接启动一个测试用的 FastAPI 服务。

### 2. `_local_tts_engine.py` (上层服务与 API 封装层)

这个文件将底层的模型能力包装成了一个标准化的、兼容 OpenAI 格式的 Web API 引擎。

* **核心类**：实现了 `LocalTTSEngine` 类（继承自系统基础引擎类 `EngineBase`）。
* **API 接口定义**：使用 FastAPI 构建了与 OpenAI 完全兼容的端点（Endpoints）：
  * `/v1/models`：获取可用模型列表。
  * `/v1/audio/voices`：获取可用音色列表。
  * `/v1/audio/speech`：核心语音合成接口，接收文本并返回流式 PCM 音频数据。
  * `/health`：服务健康检查接口。
* **生命周期与配置管理**：
  * 定义了引擎所需的各项配置项（如设备类型、语言、流式块大小等）。
  * 提供了 `start()` 和 `stop()` 方法来异步启动和优雅关闭底层的 Uvicorn 服务器。
  * 实现了独立的日志记录机制，将运行日志单独输出到指定的日志文件中。

**总结来说：**
`_local_qwen_tts_model.py` 负责**“如何把文本变成声音”**（加载模型、推理、处理音频流）；
`_local_tts_engine.py` 负责**“如何让外部程序调用这个能力”**（提供标准的 HTTP API、管理服务启停和配置）。

---

# 上层 API 与各参数对底层引擎的影响（`_local_tts_engine.py`）

## 声音克隆与 HTTP API 的关系（补充）

`_local_tts_engine.py` **没有在 HTTP 路由里提供“按请求动态克隆”的专用 API**。`ref_audio`、`ref_text` 等出现在 **引擎配置**（`get_config` / `start()` 时合并的 `config`）里，在 `LocalQwenTTSModel` **初始化时**传入；`POST /v1/audio/speech` 的请求体 **不包含** 参考音频或参考文本字段。因此克隆音色是「启动引擎时静态配置」，不是「每次请求传参切换」。

## 暴露的端点一览

| 方法   | 路径                 | 作用简述                                       |
| ---- | ------------------ | ------------------------------------------ |
| GET  | `/v1/models`       | 返回固定列表（当前实现为 `qwen3-tts`）                  |
| GET  | `/v1/audio/voices` | 返回音色列表；若启动配置里配置了 `speaker`，会多返回一条该 speaker |
| GET  | `/health`          | 健康检查：状态、`model_name`、采样率 `PIPELINE_SR`     |
| POST | `/v1/audio/speech` | 核心：按请求文本流式返回 PCM                           |

## `POST /v1/audio/speech` 请求体（`TTSRequest`）各字段

请求 JSON 对应 `TTSRequest`：

- `model`（默认 `qwen3-tts`）：OpenAI 风格字段；**路由逻辑未读取**，实际模型在引擎 `start()` 时由配置项 `model_name` 固定加载。
- `input`（必填）：待合成文本；**唯一在每次请求中传入底层**的参数，经 `_stream_audio(request.input)` 进入 `LocalQwenTTSModel.synthesize(text)`。
- `voice`（默认 `default`）：OpenAI 风格字段；**`create_speech` 未使用**；音色由启动时的 `speaker` / `ref_audio` / `instruct` 等与模型类型共同决定，与单次请求的 `voice` 无关。
- `response_format`（默认 `pcm`，注释里还写了 wav/mp3）：**未参与分支**；响应写死为 `audio/pcm;rate=PIPELINE_SR`（24000），底层也只产出 int16 PCM 分块。
- `speed`（默认 `1.0`）：**未传入底层**；当前封装未实现语速调节。

## 参数与底层行为的对应关系（简表）

| `TTSRequest` 字段   | 设计意图 | 对 `LocalQwenTTSModel` / 合成的实际影响 |
| ----------------- | ---- | ------------------------------- |
| `input`           | 合成内容 | **生效**：驱动本次合成的文本                |
| `model`           | 选模型  | **不生效**：模型在 `start()` 时已定       |
| `voice`           | 选音色  | **不生效**：音色/克隆等在 `start()` 时已定   |
| `response_format` | 输出编码 | **不生效**：固定 PCM 流                |
| `speed`           | 语速   | **不生效**：未接到底层                   |

## 小结

当前 HTTP 层为兼容 OpenAI 形态定义了完整请求体，但实现上相当于 **只把 `input` 动态交给底层**；模型、音色（含克隆）、输出格式与语速等，均依赖 **引擎启动时的配置**，而非单次 `POST /v1/audio/speech` 的请求参数。
