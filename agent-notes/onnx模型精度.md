# ONNX 模型精度（对话整理）

本文整理一次对话里关于 **MOSS-TTS-Nano ONNX CPU 路径** 的「数值精度 / dtype」结论，便于日后查阅。

---

## 1. 问题背景

关心点包括：

- `infer_onnx.py`、`app_onnx.py`（及底层引用）里，**推理精度能否配置**、**支持哪些精度**、**默认是什么**。
- 文档（`README_zh.md`、`finetuning/README_zh.md`）、Hugging Face 模型卡、`tts_browser_onnx_meta.json`、具体 `.onnx` 里是否写明精度。

---

## 2. 脚本与运行时（仓库代码）

### 2.1 结论（脚本层）

- **`infer_onnx.py` / `app_onnx.py` 没有提供「ONNX 计算 dtype / FP16」之类的命令行或配置开关。**
- 底层 `ort_cpu_runtime.py` 创建会话时只设置图优化与线程数，使用 **`CPUExecutionProvider`**，未暴露混合精度、FP16 推理等选项。
- Python 侧喂给 ONNX 的浮点张量，代码里普遍按 **`numpy.float32`** 准备（例如 `global_hidden`、`repetition_penalty`、固定采样用的随机数 `assistant_random_u` / `audio_random_u` 等）；主机侧采样里 softmax 等实现可能用到 **`float64`** 做数值稳定，属于实现细节，不是用户可配的「模型精度」。
- **`app_onnx.py`** 里 `OnnxNanoTTSServiceAdapter.dtype = "float32"` 主要是与旧版 Web 演示接口对齐的**展示字段**，不是 ORT 的精度策略 API。
- **写 WAV**：内部波形按 float 处理，落盘为 **16-bit PCM**（`onnx_tts_runtime._write_waveform_to_wav`），这是**文件格式**，与 ONNX 图内算子精度不是同一层概念。

### 2.2 含义说明（「对接 ONNX 的浮点侧默认按 float32 准备数据」）

指：调用 `session.run` 时，由本仓库构造的**浮点输入数组**会按 **float32** 组装，以与常见 ONNX 导出（`tensor(float)`）一致；**不是**指仓库提供了可在 FP16/FP32 之间切换的配置项。

---

## 3. 文档与元数据

| 来源 | 是否说明 ONNX 推理 dtype/精度 |
|------|-------------------------------|
| 仓库根目录 `README_zh.md` | **未写** ONNX 的 FP32/FP16 等；ONNX CPU 章节只讲入口、下载、ORT CPU。 |
| `finetuning/README_zh.md` | **未写** ONNX；仅有训练用 `--mixed-precision bf16`（PyTorch 侧），与 ONNX 导出 dtype 无直接对应说明。 |
| Hugging Face `OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX` 模型卡 | **未写** FP32/FP16/BF16 等；只描述用途、文件列表、后端。 |
| `model_files/tts_browser_onnx_meta.json` | **无** `dtype` 字段；含 `opset`、张量**名字**、结构超参、`fixed_sampled_frame_constants` 等，**不替代** ONNX 图内的类型信息。 |

---

## 4. ONNX 图内的类型（以实测子图为例）

### 4.1 ONNX 里 `FLOAT` 的含义

在 ONNX 中，**`FLOAT` 表示单精度浮点，即通常所说的 float32**（不是 `FLOAT16`，也不是 `DOUBLE`/float64）。

### 4.2 `moss_tts_local_fixed_sampled_frame.onnx` 的 Graph I/O（实测）

对路径 `model_files/moss_tts_local_fixed_sampled_frame.onnx`，使用 `onnx.load(..., load_external_data=False)` 仅读图结构（避免缺少同目录 `moss_tts_local_shared.data` 时默认加载失败），打印得到的 **输入/输出元素类型** 为：

| 方向 | 名称 | `elem_type` |
|------|------|----------------|
| IN | `global_hidden` | **FLOAT**（float32） |
| IN | `repetition_seen_mask` | **INT32** |
| IN | `assistant_random_u` | **FLOAT** |
| IN | `audio_random_u` | **FLOAT** |
| OUT | `should_continue` | **INT32** |
| OUT | `frame_token_ids` | **INT32** |

说明：这是**多张 ONNX 子图之一**；其余如 `prefill`、`decode_step`、`local_decoder` 等，按同一套 browser/CPU 导出习惯，浮点侧一般亦为 `FLOAT`，若要做「整套导出是否含半精度」的严格结论，应对各 `.onnx` 分别检查或抽样检查中间 value_info。

### 4.3 本地复现命令提示

- 若本地**仅有 `.onnx`、缺少外部 `.data`**，需 **`load_external_data=False`** 才能只读图结构并打印 I/O 类型；否则 `onnx.load` 可能因外部权重路径校验失败。
- 示例（单行）：

```bash
python -c "import onnx; m=onnx.load('model_files/moss_tts_local_fixed_sampled_frame.onnx', load_external_data=False); [print('IN', i.name, onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)) for i in m.graph.input]; [print('OUT', o.name, onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type)) for o in m.graph.output]"
```

仓库中曾用于同样目的的辅助脚本：`model_files/_print_onnx_io_types.py`（若不需要可自行手动删除，避免与正式代码混淆）。

---

## 5. 总括结论

1. **本仓库 ONNX CPU 推理路径：没有面向用户的「精度枚举」配置；Python 侧浮点喂数按 float32 习惯准备；WAV 输出为 int16 PCM。**
2. **README / HF 模型卡 / `tts_browser_onnx_meta.json` 均未声明 ONNX 张量 dtype；dtype 以各 `.onnx` 的 graph 为准。**
3. **已核查子图 `moss_tts_local_fixed_sampled_frame.onnx`：浮点接口为 ONNX `FLOAT`（即 float32），离散/控制类为 INT32。**

---

*整理自对话：infer/app ONNX 路径、`README_zh.md`、HF 页面、meta JSON、子图 I/O 类型打印与 ONNX `FLOAT` 语义。*
