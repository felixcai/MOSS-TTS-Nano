# ONNX / ONNX Runtime 调优思路摘要

本文整理自 `调优思路-explore.md`，按**决策 → 环境 → 代码 → 排错 → TensorRT**顺序归纳关键步骤。

---

## 1. 总体策略（先读这条）

- **有 NVIDIA GPU（含 Jetson Orin）时**：优先让 **ONNX Runtime 走 GPU（CUDA / TensorRT）** 降 RTF，而不是先对拆分 + 外部权重的 ONNX 做简单 INT8 量化。
- **不要轻易改模型目录**：`prefill` / `decode_step` 等多份 `.onnx` + **`moss_tts_global_shared.data` 外部权重**，是官方为流式与省内存设计的；用「单文件量化脚本」容易破坏外部张量或共享关系。

---

## 2. 运行时：用 GPU 跑 ONNX

- **ONNX 是中间格式**；是否用 GPU 取决于 **`onnxruntime` 的 Execution Provider**，不是 ONNX 文件本身「只能 CPU」。
- 加载会话时显式指定 Provider，例如优先 CUDA：

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession("model.onnx", providers=providers)
```

- **可选**：在 NVIDIA 环境可再叠加 `TensorrtExecutionProvider`（见第 6 节）。

---

## 3. Python 包：`onnxruntime-gpu` 与冲突

- **必须用 `onnxruntime-gpu`** 才能稳定使用 `CUDAExecutionProvider`；仅装 `onnxruntime` 往往无法真正启用 GPU。
- **`onnxruntime` 与 `onnxruntime-gpu` 在 pip 里是不同包名**，并存时 `import onnxruntime` 可能仍落到 CPU 版 → 表现为「装了 GPU 包还是很慢」。
- **推荐做法**：先卸载再只装 GPU 版（可重复执行直到干净）：

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

- **Conda 环境**：用 **`python -m pip`** 安装到当前解释器对应环境，避免装到系统 Python。

---

## 4. Jetson Orin + JetPack 6.x（示例 JP6 / cu126）

- **一般不需要自编译 ORT**；用 NVIDIA 为 Jetson 提供的 wheel / 索引即可。
- **不要指望普通 PyPI 在 ARM64 上直接 `pip install` 出完整 GPU 能力**；使用例如：

```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

- 索引里需安装 **`onnxruntime-gpu`**，不要只盯 `onnxruntime` 目录名。
- **验证**：

```python
import onnxruntime as ort
print(ort.get_device())
print(ort.get_available_providers())
```

期望列表中出现 **`CUDAExecutionProvider`**（若构建带 TRT，还可能有 **`TensorrtExecutionProvider`**）。

---

## 5. 常见环境问题（按现象处理）

| 现象 | 处理要点 |
|------|----------|
| `import onnxruntime` 正常但缺属性 / 行为异常 | 检查当前工程目录下是否有同名文件夹 **`onnxruntime`** 遮蔽真包；`print(ort.__path__)` 看是否指到项目目录 |
| `ort.__path__` 为 `_NamespacePath(...site-packages/onnxruntime)` | 多为 **卸载残留空壳**；需在对应 `site-packages` 下 **删除残留 `onnxruntime` 目录** 后重装 `onnxruntime-gpu`（操作在目标机器上由人工执行） |
| 二进制与 NumPy 2.x 不兼容报错 | 将 NumPy 降到 1.x：`python -m pip install "numpy<2.0"` |

---

## 6. 项目侧代码修改（与本仓库相关）

- **强制 CPU 的代码**（例如 `providers=["CPUExecutionProvider"]`）会封印 GPU；应改为 **`["CUDAExecutionProvider", "CPUExecutionProvider"]`** 等。
- **CUDA + 外部 `.data` 权重** 时，若遇 `Cannot parse data from external tensors` / `GatherSliceToSplitFusion` 等 **ShapeInference / Fusion** 相关错误：将 **`SessionOptions.graph_optimization_level`** 从 **`ORT_ENABLE_ALL`** 降为 **`ORT_ENABLE_BASIC`**，以规避部分 GPU 路径下对外部张量不友好的高级图融合。

---

## 7. TensorRT 使用要点

- **首次运行慢**属正常：TRT 会为实际输入形状 **构建引擎**；日志里大量 Normalization / precision 提示多为信息级，可忽略。
- **Jetson DLA**：若日志提示 **fallback 到 GPU**，通常表示复杂动态序列不适合 DLA，**由主 GPU 执行是预期行为**。
- **动态句长**：比缓存过的形状 **更长** 的输入可能再次触发编译卡顿。
- **缓解**：
  - 配置 **`trt_engine_cache_enable`**、**`trt_engine_cache_path`**，把引擎落盘；
  - 可选 **`trt_fp16_enable`: True**；
  - **启动后主动用接近上限的长文本跑一次**，生成大 profile 的引擎并缓存，减少线上「突然变长再卡」的概率。

---

## 8. 效果判读（RTF）

- **`rtf_steady < 1.0`**：生成 1 秒音频耗时小于 1 秒，通常可满足实时边合成边播。
- **首包延迟**：关注 **`first_audio_latency_s`** 对流式体验的影响。
- **GPU 上的 Memcpy 等 Warning**：多为 CPU/GPU 混合执行时的常规提示，不等于失败。

---

## 9. 不推荐路径（上下文提醒）

- 在 **多 ONNX + 共享 `.data`** 的布局下，**不要盲目套用「单文件 ONNX 量化脚本」**，除非有完整链路处理外部权重与多会话一致性。
