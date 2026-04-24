# Conda / uv / venv 部署与选型笔记

整理自一次对话中的结论与操作要点，便于在本仓库（MOSS-TTS-Nano）相关场景查阅。日期：2026-04-23。

---

## 1. `venv`、`uv`、`conda` 各是什么、怎么对比

| 维度 | **venv**（标准库） | **uv**（Astral） | **conda** / **mamba** |
|------|-------------------|------------------|------------------------|
| 身份 | 只做虚拟环境 | 包安装 + 项目管理（常作 pip 系更快替代），底层仍是常见 venv 语义 | 独立生态：环境 + 二进制包 |
| 隔离对象 | 主要是 Python `site-packages` | 同上 | Python + 大量非 Python 预编译依赖 |
| 默认「货源」 | 配合 pip → PyPI | PyPI（及 wheel / 锁文件工作流） | channel（conda-forge 等） |
| 典型强项 | 官方、最小、无额外工具 | 解析与安装快、现代依赖管理 | 科学计算栈、复杂原生依赖成套求解 |
| 学习成本 | 低 | 中 | 中高 |

**选型一句话**

- 只要官方最小方案：`venv` + `pip`。
- 要快、要现代工作流：`uv`（常与 venv 语义并存）。
- 要整栈二进制都管、solver 成套配：`conda` / `mamba`。

---

## 2. `uv` 会不会「隔离」Torch 这类二进制包

**会隔离的是「装在哪个环境里」**，不是「包办整台机器的二进制栈」。

- `torch` 会进**当前虚拟环境**的 `site-packages`（以及 wheel 自带的 `.so` / `.dll` 等），与其它环境的 Python 包路径分离；这一点与 `pip` + `venv` 同类。
- **系统级**内容（显卡驱动、部分 CUDA 变体依赖的系统库、VC++ 运行库等）仍由系统/本机决定，`uv` 不负责。
- **相对 conda**：`uv` 更接近 pip，隔离的是 **Python 环境目录**；conda 常把 **Python + 多份原生库** 统一解在同一前缀并由 solver 管理。

---

## 3. conda 与 uv 在「装什么东西」上有什么不一样

核心：**谁在决定「这一环境里到底有什么文件」**，两者答案不同。

1. **默认货源**  
   - **uv**：主要是 **PyPI 上的 Python 包**（多为 **wheel**）。  
   - **conda**：**channel 里的 conda 包**（ tarball / `.conda`，布局是 conda 前缀下的 `lib/`、`bin/` 等）。

2. **一坨里含什么**  
   - **uv**：把 wheel 里已有内容解压到当前环境的 `site-packages` 等；不会自动替你装「整套系统 CUDA / 编译器全家桶」，除非 wheel 自带或其它 PyPI 包拉进来。  
   - **conda**：常按元数据一次落下**多个 conda 包**（如 BLAS、运行库等），**同一前缀**内多库共存。

3. **解依赖**  
   - **uv**：按 Python 包元数据（如 `Requires-Dist`），与 pip 一脉。  
   - **conda**：channel 优先级 + **全局 solver**，同一前缀内要自洽。

因此：装「纯 Python / 小 wheel」时差別小；装「重二进制、强绑定系统或其它库」时，conda 常是**成套 conda 包**；uv 常是 **PyPI wheel 组合**。

---

## 4. 结合本仓库 `README_zh.md`：更适合 uv 还是 conda

本仓库安装主线是：**Python 3.12 环境 → `pip install -r requirements.txt` → `pip install -e .`**。

**痛点在 `pynini` / `WeTextProcessing`**：README 写明若 pip 装不上，**推荐用 conda-forge 装 `pynini`**，不用 conda 则需**与 Python 版本和平台匹配的 `pynini` wheel**（见仓库 Issue #6）。

| 情况 | 建议 |
|------|------|
| `pynini` 能顺利装上（有合适 wheel 或已按 Issue 处理） | **uv 很合适**：`uv venv` + `uv pip install -r requirements.txt` + `uv pip install -e .`，与文档 pip 流程一致、更快。 |
| `pynini` / `WeTextProcessing` 在 pip 上反复失败 | **按 README 用 conda-forge 装 `pynini`** 更省事。 |
| 只跑 ONNX CPU 且依赖链不涉及该问题 | 仍取决于 `requirements.txt` 是否包含 `pynini`；若仍包含，问题可能存在。 |

**结论**：`uv` **不能替代** conda-forge 对 `pynini` 那种 **conda 包构建**；`uv` 主要是 **更快的 pip 系工具**。卡在 `pynini` 上时，按 README 用 conda（或自行解决 wheel）更现实。

---

## 5. 资源使用：conda 与 uv 差别大吗

- **磁盘**：`uv` + venv 通常**更瘦**（主要是 Python + `site-packages`）。conda 环境常为**完整前缀**，solver 拉多包 + **包缓存**，多环境时重复占盘可能更明显（缓存硬链接可缓解，但仍常比纯 venv 胖）。  
- **安装过程**：`uv` 一般**更快、CPU 峰值时间更短**；conda 经典 solver 在复杂环境上可能更慢、更吃 CPU，`mamba` 会改善。  
- **运行时**：跑同一模型/服务时，**峰值内存主要看业务与库（如 torch）**，与「用 uv 装还是用 conda 装」关系不大，只要加载的是同类二进制。

---

## 6. `conda install -c conda-forge pynini=...` 能否下载后交给 `uv` 本地安装

**不能**把 conda-forge 的 **`.conda` / `.tar.bz2`** 当作 `uv pip install` 的输入；格式与元数据体系不同。

可行方向：

- 弄到 **`.whl`**（PyPI 或可信镜像），再 `uv pip install /path/to/pynini-....whl` 或 `--find-links` 指向本地目录。  
- 若只有 conda-forge 才有合适构建、又没有 wheel：要么 **conda 只负责 `pynini`**，其余仍 pip/uv；要么本地 **从源码构建 wheel**（依赖链更重）。

---

## 7. conda 只装 `pynini`，其余用 `uv pip` 装进同一解释器

**环境仍只有一个 conda 环境**；不要另建 `uv venv`，否则变成两套环境。

`conda activate` 后，用 **`--python` 指向该环境的解释器**，避免 `uv pip` 装到别的 Python（conda 未必设置 `VIRTUAL_ENV`）。

**PowerShell（Windows）示例：**

```powershell
conda create -n moss-tts-nano python=3.12 -y
conda activate moss-tts-nano

conda install -c conda-forge pynini=2.1.6.post1 -y

uv pip install --python "$env:CONDA_PREFIX\python.exe" git+https://github.com/WhizZest/WeTextProcessing.git
uv pip install --python "$env:CONDA_PREFIX\python.exe" -r requirements.txt
uv pip install --python "$env:CONDA_PREFIX\python.exe" -e .
```

**Bash 示例：**

```bash
conda activate moss-tts-nano
uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements.txt
uv pip install --python "$CONDA_PREFIX/bin/python" -e .
```

顺序与 README 一致即可：**先 conda 的 `pynini` → WeTextProcessing（git）→ `requirements.txt` → `-e .`**。

---

## 8. 这样混用和「conda 里直接 pip」比，好处在哪

- **环境仍是同一个 conda 前缀**，没有额外「隔离层」红利。  
- **差别主要在**：`uv pip` 通常 **安装更快**、**缓存与解析体验更好**；若项目别处已用 uv，命令习惯统一。  
- **若依赖少、很少重装**：直接用 **pip 完全够用**，少一个工具也可。

---

## 9. Jetson Orin + JetPack 6.2 上安装 Conda（Miniconda）

JetPack 6.2 常见为 **Ubuntu 22.04（L4T）+ `aarch64`**。必须安装 **Linux aarch64** 的 Miniconda，**不能**用 `x86_64` 安装包。

1. 确认架构：`uname -m` → 应为 **`aarch64`**。  
2. 下载并执行官方脚本（注意文件名含 **`Linux-aarch64`**）：  
   - 索引目录：<https://repo.anaconda.com/miniconda/>  
   - 例如：`Miniconda3-latest-Linux-aarch64.sh`  
3. `bash Miniconda3-latest-Linux-aarch64.sh` 按提示选择路径与是否 `conda init`；完成后 `source ~/.bashrc` 或重开终端。  
4. **ARM64 上优先 `conda-forge`**；并非所有包都有 `linux-aarch64` 构建。  
5. **Jetson 上的 PyTorch / CUDA** 常与 **JetPack / NVIDIA 提供的 wheel 或系统栈** 绑定，不要默认与 x86 GPU 的 `conda install pytorch` 体验一致。

---

## 10. 参考链接

- Miniconda 安装包目录：<https://repo.anaconda.com/miniconda/>  
- Anaconda 下载页（含 ARM64）：<https://www.anaconda.com/download>  
- JetPack 6.2 说明（NVIDIA）：<https://developer.nvidia.com/embedded/jetpack-sdk-62>  
- 本仓库中文 README：`README_zh.md`  
- `pynini` 非 conda 安装讨论：仓库 Issue #6（README 中有引用）
