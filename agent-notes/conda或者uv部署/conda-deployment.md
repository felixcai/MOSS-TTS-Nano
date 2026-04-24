# Conda 部署说明

## 系统级工具：Miniforge 安装

### 下载安装脚本

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
```

### 静默安装

使用 `-b` 进行非交互静默安装：

```bash
bash Miniforge3-Linux-aarch64.sh -b
```

### 初始化环境变量

安装完成后初始化 Shell，使 `conda` 命令生效：

```bash
~/miniforge3/bin/conda init bash
```

刷新当前终端或新开一个终端：

```bash
source ~/.bashrc
```

## MOSS-TTS-Nano：Python 环境与依赖

### 创建并激活 Conda 环境

```bash
conda create -n moss-tts-nano python=3.10 -y
conda activate moss-tts-nano
```

### 克隆仓库并切换分支

```bash
git clone https://github.com/felixcai/MOSS-TTS-Nano.git
cd MOSS-TTS-Nano
git checkout -b felix origin/felix
```

### 安装 Python 包

按项目 README 安装依赖：

```bash
conda install -c conda-forge pynini=2.1.6.post1 -y
pip install git+https://github.com/WhizZest/WeTextProcessing.git
pip install -r requirements.txt
```

### 安装 onnxruntime-gpu

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

## 验证

在终端进入 Python 交互环境后执行：

```python
import onnxruntime as ort

print(ort.get_device())
print(ort.get_available_providers())

import torch
import torchaudio
```
