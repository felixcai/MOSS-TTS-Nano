# 建议使用 Ubuntu 22.04 基础镜像，适配 JetPack 6 
# FROM ubuntu:22.04
# 使用官方 JetPack 6.1 (CUDA 12.6) 的基础运行时镜像
# FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime
# 适用于 JetPack 6 / CUDA 12.6 的官方 PyTorch 镜像
FROM nvcr.io/nvidia/pytorch:24.09-py3-igpu

# 防止交互式提示导致构建卡死
ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装系统级基础依赖，并【彻底修复 PyTorch 缺失 cuDSS 库的 Bug】
RUN apt-get update && apt-get install -y \
    wget git build-essential ffmpeg libsndfile1 \
    && wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb \
    && dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb \
    && cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-local-tegra-0777B0CD-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get install -y cudss \
    && rm cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Miniforge3 (专为 ARM64 提供的 Conda 环境)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O ~/miniforge.sh \
    && bash ~/miniforge.sh -b -p /opt/miniforge \
    && rm ~/miniforge.sh
ENV PATH="/opt/miniforge/bin:$PATH"

# 3. 创建 conda 环境 (Python 3.10 匹配你的 whl 包)
RUN conda create -n moss python=3.10 -y
ENV CONDA_DEFAULT_ENV=moss
ENV PATH="/opt/miniforge/envs/moss/bin:$PATH"

# 4. 拷贝并安装【所有本地准备好的 Jetson 专属 Wheel 包】(包含 torchcodec)
COPY torch-*.whl /tmp/
COPY torchaudio-*.whl /tmp/
COPY torchcodec-*.whl /tmp/
RUN pip install /tmp/torch-*.whl /tmp/torchaudio-*.whl /tmp/torchcodec-*.whl

# 5. 【核心避坑】使用 conda 安装 ARM 极难编译的包，并提供专属 FFmpeg
# - pynini: 解决文本正则化 C++ 编译地狱
# - ffmpeg=7.1.0: 完美满足 torchcodec 0.10.0 的底层强依赖 (libavfilter.so.10)
RUN conda install -c conda-forge pynini=2.1.6.post1 ffmpeg=7.1.0 -y

# 6. 预创建 Hugging Face / ModelScope 缓存目录（默认 root 用户路径，与 docker run -v 挂载对齐，便于与 vLLM 等共用宿主缓存）
RUN mkdir -p /root/.cache/huggingface /root/.cache/modelscope
ENV HF_HOME=/root/.cache/huggingface
# 配置 Hugging Face 国内镜像源加速，解决国内网络下载模型慢或报错的问题
ENV HF_ENDPOINT=https://hf-mirror.com

# 7. 下载并安装 MOSS-TTS-Nano 及其剩余依赖
WORKDIR /workspace
RUN git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git
WORKDIR /workspace/MOSS-TTS-Nano

# 按照官方指南顺序安装相关库，安装 WeTextProcessing
RUN pip install git+https://github.com/WhizZest/WeTextProcessing.git

# 【关键保护】从 requirements 中剔除这三剑客，防止 pip 擅自覆盖我们装好的专属版
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchaudio/d' requirements.txt && \
    sed -i '/torchcodec/d' requirements.txt

RUN pip install -r requirements.txt

# 使用可编辑模式安装，激活 `moss-tts-nano` 命令行工具
RUN pip install --no-deps -e .

CMD ["/bin/bash"]