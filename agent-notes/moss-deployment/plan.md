在 Jetson Orin (ARM64 架构) 上安装复杂的 AI 模型时，通常最大的挑战在于**系统架构兼容性**以及**底层依赖（特别是 C++ 扩展包）的编译**。MOSS-TTS-Nano 官方虽然非常轻量，但它的前端依赖项 `pynini` 在 ARM 平台上直接用 `pip` 编译极易失败。

既然你已经下好了 Jetson 版的 PyTorch 和 Torchaudio 的 `.whl` 文件，我们将采用 **Docker + Miniforge (ARM64版的Conda)** 的方案。这样既可以完美隔离环境，又能通过 Conda 直接拉取预编译好的 `pynini`，免去漫长的编译折磨。

以下是为你规划的分步执行指南：

### 🚨 前期避坑确认（非常重要）

MOSS-TTS-Nano 官方建议使用 Python 3.12，但 **你手头的 `.whl` 包大概率是基于 Python 3.10 的**（由于 JetPack 6.x 默认环境是 Python 3.10，Nvidia 官方提供的 wheel 包文件名中通常会带有 `cp310` 字样）。
**请务必以你的 wheel 包支持的 Python 版本为准！** 以下步骤我将以 Python 3.10 为例，如果你的包是 `cp311` 等，请在下面 Dockerfile 中对应修改 Python 版本。

---

### 第一步：准备构建目录

在你的 Jetson 宿主机上，创建一个用于构建 Docker 镜像的文件夹，并将你下载好的两个 `.whl` 包放进去。

```bash
mkdir moss_docker
cd moss_docker
# 将你的包拷贝到当前目录，假设文件名如下：
# cp /path/to/torch-2.x.x-cp310-cp310-linux_aarch64.whl .
# cp /path/to/torchaudio-2.x.x-cp310-cp310-linux_aarch64.whl .
```

---

### 第二步：编写 Dockerfile

在 `moss_docker` 目录下创建一个名为 `Dockerfile` 的文件，填入以下内容。这个脚本帮你完成了系统依赖、Conda环境、本地Wheel安装以及MOSS-TTS-Nano的克隆部署。

```dockerfile
具体内容参看Dockerfile文件
```

---

### 第三步：构建 Docker 镜像

在包含 `Dockerfile` 和 `.whl` 文件的目录下，执行以下命令开始构建：

```bash
sudo docker build -t moss-tts-nano:jetson .
```

如果要用代理，并且修复网络的问题，用这条命令：

```bash
sudo docker build --network=host --build-arg HTTP_PROXY="http://192.168.0.103:7890" --build-arg HTTPS_PROXY="http://192.168.0.103:7890" -t moss-tts-nano:jetson .
```

原因说明：

```markdown
看到 `FATAL: Module not found`，并且带有 `5.15.148-tegra` 的字眼，这说明**问题出在 NVIDIA 官方的系统内核上。**

你使用的是较新的 JetPack 6 系统，NVIDIA 在编译这个版本的定制系统内核（`5.15.148-tegra`）时，**官方为了精简，直接把 `iptable_raw` 这个防火墙模块给物理删除了**。

因为底层内核根本就没有这个功能，所以 Docker 在尝试创建它的“默认虚拟隔离网络（Bridge）”时，因为找不到这个模块，就直接报错罢工了。

但这完全难不倒我们！对于构建镜像来说，我们有一个**最优雅、最直接的“降维打击”绕过方法**：

**终极绕过方案：使用“宿主机网络 (Host Network)”**

既然 Docker 自己创建虚拟网络会因为缺少内核模块而报错，那我们就在构建命令里加一个参数：`--network=host`。

这个参数的意思是：**“Docker 你别费劲去弄什么虚拟局域网和防火墙了，直接借用 Jetson 这台机子本身的真实网络环境！”**

这样一来，不仅完全绕过了那个报错，而且网络连通性会更好！
```

*(构建过程可能需要几分钟，特别是拉取镜像和安装 `WeTextProcessing` 的阶段，请耐心等待)*

---

### 第四步：启动 Docker 容器

构建完成后，你需要使用 NVIDIA Container Runtime 来启动它，这样容器内部才能调用到 Orin 的 GPU 算力。

```bash
sudo docker run -itd \
  --name moss-tts \
  --runtime nvidia \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e LD_LIBRARY_PATH=/opt/miniforge/envs/moss/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  -v "$(pwd)/model_data:/workspace/MOSS-TTS-Nano/models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/modelscope:/root/.cache/modelscope" \
  moss-tts-nano:jetson
```

**参数说明：**

- `--name moss-tts`：为容器指定固定名称，便于 `docker stop` / `docker logs` 管理；若希望退出后自动删除容器，可去掉 `--name` 并在命令末尾加上 `--rm`。
- `--runtime nvidia`：将 Jetson Orin 的 GPU 暴露给 Docker 容器。
- `--network host`：让容器直接使用主机的网络，方便你在外部浏览器访问测试页（默认端口 **18083**）。与同机运行的 vLLM（如 **8081**）等可同时使用，端口不冲突。
- `-v "$(pwd)/model_data:..."`：挂载模型目录，重启容器后仍保留 MOSS 相关模型文件；路径含空格时比 `$(pwd)` 更稳妥。
- `-v "$HOME/.cache/huggingface:..."`：将宿主 Hugging Face 缓存挂入容器（与 Dockerfile 中 `HF_HOME` 一致），重建容器不必重复下载权重；与 vLLM 侧常用挂载方式一致，可共用同一份宿主缓存。
- `-v "$HOME/.cache/modelscope:..."`：若从 ModelScope 下载模型，与 vLLM 笔记中的挂法对齐；若只使用 Hugging Face，可删去该 `-v` 行。

**与同机 vLLM 共存时的建议**：若 vLLM 已占用较多 GPU 显存，优先在运行 MOSS 时使用 `--backend onnx`，在 CPU 上推理以降低显存争抢；如需双服务同时调 GPU，需在实际负载下观察显存并酌情调整 vLLM 的 `--gpu-memory-utilization` 等参数。

---

### 第五步：进行克隆和推理测试

进入容器后，你已经在 `/workspace/MOSS-TTS-Nano` 目录下了，并且所需依赖全部就绪。

**1. 命令行测试（CLI 生成）**

```bash
moss-tts-nano generate \
    --prompt-speech assets/audio/zh_1.wav \
    --text "你好，我已经成功在 Orin Jetson 上运行了小体积的语音合成模型。"
```

此命令会将生成的音频保存在默认的 `generated_audio/moss_tts_nano_output.wav` 路径中。

**2. 启动 Web 网页演示**
如果你想通过直观的 UI 进行调整和测试，只需输入：

```bash
moss-tts-nano serve
```

然后在与你处于同一局域网的电脑浏览器中，输入你的 Jetson IP 地址：
`http://<Jetson的IP地址>:18083` 即可访问可视化 Web 页面！

**💡 性能提示**：
Orin 6.2 版性能已经非常强大，但如果你同时还需要跑视觉等其他占用大量 GPU 显存的模型，MOSS-TTS-Nano 官方也提供了一个对 CPU 十分友好的推理方案。只需在执行命令时加上 `--backend onnx` 参数，它就会纯利用 Orin 强大的 ARM CPU 核心进行推理，且速度也非常理想！