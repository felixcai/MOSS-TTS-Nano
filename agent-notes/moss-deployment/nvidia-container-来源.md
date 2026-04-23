这两个镜像都来自于 **NVIDIA 官方的容器镜像注册表（NVIDIA Container Registry）**，即大名鼎鼎的 **NGC (NVIDIA GPU Cloud)** 平台。

域名中的 `nvcr.io` 正是 **NVIDIA Container Registry** 的缩写。

你可以通过浏览器访问 **NGC Catalog (NGC 目录)** 来搜索和浏览所有的官方镜像：[https://catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com)

以下是这两个镜像在 NGC 中的具体来源和命名背景：

### 1. 基础运行时镜像：`l4t-cuda`

* **来源页面**：在 NGC 网站搜索 `L4T CUDA`，对应的页面是 **NVIDIA L4T CUDA**。
* **命名含义**：
  * **`l4t`**：代表 **L**inux **4** **T**egra。Tegra 是英伟达为 Jetson 系列（如 Orin、Nano 等）开发的 SoC 芯片系列，L4T 是为其专门定制的底层 Linux 系统。
  * **`12.6.11-runtime`**：指的是内部包含 CUDA 12.6.11 版本，并且是 `runtime`（运行时）版本，体积较小，只包含运行编译好的 CUDA 程序所需的基础库（与之相对的是包含完整编译工具链的 `devel` 镜像）。

### 2. PyTorch 镜像：`pytorch:24.09-py3-igpu`

* **来源页面**：在 NGC 网站搜索 `PyTorch`，进入英伟达主打的 **PyTorch** 镜像仓库。
* **命名背景与变化（非常重要）**：
  * **`-igpu` 后缀**：在早期的 JetPack 5 时代，英伟达为 Jetson 专门提供了一个独立的镜像库叫 `l4t-pytorch`。**但从 JetPack 6 开始，英伟达改变了策略**，将 Jetson 的 PyTorch 镜像合并到了主线的 `pytorch` 仓库中。
  * 为了区分服务器独立显卡（dGPU）和 Jetson 的集成显卡，他们引入了 **`-igpu`（Integrated GPU）** 标签。所以凡是带有 `-igpu` 后缀的，都是专为 Jetson 优化的 ARM64 镜像。
  * **`24.09`**：代表 2024 年 09 月发布的版本。英伟达每月都会更新这些镜像框架。

---

### 💡 以后如何自己找最新版本的镜像？

如果你未来升级了系统，想找最新版本的镜像，可以这样做：

1. 访问 [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/containers)。
2. 搜索你需要的库，比如 `PyTorch`。
3. 点击进入后，找到页面上的 **"Tags"（标签）** 选项卡。
4. 在搜索框里过滤你要的关键词（比如 `igpu`），就能看到支持 Jetson / JetPack 6 平台的所有历史版本和最新版本的拉取命令了。