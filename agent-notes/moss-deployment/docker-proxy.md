在 Linux 下使用 Docker 时，网络代理的设置是一个经典的“坑”。因为 Docker 的架构是**客户端-服务端（C/S）架构**，这就导致了你遇到的两种不同场景，需要分别设置不同的代理：

1. **拉取镜像（`docker pull`）**：这是 **Docker Daemon（后台服务）**在工作，它不归你的当前终端管，所以你在终端里执行 `export HTTP_PROXY` 对它毫无用处。
2. **容器内网（如下载工具）**：这是 **Docker 容器**在工作，容器是一个隔离的环境，它默认不会继承宿主机（宿主Linux）的系统环境变量。
   ⚠️ **特别警告（新手必踩坑）：**
   如果你的代理软件（比如 Clash/V2ray）就运行在这台 Linux 宿主机上，代理地址是 `127.0.0.1:7890`，**在配置 Docker 代理时，绝不能写 `127.0.0.1` 或 `localhost`！** 因为对 Docker 而言，`127.0.0.1` 指的是 Docker 自己（或容器内部），而不是你的 Linux 宿主机。
   👉 **正确做法**：填写你的 **宿主机局域网 IP**（如 `192.168.1.100`），或者使用 Docker 的默认网关 IP（通常是 `172.17.0.1`）。并确保你的代理软件开启了**“允许局域网连接 (Allow LAN)”**。
   下面为你拆解各个场景的设置方法：

---

### 场景一：解决 `docker pull` 下载镜像慢/不通的问题

我们要给 Docker Daemon 配置代理。在 Linux 下最稳妥的方法是修改 `systemd` 的服务配置。
**1. 创建 Docker 的 systemd 配置目录**

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
```

**2. 创建/编辑代理配置文件**

```bash
sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf
```

**3. 写入代理配置（请把 IP 和端口换成你实际的代理地址）**

```ini
[Service]
Environment="HTTP_PROXY=http://192.168.x.x:7890"
Environment="HTTPS_PROXY=http://192.168.x.x:7890"
Environment="NO_PROXY=localhost,127.0.0.1,::1,.my-company.com,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
```

*(注：`NO_PROXY` 里的内网网段是为了防止本地容器间的通信或者局域网通信也被代理转发出去而导致失联)*
**4. 重新加载并重启 Docker 服务**

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

**5. 验证是否成功生效**
运行 `sudo docker info | grep -i proxy`，如果能看到你配置的 HTTP/HTTPS 代理地址，就说明成功了，现在你可以顺畅地 `docker pull` 了。

---

### 场景二：解决“容器运行后”在里面下载工具不通的问题

假设你已经 `docker run` 启动了一个容器，进入容器后想用 `apt-get` 或 `wget`，发现网络不通。你有两种方法：

#### 方法 A：一劳永逸法（配置 Docker 客户端全局属性）

如果你希望以后所有**新建的容器**都能自动连上代理，可以修改当前用户的 docker 配置文件。

1. 编辑当前用户目录下的 config.json：
   
   ```bash
   nano ~/.docker/config.json
   ```
   
   *(如果文件不存在，可以直接新建并写入大括号)*
2. 在里面加上 `proxies` 字段：
   
   ```json
   {
     "proxies": {
       "default": {
         "httpProxy": "http://192.168.x.x:7890",
         "httpsProxy": "http://192.168.x.x:7890",
         "noProxy": "localhost,127.0.0.1"
       }
     }
   }
   ```
   
   保存后，你下次再用 `docker run` 启动任何新容器时，Docker 会自动把这些设为容器内部的全局环境变量（对已存在的老容器无效）。
   
   #### 方法 B：临时起意法（在 `docker run` 时带上参数）
   
   如果你只是偶尔某个容器需要代理，可以在运行容器时直接把代理作为环境变量塞进去：
   
   ```bash
   docker run -it \
     -e HTTP_PROXY="http://192.168.x.x:7890" \
     -e HTTPS_PROXY="http://192.168.x.x:7890" \
     ubuntu:latest bash
   ```
   
   这样进去以后，这个容器的环境变量就已经有代理了。

---

### 场景三：解决 `docker build` 构建镜像时网络不通的问题

有时候你的需求不是进入容器，而是写了一个 `Dockerfile`，里面的 `RUN apt-get update` 卡住了。
由于构建过程是在临时容器里进行的，它既不读取 systemd 的代理，也不理会宿主机的环境变量。
**正确做法**是在使用 `docker build` 命令时，通过 `--build-arg` 传入代理：

```bash
docker build \
  --build-arg HTTP_PROXY="http://192.168.x.x:7890" \
  --build-arg HTTPS_PROXY="http://192.168.x.x:7890" \
  -t my-app:latest .
```

*(提示：在 Dockerfile 里千万不要用 `ENV HTTP_PROXY=...` 把代理写死，否则其他人拉取你的镜像后，会因为连不上你的本地代理而导致容器彻底断网)*