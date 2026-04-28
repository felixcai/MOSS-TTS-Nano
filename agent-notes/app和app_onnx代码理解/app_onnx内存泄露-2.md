# MOSS-TTS-Nano ONNX 内存管理与预热机制问答记录

## 1. 内存预热机制

**Q1：所谓的内存预热，包括哪些是要预热的？**

内存预热（Warmup）的主要目的是让 ONNX Session 完成首次 JIT 编译以及 **ORT Arena 的内存预分配**，从而降低首个请求的冷启动延迟。
根据文档记录，预热覆盖了以下 ONNX Session：

* `prefill`（计算全局 Attention）
* `local_cached_step` / `local_decoder` 等（自回归解码）
* `codec_encode`（参考音频编码）
* `codec_decode`（全量解码，仅非流式路径用到）

*(注：在当前的流式路径中，特有的 `codec_decode_step` 在 `app_onnx.py` 的 warmup 实现中被遗漏了，没有被预热到。这意味着在服务启动后的第一次流式生成请求中，当代码执行到流式解码阶段时，ONNX Runtime 会进行首次 JIT 编译，并且 ORT Arena 会在此时才去向操作系统申请所需的内存，导致首个流式请求产生额外的首帧延迟。)*

**Q2：这些预热的环节，如果在换了音色，或者换了文本（更长或者更短的文本）之后，是否还有效？**

**依然有效，但视情况可能需要申请新内存。**
ORT Arena 采用的是“高水位策略”（只进不出）。预热环节会在 Arena 中撑起一个初始的“内存水位”（预分配了一定大小的连续内存块）。

* **如果换了更短的文本或音色**：需要的内存小于或等于预热时分配的内存，预热的内存完全有效，直接复用，不会申请新内存。
* **如果换了更长的文本或音色**：导致单次需要的**单块连续内存**超过了预热时 Arena 中的最大空闲块（打破了历史最大需求纪录），Arena 就会向操作系统申请一块**全新**的大内存。此时预热的内存依然在池子里，只是不够用了。

---

## 2. 生成过程中的内存申请与释放

**Q3：在实际的生成音频的过程中，会申请哪些内存？**

生成过程中会申请两层内存：

* **底层 C++ 层（ORT Arena 内存池）**：
  * **Prefill 阶段**：为计算全局 Attention 申请庞大的张量内存（受参考音频长度和文本长度影响）。
  * **Decode 阶段**：为自回归生成和累积的 KV Cache 申请内存（受文本切分 chunk 的长度影响，步数越多 KV Cache 越大）。
* **上层 Python 层（Python Heap）**：
  * `all_generated_frames`：保存所有生成的 token 帧（纯浪费的死代码）。
  * 双份的 PCM 音频拷贝：在通过队列 yield 音频块时产生的 numpy 数组拷贝。
  * `all_waveforms` 及最终拼接的 `waveform` 大数组：在请求末尾会同时持有约 2 倍总音频 PCM 数据的内存。

**Q4：这些内存，在后续换了音色，或者换了文本之后，还会占用着么？**

* **Python 层的内存**：**不会占用**。Python 层的这些内存积累点生命周期仅限于单次请求。请求结束后，Python 引用归零，会被垃圾回收（GC）释放。
* **底层 ORT Arena 的内存**：**会一直占用着**。ORT Arena 采用高水位策略，向操作系统申请的内存**绝对不会归还**。无论后续换了多短的音色或文本，之前撑大的 Arena 内存池都会永久保留在进程中（表现为进程 RSS 居高不下）。

**Q5：哪些 arena 内存，其实是可以释放，但受限于当前 arena 机制，一直持有着不释放的？**

主要是**历史请求中为了应对“峰值需求”而申请的大块连续内存**，以及**已经碎片化的老内存**。
具体来说：

* 在处理某个超长 `text_chunk` 时累积的巨大 **KV Cache** 所占用的内存块。
* 在处理某个超长参考音频+长文本时，`Prefill` 阶段计算 **全局 Attention** 所申请的巨大临时张量内存。

当这次超长请求结束后，这些张量在逻辑上已经被销毁，其占用的内存块在 Arena 内部已经处于“空闲（Idle）”状态。在普通的 PyTorch CPU 模式下，这些空闲内存最终会被底层的 malloc 库通过 `munmap` 等方式归还给操作系统；但受限于 ORT Arena 的机制，这些实际上已经空闲的巨大内存块被 Arena 永久持有，作为备用池，导致系统内存无法回落。

---

## 3. 各阶段 Arena 内存与输入长度的关系

这四个 ONNX Session 在推理时所需的 Arena 内存大小，分别与不同的输入长度相关：

**1. `codec_encode`（参考音频编码）**

* **内存相关因素**：**参考音频（Clone 音色录音）的物理时长**。
* **原因**：将参考音频波形提取为离散的声学 token。录音越长，输入的波形数组越大，模型前向传播时产生的中间特征图（Feature Map）就越大。

**2. `prefill`（全局注意力计算）**

* **内存相关因素**：**参考音频的长度 + 当前 `text_chunk`（切分后的文本块）的长度**。
* **原因**：系统会把“参考音频的 token”和“当前这段文本的 token”拼接在一起，作为 `inputIds` 一次性送入模型计算全局 Attention。这两个部分的长度加起来越长，Prefill 阶段申请的临时张量（尤其是 Attention 矩阵，其大小与序列长度的平方成正比）就越庞大。

**3. `local_cached_step` / `local_decoder`（Decode 自回归生成阶段）**

* **内存相关因素**：**当前 `text_chunk`（切分后的文本块）的长度**。
* **原因**：逐帧生成音频 token 的阶段。模型每生成一帧，就会把当前状态存入 **KV Cache** 中。一个 `text_chunk` 越长，需要生成的音频帧数就越多，当生成到最后一帧时，积累的 KV Cache 达到峰值，占用的内存最大。
* *注意*：因为系统会对长文本进行切分，所以这里的内存上限取决于**切分后最长的那一个 `text_chunk`**，而不是传入的总文本长度。

**4. `codec_decode`（非流式全量解码）**

* **内存相关因素**：**送入解码的声学 token 总数（即生成的音频总时长）**。
* **原因**：在非流式路径下，一次性将离散声学 token 还原成连续的音频波形。送入的 token 序列越长，一次性分配的中间计算张量和最终输出的音频数组就越大。

---

## 4. 为什么启动时还没生成音频，内存就上涨了约 3GB？

在 MOSS-TTS-Nano 启动阶段（未处理任何请求），系统内存通常会上涨约 3GB。通过 `du -h` 命令可以观察到，磁盘上的模型文件（TTS + Codec）总大小仅为 728MB，内存暴涨的原因并非简单的文件映射，而是由 ONNX Runtime (ORT) 的加载机制和 CUDA 初始化开销共同造成的。

### 4.1 ORT 加载机制导致的内存膨胀（Memory Amplification）

* **反序列化与对象树**：磁盘上的 `.onnx` 是高度压缩的 Protobuf 文件。ORT 读取时必须将其解析为 C++ 内存中的节点、张量和属性对象树。对于包含大量算子的 Transformer 模型，这部分元数据的内存占用非常大。
* **权重的重排与对齐（Weight Repacking & Alignment）**：为了让 CPU/GPU 能最高效地执行矩阵乘法，ORT 会在初始化时申请一块新的、内存对齐的连续空间，把原始权重数据重新排列进去。这意味着加载阶段同一份权重在内存中可能同时存在两份。
* **多个 Session 独立加载（最关键的放大器）**：代码在 `_create_sessions` 中一口气创建了 `prefill`、`decode`、`local_decoder`、`codec_encode` 等 6~8 个独立的 `InferenceSession`。在 ORT 默认行为下，**不同的 Session 之间完全隔离，不共享内存中的权重副本**。即使 `prefill` 和 `decode` 底层共用了极大部分权重，ORT 也会为它们各自在内存中展开一份完整的图结构和重排后的权重。

### 4.2 CUDA 初始化与双重驻留的开销

* **CUDA Context 的“隐形税”**：当在 Python 进程中第一次调用初始化 GPU 的代码（如启用 `CUDAExecutionProvider`）时，NVIDIA 驱动会建立 CUDA Context。仅仅建立 Context，通常就会让进程的 CPU 内存（RSS）瞬间上涨 500MB 到 1GB 左右。
* **Host (CPU) 与 Device (GPU) 的双重驻留**：使用 `CUDAExecutionProvider` 时，模型权重最终要拷贝到 GPU 显存。但 ORT 在解析图结构、执行 CPU Fallback 算子，或维护张量元数据时，往往会**在 CPU 内存（Host Memory）中保留一份权重的副本或映射（Pinned Memory）**。

### 4.3 Warmup（预热）触发的 Arena 预分配

在服务启动的最后，`warmup()` 函数会执行一次真实的空推理。在预热的 `prefill` 阶段，计算全局 Attention 需要分配巨大的临时张量（如 $Q \times K^T$ 矩阵）。这些临时张量在预热瞬间被分配，撑大了进程的工作集（RSS）。

### 总结

这 3GB 的启动内存上涨中：

* **约 500MB ~ 1GB**：CUDA 驱动初始化（Context）的固定开销。
* **约 1.5GB ~ 2GB**：728MB 的模型文件被反序列化、权重重排，并且被 6 个独立的 InferenceSession 分别加载、互不共享所产生的内存膨胀。
* **剩余部分**：`warmup` 预热时申请的临时张量撑大的工作集。

---

# 释放 ORT Arena 内存的解决方案（Jetson Orin 等 CPU/GPU 共享内存环境）

在 Jetson Orin 这种 CPU 和 GPU 物理共享同一块内存的特殊背景下，任何一端的内存膨胀都会直接挤占整机的可用 RAM。因此，同时对 CPU 和 GPU 的内存分配器进行严格限制是非常必要的。

以下是**最完整、最严格的 ONNX Runtime 内存管理方案**，通过配置 `SessionOptions` 和 `CUDAExecutionProvider` 的参数，将内存膨胀压制到最低限度。具体是在 `ort_cpu_runtime.py` 创建 `_session` 的地方进行修改。

## 核心修改点

**1. 彻底禁用 CPU Arena**

```python
options.enable_cpu_mem_arena = False
```

* **作用**：关闭 CPU 端的内存池。对于所有在 CPU 上分配的张量（如输入/输出拷贝、Fallback 算子），用完立刻通过底层 `free()` 释放，绝不保留高水位。

**2. 禁止将模型权重（Initializers）放入 GPU Arena**

```python
options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
```

* **作用**：默认情况下，ORT 会把模型权重也塞进 GPU 的 Arena 内存池里。一旦权重进了池子，这个池子就变成了“活跃状态”，极难被收缩或释放。加上这行配置后，模型权重会绕过 Arena 直接通过底层 CUDA API 分配，从而让 Arena 保持纯粹（只存放推理时的临时张量），为后续的内存控制打下基础。

**3. 严格限制 CUDA Arena 的增长策略**
仅仅做前两步还不够，因为在推理过程中，GPU 依然会为临时张量（如 Attention 矩阵、KV Cache）在 Arena 中申请大量显存。默认情况下，CUDA Arena 是按 **2 的指数倍（Power of Two）** 激进增长的（比如需要 500MB，它可能会直接申请 1GB 备用）。

需要通过 `CUDAExecutionProvider` 的参数，将增长策略改为 **“按需申请（Same As Requested）”**：

```python
cuda_provider_options = {
    "arena_extend_strategy": "kSameAsRequested",  # 拒绝指数级暴涨，需要多少申请多少
}
```

## 完整的代码实现

将 `ort_cpu_runtime.py` 中的 `_session` 方法替换为以下代码：

```python
    def _session(self, path_value: Path) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.intra_op_num_threads = self.thread_count
        options.inter_op_num_threads = 1

        # 1. 彻底禁用 CPU 内存 Arena，内存用完即还给系统
        options.enable_cpu_mem_arena = False

        # 2. (已废弃) 禁止将模型权重分配到 GPU Arena 中
        # options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

        # 3. (已废弃) 严格配置 CUDA Execution Provider 的内存策略
        # cuda_provider_options = {
        #     "arena_extend_strategy": "kSameAsRequested",
        # }

        return ort.InferenceSession(
            str(path_value), 
            sess_options=options, 
            providers=["CUDAExecutionProvider"]
        )
```

## ⚠️ 严重警告：为什么放弃第 2 点和第 3 点？（14GB 内存暴涨之谜）

在实际测试中，同时开启第 2 点和第 3 点会导致 GPU 内存瞬间飙升至 14GB 甚至 15GB 最终 OOM。原因如下：

**1. 为什么放弃第 2 点（剥离权重）？**
在 MOSS-TTS-Nano 的架构中，代码一口气创建了 6~8 个独立的 `InferenceSession`（`prefill`, `decode`, `local_decoder` 等）。

* **默认行为（权重进入 Arena）**：虽然每个 Session 都有权重副本，但在同一个 Arena 内存池中，整体膨胀是受控的（约 3GB）。
* **剥离权重后**：每个 Session 都会**独立、毫无节制地**调用底层的 `cudaMalloc` 来加载权重。由于 `prefill` 和 `decode` 包含大量重复的 Transformer 权重，这导致同一份权重在显存中被硬生生复制了 6~8 份，瞬间撑爆显存（14GB）。

**2. 为什么放弃第 3 点（kSameAsRequested）？**
在 Decode 阶段，音频是一帧一帧生成的，KV Cache 的张量大小在**每一步都在动态变化**（比如 100, 101, 102...）。

* **kSameAsRequested**：强迫 Arena 每次都去切出极其精确的内存块。这导致不同步骤、不同请求之间的内存块**完全无法互相复用**。Arena 内部迅速产生海量碎片，只能无限向 GPU 申请新内存，导致内存像阶梯一样逐步上涨到 15GB。
* **默认的 kNextPowerOfTwo**：按 2 的指数倍扩展（如 128KB, 256KB）。虽然单个块浪费了一点空间，但这些“标准化尺寸”的块可以被完美复用，度过前期的碎片积累期后，内存增长会触顶停滞。

**结论**：在不开启每次 Run 之后的 Shrinkage（收缩）的前提下，最稳妥的策略是**仅保留第 1 点（禁用 CPU Arena）**，让 GPU Arena 保持默认行为（权重进 Arena，按 2 的指数倍扩展）。

---

**为什么没有加 `memory.enable_memory_arena_shrinkage`？**
因为已经彻底禁用了 CPU Arena，而 GPU 端的收缩策略需要修改每次 `session.run()` 的调用（传入 `RunOptions`），改动面太大。通过上述 3 点配置（尤其是 `kSameAsRequested` 和剥离权重），已经能将 GPU Arena 的膨胀压制到最低限度，通常足够解决 Jetson 上的 OOM 问题。

### 补充：如果要在 stream generate 路径中开启 GPU Arena 收缩，需要改动哪些地方？

**可以，但效果会大打折扣，甚至基本无效。**

如果不开启第 2 点（剥离权重）和第 3 点（按需申请），仅在 `RunOptions` 中开启 `memory.enable_memory_arena_shrinkage`，会发生：

1. **权重的“锚定”效应（最致命的问题）**
   模型权重被分配在 GPU Arena 内存池中。当一次推理结束后调用 Shrinkage，Arena 会尝试把空闲的内存块还给操作系统。**但是**，Arena 只能归还**位于内存池末尾的连续空闲块**。如果一块空闲内存的旁边紧挨着活跃的模型权重，这块空闲内存就会被“卡住”，无法归还给系统。
2. **默认的指数级暴涨依然存在**
   不开启第 3 点，Arena 依然会按 2 的指数倍（`kNextPowerOfTwo`）申请显存。这种频繁的“暴涨 -> 收缩 -> 暴涨”会带来极大的性能开销，而且因为“锚定”效应，收缩往往是不彻底的。

**最有效的做法是：**
不需要在每一次微小的 `session.run`（比如每一帧的 `local_cached_step`）后都去调用 Shrinkage，因为那样会严重拖慢推理速度。
你应该在**一个完整的 `text_chunk` 生成结束之后**（或者整个请求结束之后），调用一次带 Shrinkage 的空跑（或者在最后一次推理时带上 Shrinkage 参数），让 Arena 集中清理一下这一大段 Decode 过程中产生的垃圾。

### 补充：如果要 GPU Arena 收缩，最主要就是在 Decode 阶段吗？

**是的，最主要、最需要加的地方就是在 Decode 阶段。**

1. **内存泄漏/碎片化的重灾区**：Decode 阶段（自回归生成）是一帧一帧进行的。每生成一帧，序列长度加 1，KV Cache 的大小就变大一点。这种**几百上千次连续的、大小不断变化的张量分配和释放**，是导致 GPU Arena 碎片化和内存无限制增长的罪魁祸首。
2. **Prefill 阶段相对“干净”**：Prefill 阶段虽然会瞬间申请一块巨大的内存（用于计算全局 Attention），但它通常在一次请求中**只执行一次**。它申请了一块大内存，用完就释放了，行为非常规律，不容易产生大量细碎的碎片。
3. **Codec 阶段也是一次性的**：无论是编码参考音频（`codec_encode`）还是最终的波形解码（`codec_decode`），通常也是一次性处理一个大块数据，不会像 Decode 阶段那样陷入死循环式的动态分配。

因此，如果你要用 Shrinkage，目标就是清理 Decode 阶段留下的烂摊子。但再次强调，由于权重的锚定效应，即使你在 Decode 之后调用了 Shrinkage，它能回收的内存比例可能依然不理想。控制 `voice_clone_max_text_tokens` 依然是最根本、最有效的手段。

在 MOSS-TTS-Nano 的流式生成链路中，ONNX Runtime 被拆分成了多个独立的 Session。你不能像 `SessionOptions` 那样在初始化时统一配置一次就一劳永逸，而是必须深入到 `ort_cpu_runtime.py` 和 `onnx_tts_runtime.py` 的**每一个执行具体推理的底层函数中**，手动实例化 `ort.RunOptions()` 并将其作为关键字参数显式传递给每一个 `session.run()` 调用。

具体需要修改的地方包括：

1. **流式解码会话**：`ort_cpu_runtime.py` 中的 `CodecStreamingDecodeSession.run_frames`。
2. **自回归生成循环**：`ort_cpu_runtime.py` 中的 `OrtCpuRuntime.generate_audio_frames` 内部调用的辅助方法（如 `run_local_cached_step`、`run_local_greedy_frame`、`run_local_fixed_sampled_frame`）。
3. **Prefill 阶段**：`ort_cpu_runtime.py` 中的 `OrtCpuRuntime._run_prefill`。
4. **参考音频编码**：`onnx_tts_runtime.py` 中的 `OnnxTtsRuntime.encode_reference_audio`。

**示例代码**：

```python
run_options = ort.RunOptions()
run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
outputs = self.session.run(None, feeds, run_options=run_options)
```

**为什么不推荐这样做？**
这不仅破坏了代码的整洁性，而且在每次自回归生成（每秒可能执行几十上百次 `run_local_cached_step`）时频繁创建和销毁 `RunOptions` 对象，可能会带来额外的 Python 层面开销。

* **代价**：由于失去了内存池的缓存优势，每次推理（尤其是自回归 Decode 阶段频繁的小张量分配）都会产生大量的系统调用，可能会导致**推理速度（RTF）变慢**。

## 其他非接口层面的缓解方式

通过**控制输入长度**来降低内存峰值（因为 Arena 的高水位是由历史最大峰值决定的）：

1. **减小 `voice_clone_max_text_tokens`**：让每次切分出来的 `text_chunk` 更短，从而降低 Decode 阶段 KV Cache 的内存峰值。
2. **限制参考音频（Prompt Audio）的时长**：参考音频越短，Prefill 阶段的初始序列就越短，能大幅压低全局 Attention 计算时的内存尖峰。
