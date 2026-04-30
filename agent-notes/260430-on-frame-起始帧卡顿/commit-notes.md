# 修复首次执行会有轻微卡顿的问题

**Commit Hash**: `08a526f399f084e3a26c44d944d50a36dfcf569b`
**Author**: caiyifeng <caiyifeng@axhy.com>
**Date**: Thu Apr 30 12:20:24 2026 +0800

**Commit Message**:
修复首次执行会有轻微卡顿的问题

**Diff**:

```diff
diff --git a/ort_cpu_runtime.py b/ort_cpu_runtime.py
index f0a253a..2549c46 100644
--- a/ort_cpu_runtime.py
+++ b/ort_cpu_runtime.py
@@ -287,9 +287,11 @@ def _resolve_stream_decode_frame_budget(
 ) -> int:
     lead_seconds = _compute_stream_lead_seconds(emitted_samples_total, sample_rate, first_audio_emitted_at_seconds)
     if not first_audio_emitted_at_seconds or lead_seconds < 0.20:
-        return 1
+        # return 1
+        return 4
     if lead_seconds < 0.55:
-        return 2
+        # return 2
+        return 4
     if lead_seconds < 1.10:
         return 4
     return 8
```

## Diff 代码变更说明

**1. 修改了什么？**
在 `ort_cpu_runtime.py` 文件中的 `_resolve_stream_decode_frame_budget` 函数里，调整了流式解码的初始帧预算（Batch Size）：

* 当超前缓冲时间 `< 0.20` 秒（或首次发送）时，将原本每次解码 **1 帧** 的策略改为了 **4 帧**。
* 当超前缓冲时间 `< 0.55` 秒时，将原本每次解码 **2 帧** 的策略改为了 **4 帧**。

**2. 为什么这么改（解决什么问题）？**

* **原逻辑缺陷**：原代码为了追求极致的首次响应延迟（TTFB），在初始阶段只攒 1 帧（0.04秒）就发给前端。由于网络传输或前端处理的微小波动，这极短的 0.04 秒音频很容易被瞬间播完，导致前端播放器缓冲区耗尽（Starvation），从而让用户听感上产生“起始帧卡顿”或“吞字”的现象。
* **修改后效果**：强制要求系统在初始阶段至少攒够 4 帧（即 0.16 秒的音频）后再进行 Codec 解码并发送。虽然在理论上牺牲了约一百多毫秒的首字节延迟，但为前端一口气提供了 0.16 秒的初始音频缓冲，大大增强了抗网络抖动的能力，从根本上消除了首次执行时的轻微卡顿感，让后续的音频流接续更加平滑。