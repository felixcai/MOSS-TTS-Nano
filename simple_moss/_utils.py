from __future__ import annotations

# ============================================================
# _utils.py — simple_moss 包内共用工具
#
# 当前内容：
#   - _log_memory：内存打点日志，在 simple_ort_gpu_runtime.py、
#     simple_onnx_tts_runtime.py、simple_app_onnx.py 中均有使用。
# ============================================================

import logging


def _log_memory(label: str) -> None:
    """打印当前进程 RSS 和系统已用内存到日志，用于追踪推理过程中的内存变化。
    若未安装 psutil 则静默忽略。

    调用方：simple_ort_gpu_runtime.py（OrtCpuRuntime.__init__、_create_sessions、
            warmup、generate_audio_frames），
            simple_onnx_tts_runtime.py（OnnxTtsRuntime.__init__），
            simple_app_onnx.py（OnnxNanoTTSServiceAdapter.__init__、warmup、
            synthesize_stream._worker）。
    """
    try:
        import psutil

        proc_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        sys_used_mb = psutil.virtual_memory().used / (1024 * 1024)
        logging.info(
            "[MEM] %s | proc_rss=%.1f MB | sys_used=%.1f MB",
            label,
            proc_rss_mb,
            sys_used_mb,
        )
    except Exception:
        pass
