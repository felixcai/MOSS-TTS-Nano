"""
simple_moss 本地全流程测试脚本

运行方式（在项目根目录下）：
    python -m simple_moss.test_local_api
    python -m simple_moss.test_local_api --model-dir ./models --text "你好，世界。"

流程：
    1. 初始化（create_default_adapter）
    2. Warmup（warmup_runtime）
    3. Stream Generate（MossStreamApiFacade.start → audio → status → result → close）
    4. 将收到的 PCM s16le 写成 WAV，保存到 ./simple_moss_test_output.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import wave
from pathlib import Path

# 允许在项目根目录下直接 python -m simple_moss.test_local_api 运行
from .api_facade import MossStreamApiFacade
from .simple_app_onnx import create_default_adapter, warmup_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _write_pcm_to_wav(
    output_path: Path,
    pcm_chunks: list[bytes],
    sample_rate: int,
    channels: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # s16le = 2 bytes per sample
        wav.setframerate(sample_rate)
        for chunk in pcm_chunks:
            wav.writeframes(chunk)
    total_bytes = sum(len(c) for c in pcm_chunks)
    total_seconds = total_bytes / (sample_rate * channels * 2)
    log.info("WAV 已保存 → %s（%.2f 秒）", output_path, total_seconds)


def run_test(
    model_dir: str | None,
    text: str,
    output_wav: str,
    max_new_frames: int,
    voice_clone_max_text_tokens: int,
) -> None:
    # ── Step 1: 初始化 ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 1: 初始化 adapter（model_dir=%s）", model_dir or "默认")
    t0 = time.perf_counter()
    adapter = create_default_adapter(
        model_dir=model_dir,
        cpu_threads=1,
        max_new_frames=max_new_frames,
    )
    log.info("初始化完成，耗时 %.2f s", time.perf_counter() - t0)

    # ── Step 2: Warmup ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 2: Warmup")
    t0 = time.perf_counter()
    warmup_result = warmup_runtime(adapter)
    log.info("Warmup 完成，耗时 %.2f s", warmup_result.get("elapsed_seconds", time.perf_counter() - t0))

    # ── Step 3: 创建 facade ───────────────────────────────────────────
    facade = MossStreamApiFacade(adapter, max_new_frames=max_new_frames, voice_clone_max_text_tokens=voice_clone_max_text_tokens)

    # ── Step 4: start ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 3 (start): text=%r", text)
    t_start = time.perf_counter()
    start_resp = facade.start(text)
    stream_id = str(start_resp["stream_id"])
    log.info("start 返回: stream_id=%s | state=%s", stream_id, start_resp["state"])

    # ── Step 5: audio（主消费循环） ────────────────────────────────────
    log.info("Step 4 (audio): 开始消费 PCM 流...")
    pcm_chunks: list[bytes] = []
    sample_rate = 48000
    channels = 2
    first_chunk_at: float | None = None

    for pcm_bytes in facade.audio(stream_id):
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter()
            log.info("首帧 PCM 到达，首帧延迟 = %.3f s", first_chunk_at - t_start)
        pcm_chunks.append(pcm_bytes)

    log.info(
        "音频流结束，共收到 %d 个 PCM chunk，总字节 %d",
        len(pcm_chunks),
        sum(len(c) for c in pcm_chunks),
    )

    # ── Step 6: status ────────────────────────────────────────────────
    log.info("Step 5 (status): 查询任务状态")
    try:
        snap = facade.status(stream_id)
        log.info(
            "status → state=%s | emitted=%.2f s | text_chunks=%d | status_text=%r",
            snap.get("state"),
            snap.get("emitted_audio_seconds", 0.0),
            len(snap.get("text_chunks") or []),
            snap.get("status_text"),
        )
        sample_rate = int(snap.get("sample_rate") or sample_rate)
        channels = int(snap.get("channels") or channels)
    except KeyError as exc:
        log.warning("status 查询失败（job 可能已被删除）: %s", exc)

    # ── Step 7: result ────────────────────────────────────────────────
    log.info("Step 6 (result): 获取最终结果")
    try:
        final = facade.result(stream_id)
        if final is None:
            log.warning("result 返回 None（任务尚未完成？）")
        else:
            log.info(
                "result → run_status=%r | text_chunks=%d | emitted=%.2f s",
                final.get("run_status"),
                len(final.get("text_chunks") or []),
                final.get("emitted_audio_seconds", 0.0),
            )
    except KeyError as exc:
        log.warning("result 查询失败: %s", exc)

    # ── Step 8: close ─────────────────────────────────────────────────
    log.info("Step 7 (close): 清理任务")
    try:
        facade.close(stream_id)
        log.info("close 完成")
    except KeyError as exc:
        log.warning("close 失败（job 可能已被删除）: %s", exc)

    # ── Step 9: 保存 WAV ──────────────────────────────────────────────
    log.info("=" * 60)
    if pcm_chunks:
        output_path = Path(output_wav).expanduser().resolve()
        _write_pcm_to_wav(output_path, pcm_chunks, sample_rate, channels)
    else:
        log.warning("未收到任何 PCM 数据，跳过 WAV 写入")

    total_elapsed = time.perf_counter() - t_start
    log.info("全流程完成，总耗时 %.2f s", total_elapsed)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="simple_moss 本地全流程测试")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="ONNX 模型目录（默认使用 models/ 下的默认路径）",
    )
    parser.add_argument(
        "--text",
        default="你好，这是一段来自 simple_moss 的本地合成测试语音。",
        help="要合成的文本",
    )
    parser.add_argument(
        "--output",
        default="simple_moss_test_output.wav",
        help="输出 WAV 文件路径",
    )
    parser.add_argument(
        "--max-new-frames",
        type=int,
        default=375,
        help="最大生成帧数（默认 375）",
    )
    parser.add_argument(
        "--voice-clone-max-text-tokens",
        type=int,
        default=32,
        help="每个文本 chunk 最大 token 数（默认 32）",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_test(
        model_dir=args.model_dir,
        text=args.text,
        output_wav=args.output,
        max_new_frames=args.max_new_frames,
        voice_clone_max_text_tokens=args.voice_clone_max_text_tokens,
    )
