from __future__ import annotations

# ============================================================
# api_facade.py — 本地五接口协议层
#
# 从 app.py 剥离而来，HTTP/UI 层替换为本地方法：
#   - 保留 StreamingJob / StreamingJobManager / 状态工具函数
#   - _run_streaming_job 提升为顶层函数，去掉 HTTP/文件/上传相关参数
#   - _put_stream_audio 保留不变
#   - 新增 MossStreamFacade（持有 adapter + execution_lock）
#   - 新增 MossStreamApiFacade（start / audio / status / result / close）
# ============================================================

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .simple_app_onnx import OnnxNanoTTSServiceAdapter


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数（从 app.py 筛出，保留不变）
# ══════════════════════════════════════════════════════════════════════════════

def _format_run_status(result: dict[str, object]) -> str:
    """将 result 事件格式化为人类可读的完成状态文本。
    输出格式：Done | chunks=N | audio=X.XXs | elapsed=Y.YYs

    调用方：本文件 _run_streaming_job（收到 result 事件时生成状态文本并写入 job）。
    """
    sample_rate = int(result.get("sample_rate") or 48000)
    emitted_audio_seconds = float(result.get("emitted_audio_seconds") or 0.0)
    elapsed_seconds = float(result.get("elapsed_seconds") or 0.0)
    text_chunks = list(result.get("text_chunks") or [])
    return (
        f"Done | chunks={len(text_chunks)} | audio={emitted_audio_seconds:.2f}s | elapsed={elapsed_seconds:.2f}s"
    )


def _format_stream_status(snapshot: dict[str, object]) -> str:
    """根据 job snapshot 中的 failed / ready / closed / run_status 字段，
    生成面向用户的当前流式状态描述文本。

    调用方：本文件 MossStreamApiFacade.status（在 snapshot 上附加 status_text 字段）。
    """
    if bool(snapshot.get("failed")):
        return f"Stream failed: {snapshot.get('error') or snapshot.get('run_status') or 'Unknown error'}"
    if bool(snapshot.get("ready")):
        return str(snapshot.get("run_status") or "Stream complete.")
    if bool(snapshot.get("closed")):
        return "Stream closed."
    return str(snapshot.get("run_status") or "Streaming...")


def _normalize_stream_chunk_index(
    raw_chunk_index: object,
    *,
    chunk_count: int,
    current_base: int | None,
) -> tuple[int | None, int | None]:
    """将 synthesize_stream 事件中可能以 0 或 1 为起始的 chunk_index 规范化为 0-based 索引。
    通过自动检测第一个 chunk_index 的值来推断 base（0 还是 1），并在后续帧中保持一致。
    返回 (normalized_chunk_index, updated_base)；无法规范化时第一项返回 None。

    调用方：本文件 _run_streaming_job（处理每个 audio 事件时规范化 chunk_index）。
    """
    try:
        numeric_chunk_index = int(raw_chunk_index)
    except Exception:
        return None, current_base

    if chunk_count <= 0:
        return max(0, numeric_chunk_index), current_base

    normalized_base = current_base
    if normalized_base is None:
        if numeric_chunk_index == 0:
            normalized_base = 0
        elif numeric_chunk_index == chunk_count:
            normalized_base = 1
        elif numeric_chunk_index == 1:
            normalized_base = 1
        else:
            normalized_base = 0

    normalized_chunk_index = numeric_chunk_index - normalized_base
    if 0 <= normalized_chunk_index < chunk_count:
        return normalized_chunk_index, normalized_base
    if 0 <= numeric_chunk_index < chunk_count:
        return numeric_chunk_index, 0
    if 1 <= numeric_chunk_index <= chunk_count:
        return numeric_chunk_index - 1, 1
    return None, normalized_base


def _audio_to_pcm16le_bytes(audio_array) -> bytes:
    """将 float32 波形数组（值域 [-1, 1]）转换为 PCM s16le 字节串，供音频队列传输。
    自动处理 1D/2D 形状，并对 channel-major 格式（channels < samples）进行转置。

    调用方：本文件 _run_streaming_job（处理每个 audio 事件，将 numpy 波形转为 PCM 字节）。
    """
    audio_np = np.asarray(audio_array, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T
    elif audio_np.ndim != 2:
        raise ValueError(f"Unsupported audio array shape: {audio_np.shape}")
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    return audio_int16.tobytes()


def _coerce_bool(value: str | None, default: bool) -> bool:
    """将字符串形式的布尔值（"true"/"false"/"1"/"0" 等）转换为 Python bool。
    value 为 None 或无法识别时返回 default。

    调用方：此函数在 simple_moss 内部当前未被调用，保留以兼容原 app.py 的调用约定。
    """
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


# ══════════════════════════════════════════════════════════════════════════════
# StreamingJob / StreamingJobManager（从 app.py 保留不变）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StreamingJob:
    """表示一次流式 TTS 任务的完整状态，包括：
      - audio_queue：PCM 字节流队列，消费方通过 MossStreamApiFacade.audio 读取；
      - state / run_status / error：当前任务状态和错误信息；
      - emitted_audio_seconds / lead_seconds：实时音频进度统计；
      - final_result：任务完成后由 _run_streaming_job 写入的汇总结果。
    每个 StreamingJob 实例由 StreamingJobManager 创建和管理，
    生命周期从 MossStreamApiFacade.start 开始，到 MossStreamApiFacade.close 结束。
    """
    stream_id: str
    audio_queue: "queue.Queue[bytes | None]" = field(default_factory=lambda: queue.Queue(maxsize=64))
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    first_audio_at: float | None = None
    completed_at: float | None = None
    state: str = "starting"
    run_status: str = "Starting realtime synthesis..."
    error: str | None = None
    sample_rate: int = 48000
    channels: int = 2
    emitted_audio_seconds: float = 0.0
    lead_seconds: float = 0.0
    current_chunk_index: int | None = None
    text_chunks: list[str] = field(default_factory=list)
    chunk_index_base: int | None = None
    audio_chunk_ranges: list[tuple[float, float, int]] = field(default_factory=list)
    is_closed: bool = False
    final_result: dict[str, object] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _resolve_playback_chunk_index_locked(self) -> int | None:
        """根据 emitted_audio_seconds 和 lead_seconds 估算当前实际播放到的 chunk index。
        必须在持有 self.lock 的情况下调用（方法名 _locked 后缀表明此约定）。

        调用方：本类 snapshot（在 with self.lock 块内调用）。
        """
        if not self.audio_chunk_ranges:
            return self.current_chunk_index
        playback_audio_seconds = max(0.0, float(self.emitted_audio_seconds) - float(self.lead_seconds))
        for start_seconds, end_seconds, chunk_index in self.audio_chunk_ranges:
            if playback_audio_seconds <= end_seconds + 1e-6:
                return chunk_index
        return self.audio_chunk_ranges[-1][2]

    def snapshot(self) -> dict[str, object]:
        """在锁保护下读取 job 的全部状态字段，返回可安全传递给外部调用方的字典快照。
        包含 ready / failed / closed 等布尔标志，以及首帧延迟统计。

        调用方：本文件 MossStreamApiFacade.status。
        """
        with self.lock:
            return {
                "stream_id": self.stream_id,
                "state": self.state,
                "run_status": self.run_status,
                "error": self.error,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "emitted_audio_seconds": self.emitted_audio_seconds,
                "lead_seconds": self.lead_seconds,
                "current_chunk_index": self.current_chunk_index,
                "playback_chunk_index": self._resolve_playback_chunk_index_locked(),
                "text_chunks": list(self.text_chunks),
                "first_audio_latency_seconds": (
                    None
                    if self.started_at is None or self.first_audio_at is None
                    else max(0.0, self.first_audio_at - self.started_at)
                ),
                "completed_at": self.completed_at,
                "ready": self.state == "done",
                "failed": self.state == "failed",
                "closed": self.is_closed,
            }


class StreamingJobManager:
    """管理所有活跃的 StreamingJob 实例，提供线程安全的 CRUD 操作。
    被 MossStreamApiFacade 持有，作为 stream_id → StreamingJob 的注册表。
    """

    def __init__(self) -> None:
        """初始化内部锁和 job 字典。

        调用方：本文件 MossStreamApiFacade.__init__（实例化时自动调用）。
        """
        self._lock = threading.Lock()
        self._jobs: dict[str, StreamingJob] = {}

    def create(self) -> StreamingJob:
        """生成唯一的 stream_id，创建 StreamingJob 并注册到内部字典，返回新建的 job。

        调用方：本文件 MossStreamApiFacade.start（每次发起新推理请求时调用）。
        """
        stream_id = f"stream-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        job = StreamingJob(stream_id=stream_id)
        with self._lock:
            self._jobs[stream_id] = job
        return job

    def get(self, stream_id: str) -> StreamingJob | None:
        """按 stream_id 查找并返回对应的 StreamingJob，不存在时返回 None。

        调用方：本文件 MossStreamApiFacade.audio / status / result / close。
        """
        with self._lock:
            return self._jobs.get(stream_id)

    def close(self, stream_id: str) -> StreamingJob | None:
        """将 job 标记为已关闭（is_closed=True），并向 audio_queue 放入 sentinel（None）
        以通知消费方（MossStreamApiFacade.audio 的迭代循环）退出。

        调用方：本文件 MossStreamApiFacade.close（触发任务关闭序列的第一步）。
        """
        with self._lock:
            job = self._jobs.get(stream_id)
        if job is None:
            return None
        with job.lock:
            job.is_closed = True
            job.state = "closed" if job.state not in {"done", "failed"} else job.state
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass
        return job

    def delete(self, stream_id: str) -> StreamingJob | None:
        """从内部字典中删除并返回指定的 StreamingJob，不存在时返回 None。

        调用方：本文件 MossStreamApiFacade.close（close 后立即 delete，释放内存）。
        """
        with self._lock:
            return self._jobs.pop(stream_id, None)


# ══════════════════════════════════════════════════════════════════════════════
# _put_stream_audio（从 app.py 保留不变）
# ══════════════════════════════════════════════════════════════════════════════

def _put_stream_audio(job: StreamingJob, pcm_bytes: bytes) -> None:
    """将 PCM 字节块放入 job.audio_queue，以 0.1s 超时轮询（队列满时重试）。
    检测到 job.is_closed 时立即返回，不再入队。

    调用方：本文件 _run_streaming_job（每收到一个 audio 事件时调用）。
    """
    while True:
        with job.lock:
            if job.is_closed:
                return
        try:
            job.audio_queue.put(pcm_bytes, timeout=0.1)
            return
        except queue.Full:
            continue


# ══════════════════════════════════════════════════════════════════════════════
# _run_streaming_job（收窄版本，去掉 HTTP/上传/设备路由相关参数）
# ══════════════════════════════════════════════════════════════════════════════

def _run_streaming_job(
    job: StreamingJob,
    stream_facade: "MossStreamFacade",
    *,
    text: str,
    max_new_frames: int,
    voice_clone_max_text_tokens: int,
    attn_implementation: str,
    do_sample: bool,
    text_temperature: float,
    text_top_p: float,
    text_top_k: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    seed: int | None,
    chunk_pause_seconds: float,
) -> None:
    """在后台线程中驱动一次完整的流式 TTS 任务：
      - 调用 stream_facade.stream_generate 获取事件迭代器；
      - 对每个 audio 事件：转换 PCM 字节，更新 job 状态，调用 _put_stream_audio 入队；
      - 对 result 事件：计算 RTF（Real-Time Factor）并写入 job.final_result；
      - 异常时将 job 标记为 failed；
      - 最终向 audio_queue 放入 sentinel（None）通知消费方结束。

    调用方：本文件 MossStreamApiFacade.start，通过 threading.Thread 在后台启动。
    """
    try:
        with job.lock:
            job.started_at = time.monotonic()
            job.state = "running"
            job.run_status = "Streaming realtime audio..."

        last_end = time.monotonic()
        rtf_pending_lead_gen_s = 0.0
        rtf_first_gen_s: float | None = None
        rtf_first_audio_s: float | None = None
        rtf_steady_gen_s_sum = 0.0
        rtf_steady_audio_s_sum = 0.0
        rtf_audio_chunk_count = 0

        for event in stream_facade.stream_generate(
            text=text,
            max_new_frames=int(max_new_frames),
            voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
            attn_implementation=attn_implementation,
            do_sample=bool(do_sample),
            text_temperature=float(text_temperature),
            text_top_p=float(text_top_p),
            text_top_k=int(text_top_k),
            audio_temperature=float(audio_temperature),
            audio_top_p=float(audio_top_p),
            audio_top_k=int(audio_top_k),
            audio_repetition_penalty=float(audio_repetition_penalty),
            seed=seed,
            chunk_pause_seconds=float(chunk_pause_seconds),
        ):
            t_receive = time.monotonic()
            gen_time_s = t_receive - last_end
            try:
                event_type = str(event.get("type", ""))
                with job.lock:
                    if job.is_closed:
                        break

                if event_type == "audio":
                    waveform_numpy = np.asarray(event["waveform_numpy"], dtype=np.float32)
                    pcm_bytes = _audio_to_pcm16le_bytes(waveform_numpy)
                    sample_rate = int(event["sample_rate"])
                    channels = 1 if waveform_numpy.ndim == 1 else int(waveform_numpy.shape[1])
                    is_pause = bool(event.get("is_pause", False))
                    event_duration_seconds = (
                        float(waveform_numpy.shape[0]) / float(sample_rate)
                        if sample_rate > 0 and waveform_numpy.ndim >= 1
                        else 0.0
                    )
                    if pcm_bytes and event_duration_seconds > 1e-9:
                        if rtf_first_gen_s is None:
                            rtf_first_gen_s = rtf_pending_lead_gen_s + gen_time_s
                            rtf_first_audio_s = event_duration_seconds
                            rtf_pending_lead_gen_s = 0.0
                        else:
                            rtf_steady_gen_s_sum += gen_time_s
                            rtf_steady_audio_s_sum += event_duration_seconds
                        rtf_audio_chunk_count += 1
                    elif rtf_first_gen_s is None:
                        rtf_pending_lead_gen_s += gen_time_s

                    if not pcm_bytes:
                        continue
                    with job.lock:
                        job.sample_rate = sample_rate
                        job.channels = channels
                        job.emitted_audio_seconds = float(event.get("emitted_audio_seconds", 0.0))
                        job.lead_seconds = float(event.get("lead_seconds", 0.0))
                        normalized_chunk_index, job.chunk_index_base = _normalize_stream_chunk_index(
                            event.get("chunk_index"),
                            chunk_count=len(job.text_chunks),
                            current_base=job.chunk_index_base,
                        )
                        if normalized_chunk_index is not None:
                            job.current_chunk_index = normalized_chunk_index
                            if not is_pause and event_duration_seconds > 0.0:
                                chunk_end_seconds = job.emitted_audio_seconds
                                chunk_start_seconds = max(0.0, chunk_end_seconds - event_duration_seconds)
                                job.audio_chunk_ranges.append(
                                    (chunk_start_seconds, chunk_end_seconds, normalized_chunk_index)
                                )
                        if job.first_audio_at is None and not is_pause:
                            job.first_audio_at = time.monotonic()
                        job.run_status = (
                            f"Streaming | emitted={job.emitted_audio_seconds:.2f}s | lead={job.lead_seconds:.2f}s"
                        )
                    _put_stream_audio(job, pcm_bytes)
                    continue

                if event_type == "result":
                    formatted_run_status = _format_run_status(event)
                    text_chunks = list(event.get("text_chunks") or [])
                    audio_chunk_ranges = list(event.get("audio_chunk_ranges") or [])
                    sample_rate = int(event.get("sample_rate") or 48000)
                    channels = int(event.get("channels") or 2)
                    emitted_audio_seconds = float(event.get("emitted_audio_seconds") or 0.0)
                    with job.lock:
                        job.text_chunks = text_chunks
                        job.final_result = {
                            "run_status": formatted_run_status,
                            "text_chunks": text_chunks,
                            "audio_chunk_ranges": audio_chunk_ranges,
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "emitted_audio_seconds": emitted_audio_seconds,
                        }
                        job.state = "done"
                        job.completed_at = time.monotonic()
                        job.run_status = formatted_run_status

                    total_audio_s = (rtf_first_audio_s or 0.0) + rtf_steady_audio_s_sum
                    rtf_first = (
                        rtf_first_gen_s / rtf_first_audio_s
                        if rtf_first_gen_s is not None
                        and rtf_first_audio_s is not None
                        and rtf_first_audio_s > 1e-9
                        else None
                    )
                    rtf_steady = (
                        rtf_steady_gen_s_sum / rtf_steady_audio_s_sum
                        if rtf_steady_audio_s_sum > 1e-9
                        else None
                    )
                    with job.lock:
                        first_audio_latency_s = (
                            None
                            if job.started_at is None or job.first_audio_at is None
                            else max(0.0, job.first_audio_at - job.started_at)
                        )
                    logging.info(
                        "Nano-TTS stream RTF | stream_id=%s | audio_chunks=%d | total_audio_s=%.3f | "
                        "first_audio_latency_s=%s | rtf_first=%s | rtf_steady=%s",
                        job.stream_id,
                        rtf_audio_chunk_count,
                        total_audio_s,
                        f"{first_audio_latency_s:.4f}" if first_audio_latency_s is not None else "n/a",
                        f"{rtf_first:.4f}" if rtf_first is not None else "n/a",
                        f"{rtf_steady:.4f}" if rtf_steady is not None else "n/a",
                    )
            finally:
                last_end = time.monotonic()

    except Exception as exc:
        logging.exception("Nano-TTS realtime streaming job failed")
        with job.lock:
            job.state = "failed"
            job.error = str(exc)
            job.completed_at = time.monotonic()
            job.run_status = f"Stream failed: {exc}"
    finally:
        try:
            job.audio_queue.put_nowait(None)
        except queue.Full:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# MossStreamFacade — 单 adapter + 单 execution_lock 串行推理
# ══════════════════════════════════════════════════════════════════════════════

class MossStreamFacade:
    """持有 OnnxNanoTTSServiceAdapter 和 execution_lock，保证同一时刻只有一个推理任务使用 ONNX runtime。

    第二个请求的后台线程会阻塞在 _execution_lock 上排队等待，与原 OnnxRequestRuntimeManager 行为一致。
    """

    def __init__(self, adapter: OnnxNanoTTSServiceAdapter) -> None:
        """初始化 adapter 引用和推理串行锁。

        调用方：本文件 MossStreamApiFacade.__init__（创建 facade 时自动调用）。
        """
        self.adapter = adapter
        self._execution_lock = threading.Lock()

    def stream_generate(self, **kwargs) -> Iterator[dict[str, object]]:
        """在 _execution_lock 保护下调用 adapter.synthesize_stream，串行化 ONNX 推理。
        持有锁期间 yield 音频事件；锁释放后下一个排队的推理任务才能开始。

        调用方：本文件 _run_streaming_job（在后台线程中迭代消费事件）。
        """
        with self._execution_lock:
            yield from self.adapter.synthesize_stream(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# MossStreamApiFacade — 本地五接口
# ══════════════════════════════════════════════════════════════════════════════

class MossStreamApiFacade:
    """将原 HTTP 五接口（start / audio / status / result / close）转为本地方法调用。
    这是 simple_moss 包对外暴露的主要入口类，供 test_local_api.py 等调用方使用。
    """

    def __init__(
        self,
        adapter: OnnxNanoTTSServiceAdapter,
        *,
        max_new_frames: int = 375,
        voice_clone_max_text_tokens: int = 75,
        attn_implementation: str = "model_default",
        do_sample: bool = True,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_p: float = 0.95,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.2,
        seed: int | None = None,
        chunk_pause_seconds: float = 2.0,
    ) -> None:
        """创建 MossStreamFacade 和 StreamingJobManager，保存推理参数默认值。

        调用方：test_local_api.py 的 run_test 函数（直接实例化）。
        """
        self._stream_facade = MossStreamFacade(adapter)
        self._job_manager = StreamingJobManager()
        self._defaults = {
            "max_new_frames": max_new_frames,
            "voice_clone_max_text_tokens": voice_clone_max_text_tokens,
            "attn_implementation": attn_implementation,
            "do_sample": do_sample,
            "text_temperature": text_temperature,
            "text_top_p": text_top_p,
            "text_top_k": text_top_k,
            "audio_temperature": audio_temperature,
            "audio_top_p": audio_top_p,
            "audio_top_k": audio_top_k,
            "audio_repetition_penalty": audio_repetition_penalty,
            "seed": seed,
            "chunk_pause_seconds": chunk_pause_seconds,
        }

    def start(
        self,
        text: str,
        *,
        max_new_frames: int | None = None,
        voice_clone_max_text_tokens: int | None = None,
        attn_implementation: str | None = None,
        do_sample: bool | None = None,
        text_temperature: float | None = None,
        text_top_p: float | None = None,
        text_top_k: int | None = None,
        audio_temperature: float | None = None,
        audio_top_p: float | None = None,
        audio_top_k: int | None = None,
        audio_repetition_penalty: float | None = None,
        seed: int | None = None,
        chunk_pause_seconds: float | None = None,
    ) -> dict[str, object]:
        """创建 StreamingJob，在后台 daemon 线程中启动 _run_streaming_job，
        立即返回包含 stream_id 和初始状态的字典（对应原 HTTP /start 接口）。

        调用方：test_local_api.py 的 run_test 函数（发起推理请求的入口）。
        """
        job = self._job_manager.create()
        
        normalized_text = self._stream_facade.adapter.normalize_text(text)
        logging.info("经过文本正则化后的文本: %s", normalized_text)

        params = {
            "text": normalized_text,
            "max_new_frames": max_new_frames if max_new_frames is not None else self._defaults["max_new_frames"],
            "voice_clone_max_text_tokens": voice_clone_max_text_tokens if voice_clone_max_text_tokens is not None else self._defaults["voice_clone_max_text_tokens"],
            "attn_implementation": attn_implementation if attn_implementation is not None else self._defaults["attn_implementation"],
            "do_sample": do_sample if do_sample is not None else self._defaults["do_sample"],
            "text_temperature": text_temperature if text_temperature is not None else self._defaults["text_temperature"],
            "text_top_p": text_top_p if text_top_p is not None else self._defaults["text_top_p"],
            "text_top_k": text_top_k if text_top_k is not None else self._defaults["text_top_k"],
            "audio_temperature": audio_temperature if audio_temperature is not None else self._defaults["audio_temperature"],
            "audio_top_p": audio_top_p if audio_top_p is not None else self._defaults["audio_top_p"],
            "audio_top_k": audio_top_k if audio_top_k is not None else self._defaults["audio_top_k"],
            "audio_repetition_penalty": audio_repetition_penalty if audio_repetition_penalty is not None else self._defaults["audio_repetition_penalty"],
            "seed": seed if seed is not None else self._defaults["seed"],
            "chunk_pause_seconds": chunk_pause_seconds if chunk_pause_seconds is not None else self._defaults["chunk_pause_seconds"],
        }

        thread = threading.Thread(
            target=_run_streaming_job,
            args=(job, self._stream_facade),
            kwargs=params,
            name=f"moss-stream-{job.stream_id}",
            daemon=True,
        )
        thread.start()

        return {
            "stream_id": job.stream_id,
            "state": job.state,
            "run_status": job.run_status,
        }

    def audio(self, stream_id: str) -> Iterator[bytes]:
        """从 job.audio_queue 中逐块消费 PCM s16le 字节，yield 给调用方，直到收到 sentinel（None）。
        对应原 HTTP /audio 流式接口，是消费音频数据的唯一入口。

        调用方：test_local_api.py 的 run_test 函数（for 循环迭代收集 PCM 字节）。
        """
        job = self._job_manager.get(stream_id)
        if job is None:
            raise KeyError(f"stream_id not found: {stream_id}")
        while True:
            item = job.audio_queue.get()
            if item is None:
                break
            yield item

    def status(self, stream_id: str) -> dict[str, object]:
        """返回 job.snapshot() 并附加 status_text 字段，供调用方轮询任务进度。
        对应原 HTTP /status 接口。

        调用方：test_local_api.py 的 run_test 函数（audio 消费完成后查询状态）。
        """
        job = self._job_manager.get(stream_id)
        if job is None:
            raise KeyError(f"stream_id not found: {stream_id}")
        snapshot = job.snapshot()
        snapshot["status_text"] = _format_stream_status(snapshot)
        return snapshot

    def result(self, stream_id: str) -> dict[str, object] | None:
        """返回任务的最终汇总结果（text_chunks、audio_chunk_ranges、emitted_audio_seconds 等）。
        任务尚未完成时返回 None（对应原 HTTP 202 语义）；失败时返回含 error 字段的字典。

        调用方：test_local_api.py 的 run_test 函数（任务完成后获取汇总统计）。
        """
        job = self._job_manager.get(stream_id)
        if job is None:
            raise KeyError(f"stream_id not found: {stream_id}")
        with job.lock:
            if job.state not in {"done", "failed"}:
                return None
            if job.state == "failed":
                return {
                    "run_status": job.run_status,
                    "error": job.error,
                    "text_chunks": list(job.text_chunks),
                    "audio_chunk_ranges": list(job.audio_chunk_ranges),
                    "sample_rate": job.sample_rate,
                    "channels": job.channels,
                    "emitted_audio_seconds": job.emitted_audio_seconds,
                }
            return dict(job.final_result) if job.final_result else {}

    def close(self, stream_id: str) -> None:
        """取消或清理任务：先调用 manager.close 设置 is_closed 并发送 sentinel，
        再调用 manager.delete 从注册表中移除 job 释放内存。
        对应原 HTTP /close 接口。

        调用方：test_local_api.py 的 run_test 函数（流程最后一步，清理资源）。
        """
        self._job_manager.close(stream_id)
        self._job_manager.delete(stream_id)
