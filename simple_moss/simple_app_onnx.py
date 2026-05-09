from __future__ import annotations

# ============================================================
# simple_app_onnx.py — ONNX Stream Generate 适配层
#
# 从 app_onnx.py 剥离而来：
#   - 去掉 HTTP/UI 层（uvicorn, legacy_app, _render_index_html_onnx, parse_args, main）
#   - 去掉 OnnxRequestRuntimeManager（用 api_facade.py 中的单实例锁替代）
#   - 去掉非流式接口 synthesize
#   - 去掉 _format_result_payload / _write_waveform_to_wav（result 不返回 WAV/base64）
#   - synthesize_stream 固定走 FIXED_BUILTIN_VOICE，去掉 voice/prompt_audio_path 动态分支
#   - 新增 create_default_adapter / warmup_runtime 本地初始化入口
# ============================================================

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from ._utils import _log_memory
from .simple_onnx_tts_runtime import (
    DEFAULT_BROWSER_ONNX_MODEL_DIR,
    OnnxTtsRuntime,
    _concat_waveforms,
    _merge_audio_channels,
)
from .simple_ort_gpu_runtime import _resolve_stream_decode_frame_budget
from text_normalization_pipeline import WeTextProcessingManager, prepare_tts_request_texts

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# 固定使用的内置音色名称。
FIXED_BUILTIN_VOICE: str | None = "Lingyu"


class _CpuDeviceInfo:
    """设备描述占位符，将 device 字段统一表示为 "cpu" 字符串，供日志和元数据使用。"""
    type = "cpu"

    def __str__(self) -> str:
        """返回 "cpu" 字符串，用于日志/元数据格式化。

        调用方：任何对 self.device 调用 str() 的场合（目前为属性存储，无直接调用）。
        """
        return "cpu"


class OnnxNanoTTSServiceAdapter:
    """ONNX TTS 适配层，持有 OnnxTtsRuntime 实例，提供：
      - warmup：触发推理预热；
      - synthesize_stream：流式生成音频事件迭代器；
      - split_voice_clone_text：文本分句包装。
    由 api_facade.py 中的 MossStreamFacade 持有并通过 execution_lock 串行调用。
    """

    def __init__(
        self,
        *,
        model_dir: str | Path | None,
        output_dir: str | Path | None = None,
        cpu_threads: int = 4,
        max_new_frames: int = 375,
        enable_wetext: bool = True,
    ) -> None:
        """创建 OnnxTtsRuntime 并初始化输出目录、设备元数据等属性。

        调用方：本文件 create_default_adapter（作为唯一的初始化入口）。
        """
        _log_memory("runtime_init: OnnxNanoTTSServiceAdapter __init__ entry")
        self.output_dir = Path(output_dir or (REPO_ROOT / "generated_audio")).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _log_memory("runtime_init: OnnxNanoTTSServiceAdapter output_dir ready, before OnnxTtsRuntime")
        self.runtime = OnnxTtsRuntime(
            model_dir=model_dir,
            thread_count=max(1, int(cpu_threads)),
            max_new_frames=int(max_new_frames),
            output_dir=self.output_dir,
        )
        _log_memory("runtime_init: OnnxNanoTTSServiceAdapter after OnnxTtsRuntime, wiring paths")
        self.model_dir = self.runtime.model_dir
        self.device = _CpuDeviceInfo()
        self.dtype = "float32"
        self.attn_implementation = "fixed"
        self._checkpoint_global_attn_implementation = "onnxruntime_cpu"
        self._checkpoint_local_attn_implementation = "onnxruntime_cpu"
        self._configured_global_attn_implementation = "onnxruntime_cpu"
        self._configured_local_attn_implementation = "onnxruntime_cpu"
        self.checkpoint_path = self.runtime.tts_meta_path.parent.resolve()
        self.audio_tokenizer_path = self.runtime.codec_meta_path.parent.resolve()
        self.thread_count = max(1, int(cpu_threads))
        self.enable_wetext = enable_wetext
        self.text_normalizer_manager = WeTextProcessingManager() if enable_wetext else None
        if self.text_normalizer_manager:
            self.text_normalizer_manager.start()
        _log_memory("runtime_init: OnnxNanoTTSServiceAdapter __init__ complete")

    def normalize_text(self, text: str, voice: str | None = None) -> str:
        """调用文本正则化管线。"""
        if not self.enable_wetext:
            return str(text or "")
        effective_voice = FIXED_BUILTIN_VOICE if FIXED_BUILTIN_VOICE is not None else (voice or "")
        prepared_texts = prepare_tts_request_texts(
            text=str(text or ""),
            voice=str(effective_voice or ""),
            enable_wetext=self.enable_wetext,
            enable_normalize_tts_text=True,
            text_normalizer_manager=self.text_normalizer_manager,
        )
        return str(prepared_texts["text"])

    def warmup(self) -> dict[str, object]:
        """对底层 OnnxTtsRuntime 执行一次预热推理，降低首次真实请求延迟。
        返回包含 elapsed_seconds 和 audio_path（始终为 None）的结果字典。

        调用方：本文件 warmup_runtime（对外暴露的预热入口）。
        """
        _log_memory("warmup: start")
        voice_name = (
            FIXED_BUILTIN_VOICE
            if FIXED_BUILTIN_VOICE is not None
            else str(self.runtime.list_builtin_voices()[0]["voice"])
        )
        t0 = time.perf_counter()
        self.runtime.warmup(voice_name=voice_name)
        t1 = time.perf_counter()
        _log_memory("warmup: complete (codec_decode_step session done)")
        return {
            "elapsed_seconds": t1 - t0,
            "audio_path": None,
        }

    def split_voice_clone_text(self, *, text: str, voice_clone_max_text_tokens: int) -> list[str]:
        """将文本按 token 预算切分为若干 chunk，包装 OnnxTtsRuntime.split_voice_clone_text。

        调用方：此方法在 simple_moss 内部当前未被直接调用；
                保留以便外部调用方（如 test_local_api.py）需要时直接使用。
        """
        return self.runtime.split_voice_clone_text(str(text or ""), max_tokens=int(voice_clone_max_text_tokens))

    def _apply_generation_options(
        self,
        *,
        sample_mode: str | None,
        max_new_frames: int,
        do_sample: bool,
        text_temperature: float,
        text_top_p: float,
        text_top_k: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
        seed: int | None,
    ) -> None:
        """将推理参数写入 runtime.manifest["generation_defaults"]，并按需重置随机数生成器。
        这些参数在下一次 generate_audio_frames 调用时生效。

        调用方：本文件 synthesize_stream 内的 _worker 闭包，在推理开始前调用。
        """
        resolved_sample_mode = self._resolve_sample_mode(sample_mode, do_sample=do_sample)
        generation_defaults = self.runtime.manifest["generation_defaults"]
        generation_defaults["max_new_frames"] = int(max_new_frames)
        generation_defaults["sample_mode"] = resolved_sample_mode
        generation_defaults["do_sample"] = resolved_sample_mode != "greedy"
        generation_defaults["text_temperature"] = float(text_temperature)
        generation_defaults["text_top_p"] = float(text_top_p)
        generation_defaults["text_top_k"] = int(text_top_k)
        generation_defaults["audio_temperature"] = float(audio_temperature)
        generation_defaults["audio_top_p"] = float(audio_top_p)
        generation_defaults["audio_top_k"] = int(audio_top_k)
        generation_defaults["audio_repetition_penalty"] = float(audio_repetition_penalty)
        if seed is not None:
            self.runtime.rng = np.random.default_rng(int(seed))

    @staticmethod
    def _resolve_sample_mode(raw_sample_mode: str | None, *, do_sample: bool) -> str:
        """将调用方传入的 attn_implementation / sample_mode 字符串规范化为 "fixed" / "full" / "greedy"。
        do_sample=False 时强制回退到 "greedy"。

        调用方：本文件 _apply_generation_options 和 synthesize_stream._worker（两处均调用）。
        """
        normalized = str(raw_sample_mode or "").strip().lower()
        if normalized in {"fixed", "full", "greedy"}:
            if normalized == "greedy":
                return "greedy"
            return normalized if bool(do_sample) else "greedy"
        return "fixed" if bool(do_sample) else "greedy"

    def synthesize_stream(
        self,
        *,
        text: str,
        mode: str = "voice_clone",
        voice: str | None = None,
        prompt_audio_path: str | None = None,
        max_new_frames: int = 375,
        voice_clone_max_text_tokens: int = 75,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
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
    ) -> Iterator[dict[str, object]]:
        """在后台线程中启动流式 TTS 推理，通过 Generator 逐步 yield 音频事件和结果事件。

        内部结构：
          - event_queue：推理线程与主线程之间的有界队列；
          - _stop_event：主线程发出停止信号的 threading.Event；
          - _safe_put：带超时重试的入队函数，防止推理线程在队列满时永久阻塞；
          - _worker：实际执行推理的后台 daemon 线程，负责文本分句、构造请求、调用推理、发送事件；
            - _emit_waveform：将解码后的波形封装为 audio 事件入队；
            - _decode_pending：将待解码的声学 token 帧批量送入 codec，控制批大小；
            - _on_frame：每生成一帧声学 token 后的回调，追加到 pending_decode_frames 并触发解码。
        主线程从 event_queue 消费事件并 yield，error 事件则抛出异常，None（sentinel）表示结束。

        调用方：api_facade.py 中 MossStreamFacade.stream_generate，在 execution_lock 保护下调用。
        """
        del mode, tts_max_batch_size, codec_max_batch_size
        event_queue: "queue.Queue[dict[str, object] | None]" = queue.Queue(maxsize=128)
        _stop_event = threading.Event()

        def _safe_put(item: "dict[str, object] | None") -> bool:
            """将事件安全入队，以 0.5s 超时轮询，stop_event 被设置时放弃入队并返回 False。

            调用方：本闭包内 _emit_waveform 和 _worker（result/error/sentinel 入队）。
            """
            while not _stop_event.is_set():
                try:
                    event_queue.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def _worker() -> None:
            """推理主循环：初始化参数 → 分句 → 逐 chunk 推理 → 发送 result 事件 → 放 sentinel。
            固定使用 FIXED_BUILTIN_VOICE，忽略请求中的 voice / prompt_audio_path 参数。

            调用方：本函数 synthesize_stream，通过 threading.Thread 在后台启动。
            """
            try:
                _log_memory("stream_worker: start")
                resolved_sample_mode = self._resolve_sample_mode(attn_implementation, do_sample=do_sample)
                self._apply_generation_options(
                    sample_mode=resolved_sample_mode,
                    max_new_frames=max_new_frames,
                    do_sample=do_sample,
                    text_temperature=text_temperature,
                    text_top_p=text_top_p,
                    text_top_k=text_top_k,
                    audio_temperature=audio_temperature,
                    audio_top_p=audio_top_p,
                    audio_top_k=audio_top_k,
                    audio_repetition_penalty=audio_repetition_penalty,
                    seed=seed,
                )
                start_time = time.perf_counter()
                # 固定使用 FIXED_BUILTIN_VOICE，忽略请求中的 voice / prompt_audio_path
                effective_voice = FIXED_BUILTIN_VOICE if FIXED_BUILTIN_VOICE is not None else voice
                prompt_audio_codes = self.runtime.resolve_builtin_voice_prompt_audio_codes(effective_voice)
                text_chunks = self.runtime.split_voice_clone_text(str(text or ""), max_tokens=int(voice_clone_max_text_tokens))
                sample_rate, channels = self.runtime.get_codec_audio_format()
                emitted_samples_total = 0
                first_audio_emitted_at_perf: float | None = None
                all_waveforms: list[np.ndarray] = []
                audio_chunk_ranges: list[dict[str, object]] = []

                for chunk_index, chunk_text in enumerate(text_chunks):
                    if _stop_event.is_set():
                        break
                    text_token_ids = self.runtime.encode_text(chunk_text)
                    request_rows = self.runtime.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
                    pending_decode_frames: list[list[int]] = []
                    emitted_chunks: list[np.ndarray] = []
                    chunk_audio_start_sample = emitted_samples_total
                    self.runtime.codec_streaming_session.reset()

                    def _emit_waveform(waveform: np.ndarray, *, is_pause: bool) -> None:
                        """将一段解码好的波形封装为 audio 事件，更新统计数据并入队。

                        调用方：本 chunk 上下文中的 _decode_pending（正常解码输出）和
                                _worker（静音 pause 片段）。
                        """
                        nonlocal emitted_samples_total, first_audio_emitted_at_perf
                        audio_length = int(waveform.shape[0])
                        if first_audio_emitted_at_perf is None and not is_pause:
                            first_audio_emitted_at_perf = time.perf_counter()
                        emitted_samples_total += audio_length
                        lead_seconds = 0.0
                        if first_audio_emitted_at_perf is not None:
                            elapsed_since_first_audio = max(0.0, time.perf_counter() - first_audio_emitted_at_perf)
                            lead_seconds = (emitted_samples_total / float(sample_rate)) - elapsed_since_first_audio
                        emitted_chunks.append(np.asarray(waveform, dtype=np.float32))
                        _safe_put(
                            {
                                "type": "audio",
                                "waveform_numpy": np.asarray(waveform, dtype=np.float32),
                                "sample_rate": sample_rate,
                                "channels": channels,
                                "chunk_index": chunk_index,
                                "emitted_audio_seconds": emitted_samples_total / float(sample_rate),
                                "lead_seconds": lead_seconds,
                                "is_pause": bool(is_pause),
                            }
                        )

                    def _decode_pending(force: bool) -> None:
                        """将 pending_decode_frames 中积攒的声学 token 批量送入 codec 解码。
                        force=True 时强制清空所有待解码帧；否则按 frame_budget 控制批大小。

                        调用方：本 chunk 上下文中的 _on_frame（每帧推理后触发）和
                                _worker（每个 chunk 推理完成后强制刷新）。
                        """
                        pending_count = len(pending_decode_frames)
                        if pending_count <= 0:
                            return
                        decode_budget = _resolve_stream_decode_frame_budget(
                            emitted_samples_total,
                            sample_rate,
                            first_audio_emitted_at_perf,
                        )
                        if not force and pending_count < max(1, decode_budget):
                            return
                        frame_budget = pending_count if force else min(pending_count, max(1, decode_budget))
                        frame_chunk = pending_decode_frames[:frame_budget]
                        del pending_decode_frames[:frame_budget]
                        decoded = self.runtime.codec_streaming_session.run_frames(frame_chunk)
                        if decoded is None:
                            return
                        audio, audio_length = decoded
                        if audio_length <= 0:
                            return
                        waveform = _merge_audio_channels(
                            [audio[0, channel_index, :audio_length] for channel_index in range(audio.shape[1])]
                        )
                        _emit_waveform(waveform, is_pause=False)

                    def _on_frame(_generated_frames: list[list[int]], _step_index: int, frame: list[int]) -> None:
                        """TTS decode 每生成一帧声学 token 后的回调，将帧追加到待解码队列并尝试解码。

                        调用方：simple_ort_gpu_runtime.py 中 generate_audio_frames 的 decode 循环，
                                通过 on_frame 参数传入并在每步末尾调用。
                        """
                        pending_decode_frames.append(list(frame))
                        _decode_pending(False)

                    try:
                        self.runtime.generate_audio_frames(request_rows, on_frame=_on_frame)
                        _decode_pending(True)
                    finally:
                        self.runtime.codec_streaming_session.reset()

                    chunk_waveform = _concat_waveforms(emitted_chunks)
                    all_waveforms.append(chunk_waveform)
                    chunk_audio_end_sample = emitted_samples_total
                    audio_chunk_ranges.append({
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "start_sample": chunk_audio_start_sample,
                        "end_sample": chunk_audio_end_sample,
                    })
                    _log_memory(f"stream_worker: chunk {chunk_index} done")

                    if _stop_event.is_set():
                        break

                    if chunk_index < len(text_chunks) - 1:
                        pause_seconds = self.runtime.estimate_voice_clone_inter_chunk_pause_seconds(
                            chunk_text,
                            chunk_pause_seconds=chunk_pause_seconds
                        )
                        pause_samples = max(0, int(round(sample_rate * pause_seconds)))
                        if pause_samples > 0:
                            pause_waveform = np.zeros((pause_samples, channels), dtype=np.float32)
                            _emit_waveform(pause_waveform, is_pause=True)
                            all_waveforms.append(pause_waveform)

                if not _stop_event.is_set():
                    elapsed_seconds = time.perf_counter() - start_time
                    _safe_put(
                        {
                            "type": "result",
                            "run_status": "done",
                            "text_chunks": text_chunks,
                            "audio_chunk_ranges": audio_chunk_ranges,
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "emitted_audio_seconds": emitted_samples_total / float(sample_rate),
                            "elapsed_seconds": elapsed_seconds,
                        }
                    )
            except Exception as exc:
                _safe_put({"type": "error", "error": str(exc)})
            finally:
                _log_memory("stream_worker: finally exit")
                _safe_put(None)

        worker = threading.Thread(target=_worker, name="onnx-synthesize-stream", daemon=True)
        worker.start()
        try:
            while True:
                item = event_queue.get()
                if item is None:
                    break
                if str(item.get("type")) == "error":
                    raise RuntimeError(str(item.get("error") or "Unknown ONNX streaming error"))
                yield item
        finally:
            _stop_event.set()
            while True:
                try:
                    event_queue.get_nowait()
                except queue.Empty:
                    break


# ============================================================
# 本地初始化入口（替代原 main()）
# ============================================================

def create_default_adapter(
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    cpu_threads: int = 1,
    max_new_frames: int = 375,
    enable_wetext: bool = True,
) -> OnnxNanoTTSServiceAdapter:
    """创建并返回一个 OnnxNanoTTSServiceAdapter 实例，不启动 HTTP 服务器。
    这是 simple_moss 包的唯一初始化入口，供外部调用方（如 test_local_api.py）使用。

    调用方：test_local_api.py 的 run_test 函数；api_facade.py 的 MossStreamApiFacade.__init__（间接）。
    """
    return OnnxNanoTTSServiceAdapter(
        model_dir=model_dir,
        output_dir=output_dir,
        cpu_threads=cpu_threads,
        max_new_frames=max_new_frames,
        enable_wetext=enable_wetext,
    )


def warmup_runtime(adapter: OnnxNanoTTSServiceAdapter) -> dict[str, object]:
    """对已创建的 adapter 执行 warmup，返回包含 elapsed_seconds 的耗时字典。
    调用方无需了解 adapter 内部结构，直接传入 adapter 即可完成预热。

    调用方：test_local_api.py 的 run_test 函数。
    """
    return adapter.warmup()
