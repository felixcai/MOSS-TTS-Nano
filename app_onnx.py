from __future__ import annotations

import argparse
import logging
import os
import queue
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import uvicorn

import app as legacy_app
from onnx_tts_runtime import (
    DEFAULT_BROWSER_ONNX_MODEL_DIR,
    OnnxTtsRuntime,
    _concat_waveforms,
    _merge_audio_channels,
    _write_waveform_to_wav,
)
from text_normalization_pipeline import WeTextProcessingManager

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR
from ort_cpu_runtime import _resolve_stream_decode_frame_budget

# 固定使用的内置音色名称。设为 None 时保持原始行为（使用请求传入的 voice / prompt_audio_path 参数）；
# 设为具体字符串（如 "zh_female_1"）时强制使用该内置音色，忽略请求中的 voice / prompt_audio_path 参数。
FIXED_BUILTIN_VOICE: str | None = "Lingyu"

_LEGACY_RENDER_INDEX_HTML = legacy_app._render_index_html


# [调用链内部] 在 synthesize_stream._worker 内被调用（start / chunk N done / finally exit 三处），
# 记录进程 RSS 和系统内存用量，用于追踪 stream generate 过程中的内存阶梯式变化。
def _log_memory(label: str) -> None:
    try:
        import psutil
        proc_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        sys_used_mb = psutil.virtual_memory().used / (1024 * 1024)
        logging.info(
            "[MEM] %s | proc_rss=%.1f MB | sys_used=%.1f MB",
            label, proc_rss_mb, sys_used_mb,
        )
    except Exception:
        pass


# [非调用链-初始化] 仅在 OnnxNanoTTSServiceAdapter.__init__ 中被实例化，
# 用于填充 self.device 属性以兼容 app.py 对 runtime.device.type 的访问。
class _CpuDeviceInfo:
    type = "cpu"

    def __str__(self) -> str:
        return "cpu"


# ============================================================
# OnnxNanoTTSServiceAdapter
# [调用链入口] __init__ 被 main() 调用，创建唯一的 ONNX 适配器实例；
# 该实例随后通过 OnnxRequestRuntimeManager 注入 app.py 的请求路由层。
# ============================================================
class OnnxNanoTTSServiceAdapter:
    def __init__(
        self,
        *,
        model_dir: str | Path | None,
        output_dir: str | Path | None = None,
        cpu_threads: int = 4,
        max_new_frames: int = 375,
        text_normalizer_manager: WeTextProcessingManager | None = None,
    ) -> None:
        _log_memory("runtime_init: OnnxNanoTTSServiceAdapter __init__ entry")
        self.output_dir = Path(output_dir or (APP_DIR / "generated_audio")).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _log_memory("runtime_init: OnnxNanoTTSServiceAdapter output_dir ready, before OnnxTtsRuntime")
        # 创建 OnnxTtsRuntime（内部创建全部 ONNX InferenceSession 和 CodecStreamingDecodeSession）
        self.runtime = OnnxTtsRuntime(
            model_dir=model_dir,
            thread_count=max(1, int(cpu_threads)),
            max_new_frames=int(max_new_frames),
            output_dir=self.output_dir,
        )
        _log_memory("runtime_init: OnnxNanoTTSServiceAdapter after OnnxTtsRuntime, wiring paths")
        self.model_dir = self.runtime.model_dir
        self.runtime._text_normalizer_manager = text_normalizer_manager
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
        _log_memory("runtime_init: OnnxNanoTTSServiceAdapter __init__ complete")

    # [非调用链] 兼容接口：app.py 在某些路径下访问 runtime.get_model()，
    # stream generate / 非流式推理 均不经过此方法
    def get_model(self) -> "OnnxNanoTTSServiceAdapter":
        return self

    # [非调用链-初始化] 服务启动后由 WarmupManager 触发一次完整的非流式推理，
    # 目的是让 ONNX Session 完成首次 JIT 编译 / Arena 预分配，降低首请求冷启动延迟。
    # self.synthesize() 覆盖 prefill / local_cached_step / local_decoder / codec_encode（自定义音频路径）
    # codec_decode_step（stream generate 流式解码专用）在 synthesize() 路径中不被触发，
    # 因此在 synthesize() 完成后额外预热一次 codec_streaming_session。
    def warmup(self) -> dict[str, object]:
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
        
        # result = self.synthesize(
        #     text="Warmup.",
        #     mode="voice_clone",
        #     voice=voice_name,
        #     prompt_audio_path=None,
        #     max_new_frames=min(16, int(self.runtime.manifest["generation_defaults"]["max_new_frames"])),
        #     voice_clone_max_text_tokens=75,
        #     do_sample=True,
        #     text_temperature=1.0,
        #     text_top_p=1.0,
        #     text_top_k=50,
        #     audio_temperature=0.8,
        #     audio_top_p=0.95,
        #     audio_top_k=25,
        #     audio_repetition_penalty=1.2,
        #     seed=1234,
        # )
        # _log_memory("warmup: synthesize done (prefill/local_decode/codec_decode sessions)")
        # # 补充预热 codec_decode_step（stream generate 流式解码 Session）：
        # # synthesize() 走全量 codec_decode 路径，不触发 codec_decode_step，
        # # 若不预热则首次 synthesize_stream 请求会产生额外的首帧延迟
        # n_vq = int(self.runtime.manifest["tts_config"]["n_vq"])
        # empty_frames = [([0] * n_vq)]
        # self.runtime.codec_streaming_session.reset()
        # self.runtime.codec_streaming_session.run_frames(empty_frames)
        # self.runtime.codec_streaming_session.reset()
        # _log_memory("warmup: complete (codec_decode_step session done)")
        # return result

    # [非调用链] 对 runtime.split_voice_clone_text 的公开包装方法，供 app.py 直接调用；
    # stream generate 路径中 _worker 直接调用 self.runtime.split_voice_clone_text，不经过此方法
    def split_voice_clone_text(self, *, text: str, voice_clone_max_text_tokens: int) -> list[str]:
        return self.runtime.split_voice_clone_text(str(text or ""), max_tokens=int(voice_clone_max_text_tokens))

    # [调用链内部] 被 synthesize_stream._worker（stream 路径）和 synthesize（非 stream 路径）调用；
    # 将本次请求的推理超参数（sample_mode、温度、top_p/k 等）写入
    # runtime.manifest["generation_defaults"]，ort_cpu_runtime 在推理时从此处读取配置
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

    # [调用链内部] 被 _apply_generation_options 和 _worker 调用；
    # 将外部传入的 sample_mode / do_sample 组合规范化为 "greedy" / "fixed" / "full" 三种枚举值
    @staticmethod
    def _resolve_sample_mode(raw_sample_mode: str | None, *, do_sample: bool) -> str:
        normalized = str(raw_sample_mode or "").strip().lower()
        if normalized in {"fixed", "full", "greedy"}:
            if normalized == "greedy":
                return "greedy"
            return normalized if bool(do_sample) else "greedy"
        return "fixed" if bool(do_sample) else "greedy"

    # [调用链内部] 被 synthesize_stream._worker 末尾调用，将完整生成结果打包为 result event dict，
    # 发送给 app.py 消费（包含最终 wav 路径、完整波形、耗时等元数据）
    def _format_result_payload(
        self,
        *,
        waveform: np.ndarray,
        sample_rate: int,
        elapsed_seconds: float,
        audio_path: str,
        voice: str | None,
        prompt_audio_path: str | None,
        text_chunks: list[str],
    ) -> dict[str, object]:
        return {
            "audio_path": audio_path,
            "waveform_numpy": np.asarray(waveform, dtype=np.float32),
            "sample_rate": int(sample_rate),
            "elapsed_seconds": float(elapsed_seconds),
            "mode": "voice_clone",
            "voice": str(voice or ""),
            "prompt_audio_path": str(prompt_audio_path or ""),
            "voice_clone_text_chunks": list(text_chunks),
            "effective_global_attn_implementation": "onnxruntime_cpu",
            "effective_local_attn_implementation": "onnxruntime_cpu",
            "voice_clone_chunk_batch_size": 1,
            "voice_clone_codec_batch_size": 1,
        }

    # [非调用链] 同步（非流式）合成路径，被 app.py 的非流式 /api/generate 接口调用；
    # stream generate 路径使用 synthesize_stream，不经过此方法
    def synthesize(
        self,
        *,
        text: str,
        mode: str,
        voice: str | None,
        prompt_audio_path: str | None,
        max_new_frames: int,
        voice_clone_max_text_tokens: int,
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
    ) -> dict[str, object]:
        del mode, tts_max_batch_size, codec_max_batch_size
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
        result = self.runtime.synthesize(
            text=str(text or ""),
            voice=voice,
            prompt_audio_path=prompt_audio_path,
            sample_mode=resolved_sample_mode,
            do_sample=resolved_sample_mode != "greedy",
            streaming=False,
            max_new_frames=int(max_new_frames),
            voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
            enable_wetext=False,
            enable_normalize_tts_text=False,
            seed=seed,
        )
        elapsed_seconds = time.perf_counter() - start_time
        waveform = np.asarray(result["waveform"], dtype=np.float32)
        return self._format_result_payload(
            waveform=waveform,
            sample_rate=int(result["sample_rate"]),
            elapsed_seconds=elapsed_seconds,
            audio_path=str(result["audio_path"]),
            voice=voice,
            prompt_audio_path=prompt_audio_path,
            text_chunks=[str(chunk).strip() for chunk in result.get("text_chunks", []) if str(chunk).strip()],
        )

    # ============================================================
    # [调用链入口] stream generate 核心入口
    # 被 OnnxRequestRuntimeManager.iter_with_runtime 调用：
    #   app.py 的 _run_streaming_job 将 factory 闭包传入 iter_with_runtime，
    #   iter_with_runtime 在内部执行 factory(runtime)，即调用到此方法。
    #
    # 架构：使用 threading.Thread + queue.Queue 将推理线程与上层 yield 生成器解耦：
    #   - _worker 线程执行推理，将 audio/result/error event dict 放入 event_queue
    #   - 外层 while/yield 循环消费队列，将 event dict 逐一 yield 给 app.py
    #   - event_queue maxsize=128，兼顾背压控制与内存占用
    # ============================================================
    def synthesize_stream(
        self,
        *,
        text: str,
        mode: str,
        voice: str | None,
        prompt_audio_path: str | None,
        max_new_frames: int,
        voice_clone_max_text_tokens: int,
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
    ) -> Iterator[dict[str, object]]:
        del mode, tts_max_batch_size, codec_max_batch_size
        # event_queue：连接 _worker 推理线程与外层 yield 生成器的缓冲通道
        # _stop_event：客户端断开（GeneratorExit）或正常结束时通知 _worker 提前停止
        event_queue: "queue.Queue[dict[str, object] | None]" = queue.Queue(maxsize=128)
        _stop_event = threading.Event()

        # [调用链内部] 线程安全地将 item 放入 event_queue；
        # 队列满时自旋等待（最多 0.5s/次），收到停止信号后静默丢弃 item 并返回 False
        def _safe_put(item: "dict[str, object] | None") -> bool:
            while not _stop_event.is_set():
                try:
                    event_queue.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        # [调用链内部] 在独立线程中执行全部推理工作：
        #   1. 解析推理超参数（sample_mode、温度等）
        #   2. 编码参考音频 Prompt → prompt_audio_codes
        #   3. 三层策略切分长文本 → text_chunk 列表
        #   4. 对每个 chunk：prefill + 自回归 decode（generate_audio_frames）
        #      每生成一帧声学 token 触发 _on_frame 回调 → 批量 codec 解码 → 音频入队
        #   5. 全部 chunk 完成后拼接波形、写入 WAV 文件，发送 result event
        #   6. 无论成功/异常，最终放入 sentinel None 通知外层生成器退出
        def _worker() -> None:
            try:
                _log_memory("stream_worker: start")
                resolved_sample_mode = self._resolve_sample_mode(attn_implementation, do_sample=do_sample)
                # 将本次请求的推理超参数写入 runtime.manifest["generation_defaults"]
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
                # 步骤1：自定义音频走 codec_encode 编码，内置音色从 manifest 读取预编码 codes
                # 若 FIXED_BUILTIN_VOICE 已配置，则强制使用该内置音色，忽略请求传入的 voice / prompt_audio_path
                if FIXED_BUILTIN_VOICE is not None:
                    prompt_audio_codes = self.runtime.resolve_prompt_audio_codes(voice=FIXED_BUILTIN_VOICE, prompt_audio_path=None)
                else:
                    prompt_audio_codes = self.runtime.resolve_prompt_audio_codes(voice=voice, prompt_audio_path=prompt_audio_path)
                # 步骤2：三层策略将长文本切分为不超过 token 预算的 chunk 列表
                text_chunks = self.runtime.split_voice_clone_text(str(text or ""), max_tokens=int(voice_clone_max_text_tokens))
                sample_rate = int(self.runtime.codec_meta["codec_config"]["sample_rate"])
                channels = int(self.runtime.codec_meta["codec_config"]["channels"])
                emitted_samples_total = 0
                first_audio_emitted_at_perf: float | None = None
                all_waveforms: list[np.ndarray] = []
                # 注意：all_generated_frames 是已知的死代码（内存积累点）；
                # _format_result_payload 不使用此变量，但整个请求期间持续占用内存
                all_generated_frames: list[list[int]] = []

                for chunk_index, chunk_text in enumerate(text_chunks):
                    # 每个 chunk 独立经历完整的 prefill → decode 自回归循环
                    if _stop_event.is_set():
                        break
                    text_token_ids = self.runtime.encode_text(chunk_text)
                    # 拼接参考音频 codes 与目标文本 token，构造 prefill 所需的 inputIds / attentionMask
                    request_rows = self.runtime.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
                    pending_decode_frames: list[list[int]] = []
                    emitted_chunks: list[np.ndarray] = []
                    # 每个 chunk 开始前重置流式解码器状态，防止跨 chunk 污染 KV Cache
                    self.runtime.codec_streaming_session.reset()

                    # [调用链内部] 将一段已解码的 PCM 波形发送给上层消费者；
                    # 同时追踪 emitted_samples_total，用于计算 lead_seconds（音频超前播放量）
                    def _emit_waveform(waveform: np.ndarray, *, is_pause: bool) -> None:
                        nonlocal emitted_samples_total, first_audio_emitted_at_perf
                        audio_length = int(waveform.shape[0])
                        if first_audio_emitted_at_perf is None and not is_pause:
                            # 记录第一帧真实音频的发出时刻，用于后续计算超前秒数
                            first_audio_emitted_at_perf = time.perf_counter()
                        emitted_samples_total += audio_length
                        lead_seconds = 0.0
                        if first_audio_emitted_at_perf is not None:
                            elapsed_since_first_audio = max(0.0, time.perf_counter() - first_audio_emitted_at_perf)
                            lead_seconds = (emitted_samples_total / float(sample_rate)) - elapsed_since_first_audio
                        # 注意：同一份 PCM 数据产生两个独立 numpy 对象（内存积累点2）：
                        #   emitted_chunks 存一份（用于最终 chunk 拼接），队列 dict 存一份（供消费方使用）
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

                    # [调用链内部] 将 pending_decode_frames 中积累的声学 token 批量送入
                    # codec_streaming_session.run_frames 解码为 PCM 波形，再调用 _emit_waveform 发送；
                    # force=False：按 _resolve_stream_decode_frame_budget 的动态预算批处理（平衡首包延迟与吞吐）
                    # force=True ：chunk 推理结束后强制处理所有剩余帧
                    def _decode_pending(force: bool) -> None:
                        pending_count = len(pending_decode_frames)
                        if pending_count <= 0:
                            return
                        # 根据当前已发送音频相对实时播放的超前量，动态决定本次批处理的帧数
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
                        # 将声学 token 帧批量送入 codec_decode_step ONNX 模型，输出 PCM + 更新流式 KV Cache
                        decoded = self.runtime.codec_streaming_session.run_frames(frame_chunk)
                        if decoded is None:
                            return
                        audio, audio_length = decoded
                        if audio_length <= 0:
                            return
                        # 将 codec 解码的多声道数组合并为 (samples, channels) 波形
                        waveform = _merge_audio_channels(
                            [audio[0, channel_index, :audio_length] for channel_index in range(audio.shape[1])]
                        )
                        _emit_waveform(waveform, is_pause=False)

                    # [调用链内部] ort_cpu_runtime.generate_audio_frames 每生成一帧声学 token 时触发此回调；
                    # 将新帧追加到 pending_decode_frames 缓冲，并以非强制模式尝试批量解码
                    def _on_frame(_generated_frames: list[list[int]], _step_index: int, frame: list[int]) -> None:
                        pending_decode_frames.append(list(frame))
                        _decode_pending(False)

                    try:
                        # 步骤3：启动 prefill + 自回归 decode 循环（ort_cpu_runtime.py 实现）；
                        # 每生成一帧 token 触发 _on_frame → _decode_pending → codec 解码 → _emit_waveform → 入队
                        generated_frames = self.runtime.generate_audio_frames(request_rows, on_frame=_on_frame)
                        # 推理循环结束后，强制处理所有仍在缓冲区中的剩余声学 token 帧
                        _decode_pending(True)
                    finally:
                        # 无论成功/异常，重置流式解码器状态，防止污染下一个 chunk 的 KV Cache
                        self.runtime.codec_streaming_session.reset()

                    chunk_waveform = _concat_waveforms(emitted_chunks)
                    all_waveforms.append(chunk_waveform)
                    # 追加到 all_generated_frames（已知死代码，_format_result_payload 不使用）
                    all_generated_frames.extend(generated_frames)
                    _log_memory(f"stream_worker: chunk {chunk_index} done")

                    if _stop_event.is_set():
                        break

                    if chunk_index < len(text_chunks) - 1:
                        # 根据词数在相邻 chunk 之间插入静音停顿（0.24s 或 0.40s），模拟自然句间间隔
                        pause_seconds = self.runtime.estimate_voice_clone_inter_chunk_pause_seconds(chunk_text)
                        pause_samples = max(0, int(round(sample_rate * pause_seconds)))
                        if pause_samples > 0:
                            pause_waveform = np.zeros((pause_samples, channels), dtype=np.float32)
                            _emit_waveform(pause_waveform, is_pause=True)
                            all_waveforms.append(pause_waveform)

                if not _stop_event.is_set():
                    # 所有 chunk 完成：拼接全部波形（此时 all_waveforms 与 waveform 同时存活，内存峰值约 2×）
                    waveform = _concat_waveforms(all_waveforms)
                    output_path = _write_waveform_to_wav(
                        self.output_dir / "app_onnx_stream_output.wav",
                        waveform,
                        sample_rate,
                    )
                    _safe_put(
                        {
                            "type": "result",
                            **self._format_result_payload(
                                waveform=waveform,
                                sample_rate=sample_rate,
                                elapsed_seconds=time.perf_counter() - start_time,
                                audio_path=str(output_path),
                                voice=voice,
                                prompt_audio_path=prompt_audio_path,
                                text_chunks=text_chunks,
                            ),
                        }
                    )
            except Exception as exc:
                _safe_put({"type": "error", "error": str(exc)})
            finally:
                _log_memory("stream_worker: finally exit")
                # 无论成功/异常/取消，必须放入 sentinel None，
                # 确保外层 while/yield 生成器能正常退出（不永久阻塞在 event_queue.get()）
                _safe_put(None)

        # 启动 _worker 线程，外层循环消费 event_queue，将 audio/result event yield 给 app.py
        worker = threading.Thread(target=_worker, name="onnx-synthesize-stream", daemon=True)
        worker.start()
        try:
            while True:
                item = event_queue.get()
                if item is None:
                    # 收到 sentinel，_worker 已完成或异常退出，生成器正常结束
                    break
                if str(item.get("type")) == "error":
                    raise RuntimeError(str(item.get("error") or "Unknown ONNX streaming error"))
                yield item
        finally:
            # 客户端断开（GeneratorExit）或正常结束时：设置停止信号，
            # 并清空队列使 _worker 能在 ~0.5s 内从满队列的 _safe_put() 中解除阻塞
            _stop_event.set()
            while True:
                try:
                    event_queue.get_nowait()
                except queue.Empty:
                    break


# ============================================================
# OnnxRequestRuntimeManager
# [调用链入口] 被 app.py 实例化（main() 中通过 legacy_app.RequestRuntimeManager = OnnxRequestRuntimeManager 注入）；
# 管理 ONNX 运行时实例的生命周期与请求级并发控制。
#   - _lock：保护运行时字典的多线程构建操作
#   - _execution_lock：保证同一时刻只有一个推理任务在运行（ONNX Session 非线程安全）
# ============================================================
class OnnxRequestRuntimeManager:
    _factory_model_dir: Path | None = None
    _factory_output_dir: Path | None = None
    _factory_max_new_frames: int = 375
    _factory_text_normalizer_manager: WeTextProcessingManager | None = None

    def __init__(self, default_runtime: OnnxNanoTTSServiceAdapter) -> None:
        self.default_runtime = default_runtime
        self.default_cpu_threads = max(1, int(os.cpu_count() or 1))
        self._lock = threading.Lock()
        self._execution_lock = threading.Lock()
        self._cpu_runtimes: dict[int, OnnxNanoTTSServiceAdapter] = {default_runtime.thread_count: default_runtime}

    # [非调用链] 兼容接口：当前 ONNX 版本始终只支持 CPU，直接返回 "cpu"；
    # app.py 在设备类型解析时可能调用，不在 stream generate 请求路径上
    @staticmethod
    def normalize_requested_execution_device(requested: str | None) -> str:
        del requested
        return "cpu"

    # [非调用链] 兼容接口：ONNX 版本无专用 CPU 设备概念，始终返回 False；
    # app.py 在设备路由判断时可能调用，不在 stream generate 请求路径上
    def is_dedicated_cpu_request(self, requested: str | None) -> bool:
        del requested
        return False

    # [非调用链] 状态查询接口：供 app.py 在初始化或健康检查时判断 CPU 运行时是否已加载
    def is_cpu_runtime_loaded(self) -> bool:
        with self._lock:
            return bool(self._cpu_runtimes)

    # [调用链内部] 被 _locked_runtime 调用；将外部传入的 cpu_threads 参数规范化为有效正整数
    def _resolve_cpu_threads(self, cpu_threads: int | None) -> int:
        if cpu_threads is None:
            return self.default_cpu_threads
        try:
            normalized_threads = int(cpu_threads)
        except Exception:
            return self.default_cpu_threads
        if normalized_threads <= 0:
            return self.default_cpu_threads
        return max(1, normalized_threads)

    # [调用链内部] 被 _locked_runtime 调用，在已持有 _lock 的情况下返回目标运行时实例；
    # 当前实现始终复用 default_runtime（忽略 cpu_threads 差异，避免加载第二个 ONNX Session）
    def _build_runtime_locked(self, cpu_threads: int) -> OnnxNanoTTSServiceAdapter:
        _log_memory(f"build_runtime: cpu_threads={cpu_threads}")
        if cpu_threads != self.default_runtime.thread_count:
            logging.warning(
                "OnnxRequestRuntimeManager: ignoring cpu_threads=%d (default=%d) "
                "to avoid loading a second ONNX session; reusing default runtime.",
                cpu_threads,
                self.default_runtime.thread_count,
            )
        return self.default_runtime

    # [非调用链] 兼容接口：始终返回 (default_runtime, "cpu")；
    # stream generate 路径通过 iter_with_runtime 获取运行时，不经过此方法
    def resolve_runtime(self, requested: str | None) -> tuple[OnnxNanoTTSServiceAdapter, str]:
        del requested
        return self.default_runtime, "cpu"

    # [调用链内部] 被 iter_with_runtime 和 call_with_runtime 调用；
    # 解析 cpu_threads → 构建/获取运行时 → 持有 _execution_lock（保证同一时刻只有一个推理任务在运行）
    @contextmanager
    def _locked_runtime(self, cpu_threads: int | None) -> Iterator[tuple[OnnxNanoTTSServiceAdapter, str, int]]:
        resolved_cpu_threads = self._resolve_cpu_threads(cpu_threads)
        with self._lock:
            runtime = self._build_runtime_locked(resolved_cpu_threads)
        with self._execution_lock:
            yield runtime, "cpu", resolved_cpu_threads

    # [非调用链] 同步（非流式）请求路径使用此方法（一次性 callback）；
    # stream generate 路径使用 iter_with_runtime（迭代器），不经过此方法
    def call_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        callback,
    ) -> tuple[object, str, int]:
        del requested_execution_device
        with self._locked_runtime(cpu_threads) as (runtime, execution_device, resolved_cpu_threads):
            return callback(runtime), execution_device, resolved_cpu_threads

    # [调用链入口] stream generate 请求路径上被 app.py 调用；
    # 通过 _locked_runtime 获取运行时实例和排他执行锁，
    # 对 factory(runtime)（即 synthesize_stream 生成器）的每个 yield item 逐一转发给调用方
    def iter_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        factory,
    ) -> Iterator[tuple[object, str, int]]:
        del requested_execution_device
        with self._locked_runtime(cpu_threads) as (runtime, execution_device, resolved_cpu_threads):
            for item in factory(runtime):
                yield item, execution_device, resolved_cpu_threads


# [非调用链-初始化] 替换 app.py 默认的 HTML 渲染函数，将 ONNX 版本的 UI 差异注入前端页面
# （标题、采样模式下拉框、disabled 样式等）；
# 函数本身在用户访问首页时（GET /）被 app.py 调用，属于页面渲染路径，与音频生成无关
def _render_index_html_onnx(
    *,
    request,
    runtime,
    demo_entries,
    warmup_status: str,
    text_normalization_status: str,
) -> str:
    html = _LEGACY_RENDER_INDEX_HTML(
        request=request,
        runtime=runtime,
        demo_entries=demo_entries,
        warmup_status=warmup_status,
        text_normalization_status=text_normalization_status,
    )
    html = html.replace("MOSS-TTS-Nano Demo", "MOSS-TTS-Nano ONNX Demo")
    html = html.replace(
        '<label for="attn-implementation">Attention Backend</label>\n'
        '              <select id="attn-implementation">\n'
        '                <option value="model_default">model_default</option>\n'
        '                <option value="sdpa">sdpa</option>\n'
        '                <option value="eager">eager</option>\n'
        '              </select>',
        '<label for="attn-implementation">Sampling Mode</label>\n'
        '              <select id="attn-implementation">\n'
        '                <option value="fixed">fixed</option>\n'
        '                <option value="full">full</option>\n'
        '                <option value="greedy">greedy</option>\n'
        '              </select>\n'
        '              <div id="onnx-sampling-mode-note" class="meta">fixed uses the baked ONNX sampling constants.</div>',
    )
    html = html.replace(
        '<label><input id="do-sample" type="checkbox" checked> Do Sample</label>',
        '<label><input id="do-sample" type="checkbox" checked disabled> Do Sample (derived from Sampling Mode)</label>',
    )
    html = html.replace(
        'This app is CPU-only. CPU Threads maps to torch.set_num_threads for that request.',
        'This app is CPU-only. CPU Threads selects the cached ONNX runtime instance for that request.',
    )
    html = html.replace(
        '</style>',
        '    .field.disabled-field {\n'
        '      opacity: 0.5;\n'
        '    }\n'
        '    .field.disabled-field input {\n'
        '      cursor: not-allowed;\n'
        '      background: #f4f6fb;\n'
        '    }\n'
        '</style>',
        1,
    )
    html = html.replace(
        '    document.getElementById("attn-implementation").value = DEFAULT_ATTN_IMPLEMENTATION;\n',
        '    document.getElementById("attn-implementation").value = DEFAULT_ATTN_IMPLEMENTATION;\n'
        '    const onnxSamplingModeSelect = document.getElementById("attn-implementation");\n'
        '    const onnxDoSampleToggle = document.getElementById("do-sample");\n'
        '    const onnxSamplingModeNote = document.getElementById("onnx-sampling-mode-note");\n'
        '    const onnxSamplingParamIds = [\n'
        '      "text-temperature",\n'
        '      "text-top-p",\n'
        '      "text-top-k",\n'
        '      "audio-temperature",\n'
        '      "audio-top-p",\n'
        '      "audio-top-k",\n'
        '      "audio-repetition-penalty"\n'
        '    ];\n'
        '    function syncOnnxSamplingUi() {\n'
        '      const mode = (onnxSamplingModeSelect && onnxSamplingModeSelect.value) || "fixed";\n'
        '      const samplingParamsEnabled = mode === "full";\n'
        '      if (onnxDoSampleToggle) {\n'
        '        onnxDoSampleToggle.checked = mode !== "greedy";\n'
        '      }\n'
        '      for (const id of onnxSamplingParamIds) {\n'
        '        const input = document.getElementById(id);\n'
        '        if (!input) continue;\n'
        '        input.disabled = !samplingParamsEnabled;\n'
        '        const field = input.closest(".field");\n'
        '        if (field) field.classList.toggle("disabled-field", !samplingParamsEnabled);\n'
        '      }\n'
        '      if (onnxSamplingModeNote) {\n'
        '        if (mode === "full") {\n'
        '          onnxSamplingModeNote.textContent = "full uses the current page sampling hyperparameters.";\n'
        '        } else if (mode === "fixed") {\n'
        '          onnxSamplingModeNote.textContent = "fixed uses the baked ONNX sampling constants and ignores the hyperparameter inputs below.";\n'
        '        } else {\n'
        '          onnxSamplingModeNote.textContent = "greedy disables sampling and ignores the hyperparameter inputs below.";\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '    if (onnxSamplingModeSelect) {\n'
        '      onnxSamplingModeSelect.addEventListener("change", syncOnnxSamplingUi);\n'
        '      syncOnnxSamplingUi();\n'
        '    }\n',
        1,
    )
    return html


# [非调用链-初始化] 解析 CLI 启动参数，仅在 main() 中被调用一次
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-TTS-Nano ONNX web demo")
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "browser_onnx model directory. If omitted, the app uses "
            f"{DEFAULT_BROWSER_ONNX_MODEL_DIR} and auto-downloads the ONNX assets on first run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(APP_DIR / "generated_audio"),
        help="Directory for generated wav files.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-new-frames", type=int, default=375)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


# [非调用链-初始化] 程序入口：解析参数、创建 OnnxNanoTTSServiceAdapter 运行时、启动预热、
# 将 OnnxRequestRuntimeManager 和 _render_index_html_onnx 注入 app.py 全局引用，最后启动 uvicorn 服务
def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    _log_memory("main: startup baseline")

    text_normalizer_manager = WeTextProcessingManager()
    text_normalizer_manager.start()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _log_memory("main: before runtime init")
    runtime = OnnxNanoTTSServiceAdapter(
        model_dir=args.model_dir,
        output_dir=output_dir,
        cpu_threads=args.cpu_threads,
        max_new_frames=args.max_new_frames,
        text_normalizer_manager=text_normalizer_manager,
    )
    _log_memory("main: runtime created")
    warmup_manager = legacy_app.WarmupManager(runtime, text_normalizer_manager=text_normalizer_manager)
    warmup_manager.start()
    _log_memory("main: warmup started (port not open yet)")

    OnnxRequestRuntimeManager._factory_model_dir = runtime.model_dir
    OnnxRequestRuntimeManager._factory_output_dir = output_dir
    OnnxRequestRuntimeManager._factory_max_new_frames = int(args.max_new_frames)
    OnnxRequestRuntimeManager._factory_text_normalizer_manager = text_normalizer_manager
    legacy_app.RequestRuntimeManager = OnnxRequestRuntimeManager
    # 注入 ONNX 版本的 HTML 渲染函数，替换 app.py 中的默认实现
    legacy_app._render_index_html = _render_index_html_onnx

    vscode_proxy_uri = os.getenv("VSCODE_PROXY_URI", "")
    root_path = legacy_app._resolve_vscode_root_path(vscode_proxy_uri, args.port)
    logging.info("root_path=%s", root_path)
    if args.share:
        logging.warning("--share is ignored by the FastAPI-based ONNX app.")

    # Stream 和非流式路由均由 legacy_app._build_app（app.py）处理，
    # 它会使用注入后的 OnnxRequestRuntimeManager 路由到 OnnxNanoTTSServiceAdapter
    app = legacy_app._build_app(runtime, warmup_manager, text_normalizer_manager, root_path)
    app.title = "MOSS-TTS-Nano ONNX Demo"
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        root_path=root_path or "",
    )


if __name__ == "__main__":
    main()
