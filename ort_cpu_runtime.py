from __future__ import annotations

# =============================================================================
# ort_cpu_runtime.py — ONNX 核心推理层（调用链最底层）
#
# 在 Stream Generate 调用链中的角色：
#   app.py → app_onnx.py → onnx_tts_runtime.py → [本文件 OrtCpuRuntime]
#                                               ↑               ↓
#                              app_onnx.py._on_frame ← on_frame 回调
#                                    ↓
#                          CodecStreamingDecodeSession.run_frames（本文件）
#
# 对外暴露的核心接口（被其他文件调用）：
#   - OrtCpuRuntime.__init__ / generate_audio_frames / build_voice_clone_request_rows
#   - CodecStreamingDecodeSession.run_frames / reset
#   - _normalize_sample_mode / _resolve_stream_decode_frame_budget
# =============================================================================

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort

SAMPLE_MODE_GREEDY = "greedy"
SAMPLE_MODE_FIXED = "fixed"
SAMPLE_MODE_FULL = "full"

MANIFEST_CANDIDATE_RELATIVE_PATHS = (
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
)
MODEL_DIR_ALIAS_MAP = {
    "MOSS-TTS-Nano-ONNX-CPU": "MOSS-TTS-Nano-100M-ONNX",
    "MOSS-Audio-Tokenizer-Nano-ONNX-CPU": "MOSS-Audio-Tokenizer-Nano-ONNX",
}


# -----------------------------------------------------------------------------
# 模块级辅助函数 —— 均为 [调用链内部] 纯计算工具函数，由本文件内部调用
# -----------------------------------------------------------------------------

# [调用链内部] 被 _sample_from_scores 调用（do_sample=False 时取 argmax 作为 greedy 解码）
def _argmax(values: np.ndarray) -> int:
    return int(np.argmax(values))


# [调用链内部] 被 generate_audio_frames / decode_full_audio / warmup 调用
# 将 3D 嵌套 list 展平为 1D int32 数组，同时返回原始维度，供 ONNX session.run 构造输入张量
def _flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    dim2 = len(nested[0][0])
    data = np.zeros((dim0 * dim1 * dim2,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            for k in range(dim2):
                data[offset] = int(nested[i][j][k])
                offset += 1
    return data, [dim0, dim1, dim2]


# [调用链内部] 被 generate_audio_frames / warmup 调用
# 将 2D 嵌套 list 展平为 1D int32 数组，供构造 attention_mask 张量
def _flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    data = np.zeros((dim0 * dim1,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            data[offset] = int(nested[i][j])
            offset += 1
    return data, [dim0, dim1]


# [调用链内部] 被 decode_full_audio 调用（非流式路径）
# codec decode 输出张量形状为 (1, channels, samples)，按声道切片并截断到有效长度
def _slice_channel_major_audio(audio: np.ndarray, start_sample: int = 0, end_sample: int | None = None) -> list[np.ndarray]:
    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")
    channels = int(audio.shape[1])
    total_samples = int(audio.shape[2])
    start = max(0, int(start_sample))
    end = total_samples if end_sample is None else max(start, min(int(end_sample), total_samples))
    return [audio[0, channel_index, start:end].astype(np.float32, copy=False) for channel_index in range(channels)]


# [调用链内部] 被 generate_audio_frames / warmup 调用
# prefill/decode 模型输出 global_hidden 可能是 (seq_len, hidden) 或 (1, seq_len, hidden)
# 统一取最后一个时间步作为当前帧的全局隐状态
def _extract_last_hidden(hidden_states: np.ndarray) -> np.ndarray:
    if hidden_states.ndim == 2:
        return hidden_states.astype(np.float32, copy=False)
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"Unexpected global_hidden shape: {hidden_states.shape}")
    return hidden_states[:, -1, :].astype(np.float32, copy=False)


# [调用链内部] 被 _sample_audio_token 调用
# 对已生成 token 施加重复惩罚：score < 0 时乘以 penalty（使更负），score > 0 时除以 penalty（使更小）
def _apply_repetition_penalty(values: np.ndarray, previous_token_ids: list[int], repetition_penalty: float) -> np.ndarray:
    if not previous_token_ids or repetition_penalty == 1.0:
        return values
    result = values.copy()
    for token_id in set(int(item) for item in previous_token_ids):
        if token_id < 0 or token_id >= result.shape[0]:
            continue
        result[token_id] = result[token_id] * repetition_penalty if result[token_id] < 0 else result[token_id] / repetition_penalty
    return result


# [调用链内部] 被 _sample_audio_token 调用（do_sample=False 时的带重复惩罚 greedy 解码）
def _argmax_with_repetition_penalty(values: np.ndarray, previous_token_set: set[int], repetition_penalty: float) -> int:
    best_index = 0
    best_value = float("-inf")
    apply_penalty = bool(previous_token_set) and repetition_penalty != 1.0
    for index, value in enumerate(values):
        score = float(value)
        if apply_penalty and index in previous_token_set:
            score = score * repetition_penalty if score < 0 else score / repetition_penalty
        if score > best_value:
            best_value = score
            best_index = index
    return int(best_index)


# [调用链内部] 被 _sample_from_scores 调用，数值稳定的 softmax（先减最大值再 exp）
def _softmax(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values))
    shifted = np.asarray(values - max_value, dtype=np.float64)
    exps = np.exp(shifted)
    return exps / np.sum(exps, dtype=np.float64)


# [调用链内部] 被 _sample_assistant_text_token / _sample_audio_token 调用
# 实现 top-k + top-p 采样：先 top-k 截断，再 top-p 核采样，最后按概率分布随机采样
def _sample_from_scores(
    values: np.ndarray,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    if not do_sample:
        return _argmax(values)
    if not (temperature > 0):
        raise ValueError("temperature must be positive when do_sample=True")
    scores = np.asarray(values, dtype=np.float32).copy() / float(temperature)
    if top_k > 0 and top_k < scores.shape[0]:
        sorted_desc = np.sort(scores)[::-1]
        threshold = float(sorted_desc[top_k - 1])
        scores[scores < threshold] = float("-inf")
    if top_p > 0 and top_p < 1:
        indexed = list(enumerate(scores.tolist()))
        indexed.sort(key=lambda item: item[1], reverse=True)
        sorted_scores = np.asarray([item[1] for item in indexed], dtype=np.float32)
        sorted_probs = _softmax(sorted_scores)
        remove_mask = [False] * len(indexed)
        cumulative = 0.0
        for index, probability in enumerate(sorted_probs):
            cumulative += float(probability)
            if cumulative > float(top_p):
                remove_mask[index] = True
        for index in range(len(remove_mask) - 1, 0, -1):
            remove_mask[index] = remove_mask[index - 1]
        if remove_mask:
            remove_mask[0] = False
        for index, should_remove in enumerate(remove_mask):
            if should_remove:
                scores[indexed[index][0]] = float("-inf")
    probabilities = _softmax(scores)
    random_value = float(rng.random())
    for index, probability in enumerate(probabilities):
        random_value -= float(probability)
        if random_value <= 0:
            return int(index)
    return _argmax(scores)


# [调用链内部] 被 generate_audio_frames（local_cached_step 分支）调用
# 采样 assistant 文本 token：只在 audio_assistant_slot_token_id 和 audio_end_token_id 两个候选中选择
# 若采样结果为 audio_end_token_id，则 generate_audio_frames 会终止生成循环
def _sample_assistant_text_token(
    text_logits: np.ndarray,
    manifest: dict[str, Any],
    generation_defaults: dict[str, Any],
    rng: np.random.Generator,
) -> int:
    candidate_ids = np.asarray(
        [
            int(manifest["tts_config"]["audio_assistant_slot_token_id"]),
            int(manifest["tts_config"]["audio_end_token_id"]),
        ],
        dtype=np.int32,
    )
    candidate_scores = text_logits[candidate_ids]
    sampled_index = _sample_from_scores(
        candidate_scores,
        do_sample=bool(generation_defaults["do_sample"]),
        temperature=float(generation_defaults["text_temperature"]),
        top_k=min(int(generation_defaults["text_top_k"]), int(candidate_scores.shape[0])),
        top_p=float(generation_defaults["text_top_p"]),
        rng=rng,
    )
    return int(candidate_ids[sampled_index])


# [调用链内部] 被 generate_audio_frames 调用，对单个 VQ 声道进行采样
# 先施加重复惩罚（抑制已出现的 token），再通过 top-k/top-p 采样
def _sample_audio_token(
    audio_logits: np.ndarray,
    previous_token_ids: list[int],
    previous_token_set: set[int],
    generation_defaults: dict[str, Any],
    rng: np.random.Generator,
) -> int:
    repetition_penalty = float(generation_defaults["audio_repetition_penalty"])
    if not bool(generation_defaults["do_sample"]):
        return _argmax_with_repetition_penalty(audio_logits, previous_token_set, repetition_penalty)
    penalized_scores = _apply_repetition_penalty(audio_logits, previous_token_ids, repetition_penalty)
    return _sample_from_scores(
        penalized_scores,
        do_sample=True,
        temperature=float(generation_defaults["audio_temperature"]),
        top_k=int(generation_defaults["audio_top_k"]),
        top_p=float(generation_defaults["audio_top_p"]),
        rng=rng,
    )


# [调用链入口] 被 OrtCpuRuntime.__init__ 调用，也被 onnx_tts_runtime.py 等上层按需调用
# 将外部传入的 sample_mode 字符串规范化为三种模式之一（greedy / fixed / full）
# "mixed3" 是旧别名：do_sample=True 时等价于 fixed，否则等价于 greedy
def _normalize_sample_mode(raw_sample_mode: str | None, raw_do_sample: bool = True) -> str:
    normalized = str(raw_sample_mode or "").strip()
    if normalized in {SAMPLE_MODE_GREEDY, SAMPLE_MODE_FIXED, SAMPLE_MODE_FULL}:
        return normalized
    if normalized == "mixed3":
        return SAMPLE_MODE_FIXED if raw_do_sample else SAMPLE_MODE_GREEDY
    return SAMPLE_MODE_GREEDY if not raw_do_sample else SAMPLE_MODE_FIXED


# [调用链内部] 被 _resolve_stream_decode_frame_budget 调用
# 计算已生成音频比实时播放"超前"的秒数（lead_seconds > 0 表示缓冲充足）
def _compute_stream_lead_seconds(emitted_samples_total: int, sample_rate: int, first_audio_emitted_at_seconds: float | None) -> float:
    if not first_audio_emitted_at_seconds or sample_rate <= 0:
        return 0.0
    elapsed_seconds = max(0.0, time.perf_counter() - first_audio_emitted_at_seconds)
    emitted_seconds = emitted_samples_total / float(sample_rate)
    return emitted_seconds - elapsed_seconds


# [调用链入口] 被 app_onnx.py 的 _on_frame 回调调用
# 根据当前流式超前量动态决定每次 codec 解码的帧批大小（frame_budget）：
#   超前越多 → 可以一次解码更多帧 → 减少 ONNX session.run 调用次数，提升吞吐
#   首帧到来前或超前不足 0.20s → 每次只解码 1 帧，优先降低首帧延迟
def _resolve_stream_decode_frame_budget(
    emitted_samples_total: int,
    sample_rate: int,
    first_audio_emitted_at_seconds: float | None,
) -> int:
    lead_seconds = _compute_stream_lead_seconds(emitted_samples_total, sample_rate, first_audio_emitted_at_seconds)
    if not first_audio_emitted_at_seconds or lead_seconds < 0.20:
        return 1
    if lead_seconds < 0.55:
        return 2
    if lead_seconds < 1.10:
        return 4
    return 8


# =============================================================================
# CodecStreamingDecodeSession —— 流式 Codec 解码会话
# 职责：维护 codec_decode_step ONNX 模型的 KV Cache 状态，逐帧将声学 token 解码为 PCM
# =============================================================================

@dataclass
class CodecStreamingDecodeSession:
    codec_meta: dict[str, Any]
    session: ort.InferenceSession

    # [调用链内部] dataclass 自动调用，读取 codec_meta 中的 transformer / attention 规格并初始化状态
    def __post_init__(self) -> None:
        self.transformer_specs = list(self.codec_meta.get("streaming_decode", {}).get("transformer_offsets", []))
        self.attention_specs = list(self.codec_meta.get("streaming_decode", {}).get("attention_caches", []))
        self.state_feeds: dict[str, np.ndarray] = {}
        self.reset()

    # [调用链入口] 被 OrtCpuRuntime.warmup 和 app_onnx.py 调用（每次新请求前重置 KV Cache）
    # 将所有 transformer offset 和 attention cache（K/V/positions）清零，开启新一轮流式解码
    def reset(self) -> None:
        self.state_feeds = {}
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = np.zeros(tuple(spec["shape"]), dtype=np.int32)
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = np.zeros(tuple(spec["offset_shape"]), dtype=np.int32)
            self.state_feeds[str(spec["cached_keys_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            self.state_feeds[str(spec["cached_values_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            positions = np.full(tuple(spec["positions_shape"]), -1, dtype=np.int32)
            self.state_feeds[str(spec["cached_positions_input_name"])] = positions

    # [调用链入口] 被 app_onnx.py 的 _on_frame 回调调用（流式解码调用链第 6 步 - [实时解码音频]）
    # 将一批声学 token 帧（frame_rows，每帧含 num_quantizers 个 VQ token）解码为音频波形
    # 同时更新内部 KV Cache，保持流式解码的上下文连续性
    def run_frames(self, frame_rows: list[list[int]]) -> tuple[np.ndarray, int] | None:
        if not frame_rows:
            return None
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        frame_count = len(frame_rows)
        # 构造 (1, frame_count, num_quantizers) 的声学 token 输入张量
        audio_codes = np.zeros((1, frame_count, num_quantizers), dtype=np.int32)
        for frame_index, frame_row in enumerate(frame_rows):
            for channel_index in range(num_quantizers):
                audio_codes[0, frame_index, channel_index] = int(frame_row[channel_index] if channel_index < len(frame_row) else 0)
        feeds: dict[str, np.ndarray] = {
            "audio_codes": audio_codes,
            "audio_code_lengths": np.asarray([frame_count], dtype=np.int32),
        }
        # 追加 KV Cache 等有状态输入（首次调用时为全零初始状态）
        feeds.update(self.state_feeds)
        outputs = self.session.run(None, feeds)
        output_names = [output.name for output in self.session.get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        # 将本次输出的 KV Cache 写回 state_feeds，供下次调用续接
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = named_outputs[str(spec["output_name"])]
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = named_outputs[str(spec["offset_output_name"])]
            self.state_feeds[str(spec["cached_keys_input_name"])] = named_outputs[str(spec["cached_keys_output_name"])]
            self.state_feeds[str(spec["cached_values_input_name"])] = named_outputs[str(spec["cached_values_output_name"])]
            self.state_feeds[str(spec["cached_positions_input_name"])] = named_outputs[str(spec["cached_positions_output_name"])]
        return (
            named_outputs["audio"],
            int(named_outputs["audio_lengths"].reshape(-1)[0]),
        )


# =============================================================================
# OrtCpuRuntime —— ONNX 推理基类
# 职责：加载并管理全部 ONNX session，实现 prefill/decode 自回归推理循环
# OnnxTtsRuntime（onnx_tts_runtime.py）继承此类，添加 TTS 业务逻辑
# =============================================================================

class OrtCpuRuntime:
    # [调用链入口] 被 onnx_tts_runtime.py 中 OnnxTtsRuntime.__init__ 通过 super().__init__ 调用
    # 负责：读取 manifest / tts_meta / codec_meta，创建全部 ONNX session，初始化流式解码会话
    def __init__(
        self,
        model_dir: str | Path,
        thread_count: int = 4,
        max_new_frames: int | None = None,
        do_sample: bool | None = None,
        sample_mode: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.thread_count = max(1, int(thread_count))
        self.manifest_path = self._resolve_manifest_path(self.model_dir)
        self.manifest_dir = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest = manifest
        # 允许运行时覆盖 manifest 中的生成参数
        if max_new_frames is not None:
            self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        if do_sample is not None:
            self.manifest["generation_defaults"]["do_sample"] = bool(do_sample)
        self.manifest["generation_defaults"]["sample_mode"] = _normalize_sample_mode(
            sample_mode if sample_mode is not None else self.manifest["generation_defaults"].get("sample_mode"),
            bool(self.manifest["generation_defaults"]["do_sample"]),
        )
        # sample_mode 规范化后反写 do_sample：greedy 模式强制关闭采样
        self.manifest["generation_defaults"]["do_sample"] = (
            self.manifest["generation_defaults"]["sample_mode"] != SAMPLE_MODE_GREEDY
        )
        self.tts_meta_path = self.resolve_manifest_relative_path(manifest["model_files"]["tts_meta"])
        self.codec_meta_path = self.resolve_manifest_relative_path(manifest["model_files"]["codec_meta"])
        self.tts_meta = json.loads(self.tts_meta_path.read_text(encoding="utf-8"))
        self.codec_meta = json.loads(self.codec_meta_path.read_text(encoding="utf-8"))
        self.rng = np.random.default_rng(1234)
        # 一次性加载所有 ONNX 模型（prefill / decode / local_decoder / codec 等）
        self.sessions = self._create_sessions()
        # 创建流式 codec 解码会话，绑定 codec_decode_step session
        self.codec_streaming_session = CodecStreamingDecodeSession(
            codec_meta=self.codec_meta,
            session=self.sessions["codec_decode_step"],
        )

    # [调用链内部] 被 __init__ 调用，在模型目录下按优先级搜索 manifest 文件
    @staticmethod
    def _resolve_manifest_path(model_dir: Path) -> Path:
        tried_paths: list[Path] = []
        for relative_path in MANIFEST_CANDIDATE_RELATIVE_PATHS:
            candidate = (model_dir / relative_path).resolve()
            tried_paths.append(candidate)
            if candidate.is_file():
                return candidate
        joined = ", ".join(str(path_value) for path_value in tried_paths)
        raise FileNotFoundError(f"browser_poc_manifest.json not found. tried: {joined}")

    # [调用链入口] 被 onnx_tts_runtime.py 调用，将 manifest 中的相对路径解析为绝对路径
    # 同时处理旧版目录名兼容（MODEL_DIR_ALIAS_MAP）
    def resolve_manifest_relative_path(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        resolved = (self.manifest_dir / relative).resolve()
        if resolved.exists():
            return resolved
        relative_text = str(relative).replace("\\", "/")
        for legacy_name, canonical_name in MODEL_DIR_ALIAS_MAP.items():
            legacy_fragment = f"/{legacy_name}/"
            if legacy_fragment not in f"/{relative_text}/":
                continue
            rewritten_text = relative_text.replace(legacy_name, canonical_name)
            rewritten = (self.manifest_dir / Path(rewritten_text)).resolve()
            if rewritten.exists():
                return rewritten
        return resolved

    # [调用链内部] 被 _create_sessions 调用，创建单个 ONNX InferenceSession
    # 固定使用 ORT_ENABLE_BASIC 优化级别和 CUDAExecutionProvider
    def _session(self, path_value: Path) -> ort.InferenceSession:
        options = ort.SessionOptions()
        # options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.intra_op_num_threads = self.thread_count
        options.inter_op_num_threads = 1
        # return ort.InferenceSession(str(path_value), sess_options=options, providers=["CPUExecutionProvider"])
        return ort.InferenceSession(str(path_value), sess_options=options, providers=["CUDAExecutionProvider"])

    # [调用链内部] 被 __init__ 调用，按 tts_meta / codec_meta 中的文件路径加载所有 ONNX 模型
    # 可选 session（local_greedy_frame / local_fixed_sampled_frame / local_cached_step）按 manifest 条件加载
    def _create_sessions(self) -> dict[str, ort.InferenceSession]:
        tts_dir = self.tts_meta_path.parent
        codec_dir = self.codec_meta_path.parent
        return {
            "prefill": self._session(tts_dir / self.tts_meta["files"]["prefill"]),
            "decode": self._session(tts_dir / self.tts_meta["files"]["decode_step"]),
            "local_decoder": self._session(tts_dir / self.tts_meta["files"]["local_decoder"]),
            **(
                {"local_greedy_frame": self._session(tts_dir / self.tts_meta["files"]["local_greedy_frame"])}
                if self.tts_meta["files"].get("local_greedy_frame")
                else {}
            ),
            **(
                {"local_fixed_sampled_frame": self._session(tts_dir / self.tts_meta["files"]["local_fixed_sampled_frame"])}
                if self.tts_meta["files"].get("local_fixed_sampled_frame")
                else {}
            ),
            **(
                {"local_cached_step": self._session(tts_dir / self.tts_meta["files"]["local_cached_step"])}
                if self.tts_meta["files"].get("local_cached_step")
                else {}
            ),
            "codec_encode": self._session(codec_dir / self.codec_meta["files"]["encode"]),
            "codec_decode": self._session(codec_dir / self.codec_meta["files"]["decode_full"]),
            "codec_decode_step": self._session(codec_dir / self.codec_meta["files"]["decode_step"]),
        }

    # [非调用链] 返回 manifest 中内置语音列表，供上层接口展示可用音色
    def list_builtin_voices(self) -> list[dict[str, Any]]:
        return list(self.manifest["builtin_voices"])

    # [非调用链] 返回 manifest 中内置文本示例列表
    def list_text_samples(self) -> list[dict[str, Any]]:
        return list(self.manifest["text_samples"])

    # [非调用链] 服务启动时的预热函数，用于提前触发 ONNX 图优化和算子初始化，降低首次推理延迟
    # 不在 stream generate 调用链中，运行于服务初始化阶段
    def warmup(self) -> None:
        voice = self.list_builtin_voices()[0]
        text_sample = self.list_text_samples()[0]
        request_rows = self.build_voice_clone_request_rows(voice["prompt_audio_codes"], text_sample["text_token_ids"])
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
        outputs = self.sessions["prefill"].run(
            None,
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        output_names = [output.name for output in self.sessions["prefill"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = _extract_last_hidden(named_outputs["global_hidden"])
        if "local_cached_step" in self.sessions:
            local_past_by_name = self.create_empty_local_cached_past()
            _text_logits, _audio_logits, _next_local_past = self.run_local_cached_step(
                global_hidden,
                text_token_id=0,
                audio_token_id=0,
                channel_index=0,
                step_type=0,
                past_valid_lengths=0,
                local_past_by_name=local_past_by_name,
            )
        if "local_fixed_sampled_frame" in self.sessions and self.manifest["generation_defaults"]["sample_mode"] == SAMPLE_MODE_FIXED:
            self.run_local_fixed_sampled_frame(
                global_hidden,
                previous_token_sets_by_channel=[set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))],
            )
        elif "local_greedy_frame" in self.sessions and not bool(self.manifest["generation_defaults"]["do_sample"]):
            self.run_local_greedy_frame(
                global_hidden,
                previous_token_sets_by_channel=[set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))],
                repetition_penalty=float(self.manifest["generation_defaults"]["audio_repetition_penalty"]),
            )
        else:
            self.run_local_decoder(global_hidden, self.manifest["tts_config"]["audio_assistant_slot_token_id"], [])
        empty_frames = [([0] * int(self.manifest["tts_config"]["n_vq"]))]
        self.decode_full_audio(empty_frames)
        self.codec_streaming_session.reset()
        self.codec_streaming_session.run_frames(empty_frames)
        self.codec_streaming_session.reset()

    # [调用链内部] 被 build_voice_clone_request_rows 调用
    # 将文本 token id 列表编码为 (n_vq+1) 宽的行矩阵，音频列填充 audio_pad_token_id
    def build_text_rows(self, token_ids: list[int]) -> list[list[int]]:
        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        for token_id in token_ids:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = int(token_id)
            rows.append(row)
        return rows

    # [调用链内部] 被 build_voice_clone_request_rows 调用
    # 将参考音频的声学 token（prompt_audio_codes）编码为带 slot token 标记的行矩阵
    def build_audio_prefix_rows(self, prompt_audio_codes: list[list[int]], slot_token_id: int | None = None) -> list[list[int]]:
        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        resolved_slot_token_id = int(
            self.manifest["tts_config"]["audio_user_slot_token_id"] if slot_token_id is None else slot_token_id
        )
        for code_row in prompt_audio_codes:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = resolved_slot_token_id
            for index in range(min(len(code_row), int(self.manifest["tts_config"]["n_vq"]))):
                row[index + 1] = int(code_row[index])
            rows.append(row)
        return rows

    # [调用链入口] 被 onnx_tts_runtime.py 调用（调用链第 3 步的数据预处理输出）
    # 将参考音频 codes + 目标文本 token ids 拼接为 prefill 阶段所需的输入矩阵（inputIds + attentionMask）
    # 格式：[用户 prompt 前缀文本行] + [参考音频行] + [用户 prompt 后缀文本行] + [目标文本行] + [assistant 开头行]
    def build_voice_clone_request_rows(self, prompt_audio_codes: list[list[int]], text_token_ids: list[int]) -> dict[str, list[list[int]]]:
        prefix_text_token_ids = [
            *self.manifest["prompt_templates"]["user_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        suffix_text_token_ids = [
            int(self.manifest["tts_config"]["audio_end_token_id"]),
            *self.manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
            *text_token_ids,
            *self.manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        rows = [
            *self.build_text_rows(prefix_text_token_ids),
            *self.build_audio_prefix_rows(prompt_audio_codes),
            *self.build_text_rows(suffix_text_token_ids),
        ]
        return {
            "inputIds": rows,
            "attentionMask": [[1 for _ in rows]],
        }

    # [调用链内部] 被 generate_audio_frames 的 else 分支（无 local_cached_step）调用
    # 每次推理需提供当前帧已生成的 audio prefix，对每个 VQ 声道逐步采样
    def run_local_decoder(self, global_hidden: np.ndarray, text_token_id: int, frame_prefix: list[int]) -> tuple[np.ndarray, np.ndarray]:
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        audio_pad = int(self.manifest["tts_config"]["audio_pad_token_id"])
        padded_prefix = np.full((1, n_vq - 1), audio_pad, dtype=np.int32)
        for index in range(min(len(frame_prefix), n_vq - 1)):
            padded_prefix[0, index] = int(frame_prefix[index])
        outputs = self.sessions["local_decoder"].run(
            None,
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "text_token_id": np.asarray([int(text_token_id)], dtype=np.int32),
                "audio_prefix_token_ids": padded_prefix,
            },
        )
        output_names = [output.name for output in self.sessions["local_decoder"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        return named_outputs["text_logits"].reshape(-1), named_outputs["audio_logits"]

    # [调用链内部] 被 generate_audio_frames（local_cached_step 分支）调用
    # 构造空的 local KV Cache，形状由 tts_meta 中的 local_layers/heads/head_dim 决定
    def create_empty_local_cached_past(self) -> dict[str, np.ndarray]:
        local_layers = int(self.tts_meta["model_config"]["local_layers"])
        local_heads = int(self.tts_meta["model_config"]["local_heads"])
        local_head_dim = int(self.tts_meta["model_config"]["local_head_dim"])
        return {
            name: np.zeros((1, 0, local_heads, local_head_dim), dtype=np.float32)
            for layer_index in range(local_layers)
            for name in (f"local_past_key_{layer_index}", f"local_past_value_{layer_index}")
        }

    # [调用链内部] 被 generate_audio_frames（local_cached_step 分支）在每帧内逐步调用
    # step_type 区分当前步骤：0=text预测步, 1=首个VQ采样步, 2=后续VQ采样步
    # 每次调用后需更新 local_past_by_name（帧内局部 KV Cache）
    def run_local_cached_step(
        self,
        global_hidden: np.ndarray,
        *,
        text_token_id: int,
        audio_token_id: int,
        channel_index: int,
        step_type: int,
        past_valid_lengths: int,
        local_past_by_name: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        outputs = self.sessions["local_cached_step"].run(
            None,
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "text_token_id": np.asarray([int(text_token_id)], dtype=np.int32),
                "audio_token_id": np.asarray([int(audio_token_id)], dtype=np.int32),
                "channel_index": np.asarray([int(channel_index)], dtype=np.int32),
                "step_type": np.asarray([int(step_type)], dtype=np.int32),
                "past_valid_lengths": np.asarray([int(past_valid_lengths)], dtype=np.int32),
                **local_past_by_name,
            },
        )
        output_names = [output.name for output in self.sessions["local_cached_step"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        next_local_past = {
            output_name.replace("local_present_", "local_past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["local_cached_output_names"][2:]
        }
        return named_outputs["text_logits"].reshape(-1), named_outputs["audio_logits"], next_local_past

    # [调用链内部] 被 generate_audio_frames（greedy 模式）调用
    # 一次 ONNX 推理即可完整输出一帧全部 VQ token（ONNX 内部已完成跨声道 greedy 采样）
    # should_continue=False 表示模型预测生成结束，终止推理循环
    def run_local_greedy_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
        repetition_penalty: float,
    ) -> tuple[bool, list[int]]:
        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        # 将每个声道历史 token 集合编码为 mask 矩阵，供 ONNX 内部施加重复惩罚
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
        outputs = self.sessions["local_greedy_frame"].run(
            None,
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "repetition_penalty": np.asarray([float(repetition_penalty)], dtype=np.float32),
            },
        )
        output_names = [output.name for output in self.sessions["local_greedy_frame"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        should_continue = bool(int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0]))
        frame_token_ids = np.asarray(named_outputs["frame_token_ids"]).reshape(-1).astype(np.int32, copy=False).tolist()
        return should_continue, [int(item) for item in frame_token_ids]

    # [调用链内部] 被 generate_audio_frames（fixed 采样模式）调用
    # 与 greedy_frame 类似，但通过外部注入随机数（assistant_random_u / audio_random_u）控制采样，
    # 使采样过程在 ONNX 内部确定性地执行（固定随机种子可复现）
    def run_local_fixed_sampled_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
    ) -> tuple[bool, list[int]]:
        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
        # 由 Python 侧 RNG 生成随机数，传入 ONNX 内部做逆变换采样，保证与 Python 采样路径一致
        assistant_random_u = np.asarray([min(0.99999994, max(0.0, float(self.rng.random())))], dtype=np.float32)
        audio_random_u = np.asarray(
            [[min(0.99999994, max(0.0, float(self.rng.random()))) for _ in range(n_vq)]],
            dtype=np.float32,
        )
        outputs = self.sessions["local_fixed_sampled_frame"].run(
            None,
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "assistant_random_u": assistant_random_u,
                "audio_random_u": audio_random_u,
            },
        )
        output_names = [output.name for output in self.sessions["local_fixed_sampled_frame"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        frame_token_ids = np.asarray(named_outputs["frame_token_ids"]).reshape(-1).astype(np.int32, copy=False).tolist()
        should_continue = bool(int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0]))
        return should_continue, [int(item) for item in frame_token_ids]

    # [调用链内部] 被 generate_audio_frames（local_cached_step 分支）调用
    # audio_logits 将所有 VQ 声道的 logits 拼接在一起，此函数按声道索引切片取出目标声道
    def slice_audio_channel_logits(self, audio_logits: np.ndarray, channel_index: int) -> np.ndarray:
        per_channel = int(audio_logits.shape[-1])
        flat = audio_logits.reshape(-1)
        start = channel_index * per_channel
        end = start + per_channel
        return flat[start:end]

    # [非调用链] 非流式全量音频解码，用于一次性将全部生成帧解码为 PCM
    # 在 stream generate 路径中不被调用（流式解码使用 CodecStreamingDecodeSession.run_frames）
    def decode_full_audio(self, generated_frames: list[list[int]]) -> tuple[list[np.ndarray], int]:
        if not generated_frames:
            return [], 0
        audio_codes, dims = _flatten3d_int32([generated_frames])
        outputs = self.sessions["codec_decode"].run(
            None,
            {
                "audio_codes": audio_codes.reshape(dims),
                "audio_code_lengths": np.asarray([len(generated_frames)], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.sessions["codec_decode"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_length = int(named_outputs["audio_lengths"].reshape(-1)[0])
        return _slice_channel_major_audio(named_outputs["audio"], 0, audio_length), audio_length

    # [调用链入口] 被 app_onnx.py 的 _worker 线程调用（调用链第 4 步）
    # 完整实现 prefill → 自回归 decode 循环，每生成一帧触发一次 on_frame 回调
    #
    # 推理分为两个阶段：
    #   1. Prefill：将 request_rows（文本 + 参考音频 prompt）一次性送入 prefill ONNX 模型，
    #      获取初始 global_hidden 和全量 KV Cache（past_by_name）
    #   2. Decode 循环：每次迭代生成一帧（n_vq 个 VQ token），共三条执行路径：
    #      - greedy_frame：整帧一次推理（greedy 模式，最快） → do_sample=true，默认条件不满足
    #      - fixed_sampled_frame：整帧一次推理（fixed 采样模式，外部注入随机数） → sample_mode=="fixed" 且 session 存在，默认命中
    #      - local_cached_step：逐声道 n_vq+1 次推理（full 模式或 fallback） → 不会到达
    #      - local_decoder（else）：无缓存逐声道推理（最慢 fallback） → 不会到达
    #      每帧生成后，用 decode ONNX 模型更新 global_hidden 和 KV Cache，进入下一帧
    def generate_audio_frames(
        self,
        request_rows: dict[str, list[list[int]]],
        on_frame: Callable[[list[list[int]], int, list[int]], None] | None = None,
    ) -> list[list[int]]:
        generation_defaults = self.manifest["generation_defaults"]
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1

        # ── 阶段 1：Prefill ──────────────────────────────────────────────────
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
        outputs = self.sessions["prefill"].run(
            None,
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        output_names = [output.name for output in self.sessions["prefill"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        # 取序列最后一步的隐状态作为第一帧生成的条件向量
        global_hidden = _extract_last_hidden(named_outputs["global_hidden"])
        # past_valid_length 记录 KV Cache 中有效 token 数（随每帧 decode 递增）
        past_valid_length = sum(int(item) for item in request_rows["attentionMask"][0])
        past_by_name = {
            output_name.replace("present_", "past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }

        # ── 阶段 2：自回归 Decode 循环 ────────────────────────────────────────
        generated_frames: list[list[int]] = []
        # 每个 VQ 声道分别维护历史 token 列表和集合，用于重复惩罚计算
        previous_tokens_by_channel = [[] for _ in range(int(self.manifest["tts_config"]["n_vq"]))]
        previous_token_sets_by_channel = [set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))]

        for step_index in range(int(generation_defaults["max_new_frames"])):
            frame: list[int] = []

            # 路径 A：local_greedy_frame（整帧一次推理，greedy 模式）
            if "local_greedy_frame" in self.sessions and not bool(generation_defaults["do_sample"]):
                should_continue, frame = self.run_local_greedy_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                    repetition_penalty=float(generation_defaults["audio_repetition_penalty"]),
                )
                if not should_continue:
                    break
                for channel_index, sampled_token in enumerate(frame):
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)

            # 路径 B：local_fixed_sampled_frame（整帧一次推理，fixed 采样模式）
            elif "local_fixed_sampled_frame" in self.sessions and generation_defaults["sample_mode"] == SAMPLE_MODE_FIXED:
                should_continue, frame = self.run_local_fixed_sampled_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                )
                if not should_continue:
                    break
                for channel_index, sampled_token in enumerate(frame):
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)

            # 路径 C：local_cached_step（逐声道推理，带帧内 KV Cache）
            elif "local_cached_step" in self.sessions:
                local_past_by_name = self.create_empty_local_cached_past()
                local_past_valid_length = 0

                # step_type=0：text 预测步，判断是继续生成音频还是结束
                local_text_logits, _ignored_audio_logits, local_past_by_name = self.run_local_cached_step(
                    global_hidden,
                    text_token_id=0,
                    audio_token_id=0,
                    channel_index=0,
                    step_type=0,
                    past_valid_lengths=local_past_valid_length,
                    local_past_by_name=local_past_by_name,
                )
                local_past_valid_length += 1
                next_text_token = _sample_assistant_text_token(
                    local_text_logits,
                    self.manifest,
                    generation_defaults,
                    self.rng,
                )
                # 若采样到 audio_end_token，表示生成已结束
                if next_text_token != int(self.manifest["tts_config"]["audio_assistant_slot_token_id"]):
                    break

                # step_type=1：第 0 声道采样步
                _unused_text_logits, audio_logits, local_past_by_name = self.run_local_cached_step(
                    global_hidden,
                    text_token_id=next_text_token,
                    audio_token_id=0,
                    channel_index=0,
                    step_type=1,
                    past_valid_lengths=local_past_valid_length,
                    local_past_by_name=local_past_by_name,
                )
                local_past_valid_length += 1
                first_channel_logits = self.slice_audio_channel_logits(audio_logits, 0).astype(np.float32, copy=False)
                sampled_token = _sample_audio_token(
                    first_channel_logits,
                    previous_tokens_by_channel[0],
                    previous_token_sets_by_channel[0],
                    generation_defaults,
                    self.rng,
                )
                frame.append(sampled_token)
                previous_tokens_by_channel[0].append(sampled_token)
                previous_token_sets_by_channel[0].add(sampled_token)

                # step_type=2：第 1..n_vq-1 声道依次采样（自回归：每步以上一声道 token 为条件）
                previous_token = sampled_token
                host_sampled_channel_limit = int(self.manifest["tts_config"]["n_vq"])
                for channel_index in range(1, host_sampled_channel_limit):
                    _unused_text_logits, audio_logits, local_past_by_name = self.run_local_cached_step(
                        global_hidden,
                        text_token_id=0,
                        audio_token_id=previous_token,
                        channel_index=channel_index - 1,
                        step_type=2,
                        past_valid_lengths=local_past_valid_length,
                        local_past_by_name=local_past_by_name,
                    )
                    local_past_valid_length += 1
                    channel_logits = self.slice_audio_channel_logits(audio_logits, channel_index).astype(np.float32, copy=False)
                    sampled_token = _sample_audio_token(
                        channel_logits,
                        previous_tokens_by_channel[channel_index],
                        previous_token_sets_by_channel[channel_index],
                        generation_defaults,
                        self.rng,
                    )
                    frame.append(sampled_token)
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
                    previous_token = sampled_token

            # 路径 D：local_decoder（无帧内 KV Cache 的 fallback，最慢）
            else:
                local_text_logits, _ = self.run_local_decoder(global_hidden, 0, [])
                next_text_token = _sample_assistant_text_token(
                    local_text_logits,
                    self.manifest,
                    generation_defaults,
                    self.rng,
                )
                if next_text_token != int(self.manifest["tts_config"]["audio_assistant_slot_token_id"]):
                    break
                for channel_index in range(int(self.manifest["tts_config"]["n_vq"])):
                    _, audio_logits = self.run_local_decoder(global_hidden, next_text_token, frame)
                    channel_logits = self.slice_audio_channel_logits(audio_logits, channel_index).astype(np.float32, copy=False)
                    sampled_token = _sample_audio_token(
                        channel_logits,
                        previous_tokens_by_channel[channel_index],
                        previous_token_sets_by_channel[channel_index],
                        generation_defaults,
                        self.rng,
                    )
                    frame.append(sampled_token)
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)

            generated_frames.append(frame)

            # ── 每帧后：运行 decode ONNX 更新 global_hidden 和 KV Cache ────────
            # 将本帧的 assistant slot token + 全部 VQ token 组成下一步输入
            next_row = np.full((1, 1, row_width), int(self.manifest["tts_config"]["audio_pad_token_id"]), dtype=np.int32)
            next_row[0, 0, 0] = int(self.manifest["tts_config"]["audio_assistant_slot_token_id"])
            for index, token in enumerate(frame):
                next_row[0, 0, index + 1] = int(token)
            decode_feeds: dict[str, np.ndarray] = {
                "input_ids": next_row,
                "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
            }
            for input_name in self.tts_meta["onnx"]["decode_input_names"][2:]:
                decode_feeds[input_name] = past_by_name[input_name]
            decode_outputs = self.sessions["decode"].run(None, decode_feeds)
            decode_output_names = [output.name for output in self.sessions["decode"].get_outputs()]
            named_decode_outputs = dict(zip(decode_output_names, decode_outputs, strict=True))
            global_hidden = _extract_last_hidden(named_decode_outputs["global_hidden"])
            past_valid_length += 1
            past_by_name = {
                output_name.replace("present_", "past_"): named_decode_outputs[output_name]
                for output_name in self.tts_meta["onnx"]["decode_output_names"][1:]
            }

            # ── 触发 on_frame 回调（调用链第 5 步）────────────────────────────
            # app_onnx.py 的 _on_frame 在此被调用，收集足够帧后调用 codec 流式解码
            if on_frame is not None:
                on_frame(generated_frames, step_index, frame)
        return generated_frames

__all__ = [
    "CodecStreamingDecodeSession",
    "OrtCpuRuntime",
    "SAMPLE_MODE_FIXED",
    "SAMPLE_MODE_FULL",
    "SAMPLE_MODE_GREEDY",
    "_normalize_sample_mode",
    "_resolve_stream_decode_frame_budget",
]
