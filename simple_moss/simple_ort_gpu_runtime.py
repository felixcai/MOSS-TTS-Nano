from __future__ import annotations

# =============================================================================
# simple_ort_gpu_runtime.py — ONNX 核心推理层（调用链最底层）
#
# 从 ort_cpu_runtime.py 剥离而来，只保留 fixed 采样路径：
#   - 去掉 greedy / full / local_cached_step / local_decoder 路径
#   - 去掉 decode_full_audio（非流式全量解码）
#   - 去掉所有 Python 侧采样辅助函数（_argmax, _softmax, _sample_* 等）
#   - 保留 CUDAExecutionProvider
#
# 在 Stream Generate 调用链中的角色：
#   api_facade.py → simple_app_onnx.py → simple_onnx_tts_runtime.py → [本文件 OrtCpuRuntime]
#                                                                    ↑               ↓
#                                        simple_app_onnx.py._on_frame ← on_frame 回调
#                                              ↓
#                                  CodecStreamingDecodeSession.run_frames（本文件）
# =============================================================================

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort

from ._utils import _log_memory

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
# 模块级辅助函数
# -----------------------------------------------------------------------------

def _flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    """将三维嵌套 list 展平为一维 int32 数组，同时返回 [dim0, dim1, dim2] 维度信息。
    用于将 inputIds（token 序列矩阵）重塑为 ONNX session.run 所需的张量格式。

    调用方：本文件 generate_audio_frames（prefill 阶段构造输入张量）和 warmup（间接通过 generate_audio_frames）。
    """
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


def _flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    """将二维嵌套 list 展平为一维 int32 数组，同时返回 [dim0, dim1] 维度信息。
    用于将 attentionMask 重塑为 ONNX session.run 所需的张量格式。

    调用方：本文件 generate_audio_frames（prefill 阶段构造 attention_mask 输入）。
    """
    dim0 = len(nested)
    dim1 = len(nested[0])
    data = np.zeros((dim0 * dim1,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            data[offset] = int(nested[i][j])
            offset += 1
    return data, [dim0, dim1]


def _extract_last_hidden(hidden_states: np.ndarray) -> np.ndarray:
    """从 prefill/decode 模型输出的 global_hidden 中取出最后时间步的隐状态向量。
    输入形状兼容 (seq_len, hidden) 或 (1, seq_len, hidden)，统一返回 (hidden,) 的 float32 数组。

    调用方：本文件 generate_audio_frames（每帧 prefill/decode 后提取隐状态）。
    """
    if hidden_states.ndim == 2:
        return hidden_states.astype(np.float32, copy=False)
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"Unexpected global_hidden shape: {hidden_states.shape}")
    return hidden_states[:, -1, :].astype(np.float32, copy=False)


def _normalize_sample_mode(raw_sample_mode: str | None, raw_do_sample: bool = True) -> str:
    """将外部传入的采样模式字符串规范化为 "greedy" / "fixed" / "full" 三种枚举值之一。
    "mixed3" 为旧别名：do_sample=True 时等价于 "fixed"，否则等价于 "greedy"。

    调用方：本文件 OrtCpuRuntime.__init__（初始化时设置 generation_defaults["sample_mode"]）。
    """
    normalized = str(raw_sample_mode or "").strip()
    if normalized in {SAMPLE_MODE_GREEDY, SAMPLE_MODE_FIXED, SAMPLE_MODE_FULL}:
        return normalized
    if normalized == "mixed3":
        return SAMPLE_MODE_FIXED if raw_do_sample else SAMPLE_MODE_GREEDY
    return SAMPLE_MODE_GREEDY if not raw_do_sample else SAMPLE_MODE_FIXED


def _compute_stream_lead_seconds(emitted_samples_total: int, sample_rate: int, first_audio_emitted_at_seconds: float | None) -> float:
    """计算已生成音频相对实时播放"超前"的秒数。
    lead_seconds > 0 表示缓冲充足，< 0 表示来不及解码。

    调用方：本文件 _resolve_stream_decode_frame_budget。
    """
    if not first_audio_emitted_at_seconds or sample_rate <= 0:
        return 0.0
    elapsed_seconds = max(0.0, time.perf_counter() - first_audio_emitted_at_seconds)
    emitted_seconds = emitted_samples_total / float(sample_rate)
    return emitted_seconds - elapsed_seconds


def _resolve_stream_decode_frame_budget(
    emitted_samples_total: int,
    sample_rate: int,
    first_audio_emitted_at_seconds: float | None,
) -> int:
    """根据当前流式超前量，动态决定本次 codec 解码应批处理多少帧。
    超前量不足时返回较小值（优先降低首帧延迟），超前量充足时返回较大值（提升吞吐）。

    调用方：simple_app_onnx.py 中 synthesize_stream 内的 _decode_pending 闭包。
    """
    lead_seconds = _compute_stream_lead_seconds(emitted_samples_total, sample_rate, first_audio_emitted_at_seconds)
    if not first_audio_emitted_at_seconds or lead_seconds < 0.20:
        return 4
    if lead_seconds < 0.55:
        return 4
    if lead_seconds < 1.10:
        return 4
    return 8


# =============================================================================
# CodecStreamingDecodeSession —— 流式 Codec 解码会话
# =============================================================================

@dataclass
class CodecStreamingDecodeSession:
    """维护 codec_decode_step ONNX 模型的 KV Cache 状态，逐帧将声学 token 解码为 PCM 音频。
    每个 text_chunk 开始前需调用 reset() 清空状态，结束后再次 reset() 防止污染下一 chunk。
    """
    codec_meta: dict[str, Any]
    session: ort.InferenceSession

    def __post_init__(self) -> None:
        """dataclass 自动调用：从 codec_meta 读取 transformer/attention 规格并调用 reset() 初始化状态。

        调用方：Python dataclass 机制在实例化时自动调用，由 OrtCpuRuntime.__init__ 触发。
        """
        self.transformer_specs = list(self.codec_meta.get("streaming_decode", {}).get("transformer_offsets", []))
        self.attention_specs = list(self.codec_meta.get("streaming_decode", {}).get("attention_caches", []))
        self.state_feeds: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        """将所有 transformer offset 和 attention cache（K/V/positions）清零，开启新一轮流式解码。

        调用方：
          - 本文件 OrtCpuRuntime.warmup（预热前后各一次）；
          - simple_app_onnx.py 的 _worker 内，每个 text_chunk 开始前和结束后各调用一次。
        """
        self.state_feeds = {}
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = np.zeros(tuple(spec["shape"]), dtype=np.int32)
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = np.zeros(tuple(spec["offset_shape"]), dtype=np.int32)
            self.state_feeds[str(spec["cached_keys_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            self.state_feeds[str(spec["cached_values_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            positions = np.full(tuple(spec["positions_shape"]), -1, dtype=np.int32)
            self.state_feeds[str(spec["cached_positions_input_name"])] = positions

    def run_frames(self, frame_rows: list[list[int]]) -> tuple[np.ndarray, int] | None:
        """将一批声学 token 帧（frame_rows）送入 codec_decode_step ONNX 模型，解码为音频波形。
        同时更新内部 KV Cache，保持流式解码的上下文连续性。
        返回 (audio_tensor, valid_audio_length)，输入为空时返回 None。

        调用方：
          - 本文件 OrtCpuRuntime.warmup（预热 codec 流式解码器）；
          - simple_app_onnx.py 的 _decode_pending 闭包（每次批量解码声学 token）。
        """
        if not frame_rows:
            return None
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        frame_count = len(frame_rows)
        audio_codes = np.zeros((1, frame_count, num_quantizers), dtype=np.int32)
        for frame_index, frame_row in enumerate(frame_rows):
            for channel_index in range(num_quantizers):
                audio_codes[0, frame_index, channel_index] = int(frame_row[channel_index] if channel_index < len(frame_row) else 0)
        feeds: dict[str, np.ndarray] = {
            "audio_codes": audio_codes,
            "audio_code_lengths": np.asarray([frame_count], dtype=np.int32),
        }
        feeds.update(self.state_feeds)
        outputs = self.session.run(None, feeds)
        output_names = [output.name for output in self.session.get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
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
# OrtCpuRuntime —— ONNX 推理基类（simple_moss 版本，仅保留 fixed 路径）
# =============================================================================

class OrtCpuRuntime:
    """ONNX 推理基类，管理全部 InferenceSession，实现 prefill → 自回归 decode 推理循环。
    OnnxTtsRuntime（simple_onnx_tts_runtime.py）继承此类，在其上添加 TTS 业务逻辑。
    """

    def __init__(
        self,
        model_dir: str | Path,
        thread_count: int = 4,
        max_new_frames: int | None = None,
        do_sample: bool | None = None,
        sample_mode: str | None = None,
    ) -> None:
        """读取 manifest / tts_meta / codec_meta，创建全部 ONNX InferenceSession 和流式解码会话。

        调用方：simple_onnx_tts_runtime.py 中 OnnxTtsRuntime.__init__ 通过 super().__init__ 调用。
        """
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.thread_count = max(1, int(thread_count))
        _log_memory("runtime_init: OrtCpuRuntime __init__ entry (model_dir+thread_count)")
        self.manifest_path = self._resolve_manifest_path(self.model_dir)
        self.manifest_dir = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest = manifest
        if max_new_frames is not None:
            self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        if do_sample is not None:
            self.manifest["generation_defaults"]["do_sample"] = bool(do_sample)
        self.manifest["generation_defaults"]["sample_mode"] = _normalize_sample_mode(
            sample_mode if sample_mode is not None else self.manifest["generation_defaults"].get("sample_mode"),
            bool(self.manifest["generation_defaults"]["do_sample"]),
        )
        self.manifest["generation_defaults"]["do_sample"] = (
            self.manifest["generation_defaults"]["sample_mode"] != SAMPLE_MODE_GREEDY
        )
        self.tts_meta_path = self.resolve_manifest_relative_path(manifest["model_files"]["tts_meta"])
        self.codec_meta_path = self.resolve_manifest_relative_path(manifest["model_files"]["codec_meta"])
        self.tts_meta = json.loads(self.tts_meta_path.read_text(encoding="utf-8"))
        self.codec_meta = json.loads(self.codec_meta_path.read_text(encoding="utf-8"))
        _log_memory("runtime_init: OrtCpuRuntime manifest+tts_meta+codec_meta loaded")
        self.rng = np.random.default_rng(1234)
        _log_memory("runtime_init: OrtCpuRuntime before _create_sessions")
        self.sessions = self._create_sessions()
        _log_memory("runtime_init: OrtCpuRuntime after _create_sessions (all InferenceSession)")
        self.codec_streaming_session = CodecStreamingDecodeSession(
            codec_meta=self.codec_meta,
            session=self.sessions["codec_decode_step"],
        )
        _log_memory("runtime_init: OrtCpuRuntime CodecStreamingDecodeSession ready")

    @staticmethod
    def _resolve_manifest_path(model_dir: Path) -> Path:
        """在模型目录下按 MANIFEST_CANDIDATE_RELATIVE_PATHS 优先级依次查找 manifest 文件。
        找不到则抛出 FileNotFoundError，并列出所有尝试的路径。

        调用方：本文件 OrtCpuRuntime.__init__。
        """
        tried_paths: list[Path] = []
        for relative_path in MANIFEST_CANDIDATE_RELATIVE_PATHS:
            candidate = (model_dir / relative_path).resolve()
            tried_paths.append(candidate)
            if candidate.is_file():
                return candidate
        joined = ", ".join(str(path_value) for path_value in tried_paths)
        raise FileNotFoundError(f"browser_poc_manifest.json not found. tried: {joined}")

    def resolve_manifest_relative_path(self, relative_path: str | Path) -> Path:
        """将 manifest 内记录的相对路径解析为绝对路径。
        若直接解析的路径不存在，尝试通过 MODEL_DIR_ALIAS_MAP 进行旧目录名兼容重写。

        调用方：本文件 OrtCpuRuntime.__init__（解析 tts_meta/codec_meta 路径），
                以及 simple_onnx_tts_runtime.py OnnxTtsRuntime.__init__（解析 tokenizer 路径）。
        """
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

    def _session(self, path_value: Path) -> ort.InferenceSession:
        """为指定 ONNX 文件路径创建 InferenceSession，使用 CUDAExecutionProvider 和基础图优化。
        禁用 CPU 内存 Arena，使内存用完后立即归还给系统。

        调用方：本文件 _create_sessions（为每个 ONNX 模型文件各调用一次）。
        """
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.intra_op_num_threads = self.thread_count
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        return ort.InferenceSession(str(path_value), sess_options=options, providers=["CUDAExecutionProvider"])

    def _create_sessions(self) -> dict[str, ort.InferenceSession]:
        """按 tts_meta / codec_meta 中的文件路径，加载推理所需的 ONNX 模型。
        仅加载 fixed 路径必需的 session：prefill、decode、local_fixed_sampled_frame、codec_decode_step。

        调用方：本文件 OrtCpuRuntime.__init__。
        """
        tts_dir = self.tts_meta_path.parent
        codec_dir = self.codec_meta_path.parent
        sessions: dict[str, ort.InferenceSession] = {
            "prefill": self._session(tts_dir / self.tts_meta["files"]["prefill"]),
            "decode": self._session(tts_dir / self.tts_meta["files"]["decode_step"]),
        }
        if self.tts_meta["files"].get("local_fixed_sampled_frame"):
            sessions["local_fixed_sampled_frame"] = self._session(
                tts_dir / self.tts_meta["files"]["local_fixed_sampled_frame"]
            )
        _log_memory("runtime_init: _create_sessions TTS part done (prefill/decode/local_fixed_sampled_frame)")
        sessions["codec_decode_step"] = self._session(codec_dir / self.codec_meta["files"]["decode_step"])
        _log_memory("runtime_init: _create_sessions codec part done (codec_decode_step)")
        return sessions

    def list_builtin_voices(self) -> list[dict[str, Any]]:
        """返回 manifest 中内置语音列表的副本，每项包含 voice 名称和 prompt_audio_codes 等字段。

        调用方：本文件 warmup（选取预热用音色），
                simple_onnx_tts_runtime.py 的 resolve_builtin_voice_prompt_audio_codes（查找音色数据）。
        """
        return list(self.manifest["builtin_voices"])

    def list_text_samples(self) -> list[dict[str, Any]]:
        """返回 manifest 中内置文本示例列表的副本，每项包含 text_token_ids 等字段，供 warmup 使用。

        调用方：本文件 warmup（取第一条示例文本触发推理预热）。
        """
        return list(self.manifest["text_samples"])

    def warmup(self, *, voice_name: str | None = None) -> None:
        """用内置音色 + 示例文本跑一次完整的 generate_audio_frames 和 codec 流式解码，
        触发 ONNX 图优化和 GPU Arena 预分配，降低首次真实请求的冷启动延迟。

        调用方：simple_app_onnx.py 中 OnnxNanoTTSServiceAdapter.warmup，
                通过 self.runtime.warmup(voice_name=...) 调用。
        """
        _log_memory("warmup: OrtCpuRuntime.warmup start")
        voices = self.list_builtin_voices()
        if voice_name is not None:
            voice = next((v for v in voices if v["voice"] == voice_name), voices[0])
        else:
            voice = voices[0]
        text_sample = self.list_text_samples()[0]
        request_rows = self.build_voice_clone_request_rows(voice["prompt_audio_codes"], text_sample["text_token_ids"])

        original_max_new_frames = self.manifest["generation_defaults"]["max_new_frames"]
        self.manifest["generation_defaults"]["max_new_frames"] = 16

        try:
            _log_memory("warmup: before generate_audio_frames (prefill/decode)")
            generated_frames = self.generate_audio_frames(
                request_rows,
                mem_trace_label="warmup: generate_audio_frames",
            )
            _log_memory("warmup: after generate_audio_frames")
        finally:
            self.manifest["generation_defaults"]["max_new_frames"] = original_max_new_frames

        _log_memory("warmup: before codec_streaming_session.run_frames")
        self.codec_streaming_session.reset()
        self.codec_streaming_session.run_frames(generated_frames)
        self.codec_streaming_session.reset()
        _log_memory("warmup: after codec_streaming_session.run_frames")

    def build_text_rows(self, token_ids: list[int]) -> list[list[int]]:
        """将文本 token id 列表编码为 (n_vq+1) 宽的行矩阵，音频列用 audio_pad_token_id 填充。
        这是 prefill 输入 inputIds 的文本部分格式。

        调用方：本文件 build_voice_clone_request_rows（构造前缀/后缀文本行）。
        """
        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        for token_id in token_ids:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = int(token_id)
            rows.append(row)
        return rows

    def build_audio_prefix_rows(self, prompt_audio_codes: list[list[int]], slot_token_id: int | None = None) -> list[list[int]]:
        """将参考音频的声学 token（prompt_audio_codes）编码为带 slot token 标记的行矩阵。
        第 0 列填充 audio_user_slot_token_id（或自定义 slot），后续列填充各 VQ 量化器的 token。

        调用方：本文件 build_voice_clone_request_rows（构造参考音频行）。
        """
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

    def build_voice_clone_request_rows(self, prompt_audio_codes: list[list[int]], text_token_ids: list[int]) -> dict[str, list[list[int]]]:
        """拼接前缀文本行、参考音频行、后缀文本行，组装为 prefill 阶段所需的 inputIds 和 attentionMask。
        格式：[用户 prompt 前缀] + [参考音频] + [目标文本 + assistant 起始标记]

        调用方：本文件 warmup（构造预热请求），
                simple_app_onnx.py 的 _worker 闭包（每个 text_chunk 构造推理请求）。
        """
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

    def run_local_fixed_sampled_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
    ) -> tuple[bool, list[int]]:
        """调用 local_fixed_sampled_frame ONNX 模型，一次推理生成整帧所有 VQ token。
        由 Python 侧 RNG 生成随机数传入 ONNX，保证采样的可重复性。
        返回 (should_continue, frame_token_ids)；should_continue=False 时推理循环应终止。

        调用方：本文件 generate_audio_frames（decode 循环内，fixed 采样路径）。
        """
        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
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

    def generate_audio_frames(
        self,
        request_rows: dict[str, list[list[int]]],
        on_frame: Callable[[list[list[int]], int, list[int]], None] | None = None,
        mem_trace_label: str | None = None,
    ) -> list[list[int]]:
        """执行 prefill → 自回归 decode 循环，生成全部声学 token 帧并返回。
        推理分两个阶段：
          1. Prefill：将 request_rows 送入 prefill ONNX，获取初始 global_hidden 和 KV Cache；
          2. Decode 循环：反复调用 run_local_fixed_sampled_frame 生成一帧 token，
             再用 decode ONNX 更新 global_hidden 和 KV Cache，直到 should_continue=False 或到达上限。
        每帧生成后通过 on_frame 回调通知外层（用于流式 codec 解码）。
        循环结束后执行一次 GPU Arena Shrinkage，释放碎片化显存。

        调用方：
          - 本文件 warmup（预热用，不传 on_frame）；
          - simple_app_onnx.py 的 _worker 闭包（每个 text_chunk 调用一次，传入 _on_frame 回调）。
        """
        def _trace_memory(step: str) -> None:
            if mem_trace_label:
                _log_memory(f"{mem_trace_label}: {step}")

        _trace_memory("entry")
        generation_defaults = self.manifest["generation_defaults"]
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1

        # ── 阶段 1：Prefill ──────────────────────────────────────────────────
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
        _trace_memory("after build prefill inputs")
        _trace_memory("before prefill.run")
        outputs = self.sessions["prefill"].run(
            None,
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        _trace_memory("after prefill.run")
        output_names = [output.name for output in self.sessions["prefill"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = _extract_last_hidden(named_outputs["global_hidden"])
        past_valid_length = sum(int(item) for item in request_rows["attentionMask"][0])
        past_by_name = {
            output_name.replace("present_", "past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }
        _trace_memory("after build prefill outputs and past cache")

        # ── 阶段 2：自回归 Decode 循环（仅 fixed 采样路径）────────────────────
        generated_frames: list[list[int]] = []
        previous_tokens_by_channel = [[] for _ in range(int(self.manifest["tts_config"]["n_vq"]))]
        previous_token_sets_by_channel = [set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))]
        _last_decode_feeds: dict[str, np.ndarray] | None = None

        for step_index in range(int(generation_defaults["max_new_frames"])):
            frame: list[int] = []
            if step_index == 0:
                _trace_memory("decode loop first frame start")

            # 路径 B：local_fixed_sampled_frame（整帧一次推理，fixed 采样模式）
            if "local_fixed_sampled_frame" in self.sessions and generation_defaults["sample_mode"] == SAMPLE_MODE_FIXED:
                if step_index == 0:
                    _trace_memory("before first local_fixed_sampled_frame.run")
                should_continue, frame = self.run_local_fixed_sampled_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                )
                if step_index == 0:
                    _trace_memory("after first local_fixed_sampled_frame.run")
                if not should_continue:
                    break
                for channel_index, sampled_token in enumerate(frame):
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
            else:
                logging.warning("simple_ort_gpu_runtime: local_fixed_sampled_frame session not available or sample_mode != fixed, skipping frame")
                break

            generated_frames.append(frame)

            # ── 每帧后：运行 decode ONNX 更新 global_hidden 和 KV Cache ────────
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
            _last_decode_feeds = decode_feeds
            if step_index == 0:
                _trace_memory("before first decode.run")
            decode_outputs = self.sessions["decode"].run(None, decode_feeds)
            if step_index == 0:
                _trace_memory("after first decode.run")
            decode_output_names = [output.name for output in self.sessions["decode"].get_outputs()]
            named_decode_outputs = dict(zip(decode_output_names, decode_outputs, strict=True))
            global_hidden = _extract_last_hidden(named_decode_outputs["global_hidden"])
            past_valid_length += 1
            past_by_name = {
                output_name.replace("present_", "past_"): named_decode_outputs[output_name]
                for output_name in self.tts_meta["onnx"]["decode_output_names"][1:]
            }

            # ── 触发 on_frame 回调 ────────────────────────────────────────────
            if on_frame is not None:
                on_frame(generated_frames, step_index, frame)

        # ── 生成结束：执行一次带 Shrinkage 的收缩空跑 ─────────────────────────
        _trace_memory(f"decode loop done, generated_frames={len(generated_frames)}")
        if _last_decode_feeds is not None:
            _shrink_run_options = ort.RunOptions()
            _shrink_run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
            _trace_memory("before decode arena shrinkage run")
            self.sessions["decode"].run(None, _last_decode_feeds, run_options=_shrink_run_options)
            _trace_memory("after decode arena shrinkage run")

        _trace_memory("return")
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
