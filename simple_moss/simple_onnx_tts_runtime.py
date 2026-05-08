from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# simple_onnx_tts_runtime.py  —— TTS 业务逻辑层
#
# 从 onnx_tts_runtime.py 剥离而来，只保留 stream generate 所需部分：
#   - 去掉文本正则化（_ensure_text_normalizer, prepare_synthesis_text）
#   - 去掉参考音频编码（_load_reference_audio, encode_reference_audio）
#   - 去掉非流式合成（decode_full_audio_safe, synthesize_single_chunk, synthesize）
#   - 去掉 _write_waveform_to_wav（result 不再返回 WAV/base64）
#   - 去掉 torch / torchaudio / WeTextProcessingManager 依赖
#   - 新增 resolve_builtin_voice_prompt_audio_codes / get_codec_audio_format
# ─────────────────────────────────────────────────────────────────────────────

import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sentencepiece as spm

from moss_tts_nano.defaults import DEFAULT_OUTPUT_DIR

from ._utils import _log_memory
from .simple_ort_gpu_runtime import (
    OrtCpuRuntime,
    _normalize_sample_mode,
    _resolve_stream_decode_frame_budget,
    SAMPLE_MODE_FIXED,
    SAMPLE_MODE_FULL,
    SAMPLE_MODE_GREEDY,
)

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent


DEFAULT_BROWSER_ONNX_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_BROWSER_ONNX_TTS_DIR = DEFAULT_BROWSER_ONNX_MODEL_DIR / "MOSS-TTS-Nano-100M-ONNX"
DEFAULT_BROWSER_ONNX_CODEC_DIR = DEFAULT_BROWSER_ONNX_MODEL_DIR / "MOSS-Audio-Tokenizer-Nano-ONNX"
DEFAULT_BROWSER_ONNX_TTS_REPO_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
DEFAULT_BROWSER_ONNX_CODEC_REPO_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
DEFAULT_BROWSER_ONNX_TTS_REPO_URL = f"https://huggingface.co/{DEFAULT_BROWSER_ONNX_TTS_REPO_ID}"
DEFAULT_BROWSER_ONNX_CODEC_REPO_URL = f"https://huggingface.co/{DEFAULT_BROWSER_ONNX_CODEC_REPO_ID}"
# DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 2.00
# DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 2.00
SENTENCE_END_PUNCTUATION = set(".!?。！？；;")
CLAUSE_SPLIT_PUNCTUATION = set(",，、；;：:")
CLOSING_PUNCTUATION = set("\"'”’)]}）】》」』")

MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS = (
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
)


# ══════════════════════════════════════════════════════════════════════════════
# 模型目录解析 / 下载辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_model_dir_path(model_dir: str | Path | None) -> Path:
    """将 model_dir 参数（可能为 None 或字符串）解析为绝对 Path 对象。
    None 时回退到 DEFAULT_BROWSER_ONNX_MODEL_DIR。

    调用方：本文件 _default_model_dir_requested 和 ensure_browser_onnx_model_dir。
    """
    if model_dir is None:
        return DEFAULT_BROWSER_ONNX_MODEL_DIR.expanduser().resolve()
    return Path(model_dir).expanduser().resolve()


def _default_model_dir_requested(model_dir: str | Path | None) -> bool:
    """判断调用方是否请求使用默认模型目录（传入 None 或等价路径时为 True）。
    用于决定路径缺失时是否自动触发下载。

    调用方：本文件 ensure_browser_onnx_model_dir。
    """
    if model_dir is None:
        return True
    return _resolve_model_dir_path(model_dir) == DEFAULT_BROWSER_ONNX_MODEL_DIR.expanduser().resolve()


def _find_manifest_path(model_dir: Path) -> Path | None:
    """在 model_dir 下按 MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS 查找 manifest 文件，存在则返回。

    调用方：本文件 ensure_browser_onnx_model_dir（检测模型是否已就绪）。
    """
    for relative_path in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS:
        candidate = (model_dir / relative_path).resolve()
        if candidate.is_file():
            return candidate
    return None


def _directory_contains_all(parent: Path, required_names: Sequence[str]) -> bool:
    """检查 parent 目录下是否同时存在 required_names 中的所有文件/子目录。

    调用方：本文件 _find_directory_with_required_names（验证候选目录是否完整）。
    """
    return all((parent / name).exists() for name in required_names)


def _find_directory_with_required_names(root_dir: Path, required_names: Sequence[str]) -> Path | None:
    """在 root_dir 及其子树中递归搜索，返回第一个同时包含 required_names 所有文件的目录。
    优先检查 root_dir 本身，再用 rglob 深搜。

    调用方：本文件 _normalize_download_layout（下载后定位实际文件所在子目录）。
    """
    if not root_dir.exists():
        return None
    if _directory_contains_all(root_dir, required_names):
        return root_dir
    sentinel_name = str(required_names[0])
    for candidate in root_dir.rglob(sentinel_name):
        parent = candidate.parent
        if _directory_contains_all(parent, required_names):
            return parent
    return None


def _promote_directory_contents(source_dir: Path, target_dir: Path) -> None:
    """将 source_dir 下的所有子项移动到 target_dir（跳过已存在的文件）。
    用于将 HuggingFace 下载产生的嵌套目录结构展平到目标位置。

    调用方：本文件 _normalize_download_layout。
    """
    if source_dir.resolve() == target_dir.resolve():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if destination.exists():
            continue
        shutil.move(str(child), str(destination))


def _normalize_download_layout(target_dir: Path, required_names: Sequence[str]) -> None:
    """下载后修复目录层级：找到包含 required_names 的实际子目录并将内容提升到 target_dir。
    解决 HuggingFace snapshot_download 有时产生额外嵌套子目录的问题。

    调用方：本文件 _download_default_browser_onnx_assets（TTS 和 Codec 各调用一次）。
    """
    candidate_dir = _find_directory_with_required_names(target_dir, required_names)
    if candidate_dir is None:
        return
    _promote_directory_contents(candidate_dir, target_dir)


def _snapshot_download_repo(
    *,
    repo_id: str,
    local_dir: Path,
    allow_patterns: Sequence[str],
) -> None:
    """调用 huggingface_hub.snapshot_download 下载指定 repo 的文件到 local_dir。
    仅下载 allow_patterns 匹配的文件（如 *.onnx, *.json）。

    调用方：本文件 _download_default_browser_onnx_assets（TTS 和 Codec 各下载一次）。
    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required to auto-download ONNX assets. Install it with `pip install huggingface_hub`."
        ) from exc
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns),
    )


def _download_default_browser_onnx_assets(model_dir: Path) -> None:
    """从 HuggingFace 下载 TTS 和 Codec 的默认 ONNX 资产到 model_dir 下的子目录，
    下载完成后调用 _normalize_download_layout 展平目录结构。

    调用方：本文件 ensure_browser_onnx_model_dir（仅在 manifest 缺失且用户未指定自定义路径时触发）。
    """
    logging.info("browser_onnx assets missing under %s; downloading from Hugging Face.", model_dir)
    logging.info("browser_onnx TTS repo: %s", DEFAULT_BROWSER_ONNX_TTS_REPO_URL)
    logging.info("browser_onnx codec repo: %s", DEFAULT_BROWSER_ONNX_CODEC_REPO_URL)
    tts_dir = model_dir / DEFAULT_BROWSER_ONNX_TTS_DIR.name
    codec_dir = model_dir / DEFAULT_BROWSER_ONNX_CODEC_DIR.name
    _snapshot_download_repo(
        repo_id=DEFAULT_BROWSER_ONNX_TTS_REPO_ID,
        local_dir=tts_dir,
        allow_patterns=("*.onnx", "*.data", "*.json", "tokenizer.model"),
    )
    _snapshot_download_repo(
        repo_id=DEFAULT_BROWSER_ONNX_CODEC_REPO_ID,
        local_dir=codec_dir,
        allow_patterns=("*.onnx", "*.data", "*.json"),
    )
    _normalize_download_layout(
        tts_dir,
        required_names=("browser_poc_manifest.json", "tts_browser_onnx_meta.json", "tokenizer.model"),
    )
    _normalize_download_layout(
        codec_dir,
        required_names=("codec_browser_onnx_meta.json",),
    )


def ensure_browser_onnx_model_dir(model_dir: str | Path | None = None) -> Path:
    """确保 ONNX 模型资产就绪并返回解析后的模型目录路径。
    若 manifest 已存在则直接返回；若用户指定了自定义路径但不存在则报错；
    若使用默认路径且不存在则自动触发下载并再次验证。

    调用方：本文件 OnnxTtsRuntime.__init__（初始化时确认模型路径）。
    """
    resolved_model_dir = _resolve_model_dir_path(model_dir)
    manifest_path = _find_manifest_path(resolved_model_dir)
    if manifest_path is not None:
        return resolved_model_dir
    if not _default_model_dir_requested(model_dir):
        tried_paths = [str((resolved_model_dir / item).resolve()) for item in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS]
        raise FileNotFoundError(
            "browser_onnx model assets not found under the provided --model-dir. tried: " + ", ".join(tried_paths)
        )
    _download_default_browser_onnx_assets(resolved_model_dir)
    manifest_path = _find_manifest_path(resolved_model_dir)
    if manifest_path is None:
        tried_paths = [str((resolved_model_dir / item).resolve()) for item in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS]
        raise FileNotFoundError(
            "browser_onnx assets were downloaded but browser_poc_manifest.json is still missing. "
            + "tried: "
            + ", ".join(tried_paths)
        )
    return resolved_model_dir


# ══════════════════════════════════════════════════════════════════════════════
# 文本切分辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _contains_cjk(text: str) -> bool:
    """检测字符串中是否包含 CJK（中文/日文/韩文）字符。
    用于决定文本预处理和句子拼接时是否添加空格。

    调用方：本文件 _prepare_text_for_sentence_chunking 和 _join_sentence_parts。
    """
    for character in str(text or ""):
        if (
            "\u4e00" <= character <= "\u9fff"
            or "\u3400" <= character <= "\u4dbf"
            or "\u3040" <= character <= "\u30ff"
            or "\uac00" <= character <= "\ud7af"
        ):
            return True
    return False


def _prepare_text_for_sentence_chunking(text: str) -> str:
    """对文本做基础正规化（去多余空白、补全尾部标点），为后续句子切分做准备。
    CJK 文本末尾补"。"，英文短文本前补空格（触发更自然的语音韵律）。

    调用方：本文件 OnnxTtsRuntime.split_voice_clone_text。
    """
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("Text prompt cannot be empty.")
    normalized_text = normalized_text.replace("\r", " ").replace("\n", " ")
    while "  " in normalized_text:
        normalized_text = normalized_text.replace("  ", " ")
    if _contains_cjk(normalized_text):
        if normalized_text[-1] not in SENTENCE_END_PUNCTUATION:
            normalized_text += "。"
        return normalized_text
    if normalized_text[:1].islower():
        normalized_text = normalized_text[:1].upper() + normalized_text[1:]
    if normalized_text[-1].isalnum():
        normalized_text += "."
    if len([item for item in normalized_text.split() if item]) < 5:
        normalized_text = f"        {normalized_text}"
    return normalized_text


def _split_text_by_punctuation(text: str, punctuation: set[str]) -> list[str]:
    """按指定标点集合将文本切分为若干片段，同时将句末闭合标点（如引号、括号）归入当前句。

    调用方：本文件 OnnxTtsRuntime.split_voice_clone_text（先按句末标点切，再按从句标点切）。
    """
    sentences: list[str] = []
    current_chars: list[str] = []
    index = 0
    normalized_text = str(text or "")
    while index < len(normalized_text):
        character = normalized_text[index]
        current_chars.append(character)
        if character in punctuation:
            lookahead = index + 1
            while lookahead < len(normalized_text) and normalized_text[lookahead] in CLOSING_PUNCTUATION:
                current_chars.append(normalized_text[lookahead])
                lookahead += 1
            sentence = "".join(current_chars).strip()
            if sentence:
                sentences.append(sentence)
            current_chars.clear()
            while lookahead < len(normalized_text) and normalized_text[lookahead].isspace():
                lookahead += 1
            index = lookahead
            continue
        index += 1
    tail = "".join(current_chars).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _join_sentence_parts(left: str, right: str) -> str:
    """将两段文本拼接为一个 chunk：中文直接相连，英文之间插入空格。

    调用方：本文件 OnnxTtsRuntime.split_voice_clone_text（合并相邻短句以填满 token 预算）。
    """
    if not left:
        return right
    if not right:
        return left
    if _contains_cjk(left) or _contains_cjk(right):
        return left + right
    return f"{left} {right}"


# ══════════════════════════════════════════════════════════════════════════════
# 音频工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _merge_audio_channels(channel_arrays: list[np.ndarray]) -> np.ndarray:
    """将多个单声道音频数组合并为 (samples, channels) 的多声道波形矩阵。
    多声道时按最短声道对齐并截断。

    调用方：simple_app_onnx.py 中 synthesize_stream 内的 _decode_pending 闭包，
            对 codec 解码输出的每个声道分量进行合并。
    """
    if not channel_arrays:
        return np.zeros((0, 1), dtype=np.float32)
    if len(channel_arrays) == 1:
        return np.asarray(channel_arrays[0], dtype=np.float32).reshape(-1, 1)
    min_length = min(int(channel.shape[0]) for channel in channel_arrays)
    trimmed = [np.asarray(channel[:min_length], dtype=np.float32) for channel in channel_arrays]
    return np.stack(trimmed, axis=1)


def _concat_waveforms(waveforms: list[np.ndarray]) -> np.ndarray:
    """将多段 (samples, channels) 波形矩阵按时间轴拼接为完整波形。
    过滤空数组，全空时返回零矩阵。

    调用方：simple_app_onnx.py 中 synthesize_stream 内的 _worker 闭包，
            用于将一个 text_chunk 的所有流式音频片段拼接为完整的 chunk_waveform。
    """
    if not waveforms:
        return np.zeros((0, 1), dtype=np.float32)
    non_empty = [waveform for waveform in waveforms if waveform.size > 0]
    if not non_empty:
        channel_count = int(waveforms[0].shape[1]) if waveforms[0].ndim == 2 and waveforms[0].shape[1] > 0 else 1
        return np.zeros((0, channel_count), dtype=np.float32)
    return np.concatenate(non_empty, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# OnnxTtsRuntime 类：TTS 业务逻辑层（simple_moss 版本）
# ══════════════════════════════════════════════════════════════════════════════

class OnnxTtsRuntime(OrtCpuRuntime):
    """TTS 业务逻辑层，继承 OrtCpuRuntime 并在其上添加：
      - SentencePiece 文本编码；
      - 文本分句/分 chunk 策略；
      - 内置音色 prompt 解析；
      - codec 音频格式元数据查询。
    由 simple_app_onnx.py 中的 OnnxNanoTTSServiceAdapter 持有并调用。
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        thread_count: int = 4,
        max_new_frames: int | None = None,
        do_sample: bool | None = None,
        sample_mode: str | None = None,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        """确认模型目录就绪，调用父类 OrtCpuRuntime.__init__ 加载 ONNX 模型，
        然后加载 SentencePiece tokenizer。

        调用方：simple_app_onnx.py 中 OnnxNanoTTSServiceAdapter.__init__（直接实例化）。
        """
        _log_memory("OnnxTtsRuntime.__init__: before ensure_browser_onnx_model_dir")
        resolved_model_dir = ensure_browser_onnx_model_dir(model_dir)
        _log_memory("OnnxTtsRuntime.__init__: after ensure_browser_onnx_model_dir, before OrtCpuRuntime")
        super().__init__(
            model_dir=resolved_model_dir,
            thread_count=thread_count,
            max_new_frames=max_new_frames,
            do_sample=do_sample,
            sample_mode=sample_mode,
        )
        _log_memory("OnnxTtsRuntime.__init__: after OrtCpuRuntime super().__init__")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_relative_path = str(self.manifest["model_files"].get("tokenizer_model", "tokenizer.model"))
        tokenizer_path = self.resolve_manifest_relative_path(tokenizer_relative_path)
        self.sp_model = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        _log_memory("OnnxTtsRuntime.__init__: after SentencePiece tokenizer load")

    def encode_text(self, text: str) -> list[int]:
        """用 SentencePiece 将文本编码为 token id 列表，供 build_voice_clone_request_rows 使用。

        调用方：simple_app_onnx.py 的 _worker 闭包，通过 self.runtime.encode_text(chunk_text) 调用；
                本文件 count_text_tokens。
        """
        return [int(token_id) for token_id in self.sp_model.encode(str(text or ""), out_type=int)]

    def count_text_tokens(self, text: str) -> int:
        """返回文本编码后的 token 数量，用于 token budget 切分判断。

        调用方：本文件 split_text_by_token_budget 和 split_voice_clone_text（大量调用，为核心切分依据）。
        """
        return len(self.encode_text(text))

    def split_text_by_token_budget(self, text: str, max_tokens: int) -> list[str]:
        """将超过 max_tokens token 的长文本按 token 数二分切割，优先在标点边界断句。
        返回一组不超过 max_tokens 的文本片段。

        调用方：本文件 split_voice_clone_text（对超长从句进行兜底切割）。
        """
        remaining_text = str(text or "").strip()
        if not remaining_text:
            return []
        pieces: list[str] = []
        preferred_boundary_chars = set(CLAUSE_SPLIT_PUNCTUATION) | set(SENTENCE_END_PUNCTUATION) | {" "}
        while remaining_text:
            if self.count_text_tokens(remaining_text) <= max_tokens:
                pieces.append(remaining_text)
                break
            low = 1
            high = len(remaining_text)
            best_prefix_length = 1
            while low <= high:
                middle = (low + high) // 2
                candidate = remaining_text[:middle].strip()
                if not candidate:
                    low = middle + 1
                    continue
                if self.count_text_tokens(candidate) <= max_tokens:
                    best_prefix_length = middle
                    low = middle + 1
                else:
                    high = middle - 1
            cut_index = best_prefix_length
            prefix = remaining_text[:best_prefix_length]
            preferred_index = -1
            scan_min = max(-1, len(prefix) - 25)
            for scan_index in range(len(prefix) - 1, scan_min, -1):
                if prefix[scan_index] in preferred_boundary_chars:
                    preferred_index = scan_index + 1
                    break
            if preferred_index > 0:
                cut_index = preferred_index
            piece = remaining_text[:cut_index].strip()
            if not piece:
                piece = remaining_text[:best_prefix_length].strip()
                cut_index = best_prefix_length
            pieces.append(piece)
            remaining_text = remaining_text[cut_index:].strip()
        return pieces

    def split_voice_clone_text(self, text: str, max_tokens: int = 75) -> list[str]:
        """将输入文本按句末标点 → 从句标点 → token budget 三级策略分割为若干 chunk，
        每个 chunk 的 token 数不超过 max_tokens。最终结果若只有一段则直接返回原文（不分割）。

        调用方：simple_app_onnx.py 的 _worker 闭包，通过 self.runtime.split_voice_clone_text 调用；
                simple_app_onnx.py 的 OnnxNanoTTSServiceAdapter.split_voice_clone_text 包装后对外暴露。
        """
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
        safe_max_tokens = max(1, int(max_tokens))
        prepared_text = _prepare_text_for_sentence_chunking(normalized_text)
        sentence_candidates = _split_text_by_punctuation(prepared_text, SENTENCE_END_PUNCTUATION) or [prepared_text.strip()]
        sentence_slices: list[tuple[int, str]] = []
        for sentence_text in sentence_candidates:
            normalized_sentence = sentence_text.strip()
            if not normalized_sentence:
                continue
            sentence_token_count = self.count_text_tokens(normalized_sentence)
            if sentence_token_count <= safe_max_tokens:
                sentence_slices.append((sentence_token_count, normalized_sentence))
                continue
            clause_candidates = _split_text_by_punctuation(normalized_sentence, CLAUSE_SPLIT_PUNCTUATION)
            if len(clause_candidates) <= 1:
                clause_candidates = [normalized_sentence]
            for clause_text in clause_candidates:
                normalized_clause = clause_text.strip()
                if not normalized_clause:
                    continue
                clause_token_count = self.count_text_tokens(normalized_clause)
                if clause_token_count <= safe_max_tokens:
                    sentence_slices.append((clause_token_count, normalized_clause))
                    continue
                for piece in self.split_text_by_token_budget(normalized_clause, safe_max_tokens):
                    normalized_piece = piece.strip()
                    if normalized_piece:
                        sentence_slices.append((self.count_text_tokens(normalized_piece), normalized_piece))
        chunks: list[str] = []
        current_chunk = ""
        current_chunk_token_count = 0
        for sentence_token_count, sentence_text in sentence_slices:
            if not current_chunk:
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
                continue
            if current_chunk_token_count + sentence_token_count > safe_max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
            else:
                current_chunk = _join_sentence_parts(current_chunk, sentence_text)
                current_chunk_token_count = self.count_text_tokens(current_chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if len(chunks) > 1 else [normalized_text]

    def estimate_voice_clone_inter_chunk_pause_seconds(self, text_chunk: str) -> float:
        """根据当前 chunk 的词数估算下一个 chunk 前应插入的静音时长（秒）。
        短句（≤4词）后插入较长静音，长句后插入较短静音。

        调用方：simple_app_onnx.py 的 _worker 闭包，用于在相邻 text_chunk 之间生成自然的停顿静音。
        """
        word_count = len([item for item in str(text_chunk or "").strip().split() if item])
        return (
            DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS
            if word_count <= 4
            else DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS
        )

    def resolve_prompt_audio_codes(
        self,
        *,
        voice: str | None,
        prompt_audio_path: str | Path | None = None,
    ) -> list[list[int]]:
        """只支持内置 voice，忽略 prompt_audio_path（simple_moss 版本移除了参考音频编码）。
        直接委托给 resolve_builtin_voice_prompt_audio_codes。

        调用方：此方法在 simple_moss 内部未被直接调用，保留以保持与原始 OnnxTtsRuntime 的接口兼容性。
        """
        return self.resolve_builtin_voice_prompt_audio_codes(voice)

    def resolve_builtin_voice_prompt_audio_codes(self, voice: str | None) -> list[list[int]]:
        """根据音色名称在 manifest 中查找对应的 prompt_audio_codes（参考音频声学 token）。
        voice 为 None 时回退到第一个内置音色，找不到则抛出 ValueError。

        调用方：simple_app_onnx.py 的 _worker 闭包，通过 self.runtime.resolve_builtin_voice_prompt_audio_codes 调用；
                本文件 resolve_prompt_audio_codes（间接委托）。
        """
        resolved_voice = str(voice or self.list_builtin_voices()[0]["voice"])
        voice_row = next((item for item in self.list_builtin_voices() if item["voice"] == resolved_voice), None)
        if voice_row is None:
            raise ValueError(f"Built-in voice not found: {resolved_voice}")
        return list(voice_row["prompt_audio_codes"])

    def get_codec_audio_format(self) -> tuple[int, int]:
        """从 codec_meta 读取并返回 (sample_rate, channels)，供上层无需硬编码音频格式。

        调用方：simple_app_onnx.py 的 _worker 闭包，通过 self.runtime.get_codec_audio_format() 调用。
        """
        sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        channels = int(self.codec_meta["codec_config"]["channels"])
        return sample_rate, channels
