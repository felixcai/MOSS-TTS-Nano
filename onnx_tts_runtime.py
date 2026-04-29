from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# onnx_tts_runtime.py  —— TTS 业务逻辑层（Stream Generate 调用链第 3 层）
#
# 在 stream generate 调用链中的位置：
#   app.py  →  app_onnx.py (synthesize_stream / _worker)
#           →  本文件 (文本切分、音频编码、Prompt 构建)
#           →  ort_cpu_runtime.py (generate_audio_frames / codec_decode_step)
#
# 函数角色说明（仅 stream generate 视角）：
#   [调用链入口]  被 app_onnx.py 直接调用或导入的函数
#   [调用链内部]  仅在本文件内被其他函数调用的函数（支撑入口函数）
#   [非调用链]    不参与单次 stream generate 请求路径的函数
# ─────────────────────────────────────────────────────────────────────────────

import logging
import shutil
import time
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sentencepiece as spm
import torch
import torchaudio

from moss_tts_nano.defaults import DEFAULT_OUTPUT_DIR
from text_normalization_pipeline import WeTextProcessingManager, prepare_tts_request_texts

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR


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


from ort_cpu_runtime import (
    OrtCpuRuntime,
    _normalize_sample_mode,
    _resolve_stream_decode_frame_budget,
    SAMPLE_MODE_FIXED,
    SAMPLE_MODE_FULL,
    SAMPLE_MODE_GREEDY,
)

DEFAULT_BROWSER_ONNX_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_BROWSER_ONNX_TTS_DIR = DEFAULT_BROWSER_ONNX_MODEL_DIR / "MOSS-TTS-Nano-100M-ONNX"
DEFAULT_BROWSER_ONNX_CODEC_DIR = DEFAULT_BROWSER_ONNX_MODEL_DIR / "MOSS-Audio-Tokenizer-Nano-ONNX"
DEFAULT_BROWSER_ONNX_TTS_REPO_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
DEFAULT_BROWSER_ONNX_CODEC_REPO_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
DEFAULT_BROWSER_ONNX_TTS_REPO_URL = f"https://huggingface.co/{DEFAULT_BROWSER_ONNX_TTS_REPO_ID}"
DEFAULT_BROWSER_ONNX_CODEC_REPO_URL = f"https://huggingface.co/{DEFAULT_BROWSER_ONNX_CODEC_REPO_ID}"
DEFAULT_BROWSER_ONNX_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "infer_onnx_output.wav"
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24
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
# 这组函数组成 ensure_browser_onnx_model_dir 的调用树，在 __init__ 初始化阶段
# 执行，属于调用链的「初始化支路」。
# ══════════════════════════════════════════════════════════════════════════════

# [调用链内部] 被 ensure_browser_onnx_model_dir 和 _default_model_dir_requested 调用
def _resolve_model_dir_path(model_dir: str | Path | None) -> Path:
    if model_dir is None:
        return DEFAULT_BROWSER_ONNX_MODEL_DIR.expanduser().resolve()
    return Path(model_dir).expanduser().resolve()


# [调用链内部] 被 ensure_browser_onnx_model_dir 调用，判断是否使用默认目录
def _default_model_dir_requested(model_dir: str | Path | None) -> bool:
    if model_dir is None:
        return True
    return _resolve_model_dir_path(model_dir) == DEFAULT_BROWSER_ONNX_MODEL_DIR.expanduser().resolve()


# [调用链内部] 被 ensure_browser_onnx_model_dir 调用，遍历候选路径找到 manifest 文件
def _find_manifest_path(model_dir: Path) -> Path | None:
    for relative_path in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS:
        candidate = (model_dir / relative_path).resolve()
        if candidate.is_file():
            return candidate
    return None


# [调用链内部] 被 _find_directory_with_required_names 调用
def _directory_contains_all(parent: Path, required_names: Sequence[str]) -> bool:
    return all((parent / name).exists() for name in required_names)


# [调用链内部] 被 _normalize_download_layout 调用，在下载后的目录树中找到包含所有
# 必要文件的子目录（解决 HuggingFace snapshot_download 可能带来的嵌套目录问题）
def _find_directory_with_required_names(root_dir: Path, required_names: Sequence[str]) -> Path | None:
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


# [调用链内部] 被 _normalize_download_layout 调用，把嵌套子目录的文件提升到目标目录
def _promote_directory_contents(source_dir: Path, target_dir: Path) -> None:
    if source_dir.resolve() == target_dir.resolve():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if destination.exists():
            continue
        shutil.move(str(child), str(destination))


# [调用链内部] 被 _download_default_browser_onnx_assets 调用，整理下载后的目录布局
def _normalize_download_layout(target_dir: Path, required_names: Sequence[str]) -> None:
    candidate_dir = _find_directory_with_required_names(target_dir, required_names)
    if candidate_dir is None:
        return
    _promote_directory_contents(candidate_dir, target_dir)


# [调用链内部] 被 _download_default_browser_onnx_assets 调用，封装 huggingface_hub
# 的 snapshot_download，按 allow_patterns 只下载必要文件
def _snapshot_download_repo(
    *,
    repo_id: str,
    local_dir: Path,
    allow_patterns: Sequence[str],
) -> None:
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


# [调用链内部] 被 ensure_browser_onnx_model_dir 调用，当默认模型目录不存在时
# 分别下载 TTS 模型和 Codec 模型，并整理目录结构
def _download_default_browser_onnx_assets(model_dir: Path) -> None:
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


# [调用链内部] 被 OnnxTtsRuntime.__init__ 调用
# 快速路径：若 manifest 已存在则直接返回；否则（仅允许默认目录）触发自动下载
def ensure_browser_onnx_model_dir(model_dir: str | Path | None = None) -> Path:
    resolved_model_dir = _resolve_model_dir_path(model_dir)
    # 快速路径：manifest 已就位，无需下载
    manifest_path = _find_manifest_path(resolved_model_dir)
    if manifest_path is not None:
        return resolved_model_dir
    # 非默认目录时不尝试自动下载，直接报错
    if not _default_model_dir_requested(model_dir):
        tried_paths = [str((resolved_model_dir / item).resolve()) for item in MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS]
        raise FileNotFoundError(
            "browser_onnx model assets not found under the provided --model-dir. tried: " + ", ".join(tried_paths)
        )
    # 默认目录 + 文件缺失：从 HuggingFace 自动下载 TTS 和 Codec ONNX 资产
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
# 这组函数组成 split_voice_clone_text 的调用树，负责将原始用户文本按句/分句/
# token 预算切分为多个 text_chunk，是 stream generate 前置数据预处理的一部分。
# ══════════════════════════════════════════════════════════════════════════════

# [调用链内部] 被 _prepare_text_for_sentence_chunking 和 _join_sentence_parts 调用
# 检测文本是否含 CJK 字符，以选择不同的标点和拼接规则
def _contains_cjk(text: str) -> bool:
    for character in str(text or ""):
        if (
            "\u4e00" <= character <= "\u9fff"
            or "\u3400" <= character <= "\u4dbf"
            or "\u3040" <= character <= "\u30ff"
            or "\uac00" <= character <= "\ud7af"
        ):
            return True
    return False


# [调用链内部] 被 split_voice_clone_text 调用，对原始文本做清理：
# 去除多余空格、补全末尾标点、对短英文文本补前导空格（避免分词歧义）
def _prepare_text_for_sentence_chunking(text: str) -> str:
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


# [调用链内部] 被 split_voice_clone_text 调用，按指定标点集对文本做第一/二层切分
# 同时处理右括号等关闭标点（将其保留在当前句末，不拆到下一句）
def _split_text_by_punctuation(text: str, punctuation: set[str]) -> list[str]:
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


# [调用链内部] 被 split_voice_clone_text 调用，拼接两段文本时根据是否 CJK 决定是否
# 插入空格（中文无需空格，英文需要）
def _join_sentence_parts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if _contains_cjk(left) or _contains_cjk(right):
        return left + right
    return f"{left} {right}"


# ══════════════════════════════════════════════════════════════════════════════
# 音频工具函数（被 app_onnx.py 直接导入使用）
# ══════════════════════════════════════════════════════════════════════════════

# [调用链入口] 被 app_onnx.py 直接导入并调用（在 _decode_pending 闭包中）
# 将 codec 解码输出的多个声道数组（各 shape=(samples,)）堆叠为 (samples, channels) 波形
# 同时对多声道情况裁剪到最短声道长度，保证各维度对齐
def _merge_audio_channels(channel_arrays: list[np.ndarray]) -> np.ndarray:
    if not channel_arrays:
        return np.zeros((0, 1), dtype=np.float32)
    if len(channel_arrays) == 1:
        return np.asarray(channel_arrays[0], dtype=np.float32).reshape(-1, 1)
    min_length = min(int(channel.shape[0]) for channel in channel_arrays)
    trimmed = [np.asarray(channel[:min_length], dtype=np.float32) for channel in channel_arrays]
    return np.stack(trimmed, axis=1)


# [调用链入口] 被 app_onnx.py 直接导入并调用（在 _worker 末尾）
# 把某个 text_chunk 所有流式解码输出的小段波形（emitted_chunks）拼接为完整波形
def _concat_waveforms(waveforms: list[np.ndarray]) -> np.ndarray:
    if not waveforms:
        return np.zeros((0, 1), dtype=np.float32)
    non_empty = [waveform for waveform in waveforms if waveform.size > 0]
    if not non_empty:
        channel_count = int(waveforms[0].shape[1]) if waveforms[0].ndim == 2 and waveforms[0].shape[1] > 0 else 1
        return np.zeros((0, channel_count), dtype=np.float32)
    return np.concatenate(non_empty, axis=0)


# [调用链入口] 被 app_onnx.py 直接导入并调用（在 _worker 最后阶段）
# 在所有 text_chunk 处理完毕后，把最终完整波形写入磁盘（app_onnx_stream_output.wav）
def _write_waveform_to_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(int(pcm16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# OnnxTtsRuntime 类：TTS 业务逻辑层，继承自 OrtCpuRuntime（底层 ONNX 推理层）
# ══════════════════════════════════════════════════════════════════════════════

class OnnxTtsRuntime(OrtCpuRuntime):

    # [调用链入口] 被 app_onnx.py 的 OnnxNanoTTSServiceAdapter.__init__ 调用
    # 初始化顺序：
    #   1. ensure_browser_onnx_model_dir  → 确认/下载 ONNX 模型文件
    #   2. super().__init__               → 读取 manifest/meta、创建全部 ONNX InferenceSession
    #   3. 加载 SentencePiece tokenizer   → 用于 encode_text 的文本分词
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
        # 确认模型目录有效，必要时触发 HuggingFace 下载
        _log_memory("OnnxTtsRuntime.__init__: before ensure_browser_onnx_model_dir")
        resolved_model_dir = ensure_browser_onnx_model_dir(model_dir)
        _log_memory("OnnxTtsRuntime.__init__: after ensure_browser_onnx_model_dir, before OrtCpuRuntime")
        # 调用父类 OrtCpuRuntime 完成所有 ONNX InferenceSession 的创建
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
        # 从 manifest 解析 tokenizer 路径并加载 SentencePiece 模型
        tokenizer_relative_path = str(self.manifest["model_files"].get("tokenizer_model", "tokenizer.model"))
        tokenizer_path = self.resolve_manifest_relative_path(tokenizer_relative_path)
        self.sp_model = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        _log_memory("OnnxTtsRuntime.__init__: after SentencePiece tokenizer load")
        self._text_normalizer_manager: WeTextProcessingManager | None = None

    # [非调用链] 仅被 prepare_synthesis_text 调用，而 prepare_synthesis_text
    # 只在非 stream 路径的 synthesize 方法中使用
    def _ensure_text_normalizer(self, enable_wetext: bool) -> WeTextProcessingManager | None:
        if not enable_wetext:
            return None
        if self._text_normalizer_manager is None:
            self._text_normalizer_manager = WeTextProcessingManager()
        snapshot = self._text_normalizer_manager.ensure_ready()
        if not snapshot.ready:
            raise RuntimeError(snapshot.error or snapshot.message)
        return self._text_normalizer_manager

    # [调用链入口] 被 app_onnx.py 的 _worker 线程调用（每个 text_chunk 调用一次）
    # 使用 SentencePiece 将文本转换为 token ID 列表，供 build_voice_clone_request_rows 使用
    def encode_text(self, text: str) -> list[int]:
        return [int(token_id) for token_id in self.sp_model.encode(str(text or ""), out_type=int)]

    # [调用链内部] 被 split_voice_clone_text 和 split_text_by_token_budget 调用
    # 通过实际 encode 计算 token 数量，用于判断分块是否超出预算
    def count_text_tokens(self, text: str) -> int:
        return len(self.encode_text(text))

    # [非调用链] 仅被非 stream 路径的 synthesize 调用，用于文本正则化预处理
    def prepare_synthesis_text(
        self,
        *,
        text: str,
        voice: str = "",
        prompt_text: str = "",
        enable_wetext: bool = True,
        enable_normalize_tts_text: bool = True,
    ) -> dict[str, object]:
        text_normalizer_manager = self._ensure_text_normalizer(enable_wetext)
        return prepare_tts_request_texts(
            text=text,
            prompt_text=prompt_text,
            voice=voice,
            enable_wetext=bool(enable_wetext),
            enable_normalize_tts_text=bool(enable_normalize_tts_text),
            text_normalizer_manager=text_normalizer_manager,
        )

    # [调用链内部] 被 split_voice_clone_text 调用（当单个分句/子句仍超出 token 预算时）
    # 用二分查找找到不超出 max_tokens 的最长字符前缀，并在最近的边界字符处截断
    def split_text_by_token_budget(self, text: str, max_tokens: int) -> list[str]:
        remaining_text = str(text or "").strip()
        if not remaining_text:
            return []
        pieces: list[str] = []
        preferred_boundary_chars = set(CLAUSE_SPLIT_PUNCTUATION) | set(SENTENCE_END_PUNCTUATION) | {" "}
        while remaining_text:
            if self.count_text_tokens(remaining_text) <= max_tokens:
                pieces.append(remaining_text)
                break
            # 二分查找满足 token 预算的最长字符前缀长度
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
            # 在最大前缀末尾的 25 个字符内，向前扫描是否有自然边界字符，优先在此断开
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

    # [调用链入口] 被 app_onnx.py 的 _worker 线程调用，是文本预处理的核心
    # 三层切分策略：
    #   第一层：按句末标点（。！？等）切分为独立句子
    #   第二层：若单句超出 token 预算，则进一步按子句标点（，、等）切分
    #   第三层：若子句仍超出预算，则调用 split_text_by_token_budget 做 token 级二分切割
    # 切分后再按 token 预算贪心合并相邻小片段，减少总 chunk 数量
    # 特殊处理：若最终只切出一个 chunk，则返回原始文本（避免过度切分改变语义）
    def split_voice_clone_text(self, text: str, max_tokens: int = 75) -> list[str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
        safe_max_tokens = max(1, int(max_tokens))
        # 清理文本并补全末尾标点，确保句子能被正确识别
        prepared_text = _prepare_text_for_sentence_chunking(normalized_text)
        # 第一层：按句末标点切分
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
            # 第二层：句子过长，按子句标点进一步切分
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
                # 第三层：子句仍超出预算，使用二分查找强制截断
                for piece in self.split_text_by_token_budget(normalized_clause, safe_max_tokens):
                    normalized_piece = piece.strip()
                    if normalized_piece:
                        sentence_slices.append((self.count_text_tokens(normalized_piece), normalized_piece))
        # 贪心合并：将相邻小片段合并，直到合并后超出 token 预算为止
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
        # 若只切出一个 chunk，返回原始文本而非处理后文本，保持原始语义
        return chunks if len(chunks) > 1 else [normalized_text]

    # [调用链入口] 被 app_onnx.py 的 _worker 线程调用（处理相邻两个 text_chunk 之间）
    # 根据 text_chunk 的单词数决定片段间静音时长：
    #   ≤4 个词（短句/标题类）→ 较长停顿（0.40s），短句本身时长短，需要更长间隔补偿呼吸感
    #   >4 个词（正常句子）→ 较短停顿（0.24s），长句已有足够时长，短停顿维持语流连贯即可
    def estimate_voice_clone_inter_chunk_pause_seconds(self, text_chunk: str) -> float:
        word_count = len([item for item in str(text_chunk or "").strip().split() if item])
        return (
            DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS
            if word_count <= 4
            else DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS
        )

    # [调用链内部] 被 encode_reference_audio 调用
    # 加载参考音频并做归一化：重采样到 codec 要求的采样率，转换到目标声道数
    def _load_reference_audio(self, reference_audio_path: str | Path) -> np.ndarray:
        waveform, sample_rate = torchaudio.load(str(Path(reference_audio_path).expanduser().resolve()))
        waveform = waveform.to(torch.float32)
        target_sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        target_channels = int(self.codec_meta["codec_config"]["channels"])
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        current_channels = int(waveform.shape[0])
        if current_channels == target_channels:
            pass
        elif current_channels == 1 and target_channels > 1:
            waveform = waveform.repeat(target_channels, 1)
        elif current_channels > 1 and target_channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported reference audio channel conversion: {current_channels} -> {target_channels}")
        return waveform.unsqueeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    # [调用链内部] 被 resolve_prompt_audio_codes 调用（当传入自定义参考音频路径时）
    # 调用 ONNX codec_encode Session 将参考音频波形编码为离散声学 token 序列
    # 返回值 prompt_audio_codes: list[list[int]]，shape 为 [frames, num_quantizers]
    def encode_reference_audio(self, reference_audio_path: str | Path) -> list[list[int]]:
        waveform = self._load_reference_audio(reference_audio_path)
        waveform_length = int(waveform.shape[-1])
        # 运行 ONNX codec_encode 会话，输入为波形张量和长度
        outputs = self.sessions["codec_encode"].run(
            None,
            {
                "waveform": waveform,
                "input_lengths": np.asarray([waveform_length], dtype=np.int32),
            },
        )
        # 通过 output name 映射取结果，避免依赖输出顺序
        output_names = [output.name for output in self.sessions["codec_encode"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_codes = np.asarray(named_outputs["audio_codes"], dtype=np.int32)
        audio_code_lengths = np.asarray(named_outputs["audio_code_lengths"], dtype=np.int32)
        # code_length 表示有效帧数（时间维度），裁剪掉 padding 部分
        code_length = int(audio_code_lengths.reshape(-1)[0])
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        # 将 [1, frames, quantizers] 张量转换为 list[list[int]] 格式
        prompt_audio_codes: list[list[int]] = []
        for frame_index in range(code_length):
            prompt_audio_codes.append(
                [int(audio_codes[0, frame_index, quantizer_index]) for quantizer_index in range(num_quantizers)]
            )
        return prompt_audio_codes

    # [调用链入口] 被 app_onnx.py 的 _worker 线程调用，是 Prompt 音频准备的统一入口
    # 两个分支：
    #   有 prompt_audio_path → 编码自定义参考音频（encode_reference_audio）
    #   无 prompt_audio_path → 从 manifest 内置音色列表中查找对应 voice 的预编码 codes
    def resolve_prompt_audio_codes(
        self,
        *,
        voice: str | None,
        prompt_audio_path: str | Path | None,
    ) -> list[list[int]]:
        if prompt_audio_path:
            # 用户提供了自定义参考音频，实时编码
            return self.encode_reference_audio(prompt_audio_path)
        # 使用内置音色：从 manifest 预编码的 prompt_audio_codes 中读取
        resolved_voice = str(voice or self.list_builtin_voices()[0]["voice"])
        voice_row = next((item for item in self.list_builtin_voices() if item["voice"] == resolved_voice), None)
        if voice_row is None:
            raise ValueError(f"Built-in voice not found: {resolved_voice}")
        return list(voice_row["prompt_audio_codes"])

    # ──────────────────────────────────────────────────────────────────────────
    # 以下函数不参与 stream generate 路径
    # ──────────────────────────────────────────────────────────────────────────

    # [非调用链] 仅被 synthesize_single_chunk 调用（通过 synthesize 的非 stream 路径）
    # 对完整 generated_frames 做一次性 codec 全量解码；stream 路径使用
    # CodecStreamingDecodeSession.run_frames 逐帧解码，不走此函数
    def decode_full_audio_safe(self, generated_frames: list[list[int]]) -> np.ndarray:
        try:
            channel_arrays, _audio_length = self.decode_full_audio(generated_frames)
            return _merge_audio_channels(channel_arrays)
        except Exception as exc:
            logging.warning("full codec decode failed, falling back to incremental decode: %s", exc)
            self.codec_streaming_session.reset()
            merged_by_channel: list[list[np.ndarray]] = [
                [] for _ in range(int(self.codec_meta["codec_config"]["channels"]))
            ]
            try:
                for start_index in range(0, len(generated_frames), 8):
                    frame_chunk = generated_frames[start_index : start_index + 8]
                    decoded = self.codec_streaming_session.run_frames(frame_chunk)
                    if decoded is None:
                        continue
                    audio, audio_length = decoded
                    if audio_length <= 0:
                        continue
                    for channel_index, channel in enumerate(audio[0, :, :audio_length]):
                        merged_by_channel[channel_index].append(np.asarray(channel, dtype=np.float32))
            finally:
                self.codec_streaming_session.reset()
            return _merge_audio_channels(
                [np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32) for chunks in merged_by_channel]
            )

    # [非调用链] 仅被 synthesize 调用
    # 注意：此方法内部包含与 app_onnx.py _worker 类似的流式解码逻辑（on_frame 回调），
    # 但 app_onnx.py 的 stream generate 路径绕过了此方法，直接在 _worker 中
    # 调用 encode_text / build_voice_clone_request_rows / generate_audio_frames。
    # synthesize_single_chunk 的流式分支仅供独立调用 synthesize(streaming=True) 时使用。
    def synthesize_single_chunk(
        self,
        *,
        text: str,
        prompt_audio_codes: list[list[int]],
        streaming: bool,
        is_first_streaming_chunk: bool = True,
    ) -> dict[str, Any]:
        text_token_ids = self.encode_text(text)
        request_rows = self.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
        if not streaming:
            generated_frames = self.generate_audio_frames(request_rows)
            waveform = self.decode_full_audio_safe(generated_frames)
            return {
                "text": text,
                "text_token_ids": text_token_ids,
                "generated_frames": generated_frames,
                "waveform": waveform,
            }

        pending_decode_frames: list[list[int]] = []
        emitted_chunks: list[np.ndarray] = []
        emitted_samples_total = 0
        first_audio_emitted_at_perf: float | None = None
        self.codec_streaming_session.reset()

        chunk_t0 = time.perf_counter()
        last_end = chunk_t0
        rtf_pending_lead_gen_s = 0.0
        rtf_first_gen_s: float | None = None
        rtf_first_audio_s: float | None = None
        rtf_steady_gen_s_sum = 0.0
        rtf_steady_audio_s_sum = 0.0
        rtf_audio_chunk_count = 0

        def decode_pending_frames(force: bool) -> None:
            nonlocal emitted_samples_total, first_audio_emitted_at_perf
            nonlocal last_end, rtf_pending_lead_gen_s, rtf_first_gen_s, rtf_first_audio_s
            nonlocal rtf_steady_gen_s_sum, rtf_steady_audio_s_sum, rtf_audio_chunk_count
            pending_count = len(pending_decode_frames)
            if pending_count <= 0:
                return
            sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
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
            decoded = self.codec_streaming_session.run_frames(frame_chunk)
            if decoded is None:
                return
            audio, audio_length = decoded
            if audio_length <= 0:
                return
            now = time.perf_counter()
            gen_time_s = now - last_end
            waveform = _merge_audio_channels([audio[0, channel_index, :audio_length] for channel_index in range(audio.shape[1])])
            event_duration_seconds = (
                float(waveform.shape[0]) / float(sample_rate) if waveform.ndim >= 1 and sample_rate > 0 else 0.0
            )
            if first_audio_emitted_at_perf is None:
                first_audio_emitted_at_perf = now
            emitted_samples_total += audio_length
            emitted_chunks.append(waveform)

            if is_first_streaming_chunk:
                if event_duration_seconds > 1e-9:
                    rtf_audio_chunk_count += 1
                    if rtf_first_gen_s is None:
                        rtf_first_gen_s = rtf_pending_lead_gen_s + gen_time_s
                        rtf_first_audio_s = event_duration_seconds
                        rtf_pending_lead_gen_s = 0.0
                    else:
                        rtf_steady_gen_s_sum += gen_time_s
                        rtf_steady_audio_s_sum += event_duration_seconds
                elif rtf_first_gen_s is None:
                    rtf_pending_lead_gen_s += gen_time_s
            else:
                if event_duration_seconds > 1e-9:
                    rtf_audio_chunk_count += 1
                    rtf_steady_gen_s_sum += gen_time_s
                    rtf_steady_audio_s_sum += event_duration_seconds
            last_end = now

        def on_frame(_generated_frames: list[list[int]], _step_index: int, frame: list[int]) -> None:
            pending_decode_frames.append(list(frame))
            decode_pending_frames(False)

        try:
            generated_frames = self.generate_audio_frames(request_rows, on_frame=on_frame)
            decode_pending_frames(True)
        finally:
            self.codec_streaming_session.reset()
        waveform = _concat_waveforms(emitted_chunks)
        first_audio_latency_s = (
            max(0.0, first_audio_emitted_at_perf - chunk_t0) if first_audio_emitted_at_perf is not None else None
        )
        rtf_first = (
            rtf_first_gen_s / rtf_first_audio_s
            if rtf_first_gen_s is not None
            and rtf_first_audio_s is not None
            and rtf_first_audio_s > 1e-9
            else None
        )
        stream_metrics: dict[str, Any] = {
            "audio_chunk_count": rtf_audio_chunk_count,
            "first_audio_latency_s": first_audio_latency_s,
            "rtf_first": rtf_first,
            "steady_gen_s_sum": rtf_steady_gen_s_sum,
            "steady_audio_s_sum": rtf_steady_audio_s_sum,
        }
        return {
            "text": text,
            "text_token_ids": text_token_ids,
            "generated_frames": generated_frames,
            "waveform": waveform,
            "stream_metrics": stream_metrics,
        }

    # [非调用链] 被 app_onnx.py 的 OnnxNanoTTSServiceAdapter.synthesize 调用（非 stream 路径）
    # 完整合成流程：文本正则化 → 文本切分 → 逐 chunk 推理 → 拼接音频 → 写入磁盘
    # stream generate 路径不经过此函数，app_onnx.py 的 synthesize_stream._worker 直接
    # 调用 resolve_prompt_audio_codes / split_voice_clone_text / encode_text 等底层方法
    def synthesize(
        self,
        *,
        text: str,
        voice: str | None = None,
        prompt_audio_path: str | Path | None = None,
        output_audio_path: str | Path | None = None,
        sample_mode: str | None = None,
        do_sample: bool = True,
        streaming: bool = False,
        max_new_frames: int | None = None,
        voice_clone_max_text_tokens: int = 75,
        enable_wetext: bool = True,
        enable_normalize_tts_text: bool = True,
        seed: int | None = None,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
        if max_new_frames is not None:
            self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        normalized_sample_mode = _normalize_sample_mode(sample_mode, do_sample)
        self.manifest["generation_defaults"]["sample_mode"] = normalized_sample_mode
        self.manifest["generation_defaults"]["do_sample"] = normalized_sample_mode != SAMPLE_MODE_GREEDY
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        prepared_texts = self.prepare_synthesis_text(
            text=text,
            voice=str(voice or ""),
            enable_wetext=enable_wetext,
            enable_normalize_tts_text=enable_normalize_tts_text,
        )
        prepared_text = str(prepared_texts["text"])
        prompt_audio_codes = self.resolve_prompt_audio_codes(voice=voice, prompt_audio_path=prompt_audio_path)
        text_chunks = self.split_voice_clone_text(prepared_text, max_tokens=int(voice_clone_max_text_tokens))
        t_before_first_chunk = time.perf_counter()
        all_waveforms: list[np.ndarray] = []
        all_generated_frames: list[list[int]] = []
        sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        channels = int(self.codec_meta["codec_config"]["channels"])
        chunk_results: list[dict[str, Any]] = []
        streaming_flag = bool(streaming)
        merged_stream_audio_chunks = 0
        merged_steady_gen_s = 0.0
        merged_steady_audio_s = 0.0
        merged_rtf_first: float | None = None
        merged_first_audio_latency_s: float | None = None
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_result = self.synthesize_single_chunk(
                text=chunk_text,
                prompt_audio_codes=prompt_audio_codes,
                streaming=streaming_flag,
                is_first_streaming_chunk=(chunk_index == 0),
            )
            chunk_results.append(chunk_result)
            all_waveforms.append(np.asarray(chunk_result["waveform"], dtype=np.float32))
            all_generated_frames.extend(chunk_result["generated_frames"])
            sm = chunk_result.get("stream_metrics")
            if streaming_flag and isinstance(sm, dict):
                merged_stream_audio_chunks += int(sm.get("audio_chunk_count", 0))
                merged_steady_gen_s += float(sm.get("steady_gen_s_sum", 0.0))
                merged_steady_audio_s += float(sm.get("steady_audio_s_sum", 0.0))
                if chunk_index == 0:
                    merged_rtf_first = sm.get("rtf_first")
                    local_lat = sm.get("first_audio_latency_s")
                    if local_lat is not None:
                        merged_first_audio_latency_s = max(0.0, (t_before_first_chunk - t_start) + float(local_lat))
            if chunk_index < len(text_chunks) - 1:
                pause_seconds = self.estimate_voice_clone_inter_chunk_pause_seconds(chunk_text)
                pause_samples = max(0, int(round(sample_rate * pause_seconds)))
                if pause_samples > 0:
                    all_waveforms.append(np.zeros((pause_samples, channels), dtype=np.float32))
        waveform = _concat_waveforms(all_waveforms)
        resolved_output_audio_path = (
            Path(output_audio_path).expanduser().resolve()
            if output_audio_path
            else (self.output_dir / DEFAULT_BROWSER_ONNX_OUTPUT_PATH.name).resolve()
        )
        audio_path = _write_waveform_to_wav(resolved_output_audio_path, waveform, sample_rate)
        elapsed_seconds = time.perf_counter() - t_start
        waveform_1d_samples = int(waveform.shape[0]) if waveform.ndim >= 1 else 0
        total_audio_s = waveform_1d_samples / float(sample_rate) if sample_rate > 0 else 0.0
        if streaming_flag:
            rtf_steady = merged_steady_gen_s / merged_steady_audio_s if merged_steady_audio_s > 1e-9 else None
            rtf_metrics: dict[str, Any] = {
                "streaming": True,
                "audio_chunks": merged_stream_audio_chunks,
                "total_audio_s": total_audio_s,
                "first_audio_latency_s": merged_first_audio_latency_s,
                "rtf_first": merged_rtf_first,
                "rtf_steady": rtf_steady,
                "elapsed_seconds": elapsed_seconds,
            }
        else:
            gen_rtf_first = elapsed_seconds / total_audio_s if total_audio_s > 1e-9 else None
            rtf_metrics = {
                "streaming": False,
                "audio_chunks": 1,
                "total_audio_s": total_audio_s,
                "first_audio_latency_s": elapsed_seconds,
                "rtf_first": gen_rtf_first,
                "rtf_steady": None,
                "elapsed_seconds": elapsed_seconds,
            }
        return {
            "audio_path": str(audio_path),
            "waveform": waveform,
            "sample_rate": sample_rate,
            "audio_token_ids": np.asarray(all_generated_frames, dtype=np.int32),
            "text_chunks": text_chunks,
            "prepared_texts": prepared_texts,
            "sample_mode": normalized_sample_mode,
            "do_sample": normalized_sample_mode != SAMPLE_MODE_GREEDY,
            "streaming": streaming_flag,
            "chunk_results": chunk_results,
            "rtf_metrics": rtf_metrics,
        }
