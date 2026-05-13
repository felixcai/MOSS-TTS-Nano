from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeInitConfig:
    """Defaults used when constructing the service runtime/adapter.

    Current simple_moss status: all fields in this config have confirmed read
    paths in the current service/runtime stack.
    """

    model_dir: str | Path | None = None
    output_dir: str | Path | None = None
    cpu_threads: int = 1
    enable_wetext: bool = True


@dataclass(frozen=True)
class VoiceConfig:
    """Default service-level voice used for text normalization, stream generation, and warmup.

    Current simple_moss status: confirmed effective for normalization and stream
    generation fallback. Warmup also reads this default voice before applying
    its own builtin-voice fallback.
    """

    default_voice: str = "Xiaoyu"


@dataclass(frozen=True)
class StreamGenerateConfig:
    """Default values for per-request stream generation parameters.

    Current simple_moss status:
    - Confirmed effective: max_new_frames, voice_clone_max_text_tokens,
      attn_implementation, do_sample, seed, chunk_pause_seconds,
      short_frame_num.
    - Not confirmed / likely not effective in the current fixed-only runtime:
      text_temperature, text_top_p, text_top_k, audio_temperature,
      audio_top_p, audio_top_k, audio_repetition_penalty. These values are
      written into runtime.manifest["generation_defaults"], but no matching
      read path was found in simple_moss's current fixed sampling path.
    - attn_implementation / do_sample are only partially supported today:
      the current simple_moss runtime only has a confirmed fixed-sampling path.
    """

    max_new_frames: int = 375
    voice_clone_max_text_tokens: int = 32
    attn_implementation: str = "model_default"
    do_sample: bool = True
    text_temperature: float = 1.0  # Not confirmed effective in current simple_moss path.
    text_top_p: float = 1.0  # Not confirmed effective in current simple_moss path.
    text_top_k: int = 50  # Not confirmed effective in current simple_moss path.
    audio_temperature: float = 0.8  # Not confirmed effective in current simple_moss path.
    audio_top_p: float = 0.95  # Not confirmed effective in current simple_moss path.
    audio_top_k: int = 25  # Not confirmed effective in current simple_moss path.
    audio_repetition_penalty: float = 1.2  # Not confirmed effective in current simple_moss path.
    seed: int | None = 123
    chunk_pause_seconds: float = 1.5
    short_frame_num: int = 6
    prompt_tail_keep_frames: int = 60
    do_shrink_after_chunk: bool = True


@dataclass(frozen=True)
class WarmupConfig:
    """Warmup-specific defaults kept separate for later debugging/tuning.

    Current simple_moss status: all fields in this config have confirmed read
    paths in the current warmup flow.
    """

    warmup_max_new_frames: int = 16
    warmup_text_sample_index: int = 0


DEFAULT_RUNTIME_INIT_CONFIG = RuntimeInitConfig()
DEFAULT_VOICE_CONFIG = VoiceConfig()
DEFAULT_STREAM_GENERATE_CONFIG = StreamGenerateConfig()
DEFAULT_WARMUP_CONFIG = WarmupConfig()


__all__ = [
    "DEFAULT_RUNTIME_INIT_CONFIG",
    "DEFAULT_STREAM_GENERATE_CONFIG",
    "DEFAULT_VOICE_CONFIG",
    "DEFAULT_WARMUP_CONFIG",
    "RuntimeInitConfig",
    "StreamGenerateConfig",
    "VoiceConfig",
    "WarmupConfig",
]
