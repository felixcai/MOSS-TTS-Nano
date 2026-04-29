# -*- coding: utf-8 -*-
"""Local Qwen3 TTS model implementation using faster-qwen3-tts backend."""
from __future__ import annotations

import os
from typing import Any, AsyncGenerator, Literal
import logging
import io
import base64
import numpy as np
from time import perf_counter
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_REF_TEXT = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."
PIPELINE_SR = 24000

ModelHubKind = Literal["modelscope", "huggingface", "none"]


def _default_modelscope_cache_dir() -> str:
    return os.environ.get(
        "MODELSCOPE_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub"),
    )


def _resolve_pretrained_path(
    model_name: str,
    model_hub: str,
    modelscope_cache_dir: str | None,
) -> str:
    """Resolve a ModelScope / Hugging Face id to a local directory for ``from_pretrained``.

    Local filesystem paths are returned unchanged. ``FasterQwen3TTS.from_pretrained`` accepts
    either a Hub id or a local snapshot directory.
    """
    if os.path.isdir(model_name):
        return model_name

    hub = (model_hub or "modelscope").lower()
    if hub == "none":
        return model_name

    if hub == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as e:
            raise ImportError(
                "modelscope is required when model_hub='modelscope'. "
                "Install with: pip install modelscope"
            ) from e
        cache_dir = modelscope_cache_dir or _default_modelscope_cache_dir()
        try:
            path = snapshot_download(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            logger.info("Using ModelScope cache for %s -> %s", model_name, path)
            return path
        except ValueError as e:
            if "local_files_only" not in str(e) and "cached path" not in str(e):
                raise
            logger.info(
                "ModelScope cache incomplete for %s; downloading from ModelScope",
                model_name,
            )
            return snapshot_download(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False,
            )

    if hub == "huggingface":
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required when model_hub='huggingface'. "
                "Install with: pip install huggingface_hub"
            ) from e
        try:
            path = snapshot_download(model_name, local_files_only=True)
            logger.info("Using Hugging Face Hub cache for %s -> %s", model_name, path)
            return path
        except LocalEntryNotFoundError:
            logger.info(
                "Hugging Face Hub cache incomplete for %s; downloading",
                model_name,
            )
            return snapshot_download(model_name, local_files_only=False)

    raise ValueError(
        "model_hub must be 'modelscope', 'huggingface', or 'none', "
        f"got {model_hub!r}"
    )


class LocalQwenTTSModel:
    """Local Qwen3 TTS model implementation using faster-qwen3-tts backend.
    
    This model runs locally on NVIDIA GPUs for real-time TTS performance.
    Requires faster-qwen3-tts package to be installed.
    
    Supports three generation modes:
      - Voice cloning (ref_audio + ref_text)
      - Custom voice (preset speakers)
      - Voice design (instruct prompt)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: str = "auto",
        attn_implementation: str = "eager",
        ref_audio: str | None = None,
        ref_text: str = DEFAULT_REF_TEXT,
        language: str = "Chinese",
        speaker: str | None = None,
        instruct: str | None = None,
        xvec_only: bool = False,
        parity_mode: bool = False,
        streaming_chunk_size: int = 8,
        max_new_tokens: int = 360,
        blocksize: int = 512,
        stream: bool = True,
        model_hub: ModelHubKind = "modelscope",
        modelscope_cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Local Qwen3 TTS model.

        Args:
            model_name (`str`, defaults to "Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
                ModelScope / Hugging Face model id, or a local directory path.
            device (`str`, defaults to "cuda"):
                The device to run the model on ("cuda" or "cpu").
            dtype (`str`, defaults to "auto"):
                The data type for model weights ("auto", "bfloat16", "float16", "float32").
            attn_implementation (`str`, defaults to "eager"):
                The attention implementation to use.
            ref_audio (`str | None`, optional):
                Path to reference audio file for voice cloning.
            ref_text (`str`, defaults to DEFAULT_REF_TEXT):
                Reference text corresponding to the reference audio.
            language (`str`, defaults to "English"):
                The language of the text to synthesize.
            speaker (`str | None`, optional):
                Speaker name for custom voice generation.
            instruct (`str | None`, optional):
                Instruction prompt for voice design mode.
            xvec_only (`bool`, defaults to False):
                Whether to use only x-vector for voice cloning.
            parity_mode (`bool`, defaults to False):
                Whether to enable parity mode (skips CUDA graph capture).
            streaming_chunk_size (`int`, defaults to 8):
                The chunk size for streaming generation.
            max_new_tokens (`int`, defaults to 360):
                Maximum number of new tokens to generate.
            blocksize (`int`, defaults to 512):
                The audio block size for output.
            stream (`bool`, defaults to True):
                Whether to use streaming synthesis.
            model_hub (`str`, defaults to "modelscope"):
                ``modelscope`` — resolve via ModelScope ``snapshot_download`` then load;
                ``huggingface`` — Hugging Face Hub; ``none`` — pass ``model_name`` through.
            modelscope_cache_dir (`str | None`, optional):
                ModelScope cache directory (default ``MODELSCOPE_CACHE`` or ``~/.cache/modelscope/hub``).
            **kwargs (`Any`):
                Additional keyword arguments passed to the model.
        """
        self.model_name = model_name
        self.model_hub: ModelHubKind = model_hub  # type: ignore[assignment]
        self.modelscope_cache_dir = modelscope_cache_dir
        self.stream = stream
        self.device = device
        self.dtype_str = dtype
        self.attn_implementation = attn_implementation
        self.ref_audio = ""
        if ref_audio:
            self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.speaker = speaker
        self.instruct = instruct
        self.xvec_only = xvec_only
        self.parity_mode = parity_mode
        self.streaming_chunk_size = streaming_chunk_size
        self.max_new_tokens = max_new_tokens
        self.blocksize = blocksize
        self.model_kwargs = kwargs

        # Model instance will be loaded in setup
        self._model = None
        self._torch = None
        self._dtype = None

        # Setup the model
        self._setup()

    def _setup(self) -> None:
        """Setup the TTS model and load resources."""
        try:
            import torch
            self._torch = torch
        except ImportError as e:
            raise ImportError("torch is required. Install with: pip install torch") from e

        # Determine dtype
        if self.dtype_str == "auto":
            self._dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif isinstance(self.dtype_str, str):
            self._dtype = getattr(torch, self.dtype_str)
        else:
            self._dtype = self.dtype_str

        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as e:
            raise ImportError(
                "faster-qwen3-tts is required for Local Qwen3 TTS. "
                "Install with: pip install faster-qwen3-tts"
            ) from e

        load_path = _resolve_pretrained_path(
            self.model_name,
            self.model_hub,
            self.modelscope_cache_dir,
        )
        logger.info(f"Loading Local Qwen3-TTS model: {self.model_name} (path={load_path})")
        self._model = FasterQwen3TTS.from_pretrained(
            load_path,
            device=self.device,
            dtype=self._dtype,
            attn_implementation=self.attn_implementation,
        )
        logger.info("Local Qwen3-TTS model loaded")

        self._warmup()

    def _warmup(self) -> None:
        """Warmup the model for better performance."""
        logger.info(f"Warming up {self.__class__.__name__}")
        if self.parity_mode:
            logger.info("Local Qwen3-TTS parity mode enabled: skipping CUDA graph capture warmup")
        else:
            try:
                self._model._warmup(prefill_len=100)
            except Exception as e:
                logger.warning(f"CUDA graph capture failed: {e}")
        try:
            for _ in self._warmup_process("Hello, this is a warmup."):
                pass
            logger.info(f"{self.__class__.__name__} warmed up")
        except Exception as e:
            logger.warning(f"Warmup generation failed: {e}")

    def _model_type(self) -> str | None:
        """Get the TTS model type from the loaded model."""
        inner = getattr(getattr(self._model, "model", None), "model", None)
        return getattr(inner, "tts_model_type", None)

    def _resolve_speaker(self) -> str | None:
        """Resolve the speaker name for custom voice generation."""
        if self.speaker:
            return self.speaker
        inner = getattr(getattr(self._model, "model", None), "model", None)
        get_speakers = getattr(inner, "get_supported_speakers", None)
        if callable(get_speakers):
            speakers = list(get_speakers() or [])
            if speakers:
                return speakers[0]
        return None

    @staticmethod
    def _to_int16(audio: np.ndarray) -> np.ndarray:
        """Convert float audio to int16 format."""
        return np.clip(audio * 32768, -32768, 32767).astype(np.int16)

    def _warmup_process(self, text: str):
        """Process warmup text through appropriate generation mode."""
        model_type = self._model_type()
        if self.ref_audio:
            yield from self._process_voice_clone(text)
        elif model_type == "custom_voice":
            yield from self._process_custom_voice(text)
        elif model_type == "voice_design":
            yield from self._process_voice_design(text)
        else:
            # Default to voice clone with no reference (will use default voice)
            yield from self._process_voice_design(text)

    def _resample_to_pipeline_sr(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resample audio to pipeline sample rate."""
        if sr == PIPELINE_SR:
            return audio
        from scipy.signal import resample_poly
        gcd = np.gcd(PIPELINE_SR, sr)
        return resample_poly(audio, up=PIPELINE_SR // gcd, down=sr // gcd)

    def _stream(self, gen, label: str):
        """Common streaming loop: yield int16 chunks."""
        start = perf_counter()
        total_samples = 0
        first_chunk = True
        found_speech = False
        leftover = np.array([], dtype=np.int16)

        for audio_chunk, sr, _timing in gen:
            if first_chunk:
                logger.info(f"Qwen3-TTS TTFA: {perf_counter() - start:.2f}s ({label})")
                first_chunk = False
            audio_chunk = self._resample_to_pipeline_sr(audio_chunk, sr)
            audio_chunk = self._to_int16(audio_chunk)

            # Trim leading silence from the very start (model often generates
            # a silent ramp-up on the first turn, causing a perceived "missing word").
            if not found_speech:
                threshold = int(32768 * 0.01)
                above = np.abs(audio_chunk) > threshold
                if not np.any(above):
                    continue  # entire chunk is silence — skip it
                start_idx = max(0, int(np.argmax(above)) - int(PIPELINE_SR * 0.005))
                audio_chunk = audio_chunk[start_idx:]
                found_speech = True

            # Concatenate with any leftover samples from the previous chunk
            audio_chunk = np.concatenate([leftover, audio_chunk])

            # Yield exactly blocksize-sized chunks (required by LocalAudioStreamer)
            n = (len(audio_chunk) // self.blocksize) * self.blocksize
            for i in range(0, n, self.blocksize):
                yield audio_chunk[i : i + self.blocksize]
                total_samples += self.blocksize
            leftover = audio_chunk[n:]

        # Flush any remaining samples with zero-padding
        if len(leftover) > 0:
            chunk = np.pad(leftover, (0, self.blocksize - len(leftover)))
            yield chunk
            total_samples += len(leftover)

        generation_time = perf_counter() - start
        audio_duration = total_samples / PIPELINE_SR
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info(
            f"Qwen3-TTS generated {audio_duration:.2f}s audio in {generation_time:.2f}s "
            f"(RTF: {rtf:.2f}, {label})"
        )

    def _process_voice_clone(self, text: str):
        """Process voice clone generation."""
        yield from self._stream(
            self._model.generate_voice_clone_streaming(
                text=text,
                language=self.language,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                xvec_only=self.xvec_only,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=self.max_new_tokens,
                parity_mode=self.parity_mode,
            ),
            label="voice_clone_parity" if self.parity_mode else "voice_clone",
        )

    def _process_custom_voice(self, text: str):
        """Process custom voice generation."""
        speaker = self._resolve_speaker()
        if not speaker:
            raise ValueError(
                "CustomVoice generation requires a speaker. "
                "Set speaker or use a voice-clone model with ref_audio."
            )
        yield from self._stream(
            self._model.generate_custom_voice_streaming(
                text=text,
                speaker=speaker,
                language=self.language,
                instruct=self.instruct,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=self.max_new_tokens,
            ),
            label="custom_voice",
        )

    def _process_voice_design(self, text: str):
        """Process voice design generation."""
        yield from self._stream(
            self._model.generate_voice_design_streaming(
                text=text,
                instruct=self.instruct,
                language=self.language,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=self.max_new_tokens,
            ),
            label="voice_design",
        )

    async def synthesize(
        self,
        text: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize speech from text.

        Args:
            text (`str | None`, optional):
                The text to be synthesized.
            **kwargs (`Any`):
                Additional keyword arguments to pass to the TTS generation.

        Returns:
            `AsyncGenerator[bytes, None]`:
                An async generator yielding audio chunks as bytes.
        """
        if not text:
            return
            yield  # Make it a generator

        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            model_type = self._model_type()
            
            # Determine generation mode
            if self.ref_audio:
                audio_generator = self._process_voice_clone(text)
            elif model_type == "custom_voice":
                audio_generator = self._process_custom_voice(text)
            elif model_type == "voice_design":
                audio_generator = self._process_voice_design(text)
            else:
                # Default to voice clone mode
                audio_generator = self._process_voice_design(text)

            # Use ThreadPoolExecutor to run synchronous generator in separate thread
            # This prevents blocking the event loop during TTS generation
            loop = asyncio.get_event_loop()
            
            def get_next_chunk(gen):
                try:
                    return next(gen)
                except StopIteration:
                    return None
            
            # Yield audio chunks as bytes with proper async handling
            while True:
                chunk = await loop.run_in_executor(None, get_next_chunk, audio_generator)
                if chunk is None:
                    break
                yield chunk.tobytes()
                # Small async sleep to yield control and prevent overwhelming the consumer
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error during Local Qwen3-TTS generation: {e}", exc_info=True)
            raise

    def cleanup(self) -> None:
        """Cleanup model resources."""
        try:
            if self._model is not None:
                del self._model
                self._model = None
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            logger.info("Local Qwen3-TTS model cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    def __del__(self):
        """Destructor to cleanup resources."""
        self.cleanup()


if __name__ == "__main__":
    import argparse
    import asyncio
    import time
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    # ---------------------------------------------------------------------------
    # CLI argument parsing
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Standalone FastAPI TTS service backed by LocalQwenTTSModel"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8090, help="Bind port (default: 7600)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Model name or local path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--ref-audio", default=None, help="Path to reference audio for voice cloning")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT, help="Reference text for voice cloning")
    parser.add_argument("--language", default="Chinese", help="Synthesis language")
    parser.add_argument("--speaker", default=None, help="Speaker name for custom voice")
    parser.add_argument("--instruct", default=None, help="Instruction prompt for voice design")
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--parity-mode", action="store_true")
    parser.add_argument("--streaming-chunk-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=360)
    parser.add_argument("--blocksize", type=int, default=512)
    parser.add_argument("--model-hub", default="modelscope", choices=["modelscope", "huggingface", "none"])
    parser.add_argument("--modelscope-cache-dir", default=None)
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Logging setup
    # ---------------------------------------------------------------------------
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.basicConfig(
        level=log_level_map[args.log_level],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ---------------------------------------------------------------------------
    # Pydantic request/response models
    # ---------------------------------------------------------------------------
    class TTSRequest(BaseModel):
        model: str = "qwen3-tts"
        input: str
        voice: str = "default"
        response_format: str = "pcm"
        speed: float = 1.0

    class TTSVoice(BaseModel):
        id: str
        name: str
        language: str = "English"

    # ---------------------------------------------------------------------------
    # Model initialisation
    # ---------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Starting Local Qwen3-TTS standalone service on {args.host}:{args.port}")
    logger.info("=" * 60)

    model_kwargs: dict = dict(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        ref_text=args.ref_text,
        language=args.language,
        xvec_only=args.xvec_only,
        parity_mode=args.parity_mode,
        streaming_chunk_size=args.streaming_chunk_size,
        max_new_tokens=args.max_new_tokens,
        blocksize=args.blocksize,
        stream=True,
        model_hub=args.model_hub,
        modelscope_cache_dir=args.modelscope_cache_dir,
    )
    if args.ref_audio:
        model_kwargs["ref_audio"] = args.ref_audio
    if args.speaker:
        model_kwargs["speaker"] = args.speaker
    if args.instruct:
        model_kwargs["instruct"] = args.instruct

    tts_model = LocalQwenTTSModel(**model_kwargs)

    # ---------------------------------------------------------------------------
    # FastAPI application
    # ---------------------------------------------------------------------------
    app = FastAPI(
        title="Local Qwen3 TTS API",
        description="OpenAI-compatible TTS API using local Qwen3-TTS model",
        version="1.0.0",
    )

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "qwen3-tts",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "permission": [],
                    "root": "qwen3-tts",
                    "parent": None,
                }
            ],
        }

    @app.get("/v1/audio/voices")
    async def list_voices():
        """List available voices."""
        voices = [TTSVoice(id="default", name="Default Voice", language=args.language)]
        if args.speaker:
            voices.append(TTSVoice(id=args.speaker, name=args.speaker, language=args.language))
        return {"voices": [v.model_dump() for v in voices]}

    async def _stream_audio(text: str):
        """Async generator that yields PCM bytes from the TTS model."""
        try:
            async for audio_bytes in tts_model.synthesize(text):
                yield audio_bytes
        except Exception as exc:
            logger.error(f"Error streaming audio: {exc}", exc_info=True)
            raise

    @app.post("/v1/audio/speech")
    async def create_speech(request: TTSRequest):
        """Create speech from text (OpenAI-compatible, streaming PCM)."""
        if not request.input:
            raise HTTPException(status_code=400, detail="Input text is required")
        try:
            return StreamingResponse(
                _stream_audio(request.input),
                media_type=f"audio/pcm;rate={PIPELINE_SR}",
                headers={
                    "Content-Disposition": "attachment; filename=speech.pcm",
                    "X-Sample-Rate": str(PIPELINE_SR),
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"TTS synthesis error: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": args.model_name,
            "sample_rate": PIPELINE_SR,
        }

    # ---------------------------------------------------------------------------
    # Start uvicorn
    # ---------------------------------------------------------------------------
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
