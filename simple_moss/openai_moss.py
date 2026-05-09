# -*- coding: utf-8 -*-
"""
simple_moss OpenAI-compatible TTS Service.

Wraps simple_moss (ONNX backend) behind an OpenAI-compatible HTTP API.
Interface mirrors stream-api-ref/_local_tts_engine.py exactly.

Run (from project root):
    python -m simple_moss.openai_moss
    python -m simple_moss.openai_moss --port 8090 --model-dir ./models
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ._config import DEFAULT_VOICE_CONFIG
from ._voice_list import BUILTIN_VOICE_MAP
from .api_facade import MossStreamApiFacade
from .simple_app_onnx import create_default_adapter, warmup_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Fallback audio format — actual values are read from the ONNX model's
# codec metadata at startup via adapter.runtime.get_codec_audio_format().
FALLBACK_SAMPLE_RATE: int = 48000
FALLBACK_CHANNELS: int = 2
MODEL_ID: str = "moss-tts"


# ---------------------------------------------------------------------------
# Pydantic models (identical schema to stream-api-ref/_local_tts_engine.py)
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request."""
    model: str = MODEL_ID
    input: str
    voice: str = "default"
    response_format: str = "pcm"
    speed: float = 1.0


class TTSVoice(BaseModel):
    """TTS voice definition."""
    id: str
    name: str
    language: str = "Unknown"
    description: str = ""
    gender: str = "Unknown"


# ---------------------------------------------------------------------------
# Global runtime state (populated in main() before uvicorn.run)
# ---------------------------------------------------------------------------

g_facade: Optional[MossStreamApiFacade] = None
g_adapter = None
g_warmup_ready: bool = False
g_warmup_error: Optional[str] = None
g_model_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Streaming bridge: simple_moss sync queue → async generator
# ---------------------------------------------------------------------------

def _resolve_request_voice(requested_voice: str | None) -> str:
    """Map OpenAI-style voice input to the actual simple_moss voice name."""
    normalized_voice = str(requested_voice or "").strip()
    if not normalized_voice or normalized_voice.lower() == "default":
        return str(DEFAULT_VOICE_CONFIG.default_voice)
    return normalized_voice


def _resolve_voice_language(voice_name: str | None) -> str:
    """Resolve voice language from the shared voice metadata table."""
    normalized_voice = str(voice_name or "").strip()
    if not normalized_voice:
        return "Unknown"

    voice_row = BUILTIN_VOICE_MAP.get(normalized_voice)
    if not voice_row:
        return "Unknown"

    group_name = str(voice_row.get("group") or "").strip()
    if not group_name:
        return "Unknown"

    return group_name.split()[0]


def _resolve_voice_description(voice_name: str | None) -> str:
    """Resolve voice description (display_name) from the shared voice metadata table."""
    normalized_voice = str(voice_name or "").strip()
    if not normalized_voice:
        return ""

    voice_row = BUILTIN_VOICE_MAP.get(normalized_voice)
    if not voice_row:
        return ""

    return str(voice_row.get("display_name") or "").strip()


def _resolve_voice_gender(voice_name: str | None) -> str:
    """Resolve voice gender from the shared voice metadata table's group field."""
    normalized_voice = str(voice_name or "").strip()
    if not normalized_voice:
        return "Unknown"

    voice_row = BUILTIN_VOICE_MAP.get(normalized_voice)
    if not voice_row:
        return "Unknown"

    group_name = str(voice_row.get("group") or "").strip()
    if not group_name:
        return "Unknown"

    parts = group_name.split()
    if len(parts) > 1:
        return parts[1]
    return "Unknown"


async def _stream_pcm(text: str, voice: str | None) -> AsyncGenerator[bytes, None]:
    """Convert simple_moss blocking audio queue to an async byte generator.

    facade.audio() internally calls queue.Queue.get() which blocks the
    calling thread. Each next() call is offloaded to the default thread-pool
    executor so the asyncio event loop is never blocked.
    """
    if g_facade is None:
        raise RuntimeError("TTS facade not initialised")

    resolved_voice = _resolve_request_voice(voice)
    start_resp = g_facade.start(text, voice=resolved_voice)
    stream_id = str(start_resp["stream_id"])
    log.info("TTS stream started: stream_id=%s voice=%s", stream_id, resolved_voice)

    try:
        loop = asyncio.get_event_loop()
        audio_iter = g_facade.audio(stream_id)

        def _next_chunk() -> bytes | None:
            try:
                return next(audio_iter)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, _next_chunk)
            if chunk is None:
                break
            yield chunk
    finally:
        try:
            g_facade.close(stream_id)
        except Exception:
            pass
        log.info("TTS stream closed: stream_id=%s", stream_id)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def build_app(
    sample_rate: int = FALLBACK_SAMPLE_RATE,
    channels: int = FALLBACK_CHANNELS,
) -> FastAPI:
    """Create and return the FastAPI application with all routes registered."""
    app = FastAPI(
        title="MOSS TTS OpenAI-compatible API",
        description="OpenAI-compatible TTS API backed by simple_moss ONNX runtime",
        version="1.0.0",
    )

    # Allow cross-origin requests so browser pages on other ports can call this API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Sample-Rate", "X-Channels"],
    )

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "permission": [],
                    "root": MODEL_ID,
                    "parent": None,
                }
            ],
        }

    @app.get("/v1/audio/voices")
    async def list_voices():
        """List available voices."""
        voices: list[TTSVoice] = []
        seen_voice_ids: set[str] = set()
        default_voice = str(DEFAULT_VOICE_CONFIG.default_voice or "default")

        def _append_voice(voice_id: str, voice_name: str) -> None:
            if voice_id in seen_voice_ids:
                return
            seen_voice_ids.add(voice_id)
            voices.append(
                TTSVoice(
                    id=voice_id,
                    name=voice_name,
                    language=_resolve_voice_language(voice_name),
                    description=_resolve_voice_description(voice_name),
                    gender=_resolve_voice_gender(voice_name),
                )
            )

        _append_voice("default", default_voice)
        if g_adapter is not None:
            try:
                for item in g_adapter.runtime.list_builtin_voices():
                    builtin_voice = str(item.get("voice") or "").strip()
                    if builtin_voice:
                        _append_voice(builtin_voice, builtin_voice)
            except Exception as exc:
                log.warning("Could not list builtin voices from runtime: %s", exc)
        return {"voices": [v.model_dump() for v in voices]}

    @app.post("/v1/audio/speech")
    async def create_speech(request: TTSRequest):
        """Create speech from text — OpenAI-compatible streaming PCM endpoint.

        Returns a streaming response of raw PCM s16le audio at the configured sample rate.
        Response headers carry X-Sample-Rate and X-Channels for client decoding.

        Fields voice, response_format, and speed are accepted for API
        compatibility. voice selects the built-in voice used by simple_moss;
        response_format and speed still have no effect.
        """
        if g_facade is None:
            raise HTTPException(status_code=503, detail="TTS service not initialised")
        if not request.input or not request.input.strip():
            raise HTTPException(status_code=400, detail="input text is required")
        try:
            return StreamingResponse(
                _stream_pcm(request.input, voice=request.voice),
                media_type=f"audio/pcm;rate={sample_rate}",
                headers={
                    "Content-Disposition": "attachment; filename=speech.pcm",
                    "X-Sample-Rate": str(sample_rate),
                    "X-Channels": str(channels),
                    "Cache-Control": "no-store",
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            log.error("TTS synthesis error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy" if g_facade is not None else "not_ready",
            "model": MODEL_ID,
            "voice": DEFAULT_VOICE_CONFIG.default_voice,
            "sample_rate": sample_rate,
            "channels": channels,
            "warmup_ready": g_warmup_ready,
            "warmup_error": g_warmup_error,
            "model_dir": g_model_dir,
        }

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="simple_moss OpenAI-compatible TTS service")
    # Web service
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8090, help="Bind port (default: 8090)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    # simple_moss backend
    parser.add_argument("--model-dir", default=None, help="ONNX model directory (default: models/)")
    parser.add_argument("--cpu-threads", type=int, default=1, help="ORT CPU thread count (default: 1)")
    parser.add_argument("--max-new-frames", type=int, default=375, help="Max generation frames (default: 375)")
    parser.add_argument("--voice-clone-max-text-tokens", type=int, default=32,
                        help="Max tokens per text chunk (default: 32)")
    parser.add_argument("--chunk-pause-seconds", type=float, default=1.5,
                        help="Silence between text chunks in seconds (default: 1.5)")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip startup warmup")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global g_adapter, g_facade, g_warmup_ready, g_warmup_error, g_model_dir

    args = _parse_args(argv)

    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.getLogger().setLevel(log_level_map[args.log_level])

    g_model_dir = args.model_dir

    log.info("=" * 60)
    log.info("simple_moss OpenAI-compatible TTS Service")
    log.info("Starting on %s:%d", args.host, args.port)
    log.info("=" * 60)

    log.info("Initialising adapter (model_dir=%s)", args.model_dir or "default")
    t0 = time.perf_counter()
    g_adapter = create_default_adapter(
        model_dir=args.model_dir,
        cpu_threads=max(1, args.cpu_threads),
        max_new_frames=max(1, args.max_new_frames),
    )
    log.info("Adapter ready in %.2f s", time.perf_counter() - t0)

    if not args.skip_warmup:
        try:
            log.info("Running warmup...")
            t0 = time.perf_counter()
            warmup_result = warmup_runtime(g_adapter)
            elapsed = float(warmup_result.get("elapsed_seconds") or time.perf_counter() - t0)
            log.info("Warmup complete in %.2f s", elapsed)
            g_warmup_ready = True
        except Exception as exc:
            g_warmup_error = str(exc)
            log.exception("Warmup failed — service will still start")
    else:
        log.info("Warmup skipped (--skip-warmup)")

    g_facade = MossStreamApiFacade(
        g_adapter,
        max_new_frames=max(1, args.max_new_frames),
        voice_clone_max_text_tokens=max(1, args.voice_clone_max_text_tokens),
        chunk_pause_seconds=max(0.0, args.chunk_pause_seconds),
    )

    # Read actual audio format from the ONNX model's codec metadata.
    # Falls back to module-level defaults if the runtime doesn't expose the method.
    try:
        _sr, _ch = g_adapter.runtime.get_codec_audio_format()
        log.info("Codec audio format from ONNX model: sample_rate=%d, channels=%d", _sr, _ch)
    except Exception as exc:
        _sr, _ch = FALLBACK_SAMPLE_RATE, FALLBACK_CHANNELS
        log.warning(
            "Could not read codec audio format, using defaults (%d Hz / %d ch): %s",
            _sr, _ch, exc,
        )

    app = build_app(sample_rate=_sr, channels=_ch)
    log.info("Endpoints: GET /v1/models  GET /v1/audio/voices  POST /v1/audio/speech  GET /health")
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level=args.log_level)


if __name__ == "__main__":
    main()
