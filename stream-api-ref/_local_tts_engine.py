# -*- coding: utf-8 -*-
"""
Local Qwen3 TTS Engine Service.

Provides an OpenAI-compatible TTS API service using the LocalQwenTTSModel.
Supports streaming audio synthesis with configurable voice parameters.
"""

import asyncio
import base64
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np

from ._engine_base import (
    EngineBase, 
    EngineConfig, 
    EngineStatus,
    ConfigField,
    EngineType,
)
from ._local_qwen_tts_model import LocalQwenTTSModel, PIPELINE_SR

logger = logging.getLogger(__name__)


def _get_log_dir() -> str:
    """Return the directory for engine log files."""
    log_dir = os.environ.get(
        "AXAGENT_LOG_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "logs", "engines"),
    )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _get_log_file_path(engine_name: str) -> str:
    """Return the log file path for a given engine."""
    return os.path.join(_get_log_dir(), f"local_engine_{engine_name}.log")

# Default configuration values
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_REF_TEXT = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request."""
    model: str = "qwen3-tts"
    input: str
    voice: str = "default"
    response_format: str = "pcm"  # pcm, wav, mp3
    speed: float = 1.0


class TTSVoice(BaseModel):
    """TTS voice definition."""
    id: str
    name: str
    language: str = "English"


class LocalTTSEngine(EngineBase):
    """
    Local Qwen3 TTS Engine Service.
    
    Provides an OpenAI-compatible TTS API with streaming support.
    Runs as a standalone FastAPI service on a configurable port.
    """
    
    # TTS model instance
    _tts_model: Optional[LocalQwenTTSModel] = None
    
    @classmethod
    def get_config(cls) -> EngineConfig:
        """Return the engine configuration schema."""
        return EngineConfig(
            name="local-qwen-tts",
            display_name="Local Qwen3 TTS",
            engine_type=EngineType.TTS,
            description="Local Qwen3 Text-to-Speech engine with OpenAI-compatible API",
            version="1.0.0",
            config_schema={
                "model_name": ConfigField(
                    name="model_name",
                    type="string",
                    default=DEFAULT_MODEL,
                    required=False,
                    description="Model name or path for Qwen3 TTS",
                ),
                "device": ConfigField(
                    name="device",
                    type="string",
                    default="cuda",
                    required=False,
                    description="Device to run the model on (cuda or cpu)",
                    options=["cuda", "cpu"],
                ),
                "dtype": ConfigField(
                    name="dtype",
                    type="string",
                    default="auto",
                    required=False,
                    description="Data type for model weights",
                    options=["auto", "float16", "bfloat16", "float32"],
                ),
                "ref_audio": ConfigField(
                    name="ref_audio",
                    type="path",
                    default=None,
                    required=False,
                    description="Path to reference audio file for voice cloning",
                ),
                "ref_text": ConfigField(
                    name="ref_text",
                    type="string",
                    default=DEFAULT_REF_TEXT,
                    required=False,
                    description="Reference text corresponding to the reference audio",
                ),
                "language": ConfigField(
                    name="language",
                    type="string",
                    default="English",
                    required=False,
                    description="Language of the text to synthesize",
                ),
                "speaker": ConfigField(
                    name="speaker",
                    type="string",
                    default=None,
                    required=False,
                    description="Speaker name for custom voice generation",
                ),
                "instruct": ConfigField(
                    name="instruct",
                    type="string",
                    default=None,
                    required=False,
                    description="Instruction prompt for voice design mode",
                ),
                "streaming_chunk_size": ConfigField(
                    name="streaming_chunk_size",
                    type="integer",
                    default=1,
                    required=False,
                    description="Chunk size for streaming generation",
                    min_value=1,
                    max_value=16,
                ),
                "max_new_tokens": ConfigField(
                    name="max_new_tokens",
                    type="integer",
                    default=360,
                    required=False,
                    description="Maximum number of new tokens to generate",
                    min_value=64,
                    max_value=1024,
                ),
                "blocksize": ConfigField(
                    name="blocksize",
                    type="integer",
                    default=512,
                    required=False,
                    description="Audio block size for output",
                    min_value=256,
                    max_value=2048,
                ),
                "model_hub": ConfigField(
                    name="model_hub",
                    type="string",
                    default="modelscope",
                    required=False,
                    description=(
                        "Where to resolve model_name: modelscope (default), huggingface, or none"
                    ),
                    options=["modelscope", "huggingface", "none"],
                ),
                "modelscope_cache_dir": ConfigField(
                    name="modelscope_cache_dir",
                    type="path",
                    default=None,
                    required=False,
                    description="ModelScope cache dir (default: MODELSCOPE_CACHE or ~/.cache/modelscope/hub)",
                ),
            },
            default_values={
                "model_name": DEFAULT_MODEL,
                "device": "cuda",
                "dtype": "auto",
                "ref_text": DEFAULT_REF_TEXT,
                "language": "English",
                "streaming_chunk_size": 1,
                "max_new_tokens": 360,
                "blocksize": 512,
                "model_hub": "modelscope",
                "modelscope_cache_dir": None,
            },
        )
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the TTS engine."""
        super().__init__(config)
        self._app: Optional[FastAPI] = None
        self._tts_model: Optional[LocalQwenTTSModel] = None
    
    def _create_app(self) -> FastAPI:
        """Create the FastAPI application."""
        app = FastAPI(
            title="Local Qwen3 TTS API",
            description="OpenAI-compatible TTS API using Qwen3 TTS",
            version="1.0.0",
        )
        
        # Register routes
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
                ]
            }
        
        @app.get("/v1/audio/voices")
        async def list_voices():
            """List available voices."""
            voices = [
                TTSVoice(id="default", name="Default Voice", language="English"),
            ]
            
            # Add speaker-based voices if configured
            speaker = self.get_config_value("speaker")
            if speaker:
                voices.append(TTSVoice(id=speaker, name=speaker, language="English"))
            
            return {"voices": [v.model_dump() for v in voices]}
        
        @app.post("/v1/audio/speech")
        async def create_speech(request: TTSRequest):
            """Create speech from text (OpenAI-compatible endpoint)."""
            if not self._tts_model:
                raise HTTPException(status_code=503, detail="TTS model not initialized")
            
            if not request.input:
                raise HTTPException(status_code=400, detail="Input text is required")
            
            try:
                # Generate audio - pass text directly
                return StreamingResponse(
                    self._stream_audio(request.input),
                    media_type=f"audio/pcm;rate={PIPELINE_SR}",
                    headers={
                        "Content-Disposition": f"attachment; filename=speech.pcm",
                        "X-Sample-Rate": str(PIPELINE_SR),
                    }
                )
            
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy" if self._tts_model else "not_ready",
                "model": self.get_config_value("model_name"),
                "sample_rate": PIPELINE_SR,
            }
        
        return app
    
    async def _stream_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks from TTS model."""
        try:
            async for audio_bytes in self._tts_model.synthesize(text):
                yield audio_bytes
        except Exception as e:
            logger.error(f"Error streaming audio: {e}", exc_info=True)
            raise
    
    async def start(self, port: int, **kwargs) -> bool:
        """Start the TTS engine service."""
        if self.is_running():
            logger.warning("TTS engine is already running")
            return True
        
        self._set_status(EngineStatus.STARTING)
        self._port = port
        
        # Setup file logging
        log_file_path = _get_log_file_path("local-qwen-tts")
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add file handler to root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        self._log_file_handler = file_handler
        self._log_file_path = log_file_path
        
        try:
            # Merge kwargs with existing config
            config = {**self._config, **kwargs}
            self._config = config
            
            # Initialize TTS model
            logger.info("=" * 60)
            logger.info(f"Starting Local Qwen3 TTS Engine on port {port}")
            logger.info(f"Log file: {log_file_path}")
            logger.info("=" * 60)
            logger.info("Initializing TTS model...")
            
            model_kwargs = {
                "model_name": config.get("model_name", DEFAULT_MODEL),
                "device": config.get("device", "cuda"),
                "dtype": config.get("dtype", "auto"),
                "language": config.get("language", "English"),
                "streaming_chunk_size": config.get("streaming_chunk_size", 1),
                "max_new_tokens": config.get("max_new_tokens", 360),
                "blocksize": config.get("blocksize", 512),
                "stream": True,
                "model_hub": config.get("model_hub", "modelscope"),
                "modelscope_cache_dir": config.get("modelscope_cache_dir"),
            }
            
            # Add optional parameters
            if config.get("ref_audio"):
                model_kwargs["ref_audio"] = config["ref_audio"]
            if config.get("ref_text"):
                model_kwargs["ref_text"] = config["ref_text"]
            if config.get("speaker"):
                model_kwargs["speaker"] = config["speaker"]
            if config.get("instruct"):
                model_kwargs["instruct"] = config["instruct"]
            
            self._tts_model = LocalQwenTTSModel(**model_kwargs)
            
            # Create FastAPI app
            self._app = self._create_app()
            
            # Start server with log file output
            import uvicorn
            uvicorn_config = uvicorn.Config(
                self._app,
                host="0.0.0.0",
                port=port,
                log_level="info",
            )
            server = uvicorn.Server(uvicorn_config)
            
            # Run server in background task
            self._server_task = asyncio.create_task(server.serve())
            self._start_time = time.time()
            self._set_status(EngineStatus.RUNNING)
            
            logger.info(f"TTS engine started successfully on port {port}")
            return True
        
        except Exception as e:
            error_msg = f"Failed to start TTS engine: {e}"
            logger.error(error_msg, exc_info=True)
            self._set_status(EngineStatus.ERROR, error_msg)
            return False
    
    async def stop(self) -> bool:
        """Stop the TTS engine service."""
        if not self.is_running():
            logger.warning("TTS engine is not running")
            return True
        
        self._set_status(EngineStatus.STOPPING)
        
        try:
            # Cancel server task
            if self._server_task:
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
                self._server_task = None
            
            # Cleanup TTS model
            if self._tts_model:
                self._tts_model.cleanup()
                self._tts_model = None
            
            self._app = None
            self._start_time = None
            
            # Remove file handler
            if hasattr(self, '_log_file_handler') and self._log_file_handler:
                root_logger = logging.getLogger()
                root_logger.removeHandler(self._log_file_handler)
                self._log_file_handler.close()
                self._log_file_handler = None
            
            self._set_status(EngineStatus.STOPPED)
            
            logger.info("TTS engine stopped")
            return True
        
        except Exception as e:
            error_msg = f"Failed to stop TTS engine: {e}"
            logger.error(error_msg, exc_info=True)
            self._set_status(EngineStatus.ERROR, error_msg)
            return False
    
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return self._status == EngineStatus.RUNNING
    
    def get_log_file_path(self) -> Optional[str]:
        """Return the log file path if available."""
        return getattr(self, '_log_file_path', None)
    
    def get_openai_endpoints(self) -> List[Dict[str, str]]:
        """Return OpenAI-compatible endpoints."""
        return [
            {
                "method": "GET",
                "path": "/v1/models",
                "description": "List available models",
            },
            {
                "method": "GET",
                "path": "/v1/audio/voices",
                "description": "List available voices",
            },
            {
                "method": "POST",
                "path": "/v1/audio/speech",
                "description": "Create speech from text (streaming)",
            },
            {
                "method": "GET",
                "path": "/health",
                "description": "Health check",
            },
        ]
