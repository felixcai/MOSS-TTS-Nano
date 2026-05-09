"""
simple_moss Web 全流程测试服务

运行方式（在项目根目录下）：
    python -m simple_moss.test_web_api
    python -m simple_moss.test_web_api --host 0.0.0.0 --port 8000

流程：
    1. 初始化（create_default_adapter）
    2. Warmup（warmup_runtime，默认执行，可用 --skip-warmup 跳过）
    3. Web 页面调用 start/audio/status/result/close
    4. 浏览器端用 WebAudio 对 PCM s16le 流做实时播放
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Iterator

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .api_facade import MossStreamApiFacade
from .simple_app_onnx import OnnxNanoTTSServiceAdapter, create_default_adapter, warmup_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


DEFAULT_TEXT = "你好，这是一段来自 simple_moss 的 Web 流式合成测试语音。"


@dataclass
class WebRuntimeState:
    """Web 服务运行时状态，集中持有 adapter/facade 和默认参数。

    调用方：build_app 创建 FastAPI routes 时读取；main 入口初始化后传入 build_app。
    """

    adapter: OnnxNanoTTSServiceAdapter
    facade: MossStreamApiFacade
    model_dir: str | None
    cpu_threads: int
    max_new_frames: int
    voice_clone_max_text_tokens: int
    chunk_pause_seconds: float
    warmup_result: dict[str, object] | None = None
    warmup_error: str | None = None

    @property
    def warmup_ready(self) -> bool:
        """返回 warmup 是否成功完成。

        调用方：/health、/api/warmup-status 和首页渲染。
        """

        return self.warmup_result is not None and self.warmup_error is None


def _coerce_int(value: object, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """将 HTTP 表单值安全转换为 int，并按需限制上下界。

    调用方：/api/generate-stream/start 路由解析表单参数。
    """

    try:
        result = int(value)
    except Exception:
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def _coerce_float(value: object, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """将 HTTP 表单值安全转换为 float，并按需限制上下界。
    """
    try:
        result = float(value)
    except Exception:
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def _status_text(state: WebRuntimeState) -> str:
    """根据 warmup 状态生成页面显示文本。

    调用方：_render_index_html 和 /api/warmup-status。
    """

    if state.warmup_error:
        return f"Warmup failed: {state.warmup_error}"
    if state.warmup_result is None:
        return "Warmup skipped."
    elapsed = float(state.warmup_result.get("elapsed_seconds") or 0.0)
    return f"Warmup ready. elapsed={elapsed:.2f}s"


def _render_index_html(state: WebRuntimeState) -> str:
    """返回内嵌 HTML/JS 页面，提供实时流式播放 UI。

    调用方：GET / 路由。
    """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>simple_moss Web Demo</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    body {{
      margin: 0;
      background: #f5f5f5;
      color: #1f2933;
    }}
    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .panel {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.8fr);
      gap: 18px;
      margin-top: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    textarea {{
      width: 100%;
      min-height: 160px;
      resize: vertical;
    }}
    input, textarea, button {{
      font: inherit;
      box-sizing: border-box;
    }}
    input[type="number"] {{
      width: 100%;
    }}
    label {{
      display: block;
      font-weight: 650;
      margin-bottom: 6px;
    }}
    .field {{
      margin: 14px 0;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .buttons {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    button {{
      border: 0;
      border-radius: 10px;
      padding: 10px 16px;
      background: #2563eb;
      color: #fff;
      cursor: pointer;
    }}
    button.secondary {{
      background: #475569;
    }}
    button:disabled {{
      opacity: 0.55;
      cursor: not-allowed;
    }}
    .status {{
      white-space: pre-wrap;
      padding: 12px;
      background: #f1f5f9;
      border-radius: 10px;
      min-height: 44px;
    }}
    .meta {{
      color: #64748b;
      font-size: 13px;
    }}
    .script {{
      padding: 12px;
      border-radius: 10px;
      background: #f8fafc;
      border: 1px dashed #cbd5e1;
      min-height: 80px;
    }}
    audio {{
      width: 100%;
      margin-top: 8px;
    }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #0f172a; color: #e5e7eb; }}
      .hero, .panel {{ background: #111827; border-color: #334155; }}
      .status, .script {{ background: #0b1220; border-color: #334155; }}
      .meta {{ color: #94a3b8; }}
    }}
    @media (max-width: 860px) {{
      .grid, .row {{ grid-template-columns: 1fr; }}
      .page {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>simple_moss Web Demo</h1>
      <div class="meta">Fixed built-in voice + ONNX fixed stream generation.</div>
      <div class="meta">Warmup: {_status_text(state)}</div>
    </div>

    <div class="grid">
      <section class="panel">
        <div class="field">
          <label for="text">Text</label>
          <textarea id="text">{DEFAULT_TEXT}</textarea>
        </div>
        <div class="row">
          <div class="field">
            <label for="max-new-frames">Max New Frames</label>
            <input id="max-new-frames" type="number" min="1" max="2048" step="1" value="{state.max_new_frames}">
          </div>
          <div class="field">
            <label for="voice-clone-max-text-tokens">Voice Clone Max Text Tokens</label>
            <input id="voice-clone-max-text-tokens" type="number" min="1" max="300" step="1" value="{state.voice_clone_max_text_tokens}">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="initial-delay">Initial Playback Delay (s)</label>
            <input id="initial-delay" type="number" min="0" max="3" step="0.01" value="0.0">
          </div>
          <div class="field">
            <label for="chunk-pause-seconds">Chunk Pause (s)</label>
            <input id="chunk-pause-seconds" type="number" min="0" max="10" step="0.1" value="{state.chunk_pause_seconds}">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label>&nbsp;</label>
            <div class="meta">PCM stream is played in realtime with WebAudio.</div>
          </div>
        </div>
        <div class="buttons">
          <button id="generate-btn" type="button">Generate</button>
          <button id="close-btn" class="secondary" type="button" disabled>Close Stream</button>
        </div>
      </section>

      <section class="panel">
        <div class="field">
          <label>Warmup Status</label>
          <div id="warmup-status" class="status">{_status_text(state)}</div>
        </div>
        <div class="field">
          <label>Run Status</label>
          <div id="run-status" class="status">Idle.</div>
        </div>
        <div class="field">
          <label>Playback Text</label>
          <div id="playback-script" class="script">等待生成。</div>
        </div>
        <div class="field">
          <label>Generated Speech (after stream completes)</label>
          <audio id="audio-output" controls></audio>
          <div id="audio-meta" class="meta"></div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const generateBtn = document.getElementById("generate-btn");
    const closeBtn = document.getElementById("close-btn");
    const textInput = document.getElementById("text");
    const maxNewFramesInput = document.getElementById("max-new-frames");
    const maxTextTokensInput = document.getElementById("voice-clone-max-text-tokens");
    const initialDelayInput = document.getElementById("initial-delay");
    const chunkPauseSecondsInput = document.getElementById("chunk-pause-seconds");
    const warmupStatus = document.getElementById("warmup-status");
    const runStatus = document.getElementById("run-status");
    const playbackScript = document.getElementById("playback-script");
    const audioOutput = document.getElementById("audio-output");
    const audioMeta = document.getElementById("audio-meta");

    let currentStreamId = null;
    let currentAbortController = null;
    let statusTimer = null;
    let audioContext = null;
    let nextPlaybackTime = 0;
    let pcmChunks = [];
    let currentObjectUrl = null;

    function setRunStatus(message) {{
      runStatus.textContent = message;
    }}

    function concatUint8Arrays(chunks) {{
      const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
      const out = new Uint8Array(total);
      let offset = 0;
      for (const chunk of chunks) {{
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }}
      return out;
    }}

    function pcm16leToAudioBuffer(bytes, sampleRate, channels) {{
      const bytesPerFrame = channels * 2;
      const frameCount = Math.floor(bytes.byteLength / bytesPerFrame);
      const audioBuffer = audioContext.createBuffer(channels, frameCount, sampleRate);
      const view = new DataView(bytes.buffer, bytes.byteOffset, frameCount * bytesPerFrame);
      for (let frame = 0; frame < frameCount; frame += 1) {{
        for (let channel = 0; channel < channels; channel += 1) {{
          const sample = view.getInt16((frame * channels + channel) * 2, true) / 32768;
          audioBuffer.getChannelData(channel)[frame] = sample;
        }}
      }}
      return audioBuffer;
    }}

    function schedulePcmChunk(bytes, sampleRate, channels) {{
      if (!bytes || bytes.byteLength <= 0) {{
        return;
      }}
      if (!audioContext) {{
        audioContext = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate }});
        const delay = Math.max(0, Number(initialDelayInput.value || 0.0));
        nextPlaybackTime = audioContext.currentTime + delay;
      }}
      const audioBuffer = pcm16leToAudioBuffer(bytes, sampleRate, channels);
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      const startAt = Math.max(nextPlaybackTime, audioContext.currentTime + 0.01);
      source.start(startAt);
      nextPlaybackTime = startAt + audioBuffer.duration;
    }}

    function encodeWav(pcmBytes, sampleRate, channels) {{
      const dataSize = pcmBytes.byteLength;
      const buffer = new ArrayBuffer(44 + dataSize);
      const view = new DataView(buffer);
      function writeString(offset, value) {{
        for (let i = 0; i < value.length; i += 1) {{
          view.setUint8(offset + i, value.charCodeAt(i));
        }}
      }}
      writeString(0, "RIFF");
      view.setUint32(4, 36 + dataSize, true);
      writeString(8, "WAVE");
      writeString(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, channels, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * channels * 2, true);
      view.setUint16(32, channels * 2, true);
      view.setUint16(34, 16, true);
      writeString(36, "data");
      view.setUint32(40, dataSize, true);
      new Uint8Array(buffer, 44).set(pcmBytes);
      return new Blob([buffer], {{ type: "audio/wav" }});
    }}

    async function refreshWarmupStatus() {{
      try {{
        const response = await fetch("/api/warmup-status");
        const payload = await response.json();
        warmupStatus.textContent = payload.status_text || JSON.stringify(payload);
      }} catch (error) {{
        warmupStatus.textContent = "Warmup status error: " + error;
      }}
    }}

    async function pollStatus(statusUrl) {{
      try {{
        const response = await fetch(statusUrl);
        if (!response.ok) {{
          return;
        }}
        const payload = await response.json();
        const chunks = Array.isArray(payload.text_chunks) ? payload.text_chunks : [];
        setRunStatus(
          payload.status_text || payload.run_status || payload.state || "Streaming"
        );
        if (chunks.length > 0) {{
          playbackScript.textContent = chunks.map((item, index) => (index + 1) + ". " + item).join("\\n");
        }}
      }} catch (error) {{
        console.warn("status polling failed", error);
      }}
    }}

    async function closeCurrentStream() {{
      if (!currentStreamId) {{
        return;
      }}
      if (currentAbortController) {{
        currentAbortController.abort();
      }}
      try {{
        await fetch(`/api/generate-stream/${{currentStreamId}}/close`, {{ method: "POST" }});
      }} catch (error) {{
        console.warn("close failed", error);
      }}
      currentStreamId = null;
      closeBtn.disabled = true;
      if (statusTimer) {{
        clearInterval(statusTimer);
        statusTimer = null;
      }}
    }}

    async function startGenerate() {{
      await closeCurrentStream();
      pcmChunks = [];
      if (currentObjectUrl) {{
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
      }}
      audioOutput.removeAttribute("src");
      audioOutput.load();
      audioMeta.textContent = "";
      playbackScript.textContent = textInput.value.trim() || "等待生成。";
      if (audioContext) {{
        try {{ await audioContext.close(); }} catch (error) {{ console.warn(error); }}
        audioContext = null;
      }}
      const formData = new FormData();
      formData.set("text", textInput.value);
      formData.set("max_new_frames", String(Number(maxNewFramesInput.value || {state.max_new_frames})));
      formData.set("voice_clone_max_text_tokens", String(Number(maxTextTokensInput.value || {state.voice_clone_max_text_tokens})));
      formData.set("chunk_pause_seconds", String(Number(chunkPauseSecondsInput.value || {state.chunk_pause_seconds})));

      generateBtn.disabled = true;
      setRunStatus("Starting...");
      try {{
        const startResponse = await fetch("/api/generate-stream/start", {{
          method: "POST",
          body: formData,
        }});
        const startPayload = await startResponse.json();
        if (!startResponse.ok) {{
          throw new Error(startPayload.error || "start failed");
        }}
        currentStreamId = startPayload.stream_id;
        closeBtn.disabled = false;
        const sampleRate = Number(startPayload.sample_rate || 48000);
        const channels = Number(startPayload.channels || 2);
        setRunStatus(startPayload.run_status || "Streaming...");
        currentAbortController = new AbortController();
        statusTimer = setInterval(() => pollStatus(startPayload.status_url), 500);

        const audioResponse = await fetch(startPayload.audio_url, {{
          signal: currentAbortController.signal,
        }});
        if (!audioResponse.ok || !audioResponse.body) {{
          throw new Error("audio stream failed");
        }}
        const reader = audioResponse.body.getReader();
        let carry = new Uint8Array(0);
        const bytesPerFrame = channels * 2;
        while (true) {{
          const {{ value, done }} = await reader.read();
          if (done) {{
            break;
          }}
          const incoming = value || new Uint8Array(0);
          const merged = new Uint8Array(carry.byteLength + incoming.byteLength);
          merged.set(carry, 0);
          merged.set(incoming, carry.byteLength);
          const playableLength = Math.floor(merged.byteLength / bytesPerFrame) * bytesPerFrame;
          const playable = merged.slice(0, playableLength);
          carry = merged.slice(playableLength);
          if (playable.byteLength > 0) {{
            pcmChunks.push(playable);
            schedulePcmChunk(playable, sampleRate, channels);
          }}
        }}
        if (statusTimer) {{
          clearInterval(statusTimer);
          statusTimer = null;
        }}
        await pollStatus(startPayload.status_url);
        const resultResponse = await fetch(startPayload.result_url);
        const resultPayload = await resultResponse.json();
        setRunStatus(resultPayload.run_status || "Done.");
        if (pcmChunks.length > 0) {{
          const allPcm = concatUint8Arrays(pcmChunks);
          const wavBlob = encodeWav(allPcm, sampleRate, channels);
          currentObjectUrl = URL.createObjectURL(wavBlob);
          audioOutput.src = currentObjectUrl;
          const totalSeconds = allPcm.byteLength / (sampleRate * channels * 2);
          audioMeta.textContent = `Buffered WAV ready: ${{totalSeconds.toFixed(2)}}s, ${{pcmChunks.length}} chunks.`;
        }}
      }} catch (error) {{
        if (String(error && error.name) === "AbortError") {{
          setRunStatus("Stream closed.");
        }} else {{
          setRunStatus("Error: " + error);
        }}
      }} finally {{
        generateBtn.disabled = false;
        closeBtn.disabled = currentStreamId === null;
      }}
    }}

    generateBtn.addEventListener("click", startGenerate);
    closeBtn.addEventListener("click", closeCurrentStream);
    refreshWarmupStatus();
  </script>
</body>
</html>
"""


def build_app(state: WebRuntimeState) -> FastAPI:
    """创建 FastAPI app，并把 simple_moss 本地五接口映射为 HTTP 接口。

    调用方：main 入口初始化 runtime 后调用。
    """

    app = FastAPI(title="simple_moss Web Demo")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_render_index_html(state))

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_dir": str(state.adapter.model_dir),
            "checkpoint_path": str(state.adapter.checkpoint_path),
            "audio_tokenizer_path": str(state.adapter.audio_tokenizer_path),
            "cpu_threads": state.cpu_threads,
            "max_new_frames": state.max_new_frames,
            "voice_clone_max_text_tokens": state.voice_clone_max_text_tokens,
            "warmup_ready": state.warmup_ready,
            "warmup_error": state.warmup_error,
        }

    @app.get("/api/warmup-status")
    async def warmup_status() -> dict[str, object]:
        return {
            "ready": state.warmup_ready,
            "error": state.warmup_error,
            "result": state.warmup_result,
            "status_text": _status_text(state),
        }

    @app.post("/api/generate-stream/start")
    async def generate_stream_start(
        request: Request,
        text: str = Form(...),
        max_new_frames: int = Form(375),
        voice_clone_max_text_tokens: int = Form(32),
        chunk_pause_seconds: float = Form(1.5),
    ):
        resolved_text = str(text or "").strip()
        if not resolved_text:
            return JSONResponse(status_code=400, content={"error": "text is required."})
        start_response = state.facade.start(
            resolved_text,
            max_new_frames=_coerce_int(max_new_frames, state.max_new_frames, minimum=1, maximum=2048),
            voice_clone_max_text_tokens=_coerce_int(
                voice_clone_max_text_tokens,
                state.voice_clone_max_text_tokens,
                minimum=1,
                maximum=300,
            ),
            chunk_pause_seconds=_coerce_float(
                chunk_pause_seconds,
                state.chunk_pause_seconds,
                minimum=0.0,
                maximum=10.0,
            ),
        )
        stream_id = str(start_response["stream_id"])
        root_path = str(request.scope.get("root_path") or "")
        return {
            "stream_id": stream_id,
            "audio_url": f"{root_path}/api/generate-stream/{stream_id}/audio",
            "status_url": f"{root_path}/api/generate-stream/{stream_id}/status",
            "result_url": f"{root_path}/api/generate-stream/{stream_id}/result",
            "sample_rate": 48000,
            "channels": 2,
            "run_status": start_response.get("run_status") or "Streaming...",
        }

    @app.get("/api/generate-stream/{stream_id}/audio")
    async def generate_stream_audio(stream_id: str):
        try:
            state.facade.status(stream_id)
            audio_iter = state.facade.audio(stream_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "stream not found"})

        def _iter_audio() -> Iterator[bytes]:
            yield from audio_iter

        return StreamingResponse(
            _iter_audio(),
            media_type="audio/L16; rate=48000; channels=2",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/generate-stream/{stream_id}/status")
    async def generate_stream_status(stream_id: str):
        try:
            return state.facade.status(stream_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "stream not found"})

    @app.get("/api/generate-stream/{stream_id}/result")
    async def generate_stream_result(stream_id: str):
        try:
            result = state.facade.result(stream_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        if result is None:
            return JSONResponse(status_code=202, content={"run_status": "Streaming..."})
        return result

    @app.post("/api/generate-stream/{stream_id}/close")
    async def generate_stream_close(stream_id: str):
        try:
            state.facade.close(stream_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        return {"closed": True}

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 Web 测试服务命令行参数。

    调用方：main。
    """

    parser = argparse.ArgumentParser(description="simple_moss Web 流式测试服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--model-dir", default=None, help="ONNX 模型目录（默认使用 models/）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认 generated_audio/）")
    parser.add_argument("--cpu-threads", type=int, default=1, help="ORT CPU 线程数")
    parser.add_argument("--max-new-frames", type=int, default=375, help="最大生成帧数")
    parser.add_argument("--voice-clone-max-text-tokens", type=int, default=32, help="每个文本 chunk 最大 token 数")
    parser.add_argument("--chunk-pause-seconds", type=float, default=1.5, help="相邻 chunk 之间的静音时长基准（秒）")
    parser.add_argument("--skip-warmup", action="store_true", help="跳过启动 warmup")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """初始化 simple_moss runtime 并启动 uvicorn。

    调用方：python -m simple_moss.test_web_api。
    """

    args = _parse_args(argv)
    log.info("初始化 adapter（model_dir=%s）", args.model_dir or "默认")
    t0 = time.perf_counter()
    adapter = create_default_adapter(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        cpu_threads=max(1, int(args.cpu_threads)),
        max_new_frames=max(1, int(args.max_new_frames)),
    )
    log.info("初始化完成，耗时 %.2f s", time.perf_counter() - t0)

    warmup_result: dict[str, object] | None = None
    warmup_error: str | None = None
    if not bool(args.skip_warmup):
        try:
            log.info("开始 warmup")
            t0 = time.perf_counter()
            warmup_result = warmup_runtime(adapter)
            log.info("Warmup 完成，耗时 %.2f s", warmup_result.get("elapsed_seconds", time.perf_counter() - t0))
        except Exception as exc:
            warmup_error = str(exc)
            log.exception("Warmup 失败，服务仍会启动")
    else:
        log.info("按参数跳过 warmup")

    facade = MossStreamApiFacade(
        adapter,
        max_new_frames=max(1, int(args.max_new_frames)),
        voice_clone_max_text_tokens=max(1, int(args.voice_clone_max_text_tokens)),
        chunk_pause_seconds=max(0.0, float(args.chunk_pause_seconds)),
    )
    state = WebRuntimeState(
        adapter=adapter,
        facade=facade,
        model_dir=args.model_dir,
        cpu_threads=max(1, int(args.cpu_threads)),
        max_new_frames=max(1, int(args.max_new_frames)),
        voice_clone_max_text_tokens=max(1, int(args.voice_clone_max_text_tokens)),
        chunk_pause_seconds=max(0.0, float(args.chunk_pause_seconds)),
        warmup_result=warmup_result,
        warmup_error=warmup_error,
    )
    app = build_app(state)
    uvicorn.run(app, host=str(args.host), port=int(args.port))


if __name__ == "__main__":
    main()
