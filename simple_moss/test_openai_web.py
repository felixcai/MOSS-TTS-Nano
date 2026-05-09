"""
simple_moss OpenAI-compatible Web 全流程测试服务

运行方式（在项目根目录下）：
    python -m simple_moss.test_openai_web
    python -m simple_moss.test_openai_web --base-url http://127.0.0.1:8090 --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


DEFAULT_TEXT = "你好，这是一段来自 simple_moss OpenAI-compatible 接口的 Web 测试语音。"


def _request_json(method: str, url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    """发起 HTTP 请求并解析返回的 JSON，逻辑参考 test_openai_local.py"""
    data_bytes = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url=url, data=data_bytes, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {method.upper()} {url} 失败: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method.upper()} {url} 失败: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method.upper()} {url} 返回的不是合法 JSON: {body[:300]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method.upper()} {url} 返回 JSON 类型异常: {type(parsed).__name__}")
    return parsed


def _render_index_html(base_url: str) -> str:
    """渲染 Web 界面 HTML，样式和 PCM 播放逻辑参考 test_web_api.py"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenAI TTS Web Demo</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    body {{ margin: 0; background: #f5f5f5; color: #1f2933; }}
    .page {{ max-width: 1080px; margin: 0 auto; padding: 28px; }}
    .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 14px; padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 0.8fr); gap: 18px; margin-top: 18px; }}
    h1, h2 {{ margin: 0 0 8px; }}
    h2 {{ font-size: 18px; border-bottom: 1px solid #eee; padding-bottom: 8px; margin-bottom: 12px; }}
    textarea {{ width: 100%; min-height: 100px; resize: vertical; font: inherit; box-sizing: border-box; padding: 8px; border-radius: 6px; border: 1px solid #ccc; }}
    button {{ border: 0; border-radius: 10px; padding: 10px 16px; background: #2563eb; color: #fff; cursor: pointer; font: inherit; }}
    button:disabled {{ opacity: 0.55; cursor: not-allowed; }}
    .status-box {{ white-space: pre-wrap; font-family: monospace; font-size: 13px; padding: 12px; background: #f1f5f9; border-radius: 10px; min-height: 44px; max-height: 300px; overflow-y: auto; word-wrap: break-word; }}
    .meta {{ color: #64748b; font-size: 13px; margin-bottom: 8px; }}
    audio {{ width: 100%; margin-top: 8px; }}
    input[type="text"] {{ padding: 4px; border: 1px solid #ccc; border-radius: 4px; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #0f172a; color: #e5e7eb; }}
      .panel {{ background: #111827; border-color: #334155; }}
      .status-box {{ background: #0b1220; border-color: #334155; }}
      .meta {{ color: #94a3b8; }}
      textarea, input[type="text"] {{ background: #1f2937; color: #e5e7eb; border-color: #4b5563; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel">
      <h1>OpenAI-compatible TTS Web Demo</h1>
      <div class="meta">目标 OpenAI 兼容服务地址: {base_url}</div>
      <button id="refresh-info-btn">刷新服务信息 (/v1/models, /v1/audio/voices)</button>
    </div>

    <div class="grid">
      <section class="panel">
        <h2>1. 接口信息</h2>
        <div class="meta">/v1/models</div>
        <div id="models-info" class="status-box">等待查询...</div>
        
        <div class="meta" style="margin-top: 12px;">/v1/audio/voices</div>
        <div id="voices-info" class="status-box">等待查询...</div>
      </section>

      <section class="panel">
        <h2>2. 合成测试</h2>
        <div style="margin-bottom: 12px;">
          <textarea id="text-input">{DEFAULT_TEXT}</textarea>
        </div>
        <div style="margin-bottom: 12px;">
          <label for="voice-input" class="meta">voice（可填 default 或内置音色名）</label>
          <input id="voice-input" type="text" value="default" style="width: 100%; box-sizing: border-box;">
        </div>
        <div class="meta" style="margin-top: 16px;">/health (TTS 前)</div>
        <div id="health-before" class="status-box">等待测试...</div>

        <button id="generate-btn" style="margin-top: 12px; margin-bottom: 12px; width: 100%;">开始调用 /v1/audio/speech</button>

        <div class="meta">实时流式状态</div>
        <div id="run-status" class="status-box" style="min-height: 20px; padding: 8px; margin-bottom: 8px;">Idle.</div>
        <audio id="audio-output" controls></audio>
        
        <div class="meta" style="margin-top: 12px;">/health (TTS 后)</div>
        <div id="health-after" class="status-box">等待测试...</div>
      </section>
    </div>
  </div>

  <script>
    const refreshInfoBtn = document.getElementById("refresh-info-btn");
    const modelsInfo = document.getElementById("models-info");
    const voicesInfo = document.getElementById("voices-info");
    const healthBefore = document.getElementById("health-before");
    const healthAfter = document.getElementById("health-after");
    
    const textInput = document.getElementById("text-input");
    const voiceInput = document.getElementById("voice-input");
    const generateBtn = document.getElementById("generate-btn");
    const runStatus = document.getElementById("run-status");
    const audioOutput = document.getElementById("audio-output");

    let audioContext = null;
    let nextPlaybackTime = 0;
    let pcmChunks = [];
    let currentObjectUrl = null;

    async function loadInfo() {{
      try {{
        modelsInfo.textContent = "Loading...";
        const mResp = await fetch("/api/models");
        modelsInfo.textContent = JSON.stringify(await mResp.json(), null, 2);
      }} catch (e) {{ modelsInfo.textContent = "Error: " + e; }}

      try {{
        voicesInfo.textContent = "Loading...";
        const vResp = await fetch("/api/voices");
        const vData = await vResp.json();
        voicesInfo.textContent = JSON.stringify(vData, null, 2);
        
      }} catch (e) {{ voicesInfo.textContent = "Error: " + e; }}
    }}

    async function fetchHealth(element) {{
      try {{
        element.textContent = "Loading health...";
        const resp = await fetch("/api/health");
        element.textContent = JSON.stringify(await resp.json(), null, 2);
      }} catch (e) {{ element.textContent = "Error: " + e; }}
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
      if (!bytes || bytes.byteLength <= 0) return;
      if (!audioContext) {{
        audioContext = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate }});
        nextPlaybackTime = audioContext.currentTime;
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
        for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
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

    async function startGenerate() {{
      generateBtn.disabled = true;
      runStatus.textContent = "开始进行流式 TTS...";
      healthAfter.textContent = "等待测试...";
      
      pcmChunks = [];
      if (currentObjectUrl) {{
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
      }}
      audioOutput.removeAttribute("src");
      audioOutput.load();
      if (audioContext) {{
        try {{ await audioContext.close(); }} catch(e) {{}}
        audioContext = null;
      }}

      // TTS 调用前，获取并显示 /health
      await fetchHealth(healthBefore);

      try {{
        const formData = new FormData();
        formData.set("text", textInput.value);
        formData.set("voice", (voiceInput.value || "default").trim() || "default");

        const response = await fetch("/api/speech", {{
          method: "POST",
          body: formData
        }});
        
        if (!response.ok) {{
          const errData = await response.json();
          throw new Error(errData.error || "HTTP " + response.status);
        }}

        const sampleRate = Number(response.headers.get("x-sample-rate") || 48000);
        const channels = Number(response.headers.get("x-channels") || 2);
        
        runStatus.textContent = "首字节到达，正在接收并播放音频流...";

        const reader = response.body.getReader();
        let carry = new Uint8Array(0);
        const bytesPerFrame = channels * 2;
        let chunkCount = 0;
        
        while (true) {{
          const {{ value, done }} = await reader.read();
          if (done) break;
          
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
            chunkCount++;
            runStatus.textContent = `流式收包并播放中... (已接收 ${{chunkCount}} 个块)`;
          }}
        }}

        runStatus.textContent = "流式结束，正在组合并生成全量 WAV...";
        
        if (pcmChunks.length > 0) {{
          const allPcm = concatUint8Arrays(pcmChunks);
          const wavBlob = encodeWav(allPcm, sampleRate, channels);
          currentObjectUrl = URL.createObjectURL(wavBlob);
          audioOutput.src = currentObjectUrl;
          runStatus.textContent = "完成。可通过播放器重复回放生成的音频。";
        }} else {{
          runStatus.textContent = "流已结束，但是没有收到任何音频数据。";
        }}

      }} catch (e) {{
        runStatus.textContent = "发生错误: " + e;
      }} finally {{
        // TTS 调用后，再次获取并显示 /health
        await fetchHealth(healthAfter);
        generateBtn.disabled = false;
      }}
    }}

    refreshInfoBtn.addEventListener("click", loadInfo);
    generateBtn.addEventListener("click", startGenerate);

    // 页面加载后自动请求一次基础信息
    loadInfo();
  </script>
</body>
</html>
"""


def build_app(base_url: str) -> FastAPI:
    """创建 FastAPI 代理服务器，提供 HTML UI 并将请求转发到实际的 OpenAI-compatible TTS 服务。"""
    app = FastAPI(title="OpenAI-compatible Web Demo")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_render_index_html(base_url))

    @app.get("/api/models")
    def get_models():
        url = urljoin(base_url, "/v1/models")
        try:
            return _request_json("GET", url)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/api/voices")
    def get_voices():
        url = urljoin(base_url, "/v1/audio/voices")
        try:
            return _request_json("GET", url)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/api/health")
    def get_health():
        url = urljoin(base_url, "/health")
        try:
            return _request_json("GET", url)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/api/speech")
    def post_speech(
        text: str = Form(...),
        voice: str = Form("default"),
    ):
        """代理调用 /v1/audio/speech 接口，并将结果通过 StreamingResponse 逐块流式返回给前端"""
        url = urljoin(base_url, "/v1/audio/speech")
        payload = {
            "model": "moss-tts",
            "input": text,
            "voice": str(voice or "default"),
            "response_format": "pcm",
            "speed": 1.0,
        }
        
        request = Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "audio/pcm",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        
        try:
            # 开启网络连接读取流
            response = urlopen(request, timeout=300)
            
            # 透传服务端的采样率和声道数
            sample_rate = response.headers.get("X-Sample-Rate", "48000")
            channels = response.headers.get("X-Channels", "2")
            
            def _iter_audio() -> Iterator[bytes]:
                with response:
                    while True:
                        chunk = response.read(4096)
                        if not chunk:
                            break
                        yield chunk
                        
            return StreamingResponse(
                _iter_audio(),
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": sample_rate,
                    "X-Channels": channels,
                    "Cache-Control": "no-store",
                }
            )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return JSONResponse(status_code=exc.code, content={"error": f"HTTP {exc.code}: {detail}"})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="simple_moss OpenAI-compatible Web 测试服务")
    parser.add_argument("--base-url", default="http://127.0.0.1:8090", help="OpenAI-compatible 服务地址")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="监听端口 (默认: 8001)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    base_url = args.base_url.rstrip("/")
    log.info("启动 Web 界面代理服务，目标后端 base_url=%s", base_url)
    log.info("将在浏览器中访问 http://%s:%d 开始测试", "127.0.0.1" if args.host == "0.0.0.0" else args.host, args.port)
    
    app = build_app(base_url)
    uvicorn.run(app, host=str(args.host), port=int(args.port))


if __name__ == "__main__":
    main()
