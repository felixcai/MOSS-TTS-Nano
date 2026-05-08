"""
simple_moss OpenAI-compatible 本地 HTTP 全流程测试脚本

运行方式（在项目根目录下）：
    python -m simple_moss.test_openai_local
    python -m simple_moss.test_openai_local --base-url http://127.0.0.1:8090 --text "你好，世界。"

前置条件：
    需要先单独启动 OpenAI-compatible TTS 服务，例如：
    python -m simple_moss.openai_moss

流程：
    1. GET /v1/models
    2. GET /v1/audio/voices
    3. GET /health（TTS 前）
    4. POST /v1/audio/speech
    5. GET /health（TTS 后）
    6. 将收到的 PCM s16le 写成 WAV，保存到 ./simple_moss_openai_test_output.wav
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _write_pcm_to_wav(
    output_path: Path,
    pcm_chunks: list[bytes],
    sample_rate: int,
    channels: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # s16le = 2 bytes per sample
        wav.setframerate(sample_rate)
        for chunk in pcm_chunks:
            wav.writeframes(chunk)
    total_bytes = sum(len(c) for c in pcm_chunks)
    total_seconds = total_bytes / (sample_rate * channels * 2)
    log.info("WAV 已保存 → %s（%.2f 秒）", output_path, total_seconds)


def _pretty_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _request_json(method: str, url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
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


def _log_step_header(title: str) -> None:
    log.info("=" * 60)
    log.info(title)


def _fetch_models(base_url: str) -> dict[str, object]:
    url = urljoin(base_url, "/v1/models")
    _log_step_header("Step 1: 访问 /v1/models")
    result = _request_json("GET", url)
    log.info("/v1/models 返回：\n%s", _pretty_json(result))
    return result


def _fetch_voices(base_url: str) -> dict[str, object]:
    url = urljoin(base_url, "/v1/audio/voices")
    _log_step_header("Step 2: 访问 /v1/audio/voices")
    result = _request_json("GET", url)
    log.info("/v1/audio/voices 返回：\n%s", _pretty_json(result))
    return result


def _fetch_health(base_url: str, stage: str) -> dict[str, object]:
    url = urljoin(base_url, "/health")
    _log_step_header(f"Step {stage}: 访问 /health")
    result = _request_json("GET", url)
    log.info("/health（%s）返回：\n%s", stage, _pretty_json(result))
    return result


def _stream_tts(
    base_url: str,
    text: str,
    model: str,
    voice: str,
    response_format: str,
    chunk_size: int,
) -> tuple[list[bytes], int, int]:
    url = urljoin(base_url, "/v1/audio/speech")
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": 1.0,
    }

    _log_step_header("Step 4: 访问 /v1/audio/speech")
    log.info("TTS 请求：\n%s", _pretty_json(payload))

    request = Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "audio/pcm",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    pcm_chunks: list[bytes] = []
    sample_rate = 48000
    channels = 2
    started_at = time.perf_counter()
    first_chunk_at: float | None = None

    try:
        with urlopen(request, timeout=300) as response:
            sample_rate = int(response.headers.get("X-Sample-Rate", sample_rate))
            channels = int(response.headers.get("X-Channels", channels))
            log.info(
                "/v1/audio/speech 响应头：sample_rate=%d, channels=%d, content_type=%s",
                sample_rate,
                channels,
                response.headers.get_content_type(),
            )

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    log.info("首帧 PCM 到达，首帧延迟 = %.3f s", first_chunk_at - started_at)
                pcm_chunks.append(chunk)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} POST {url} 失败: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {url} 失败: {exc}") from exc

    log.info(
        "TTS 完成，共收到 %d 个 PCM chunk，总字节 %d",
        len(pcm_chunks),
        sum(len(c) for c in pcm_chunks),
    )
    return pcm_chunks, sample_rate, channels


def run_test(
    base_url: str,
    text: str,
    output_wav: str,
    voice: str,
    response_format: str,
    chunk_size: int,
) -> None:
    base_url = base_url.rstrip("/") + "/"
    _log_step_header("OpenAI-compatible HTTP 测试开始")
    log.info("base_url=%s", base_url.rstrip("/"))
    log.info("text=%r", text)

    models_resp = _fetch_models(base_url)
    voices_resp = _fetch_voices(base_url)
    _fetch_health(base_url, "3（TTS 前）")

    model_list = models_resp.get("data") or []
    if not isinstance(model_list, list) or not model_list:
        raise RuntimeError("/v1/models 未返回可用模型")
    first_model = model_list[0]
    if not isinstance(first_model, dict) or not first_model.get("id"):
        raise RuntimeError("/v1/models 返回格式异常，缺少 model id")
    model_id = str(first_model["id"])

    voice_list = voices_resp.get("voices") or []
    if not isinstance(voice_list, list) or not voice_list:
        raise RuntimeError("/v1/audio/voices 未返回可用 voice")
    available_voice_ids = [
        str(item.get("id"))
        for item in voice_list
        if isinstance(item, dict) and item.get("id") is not None
    ]
    if voice not in available_voice_ids:
        log.warning("指定 voice=%r 不在服务返回列表中，仍将继续请求", voice)

    pcm_chunks, sample_rate, channels = _stream_tts(
        base_url=base_url,
        text=text,
        model=model_id,
        voice=voice,
        response_format=response_format,
        chunk_size=chunk_size,
    )

    _fetch_health(base_url, "5（TTS 后）")

    _log_step_header("Step 6: 保存 WAV")
    if pcm_chunks:
        output_path = Path(output_wav).expanduser().resolve()
        _write_pcm_to_wav(output_path, pcm_chunks, sample_rate, channels)
    else:
        log.warning("未收到任何 PCM 数据，跳过 WAV 写入")

    log.info("HTTP 全流程测试完成")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="simple_moss OpenAI-compatible 本地 HTTP 全流程测试")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8090",
        help="OpenAI-compatible 服务地址，默认 http://127.0.0.1:8090",
    )
    parser.add_argument(
        "--text",
        default="你好，这是一段来自 simple_moss OpenAI-compatible 接口的本地测试语音。",
        help="要合成的文本",
    )
    parser.add_argument(
        "--output",
        default="simple_moss_openai_test_output.wav",
        help="输出 WAV 文件路径",
    )
    parser.add_argument(
        "--voice",
        default="default",
        help="要请求的 voice id，默认 default",
    )
    parser.add_argument(
        "--response-format",
        default="pcm",
        help="请求的 response_format，默认 pcm",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="流式读取响应时的 chunk 大小（字节），默认 4096",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_test(
        base_url=args.base_url,
        text=args.text,
        output_wav=args.output,
        voice=args.voice,
        response_format=args.response_format,
        chunk_size=max(1, int(args.chunk_size)),
    )
