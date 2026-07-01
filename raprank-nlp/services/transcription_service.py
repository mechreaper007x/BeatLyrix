"""
Async HTTP client for the raprank-whisper HF Space transcription service.

Set the WHISPER_API_URL env var to override the default HF Space URL
(useful for local development with a locally-running whisper service).
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

WHISPER_BASE_URL: str = os.getenv(
    "WHISPER_API_URL",
    "https://mechreaper007x-raprank-whisper.hf.space",
).rstrip("/")


async def transcribe_audio(audio_bytes: bytes, filename: str) -> dict:
    """
    POST audio to raprank-whisper and return the parsed JSON response.

    Expected response shape:
        {
            "text": str,
            "detected_language": str,        # ISO code, e.g. "hi" or "en"
            "language_probability": float,
            "words": [{"word": str, "start": float, "end": float}, ...]
        }

    Raises:
        httpx.HTTPStatusError  — non-2xx from whisper service
        httpx.TimeoutException — service took >120 s (common on cold start)
    """
    url = f"{WHISPER_BASE_URL}/transcribe"
    logger.info("Sending %d bytes to %s", len(audio_bytes), url)

    async with httpx.AsyncClient(timeout=900.0) as client:
        response = await client.post(
            url,
            files={"file": (filename, audio_bytes, _content_type(filename))},
        )
        response.raise_for_status()
        data = response.json()

    logger.info(
        "Whisper response: lang=%s (p=%.2f), words=%d",
        data.get("detected_language"),
        data.get("language_probability", 0.0),
        len(data.get("words", [])),
    )
    return data


def _content_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
    }.get(ext, "application/octet-stream")
