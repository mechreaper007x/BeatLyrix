"""
RapRank — Whisper Transcription Microservice
FastAPI service wrapping faster-whisper for multilingual rap audio transcription.
Supports Hindi, English, and code-switched (Hinglish) audio.
Hosted as a Hugging Face Docker Space on port 7860.
"""

import os
import tempfile
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Model — loaded ONCE at module level, not per-request.
# "small"  → best accuracy/speed trade-off on free CPU tier
# "int8"   → quantised inference; halves memory, keeps speed
# device="cpu" → HF free tier has no GPU
# ─────────────────────────────────────────────────────────────
logger.info("Loading WhisperModel (small / int8 / cpu) …")
model = WhisperModel("small", device="cpu", compute_type="int8")
logger.info("WhisperModel ready.")

# ─────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="RapRank Whisper Service",
    description="Multilingual audio transcription (Hindi / English / Hinglish) powered by faster-whisper.",
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────────
# GET /health
# Spring Boot liveness probe — confirm the service is up before
# sending transcription requests.
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "raprank-whisper"}


# ─────────────────────────────────────────────────────────────
# POST /transcribe
# Accepts a multipart audio file, returns full transcript +
# word-level timestamps as JSON.
# ─────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
    # ── Validate upload ──────────────────────────────────────
    if not file or not file.filename:
        raise HTTPException(
            status_code=400,
            detail={"error": "No file uploaded. Provide an audio file via the 'file' field."},
        )

    allowed_extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unsupported file format '{ext}'. Accepted: {', '.join(allowed_extensions)}"},
        )

    # ── Read bytes and sanity-check ──────────────────────────
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "Uploaded file is empty."},
        )

    # ── Write to a temp file (faster-whisper needs a path) ───
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        logger.info("Transcribing '%s' (%d bytes) → %s", file.filename, len(audio_bytes), tmp_path)

        # ── Transcription ────────────────────────────────────
        # language=None  → auto-detect (handles Hindi, English, Hinglish)
        # task="transcribe" → do not translate, return original language
        # word_timestamps=True → required for future beat-sync features
        try:
            segments, info = model.transcribe(
                tmp_path,
                language=None,
                task="transcribe",
                word_timestamps=True,
            )
        except Exception as exc:
            logger.exception("Transcription failed for '%s': %s", file.filename, exc)
            return JSONResponse(
                status_code=500,
                content={"error": "Transcription failed. Check server logs for details."},
            )

        # ── Collect results ──────────────────────────────────
        full_text_parts: list[str] = []
        words_data: list[dict] = []

        for segment in segments:
            full_text_parts.append(segment.text)
            if segment.words:
                for word in segment.words:
                    words_data.append(
                        {
                            "word": word.word,
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                        }
                    )

        full_text = "".join(full_text_parts).strip()
        logger.info(
            "Done. lang=%s (p=%.2f), words=%d",
            info.language,
            info.language_probability,
            len(words_data),
        )

        return JSONResponse(
            content={
                "text": full_text,
                "detected_language": info.language,
                "language_probability": round(info.language_probability, 4),
                "words": words_data,
            }
        )

    finally:
        # ── Always remove the temp file, even on exception ───
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning("Could not delete temp file '%s': %s", tmp_path, e)
