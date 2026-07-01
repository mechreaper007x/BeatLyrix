"""
RapRank — Whisper Transcription Microservice
FastAPI service wrapping faster-whisper for multilingual rap audio transcription.
Supports Hindi, English, and code-switched (Hinglish) audio.
Hosted as a Hugging Face Docker Space on port 7860.
"""

import os
import tempfile
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from transformers import pipeline

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
# "large-v3-turbo" → best accuracy/speed balance for multilingual rap
# "int8"   → quantised inference; halves memory, keeps speed
# device="cpu" → HF free tier has no GPU
# ─────────────────────────────────────────────────────────────
logger.info("Loading WhisperModel (large-v3-turbo-ct2 / int8 / cpu) …")
model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cpu", compute_type="int8")
logger.info("WhisperModel ready.")

logger.info("Loading Hinglish LID transformer model...")
try:
    lid_pipeline = pipeline("token-classification", model="l3cube-pune/hing-bert-lid", device=-1)
    logger.info("LID model ready.")
except Exception as exc:
    logger.error("Failed to load LID model: %s", exc)
    lid_pipeline = None

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
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Query(default=None, description="Optional ISO-639-1 language code hint."),
    lyrics: str | None = Form(default=None, description="Optional track lyrics for LID classification.")
) -> JSONResponse:
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

        whisper_lang = None
        if language:
            lang_cleaned = language.strip().lower()
            if lang_cleaned in ("hi", "hindi"):
                whisper_lang = "hi"
            elif lang_cleaned in ("en", "english"):
                whisper_lang = "en"
        elif lyrics and lid_pipeline:
            try:
                # Classify lyrics text (up to 300 words to avoid token limit issues)
                words_list = lyrics.split()[:300]
                truncated_lyrics = " ".join(words_list)
                results = lid_pipeline(truncated_lyrics)
                
                hi_count = 0
                en_count = 0
                for r in results:
                    entity = r.get("entity", "").lower()
                    if "hi" in entity or "label_1" in entity:
                        hi_count += 1
                    elif "en" in entity or "label_0" in entity:
                        en_count += 1
                
                logger.info("LID results: hi=%d, en=%d", hi_count, en_count)
                if hi_count > 5:
                    whisper_lang = "hi"
                    logger.info("Transformer classified lyrics as Hindi/Hinglish. Forcing Whisper to 'hi'.")
                else:
                    whisper_lang = "en"
                    logger.info("Transformer classified lyrics as English. Forcing Whisper to 'en'.")
            except Exception as exc:
                logger.warning("LID classification failed: %s", exc)

        logger.info(
            "Transcribing '%s' (%d bytes) → %s (language hint: %s)", 
            file.filename, len(audio_bytes), tmp_path, whisper_lang
        )

        # ── Transcription ────────────────────────────────────
        # task="transcribe" → do not translate, return original language
        # word_timestamps=True → required for future beat-sync features
        try:
            segments, info = model.transcribe(
                tmp_path,
                language=whisper_lang,
                task="transcribe",
                word_timestamps=True,
                beam_size=1,
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
                            "probability": round(word.probability, 4)
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
