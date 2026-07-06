"""
RapRank — Whisper Transcription Microservice
FastAPI service wrapping faster-whisper for multilingual rap audio transcription.
Supports Hindi, English, and code-switched (Hinglish) audio.
Hosted as a Hugging Face Docker Space on port 7860.
"""

import asyncio
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
logger.info("Loading WhisperModel (fast-whisper-hinglish-Apex-ct2 / int8 / cpu) …")
model = WhisperModel("AlexAnoshka/fast-whisper-hinglish-Apex-ct2", device="cpu", compute_type="int8")
logger.info("WhisperModel ready.")

logger.info("Loading Hinglish LID transformer model...")
try:
    lid_pipeline = pipeline("token-classification", model="l3cube-pune/hing-bert-lid", device=-1)
    logger.info("LID model ready.")
except Exception as exc:
    logger.error("Failed to load LID model: %s", exc)
    lid_pipeline = None

# ─────────────────────────────────────────────────────────────
# Single-flight concurrency guard.
# faster-whisper's `model.transcribe()` is synchronous, CPU-bound, and this
# Space runs a single Uvicorn worker (see Dockerfile) with no GPU. Running it
# directly inside an `async def` endpoint blocks the *entire* event loop for
# the whole duration of inference -- including `/health` and any other
# concurrent request. `_transcription_lock` ensures at most one transcription
# runs at a time (queuing the rest with a clear 429 rather than silently
# piling up CPU contention that makes every request slower and more
# failure-prone); `asyncio.to_thread` in the endpoint itself moves the actual
# blocking call off the event loop so /health stays responsive meanwhile.
# ─────────────────────────────────────────────────────────────
_transcription_lock = asyncio.Semaphore(1)

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


def _run_transcription(tmp_path: str, whisper_lang: str | None, initial_prompt: str | None):
    """
    Synchronous, CPU-bound transcription work: the actual model call plus
    forcing faster-whisper's lazy `segments` generator by iterating it. Must
    be called via `asyncio.to_thread` from the endpoint, never awaited
    directly -- this is what keeps the event loop (and /health) responsive
    while a transcription is running.
    """
    segments, info = model.transcribe(
        tmp_path,
        language=whisper_lang,
        task="transcribe",
        word_timestamps=True,
        beam_size=1,
        initial_prompt=initial_prompt,
    )

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
                        "probability": round(word.probability, 4),
                    }
                )

    full_text = "".join(full_text_parts).strip()
    return full_text, info, words_data


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
        initial_prompt = None
        if language:
            lang_cleaned = language.strip().lower()
            if lang_cleaned in ("hi", "hindi"):
                whisper_lang = "hi"
            elif lang_cleaned in ("en", "english"):
                whisper_lang = "en"
            elif lang_cleaned == "hinglish":
                # To get Hinglish in Romanized characters, use the English decoder
                # but bias it with a code-switched Hinglish initial prompt
                whisper_lang = "en"
                initial_prompt = (
                    "What you do it for? (Uh) What you do it for? "
                    "poochho inse aaya kaun? jab time mera aaya to palat paaya kaaya kaun? "
                    "dilli ka launda Japan mein chill, Arigato, aram se kill, kaise ho bhai, kya chal raha hai"
                )
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
            "Transcribing '%s' (%d bytes) → %s (language hint: %s, prompt: %s)",
            file.filename, len(audio_bytes), tmp_path, whisper_lang, "Yes" if initial_prompt else "No"
        )

        # ── Transcription ────────────────────────────────────
        # task="transcribe" → do not translate, return original language
        # word_timestamps=True → required for future beat-sync features
        #
        # `model.transcribe()` only returns a lazy generator -- the actual
        # CPU-bound decoding happens while iterating `segments`, so both the
        # call AND the collection loop must run inside the same
        # `asyncio.to_thread` to actually get off the event loop (offloading
        # only the call and then iterating the generator back on the loop
        # would silently undo the fix). `_transcription_lock` caps this
        # Space to one transcription at a time so concurrent uploads queue
        # with a clear 429 instead of thrashing the single CPU worker.
        if _transcription_lock.locked():
            raise HTTPException(
                status_code=429,
                detail={"error": "Another transcription is already in progress on this instance. Please retry shortly."},
            )

        async with _transcription_lock:
            try:
                full_text, info, words_data = await asyncio.to_thread(
                    _run_transcription, tmp_path, whisper_lang, initial_prompt
                )
            except TimeoutError as exc:
                logger.exception("Transcription timed out for '%s': %s", file.filename, exc)
                return JSONResponse(
                    status_code=504,
                    content={"error": "Transcription timed out. Try a shorter clip or retry."},
                )
            except Exception as exc:
                logger.exception("Transcription failed for '%s': %s", file.filename, exc)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Transcription failed: {exc}"},
                )

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
