"""
RapRank NLP Scoring Service v2
FastAPI application exposing two endpoints:

  GET  /health                   — liveness probe
  POST /analyze                  — score pre-transcribed lyrics (text-only)
  POST /transcribe-and-analyze   — score an audio file end-to-end

Scoring weights:
  With flow  → rhyme 30 | syllable 20 | alliteration 15 | vocabulary 15 | flow 20
  Text-only  → rhyme 35 | syllable 25 | alliteration 20 | vocabulary 20
"""
from __future__ import annotations

import logging
import os
import httpx

async def update_status(track_id: int | None, status: str):
    if not track_id:
        return
    try:
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8080").rstrip("/")
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.post(f"{backend_url}/api/tracks/{track_id}/status?status={status}")
            if not res.is_success:
                logger.warning("Failed to update status on backend: %d %s", res.status_code, res.text)
    except Exception as exc:
        logger.warning("Error reporting status %s to backend: %s", status, exc)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from models.schemas import (
    AnalyzeRequest,
    FlowMetadata,
    ScoreBreakdown,
)
from services import (
    alliteration_service,
    rhyme_service,
    syllable_service,
    vocabulary_service,
    wordplay_service,
)
from services.flow_service import calculate_beat_sync
from services.alignment_service import align_structured_lyrics_to_whisper
from services.language_utils import detect_language
from services.transcription_service import transcribe_audio
from services.separation_service import separate_vocals
from config import scoring_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RapRank NLP Service",
    description=(
        "Multilingual (Hindi / English / Hinglish) rap lyric scoring. "
        "Scores syllable density, end rhyme, alliteration, vocabulary uniqueness, "
        "and beat-sync flow."
    ),
    version="2.0.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "raprank-nlp"}


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze
# Accepts pre-transcribed lyrics + optional word timestamps.
# When `words` are provided, flow/beat-sync is attempted.
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=ScoreBreakdown)
async def analyze(request: AnalyzeRequest) -> ScoreBreakdown:
    lyrics = request.lyrics.strip()
    if not lyrics:
        raise HTTPException(
            status_code=400,
            detail={"error": "lyrics cannot be empty"},
        )

    lang = (
        request.language
        if request.language and request.language != "auto"
        else detect_language(lyrics)
    )

    # ── Run all text-based scoring services ───────────────────────────────
    await update_status(request.track_id, "ANALYZING_TEXT")
    syllable_score, avg_syl, syllable_weight_score, weight_ratio = syllable_service.calculate(lyrics)
    rhyme_score, rhyme_pairs, multisyl_count, internal_score, chain_score = rhyme_service.calculate(lyrics)
    allit_score, allit_pairs = alliteration_service.calculate(lyrics)
    vocab_score, ttr = vocabulary_service.calculate(lyrics)
    wordplay_score, wordplay_meta = wordplay_service.calculate(lyrics)

    # ── Flow scoring (only when word timestamps or audio_url is provided) ─────────
    flow_score: float | None = None
    flow_meta: FlowMetadata | None = None

    if request.words:
        words_dicts = [w.model_dump() for w in request.words]
        # No audio bytes in text-only mode — skip beat detection
        # (beat detection needs audio; timestamps alone are insufficient)
        logger.info(
            "/analyze received %d word timestamps but no audio — flow skipped",
            len(words_dicts),
        )
    elif request.audio_url:
        try:
            await update_status(request.track_id, "DOWNLOADING_AUDIO")
            logger.info("Downloading audio from %s for flow scoring...", request.audio_url)
            filename = request.audio_url.split("/")[-1] or "audio.mp3"
            async with httpx.AsyncClient(timeout=60.0) as client:
                audio_res = await client.get(request.audio_url)
                audio_res.raise_for_status()
                audio_bytes = audio_res.content

            # ── Separation ──────────────────────────────────────────
            await update_status(request.track_id, "SEPARATING_AUDIO")
            logger.info("Running audio source separation...")
            separated = separate_vocals(audio_bytes, filename)
            if separated:
                vocals_bytes, accompaniment_bytes = separated
                transcribe_bytes = vocals_bytes
                flow_bytes = accompaniment_bytes
                logger.info("Source separation successful. Vocals and accompaniment isolated.")
            else:
                transcribe_bytes = audio_bytes
                flow_bytes = audio_bytes
                logger.info("Source separation skipped/failed. Using mixed audio.")

            await update_status(request.track_id, "TRANSCRIBING")
            logger.info("Transcribing audio for flow alignment (hint lang: %s)...", lang)
            transcription = await transcribe_audio(transcribe_bytes, filename, lang, lyrics)
            words_raw = transcription.get("words", [])

            if words_raw:
                await update_status(request.track_id, "ANALYZING_FLOW")
                # Map raw Whisper ASR word timestamps to the user's structured lyrics
                words_aligned = align_structured_lyrics_to_whisper(lyrics, words_raw)
                
                raw_score, meta_dict = calculate_beat_sync(
                    transcribe_bytes, flow_bytes, filename, words_aligned
                )
                if "error" not in meta_dict:
                    flow_score = raw_score
                    flow_meta = FlowMetadata(**meta_dict)
                else:
                    logger.warning("Beat sync returned error: %s", meta_dict["error"])
        except Exception as exc:
            logger.exception("Flow analysis from audio_url failed: %s", exc)

    # ── Weighted total from Config ─────────────────────────────────────────
    if flow_score is not None:
        w = scoring_config.MAIN_WEIGHTS["WITH_FLOW"]
        total = (
            rhyme_score * w["rhyme"]
            + syllable_score * w["syllable"]
            + allit_score * w["alliteration"]
            + vocab_score * w["vocabulary"]
            + wordplay_score * w["wordplay"]
            + syllable_weight_score * w["syllable_weight"]
            + flow_score * w["flow"]
        )
    else:
        w = scoring_config.MAIN_WEIGHTS["TEXT_ONLY"]
        total = (
            rhyme_score * w["rhyme"]
            + syllable_score * w["syllable"]
            + allit_score * w["alliteration"]
            + vocab_score * w["vocabulary"]
            + wordplay_score * w["wordplay"]
            + syllable_weight_score * w["syllable_weight"]
        )

    line_count = sum(
        1
        for l in lyrics.split("\n")
        if l.strip() and not (l.strip().startswith("[") and l.strip().endswith("]"))
    )

    return ScoreBreakdown(
        syllable_score=round(syllable_score, 2),
        rhyme_score=round(rhyme_score, 2),
        alliteration_score=round(allit_score, 2),
        vocabulary_score=round(vocab_score, 2),
        flow_score=flow_score,
        total_score=round(total, 2),
        wordplay_score=round(wordplay_score, 2),
        syllable_weight=round(syllable_weight_score, 2),
        word_count=len(lyrics.split()),
        line_count=line_count,
        avg_syllables_per_word=round(avg_syl, 2),
        vocabulary_uniqueness=ttr,
        detected_language=lang,
        double_entendres_count=wordplay_meta["double_entendres_count"],
        puns_count=wordplay_meta["puns_count"],
        similes_count=wordplay_meta["simile_count"],
        metaphors_count=wordplay_meta["metaphor_count"],
        alliteration_pairs=allit_pairs,
        rhyme_pairs=rhyme_pairs,
        multisyllabic_rhyme_count=multisyl_count,
        flow_metadata=flow_meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /transcribe-and-analyze
# Full pipeline: audio → whisper → NLP scoring + beat-sync flow
# ─────────────────────────────────────────────────────────────────────────────

_ALLOWED_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm"}


@app.post("/transcribe-and-analyze")
async def transcribe_and_analyze(file: UploadFile = File(...)) -> JSONResponse:
    # ── Validate upload ───────────────────────────────────────────────────
    if not file or not file.filename:
        raise HTTPException(
            status_code=400,
            detail={"error": "No file provided. Send audio via the 'file' field."},
        )

    import os
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    f"Unsupported format '{ext}'. "
                    f"Accepted: {', '.join(_ALLOWED_EXT)}"
                )
            },
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail={"error": "Uploaded file is empty."},
        )

    # ── Step 1: Transcription ─────────────────────────────────────────────
    try:
        transcription = await transcribe_audio(audio_bytes, file.filename)
    except Exception as exc:
        logger.exception("Whisper API call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={"error": f"Transcription service error: {exc}"},
        )

    lyrics = transcription.get("text", "").strip()
    if not lyrics:
        raise HTTPException(
            status_code=422,
            detail={"error": "Transcription returned empty text."},
        )

    lang = transcription.get("detected_language", "auto")
    words_raw: list[dict] = transcription.get("words", [])

    # ── Step 2: Text-based scoring ────────────────────────────────────────
    syllable_score, avg_syl, syllable_weight_score, weight_ratio = syllable_service.calculate(lyrics)
    rhyme_score, rhyme_pairs, multisyl_count, internal_score, chain_score = rhyme_service.calculate(lyrics)
    allit_score, allit_pairs = alliteration_service.calculate(lyrics)
    vocab_score, ttr = vocabulary_service.calculate(lyrics)
    wordplay_score, wordplay_meta = wordplay_service.calculate(lyrics)

    # ── Step 3: Beat-sync flow scoring ───────────────────────────────────
    flow_score: float | None = None
    flow_meta: FlowMetadata | None = None

    if words_raw:
        try:
            raw_score, meta_dict = calculate_beat_sync(
                audio_bytes, file.filename, words_raw
            )
            if "error" not in meta_dict:
                flow_score = raw_score
                flow_meta = FlowMetadata(**meta_dict)
            else:
                logger.warning("Beat sync returned error: %s", meta_dict["error"])
        except Exception as exc:
            logger.exception("Beat sync failed: %s", exc)
            # Non-fatal — continue without flow score

    # ── Step 4: Weighted total ────────────────────────────────────────────
    if flow_score is not None:
        total = (
            rhyme_score * 0.20
            + syllable_score * 0.15
            + allit_score * 0.10
            + vocab_score * 0.10
            + wordplay_score * 0.15
            + syllable_weight_score * 0.10
            + flow_score * 0.20
        )
    else:
        total = (
            rhyme_score * 0.25
            + syllable_score * 0.20
            + allit_score * 0.15
            + vocab_score * 0.15
            + wordplay_score * 0.15
            + syllable_weight_score * 0.10
        )

    line_count = sum(
        1
        for l in lyrics.split("\n")
        if l.strip() and not (l.strip().startswith("[") and l.strip().endswith("]"))
    )

    analysis = ScoreBreakdown(
        syllable_score=round(syllable_score, 2),
        rhyme_score=round(rhyme_score, 2),
        alliteration_score=round(allit_score, 2),
        vocabulary_score=round(vocab_score, 2),
        flow_score=flow_score,
        total_score=round(total, 2),
        wordplay_score=round(wordplay_score, 2),
        syllable_weight=round(syllable_weight_score, 2),
        word_count=len(lyrics.split()),
        line_count=line_count,
        avg_syllables_per_word=round(avg_syl, 2),
        vocabulary_uniqueness=ttr,
        detected_language=lang,
        double_entendres_count=wordplay_meta["double_entendres_count"],
        puns_count=wordplay_meta["puns_count"],
        similes_count=wordplay_meta["simile_count"],
        metaphors_count=wordplay_meta["metaphor_count"],
        alliteration_pairs=allit_pairs,
        rhyme_pairs=rhyme_pairs,
        multisyllabic_rhyme_count=multisyl_count,
        flow_metadata=flow_meta,
    )

    return JSONResponse(
        content={
            "transcription": {
                "text": lyrics,
                "detected_language": lang,
                "language_probability": transcription.get("language_probability"),
                "words": words_raw,
            },
            "analysis": analysis.model_dump(),
        }
    )
