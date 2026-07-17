"""
RapRank NLP Scoring Service v2
FastAPI application exposing two endpoints:

  GET  /health                   — liveness probe
  POST /analyze                  — score pre-transcribed lyrics (text-only)
  POST /transcribe-and-analyze   — score an audio file end-to-end

total_score is the decoupled LQI score from lyrical_compiler.compile_lyrics()
(density + entropy of syllables/rhyme). Assonance, consonance, onomatopoeia,
vocabulary, and wordplay are rule-based sub-scores reported alongside it for
display only -- none of them feed total_score. Alliteration was removed from
this service entirely (not just decoupled) -- the rule-based detector was
found to score incidental phonetic overlap too highly to be trustworthy.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import json
import httpx
from mistralai import Mistral
from dotenv import load_dotenv
import config.ffmpeg_patch

from services.lyrical_compiler import compile_lyrics

load_dotenv()


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

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse

async def format_lyrics_with_mistral(pasted_lyrics: str, whisper_transcript: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.warning("No MISTRAL_API_KEY found, skipping SLM formatting.")
        return pasted_lyrics
    
    try:
        client = Mistral(api_key=api_key)
        prompt = (
            "You are an expert rap lyricist. I will provide you with the user's pasted lyrics and a raw Whisper audio transcript. "
            "The user's lyrics have the correct words, but might have messy formatting. The Whisper transcript shows the natural flow "
            "and pauses of the audio. Please format the user's lyrics into perfect structural rap verses and 4-line bars, using "
            "the Whisper transcript to determine where the line breaks should go. Output ONLY the formatted user lyrics, no intro or outro.\n\n"
            f"USER LYRICS:\n{pasted_lyrics}\n\nWHISPER TRANSCRIPT:\n{whisper_transcript}"
        )
        
        response = await client.chat.complete_async(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Mistral formatting failed: %s", exc)
        return pasted_lyrics


async def transliterate_devanagari_to_hinglish(whisper_transcript: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.warning("No MISTRAL_API_KEY found, skipping Hinglish transliteration.")
        return whisper_transcript
    
    try:
        client = Mistral(api_key=api_key)
        prompt = (
            "You are an expert rap transcriber. I will provide you with a raw Whisper audio transcript written in Devanagari script. "
            "This transcript contains a mix of Hindi and English words (Hinglish) spoken in a rap song. Please transliterate "
            "the Devanagari text into clean, standard Romanized Hinglish (using the English alphabet).\n\n"
            "Rules:\n"
            "1. Convert Devanagari representations of English words back to correct English words (e.g., 'वाच्छु दूइफर' -> 'What you do it for', 'टाइम' -> 'time', 'चिल' -> 'chill', 'शिमी' -> 'shimmy').\n"
            "2. Convert Hindi/Hinglish words to standard Romanized Hindi spelling (e.g., 'पूछो' -> 'poochho', 'आया' -> 'aaya', 'लौंडा' -> 'launda', 'सुरूर' -> 'suroor', 'मैदान' -> 'maidaan').\n"
            "3. Format the output into clean, structured rap verses and lines.\n"
            "4. Output ONLY the transliterated Romanized Hinglish lyrics, no introductory or concluding remarks.\n\n"
            f"TRANSCRIPT:\n{whisper_transcript}"
        )
        
        response = await client.chat.complete_async(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Mistral transliteration failed: %s", exc)
        return whisper_transcript


async def analyze_story_and_structure_with_mistral(lyrics: str) -> dict | None:
    """
    Calls Mistral for a QUALITATIVE description of the song's story, theme,
    and structure -- NOT for any numeric scores. All scoring is done locally
    by the rule-based services; Mistral here only narrates what the song is
    about and how it is built, which is surfaced alongside the local scores.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.warning("MISTRAL_API_KEY environment variable is not set. Skipping story/structure analysis.")
        return None

    try:
        client = Mistral(api_key=api_key)
        prompt = (
            "You are an expert rap literary critic and hip-hop analyst. Describe the STORY and STRUCTURE of the "
            "provided rap lyrics, and identify two specific literary devices. Do NOT rate, score, or grade anything "
            "-- output only qualitative description and the identified devices.\n\n"
            "Cover:\n"
            "1. theme  -- the central subject/message in one short phrase.\n"
            "2. story  -- the narrative or emotional arc across the verses (2-4 sentences).\n"
            "3. structure -- how the song is organized (verses, hook/chorus, bridge, refrains, repetition) and how "
            "the sections relate (2-3 sentences).\n"
            "4. mood   -- the overall tone/emotion in one short phrase.\n"
            "5. punchlines -- lines with a clear setup-then-payoff structure where the payoff lands a witty, hard, or "
            "surprising twist. For each, give the exact quoted line(s) and a one-line reason. Empty list if none.\n"
            "6. extended_metaphors -- a single metaphor/conceit SUSTAINED across two or more lines (not a one-line "
            "simile). For each, give the quoted lines and a one-line explanation of the sustained image. Empty list if none.\n\n"
            "Only include GENUINE instances -- an empty list is correct and expected when a device is absent. Do not "
            "invent devices to fill the lists.\n\n"
            "Return your response STRICTLY as raw JSON (no intro, outro, or markdown code fences):\n"
            "{\n"
            "  \"theme\": \"<short phrase>\",\n"
            "  \"story\": \"<2-4 sentences>\",\n"
            "  \"structure\": \"<2-3 sentences>\",\n"
            "  \"mood\": \"<short phrase>\",\n"
            "  \"punchlines\": [{\"quote\": \"<line>\", \"why\": \"<short reason>\"}],\n"
            "  \"extended_metaphors\": [{\"quote\": \"<lines>\", \"why\": \"<short reason>\"}]\n"
            "}\n\n"
            f"LYRICS TO ANALYZE:\n{lyrics}"
        )

        response = await client.chat.complete_async(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()

        # Clean markdown code block wrapper if returned
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        return json.loads(content)
    except Exception as exc:
        logger.exception("Mistral story/structure analysis failed: %s", exc)
        return None


from models.schemas import (
    AnalyzeRequest,
    FlowMetadata,
    ScoreBreakdown,
)
from services import (
    assonance_service,
    consonance_service,
    onomatopoeia_service,
    rhyme_service,
    syllable_service,
    vocabulary_service,
    wordplay_service,
)
from services.flow_service import calculate_beat_sync
from services.alignment_service import align_structured_lyrics_to_whisper
from services.language_utils import detect_language
from services.semantic_service import analyze_semantics
from services import gmm_style_service
from services import rf_quality_service
from services import svm_quality_service
from services import bayesian_scoring_service
from services import prosody_service
from services import element_cluster_service
from services.transcription_service import transcribe_audio
from services.separation_service import separate_vocals
from config import scoring_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# GMM style-clustering model: load the pickled bundle once at startup (like
# alignment_service caches its model). None if untrained/absent -- score_style
# then simply returns None and the style fields stay null. Never fatal.
_GMM_STYLE_BUNDLE = None
try:
    _GMM_STYLE_BUNDLE = gmm_style_service.load()
    logger.info("Loaded GMM style model (k=%s).", _GMM_STYLE_BUNDLE.get("k"))
except Exception as _exc:
    logger.warning("GMM style model unavailable (%s) -- style fields will be null.", _exc)

# RF quality-tier classifier: same load-once-at-startup, never-fatal pattern.
_RF_QUALITY_BUNDLE = None
try:
    _RF_QUALITY_BUNDLE = rf_quality_service.load()
    logger.info("Loaded RF quality-tier model (classes=%s).", _RF_QUALITY_BUNDLE.get("classes"))
except Exception as _exc:
    logger.warning("RF quality-tier model unavailable (%s) -- tier fields will be null.", _exc)

# SVM quality-tier classifier: second comparison head, same pattern.
_SVM_QUALITY_BUNDLE = None
try:
    _SVM_QUALITY_BUNDLE = svm_quality_service.load()
    logger.info("Loaded SVM quality-tier model (classes=%s).", _SVM_QUALITY_BUNDLE.get("classes"))
except Exception as _exc:
    logger.warning("SVM quality-tier model unavailable (%s) -- svm_tier fields will be null.", _exc)

# Bayesian quality-tier network: third comparison head, same pattern.
_BAYES_QUALITY_MODEL = None
try:
    _BAYES_QUALITY_MODEL = bayesian_scoring_service.load()
    logger.info("Loaded Bayesian quality-tier model.")
except Exception as _exc:
    logger.warning("Bayesian quality-tier model unavailable (%s) -- bayes_tier fields will be null.", _exc)

# Per-element cluster bundles (rhyme/wordplay/texture/rare): same load-once
# pattern, but one independent try/except per family so a single missing
# pickle doesn't block the others from loading.
_ELEMENT_CLUSTER_BUNDLES: dict = {}
for _family in element_cluster_service.FAMILIES:
    try:
        _ELEMENT_CLUSTER_BUNDLES[_family] = element_cluster_service.load(_family)
        logger.info("Loaded element-cluster model '%s'.", _family)
    except Exception as _exc:
        logger.warning("Element-cluster model '%s' unavailable (%s).", _family, _exc)

app = FastAPI(
    title="RapRank NLP Service",
    description=(
        "Multilingual (Hindi / English / Hinglish) rap lyric scoring. "
        "Scores syllable density, end rhyme, vocabulary uniqueness, "
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


async def compute_scores_and_breakdown(
    lyrics: str,
    lang: str,
    flow_score: float | None = None,
    flow_meta: FlowMetadata | None = None,
    generated_lyrics: str | None = None,
    track_id: int | None = None,
) -> ScoreBreakdown:
    if track_id is not None:
        await update_status(track_id, "ANALYZING_TEXT")
        
    # 1. Compile lyrics using the new unsupervised Lyrical Lexer & Parser Compiler (LLPC)
    compiled = compile_lyrics(lyrics)
    
    # 2. Map all scores to the compiled LQI metrics directly, bypassing the rule-based NLP services entirely
    syllable_score = compiled["syllable_density"]
    rhyme_score = compiled["rhyme_complexity"]
    
    # Calculate specialized lexical metrics using rule/regex-based services
    assonance_score, assonance_pairs = assonance_service.calculate(lyrics)
    consonance_score, consonance_pairs = consonance_service.calculate(lyrics)
    onomatopoeia_score, onomatopoeia_hits = onomatopoeia_service.calculate(lyrics)
    
    vocab_score, ttr = vocabulary_service.calculate(lyrics)
    
    wordplay_score, wordplay_meta = wordplay_service.calculate(lyrics)
    double_entendres_count = wordplay_meta["double_entendres_count"]
    puns_count = wordplay_meta["puns_count"]
    similes_count = wordplay_meta["simile_count"]
    metaphors_count = wordplay_meta["metaphor_count"]
    allusions_count = wordplay_meta["allusions_count"]
    
    syllable_weight_score = compiled["lexical_information_density"]

    # Supervised quality-tier classification: three comparison heads (RF, SVM,
    # Bayesian) sharing ONE signature() pass via _axis_scores_from_lyrics, plus
    # a majority-vote consensus. Depends only on lyrics text (no semantics),
    # so it runs early. Each head is independently guarded: None if its model
    # is absent or scoring fails; the request never fails.
    predicted_tier = tier_confidence = tier_probabilities = None
    svm_tier = svm_tier_confidence = svm_tier_probabilities = None
    bayes_tier = bayes_tier_probabilities = None
    tier_consensus = tier_consensus_agreement = None
    if any(m is not None for m in (_RF_QUALITY_BUNDLE, _SVM_QUALITY_BUNDLE, _BAYES_QUALITY_MODEL)):
        axis_scores = None
        try:
            axis_scores = await asyncio.to_thread(
                bayesian_scoring_service._axis_scores_from_lyrics, lyrics
            )
        except Exception as exc:
            logger.warning("Tier axis-score extraction failed: %s", exc)

        if axis_scores is not None:
            if _RF_QUALITY_BUNDLE is not None:
                try:
                    rf_result = await asyncio.to_thread(
                        rf_quality_service.predict_tier_from_scores,
                        _RF_QUALITY_BUNDLE, axis_scores,
                    )
                    predicted_tier = rf_result["tier"]
                    tier_confidence = rf_result["confidence"]
                    tier_probabilities = rf_result["probabilities"]
                except Exception as exc:
                    logger.warning("RF quality-tier scoring failed: %s", exc)

            if _SVM_QUALITY_BUNDLE is not None:
                try:
                    svm_result = await asyncio.to_thread(
                        svm_quality_service.predict_tier_from_scores,
                        _SVM_QUALITY_BUNDLE, axis_scores,
                    )
                    svm_tier = svm_result["tier"]
                    svm_tier_confidence = svm_result["confidence"]
                    svm_tier_probabilities = svm_result["probabilities"]
                except Exception as exc:
                    logger.warning("SVM quality-tier scoring failed: %s", exc)

            if _BAYES_QUALITY_MODEL is not None:
                try:
                    posterior = await asyncio.to_thread(
                        bayesian_scoring_service.predict_posterior,
                        _BAYES_QUALITY_MODEL, axis_scores,
                    )
                    bayes_tier = max(posterior, key=posterior.get)
                    bayes_tier_probabilities = {t: round(float(p), 4) for t, p in posterior.items()}
                except Exception as exc:
                    logger.warning("Bayesian quality-tier scoring failed: %s", exc)

            # ── Quality override baseline ───────────────────────────────────────────
            # If key technical metrics (rhyme, vocabulary, wordplay) are simultaneously
            # low, override predicted tiers to "commercial" (repetition and ad-libs in
            # pop-rap can sometimes inflate consonance/onomatopoeia, tricking raw ML).
            rhyme_val = axis_scores.get("rhyme", 0.0)
            vocab_val = axis_scores.get("vocabulary", 0.0)
            wordplay_val = axis_scores.get("wordplay", 0.0)
            
            is_commercial_by_rule = (
                (rhyme_val < 35.0 and vocab_val < 60.0 and wordplay_val < 50.0) or
                (wordplay_val < 35.0 and vocab_val < 60.0)
            )
            
            if is_commercial_by_rule:
                if _RF_QUALITY_BUNDLE is not None and predicted_tier is not None:
                    predicted_tier = "commercial"
                    tier_confidence = 1.0
                    tier_probabilities = {"commercial": 1.0, "mid": 0.0, "elite": 0.0}
                if _SVM_QUALITY_BUNDLE is not None and svm_tier is not None:
                    svm_tier = "commercial"
                    svm_tier_confidence = 1.0
                    svm_tier_probabilities = {"commercial": 1.0, "mid": 0.0, "elite": 0.0}
                if _BAYES_QUALITY_MODEL is not None and bayes_tier is not None:
                    bayes_tier = "commercial"
                    bayes_tier_confidence = 1.0
                    bayes_tier_probabilities = {"commercial": 1.0, "mid": 0.0, "elite": 0.0}


            # Majority vote across whichever heads produced a tier. Ties break
            # toward the RF head (strongest under grouped CV), else first voter.
            votes = [t for t in (predicted_tier, svm_tier, bayes_tier) if t]
            if votes:
                from collections import Counter as _Counter
                counted = _Counter(votes)
                top_count = max(counted.values())
                winners = [t for t, n in counted.items() if n == top_count]
                tier_consensus = predicted_tier if predicted_tier in winners else winners[0]
                tier_consensus_agreement = round(top_count / len(votes), 2)

    # Map track metadata metrics
    avg_syl = compiled["avg_syllables_per_line"]
    multisyl_count = compiled["detected_rhyme_count"]
    rhyme_pairs = []
    
    # Qualitative story/structure description from Mistral
    story_structure = await analyze_story_and_structure_with_mistral(lyrics)

    # Semantic (meaning-based) axes from the Hindi BERT / MuRIL service.
    # Additive enrichment: None if the service is unavailable -- must not break
    # scoring, and does NOT feed total_score this iteration.
    semantics = await analyze_semantics(lyrics)
    coherence_score = semantics.get("coherence_score") if semantics else None
    semantic_surprisal_score = semantics.get("semantic_surprisal_score") if semantics else None
    lexical_sophistication_score = semantics.get("lexical_sophistication_score") if semantics else None
    theme_consistency_score = semantics.get("theme_consistency_score") if semantics else None
    callback_score = semantics.get("callback_score") if semantics else None

    # Literary-device counts from the Mistral story/structure result (Part A):
    # same single call, richer JSON. Defaults to 0 when absent/failed.
    punchline_count = len(story_structure.get("punchlines", [])) if story_structure else 0
    extended_metaphor_count = len(story_structure.get("extended_metaphors", [])) if story_structure else 0

    # Prosody / structural axes (local, no network): code-switching, anaphora,
    # text cadence variance. Guarded so a failure never breaks scoring.
    codeswitch_score = repetition_score = cadence_text_score = None
    try:
        prosody = prosody_service.calculate(lyrics)
        codeswitch_score = prosody["codeswitch_score"]
        repetition_score = prosody["repetition_score"]
        cadence_text_score = prosody["cadence_text_score"]
    except Exception as exc:
        logger.warning("Prosody scoring failed: %s", exc)

    # Descriptive GMM style fingerprint. Reuses the semantic raw `metrics` already
    # fetched above (no extra Space call). Guarded: None if the model is absent or
    # semantics degraded -- style fields simply stay null, request never fails.
    style_cluster = style_cluster_confidence = style_membership = None
    if _GMM_STYLE_BUNDLE is not None:
        try:
            style = await asyncio.to_thread(
                gmm_style_service.score_style,
                lyrics,
                semantics.get("metrics") if semantics else None,
                _GMM_STYLE_BUNDLE,
            )
            if style:
                style_cluster = style["cluster"]
                style_cluster_confidence = style["confidence"]
                style_membership = style["membership"]
        except Exception as exc:
            logger.warning("GMM style scoring failed: %s", exc)

    # Descriptive per-element cluster fingerprints (rhyme/wordplay/texture/rare).
    # Guarded the same way as the GMM style block above: None if no bundles
    # loaded, request never fails on a scoring error.
    element_clusters = None
    if _ELEMENT_CLUSTER_BUNDLES:
        try:
            element_clusters = await asyncio.to_thread(
                element_cluster_service.score_all_families,
                lyrics,
                _ELEMENT_CLUSTER_BUNDLES,
            )
            if not element_clusters:
                element_clusters = None
        except Exception as exc:
            logger.warning("Element-cluster scoring failed: %s", exc)

    # ── Decoupled LQI Score ───────────────────────────────────────────────
    total = compiled["lyrical_score"]

    line_count = sum(
        1
        for l in lyrics.split("\n")
        if l.strip() and not (l.strip().startswith("[") and l.strip().endswith("]"))
    )

    return ScoreBreakdown(
        syllable_score=round(syllable_score, 2),
        rhyme_score=round(rhyme_score, 2),
        vocabulary_score=round(vocab_score, 2),
        flow_score=flow_score,
        total_score=round(total, 2),
        wordplay_score=round(wordplay_score, 2),
        syllable_weight=round(syllable_weight_score, 2),
        coherence_score=coherence_score,
        semantic_surprisal_score=semantic_surprisal_score,
        lexical_sophistication_score=lexical_sophistication_score,
        theme_consistency_score=theme_consistency_score,
        callback_score=callback_score,
        punchline_count=punchline_count,
        extended_metaphor_count=extended_metaphor_count,
        codeswitch_score=codeswitch_score,
        repetition_score=repetition_score,
        cadence_text_score=cadence_text_score,
        style_cluster=style_cluster,
        style_cluster_confidence=style_cluster_confidence,
        style_membership=style_membership,
        element_clusters=element_clusters,
        predicted_tier=predicted_tier,
        tier_confidence=tier_confidence,
        tier_probabilities=tier_probabilities,
        svm_tier=svm_tier,
        svm_tier_confidence=svm_tier_confidence,
        svm_tier_probabilities=svm_tier_probabilities,
        bayes_tier=bayes_tier,
        bayes_tier_probabilities=bayes_tier_probabilities,
        tier_consensus=tier_consensus,
        tier_consensus_agreement=tier_consensus_agreement,
        assonance_score=round(assonance_score, 2),
        consonance_score=round(consonance_score, 2),
        onomatopoeia_score=round(onomatopoeia_score, 2),
        word_count=len(lyrics.split()),
        line_count=line_count,
        avg_syllables_per_word=round(avg_syl, 2),
        vocabulary_uniqueness=ttr,
        detected_language=lang,
        double_entendres_count=double_entendres_count,
        puns_count=puns_count,
        similes_count=similes_count,
        metaphors_count=metaphors_count,
        allusions_count=allusions_count,
        assonance_pairs=assonance_pairs,
        consonance_pairs=consonance_pairs,
        onomatopoeia_hits=onomatopoeia_hits,
        rhyme_pairs=rhyme_pairs,
        multisyllabic_rhyme_count=multisyl_count,
        flow_metadata=flow_meta,
        generated_lyrics=generated_lyrics,
        story_structure=story_structure,
    )


@app.post("/analyze", response_model=ScoreBreakdown)
async def analyze(request: AnalyzeRequest) -> ScoreBreakdown:
    try:
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

        generated_lyrics = None
        flow_score: float | None = None
        flow_meta: FlowMetadata | None = None

        # ── Pure Lyrical scoring (Audio/ASR transcription is completely removed) ──────
        return await compute_scores_and_breakdown(
            lyrics=lyrics,
            lang=lang,
            flow_score=None,
            flow_meta=None,
            generated_lyrics=None,
            track_id=request.track_id,
        )
    finally:
        import gc
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# POST /transcribe-and-analyze
# Full pipeline: audio → whisper → NLP scoring + beat-sync flow
# ─────────────────────────────────────────────────────────────────────────────

_ALLOWED_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm"}


@app.post("/transcribe-and-analyze")
async def transcribe_and_analyze(
    file: UploadFile = File(...),
    language: str | None = Form(default="hi"),
    lyrics: str | None = Form(default=None)
) -> JSONResponse:
    try:
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
        lyrics_input = lyrics
        try:
            transcription = await transcribe_audio(audio_bytes, file.filename, language=language, lyrics=lyrics_input)
        except Exception as exc:
            logger.exception("Whisper API call failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail={"error": f"Transcription service error: {exc}"},
            )

        whisper_text = transcription.get("text", "").strip()
        
        print("\n" + "=" * 50)
        print("RAW WHISPER TRANSCRIPTION:")
        print(whisper_text)
        print("=" * 50 + "\n")
        
        if not whisper_text:
            raise HTTPException(
                status_code=422,
                detail={"error": "Transcription returned empty text."},
            )

        lang = transcription.get("detected_language", "auto")
        words_raw: list[dict] = transcription.get("words", [])

        # If user passed ground-truth lyrics, format them with Mistral using Whisper transcript
        # for *alignment* purposes only -- scoring must always use the user's
        # original submission (lyrics_input) so this pipeline's scores match
        # what /analyze (text-only) produces for the same lyrics. Mistral's
        # output is still generated and returned for display/alignment.
        generated_lyrics = None
        lyrics_to_score = lyrics_input if lyrics_input else whisper_text
        if lyrics_input:
            logger.info("Using Mistral to format lyrics based on Whisper transcription...")
            formatted = await format_lyrics_with_mistral(lyrics_input, whisper_text)
            if formatted and formatted != lyrics_input:
                logger.info("Lyrics successfully formatted by Mistral.")
                generated_lyrics = formatted

                print("\n" + "=" * 50)
                print("MISTRAL FORMATTED LYRICS:")
                print(generated_lyrics)
                print("=" * 50 + "\n")
        else:
            # If no lyrics are provided, but the transcript contains Devanagari/Hindi characters,
            # transliterate the transcript into clean Romanized Hinglish using Mistral
            if lang == "hi" or any(ord(c) > 127 for c in whisper_text):
                logger.info("Devanagari text detected. Transliterating to Romanized Hinglish using Mistral...")
                transliterated = await transliterate_devanagari_to_hinglish(whisper_text)
                if transliterated and transliterated != whisper_text:
                    logger.info("Lyrics successfully transliterated by Mistral.")
                    generated_lyrics = transliterated
                    lyrics_to_score = generated_lyrics
                    
                    print("\n" + "=" * 50)
                    print("MISTRAL TRANSLITERATED HINGLISH LYRICS:")
                    print(generated_lyrics)
                    print("=" * 50 + "\n")

        # ── Step 2: Beat-sync flow scoring ───────────────────────────────────
        flow_score: float | None = None
        flow_meta: FlowMetadata | None = None

        if words_raw:
            try:
                words_aligned = words_raw
                if generated_lyrics:
                    # Alignment-only use of Mistral's formatted/transliterated
                    # output (better line-break match against the ASR
                    # transcript) -- lyrics_to_score (what actually gets
                    # scored below) is unaffected by this.
                    words_aligned = align_structured_lyrics_to_whisper(generated_lyrics, words_raw)
                
                raw_score, meta_dict = await asyncio.to_thread(
                    calculate_beat_sync, audio_bytes, audio_bytes, file.filename, words_aligned
                )
                if "error" not in meta_dict:
                    flow_score = raw_score
                    flow_meta = FlowMetadata(**meta_dict)
                else:
                    logger.warning("Beat sync returned error: %s", meta_dict["error"])
            except Exception as exc:
                logger.exception("Beat sync failed: %s", exc)

        # ── Step 3: Run all text-based scoring and hybrid LLM evaluation ──────
        analysis = await compute_scores_and_breakdown(
            lyrics=lyrics_to_score,
            lang=lang,
            flow_score=flow_score,
            flow_meta=flow_meta,
            generated_lyrics=generated_lyrics,
        )

        return JSONResponse(
            content={
                "transcription": {
                    "text": lyrics_to_score,
                    "detected_language": lang,
                    "language_probability": transcription.get("language_probability"),
                    "words": words_raw,
                },
                "analysis": analysis.model_dump(),
            }
        )
    finally:
        import gc
        gc.collect()
