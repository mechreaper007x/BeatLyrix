"""
RapRank · Hindi NER Microservice
Deployed on HuggingFace Docker Space (free tier, CPU basic)

Model: ai4bharat/IndicNER
  - Supports 11 Indian languages including Hindi (Devanagari)
  - bert-base-multilingual-uncased backbone — runs fine on CPU
  - Returns PERSON / ORG / LOC entities from Hindi/Hinglish rap lyrics

Endpoints:
  POST /ner   { "text": "..." }  → { "entities": [...] }
  GET  /health
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RapRank Hindi NER", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model load (baked into image at build time, instant cold start) ────────
MODEL_ID = "ai4bharat/IndicNER"
logger.info("Loading %s ...", MODEL_ID)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

# HuggingFace pipeline handles BIO tag aggregation cleanly
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",   # merges B-/I- tokens into single spans
    device=-1,                        # CPU only — free tier has no GPU
)
logger.info("Model loaded.")

# ── Regex to strip section tags like [Chorus], [Verse 1] ─────────────────
_TAG_RE = re.compile(r"\[.*?\]")

# ── Entity label filter — only types that indicate a cultural reference ────
_VALID_LABELS = {"PER", "PERSON", "ORG", "LOC", "B-PER", "B-ORG", "B-LOC"}


# ── Request/Response schemas ───────────────────────────────────────────────
class NERRequest(BaseModel):
    text: str


class EntityResult(BaseModel):
    entity: str
    label: str
    score: float


class NERResponse(BaseModel):
    entities: List[EntityResult]
    count: int


# ── Helpers ────────────────────────────────────────────────────────────────
def clean_lyrics(text: str) -> List[str]:
    """
    Split lyrics into individual lines, stripping section headers.
    IndicNER works best on sentence-length inputs, not entire tracks at once
    (long inputs get truncated at 512 tokens).
    """
    lines = []
    for line in text.split("\n"):
        line = _TAG_RE.sub("", line).strip()
        if len(line) > 3:
            lines.append(line)
    return lines


def deduplicate(entities: List[Dict]) -> List[Dict]:
    """Remove duplicate entity strings (case-insensitive)."""
    seen: set = set()
    out = []
    for e in entities:
        key = e["entity"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/ner", response_model=NERResponse)
def run_ner(req: NERRequest) -> NERResponse:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty")

    lines = clean_lyrics(req.text)
    if not lines:
        return NERResponse(entities=[], count=0)

    all_entities: List[Dict] = []

    for line in lines:
        try:
            results = ner_pipeline(line)
            for r in results:
                label = r.get("entity_group", r.get("entity", ""))
                # Normalise BIO labels: B-PER -> PER
                label_norm = label.lstrip("B-").lstrip("I-")
                if label_norm not in _VALID_LABELS and label not in _VALID_LABELS:
                    continue
                word = r.get("word", "").strip()
                # Strip subword artefacts (## prefix from BERT tokenizer)
                word = re.sub(r"^#+", "", word).strip()
                if len(word) < 2:
                    continue
                all_entities.append({
                    "entity": word,
                    "label":  label_norm,
                    "score":  round(float(r.get("score", 0.0)), 4),
                })
        except Exception as exc:
            logger.warning("NER failed on line %r: %s", line[:60], exc)
            continue

    deduped = deduplicate(all_entities)
    return NERResponse(
        entities=[EntityResult(**e) for e in deduped],
        count=len(deduped),
    )
