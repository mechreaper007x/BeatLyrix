"""
RapRank · Hindi Semantic Scoring Microservice
Deployed on HuggingFace Docker Space (free tier, CPU basic)

Models (both multilingual, cover Hindi + Romanized Hinglish, CPU-friendly):
  - EMBEDDINGS: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
      Trained so COSINE similarity is meaningful (features 1/3/4). Raw masked-LM
      models like MuRIL are anisotropic -- every sentence sits at cosine ~0.99,
      so they do NOT discriminate coherent verses from word-salad. A
      sentence-transformer is the correct backbone for the embedding features.
  - MASKED-LM: google/muril-base-cased
      Devanagari + transliterated Indian text; used only for surprisal (feat 2).

Produces four *label-free* semantic feature scores (0-100) that surface-level
phonetic/lexicon scoring can't capture:

  1. coherence               — do adjacent bars connect, or is it word salad?
  2. semantic_surprisal      — how unexpected are the word choices (cleverness proxy)?
  3. lexical_sophistication  — semantic spread of vocabulary (beyond surface TTR)
  4. theme_consistency       — how tightly lines hold the central theme

Endpoints:
  POST /semantic  { "lyrics": "..." }  → { <4 scores>, "metrics": {...} }
  GET  /health
"""
from __future__ import annotations

import re
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RapRank Hindi Semantic Scoring", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model load (baked into image at build time, instant cold start) ────────
EMBED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MLM_MODEL_ID = "google/muril-base-cased"
logger.info("Loading %s (embeddings) + %s (masked-LM) ...", EMBED_MODEL_ID, MLM_MODEL_ID)

embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_ID)
mlm_tokenizer = AutoTokenizer.from_pretrained(MLM_MODEL_ID)
mlm_model = AutoModelForMaskedLM.from_pretrained(MLM_MODEL_ID)
embed_model.eval()
mlm_model.eval()
logger.info("Models loaded.")

# ── CPU-bound safety caps (mirror raprank-nlp's >150-word skip guard) ──────
MAX_LINES = 60                # bars analysed per request
MAX_TOKENS_PER_LINE = 64      # truncation for embedding + masking passes
MAX_MASK_TOKENS = 300         # total masked-LM forward passes per request

_TAG_RE = re.compile(r"\[.*?\]")            # strip [Chorus], [Verse 1], ...
_ADLIB_RE = re.compile(r"\s*\(.*?\)\s*")     # strip (woo), (yeah) ad-libs

# ── Per-script percentile calibration (built by build_calibration.py) ──────
# calibration.json holds, per script bucket, per raw metric, a SORTED array of
# reference values observed across the real domain corpus. At runtime a raw
# metric maps to its percentile within that array -> a script-aware 0-100 score
# that fixes the fixed-anchor problems (weak EN separation, Devanagari cosine
# saturation). If the file is absent/unreadable we fall back to the heuristic
# _rescale anchors below, so the service still works uncalibrated.
import json as _json
from pathlib import Path as _Path

_DEVA_START, _DEVA_END = 0x0900, 0x097F
_DEVA_RATIO_THRESHOLD = 0.20

# Metrics -> the SBERT-cosine fallback anchors (used only when uncalibrated).
_FALLBACK_ANCHORS = {
    "coherence_cosine": (0.05, 0.35),
    "mean_surprisal_nats": (2.0, 11.0),
    "pairwise_spread": (0.55, 0.90),
    "theme_cosine": (0.45, 0.75),
    # Callback = strongest echo between DISTANT lines. A high value means a motif
    # or phrase recurs far apart. Anchored higher than adjacent coherence since a
    # deliberate callback tends to be a strong (near-repeat) match.
    "callback_cosine": (0.35, 0.80),
}

_CALIBRATION: dict | None = None
try:
    _cal_path = _Path(__file__).resolve().parent / "calibration.json"
    if _cal_path.exists():
        _CALIBRATION = _json.loads(_cal_path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded calibration.json (deva=%s, latin=%s songs).",
            _CALIBRATION.get("_meta", {}).get("deva"),
            _CALIBRATION.get("_meta", {}).get("latin"),
        )
    else:
        logger.warning("calibration.json not found -- using heuristic _rescale anchors.")
except Exception as _exc:  # pragma: no cover - defensive
    logger.warning("Failed to load calibration.json (%s) -- using heuristic anchors.", _exc)
    _CALIBRATION = None


def _script_bucket(text: str) -> str:
    """Devanagari-char ratio -> 'deva' | 'latin' (Latin covers EN + Romanized Hinglish)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "latin"
    deva = sum(1 for c in letters if _DEVA_START <= ord(c) <= _DEVA_END)
    return "deva" if (deva / len(letters)) >= _DEVA_RATIO_THRESHOLD else "latin"


def _calibrated_score(bucket: str, metric: str, raw: float) -> float:
    """Map a raw metric to its 0-100 percentile within the corpus reference
    distribution for its script bucket. Falls back to the heuristic linear
    anchors when calibration data is missing for that bucket/metric."""
    ref = None
    if _CALIBRATION is not None:
        ref = _CALIBRATION.get(bucket, {}).get(metric)
    if not ref:
        lo, hi = _FALLBACK_ANCHORS[metric]
        return _rescale(raw, lo, hi)
    # All four metrics are monotonic (higher raw = higher score), so the plain
    # percentile position is the score. np.searchsorted is O(log n) on the
    # pre-sorted reference array.
    pos = int(np.searchsorted(ref, raw, side="right"))
    return round(float(np.clip(pos / len(ref) * 100.0, 0.0, 100.0)), 2)


# ── Request/Response schemas ───────────────────────────────────────────────
class SemanticRequest(BaseModel):
    lyrics: str


class SemanticResponse(BaseModel):
    coherence_score: float
    semantic_surprisal_score: float
    lexical_sophistication_score: float
    theme_consistency_score: float
    callback_score: float = 0.0
    metrics: Dict[str, Any]  # mix of floats + the string script_bucket


# ── Helpers ────────────────────────────────────────────────────────────────
def content_lines(text: str) -> List[str]:
    """Non-empty lyric lines, section headers + parenthetical ad-libs stripped."""
    lines = []
    for raw in text.split("\n"):
        line = _TAG_RE.sub("", raw)
        line = _ADLIB_RE.sub(" ", line).strip()
        if len(line) > 3:
            lines.append(line)
    return lines[:MAX_LINES]


def _rescale(value: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] -> [0, 100], clipped. Anchors are heuristic and are
    the first thing to recalibrate once labelled/synthetic data exists."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0) * 100.0)


@torch.inference_mode()
def _embed_lines(lines: List[str]) -> np.ndarray:
    """Mean-pooled MuRIL embedding per line -> (n_lines, hidden). L2-normalised."""
    enc = embed_tokenizer(
        lines,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS_PER_LINE,
        return_tensors="pt",
    )
    out = embed_model(**enc).last_hidden_state          # (n, seq, hidden)
    mask = enc["attention_mask"].unsqueeze(-1).float()   # (n, seq, 1)
    summed = (out * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    mean_pooled = (summed / counts).cpu().numpy()
    norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return mean_pooled / norms


@torch.inference_mode()
def _pseudo_surprisal(lines: List[str]) -> float:
    """Mean masked-LM surprisal (nats) over content tokens: mask one token at a
    time, read -log p(true token). A word-choice unexpectedness proxy."""
    mask_id = mlm_tokenizer.mask_token_id
    special_ids = set(mlm_tokenizer.all_special_ids)
    surprisals: List[float] = []
    budget = MAX_MASK_TOKENS

    for line in lines:
        if budget <= 0:
            break
        enc = mlm_tokenizer(
            line,
            truncation=True,
            max_length=MAX_TOKENS_PER_LINE,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        positions = [
            i for i, tok in enumerate(ids.tolist())
            if tok not in special_ids
        ]
        if not positions:
            continue
        positions = positions[:budget]
        budget -= len(positions)

        # One batched forward pass: one masked copy of the line per position.
        batch = enc["input_ids"].repeat(len(positions), 1)
        attn = enc["attention_mask"].repeat(len(positions), 1)
        true_ids = []
        for row, pos in enumerate(positions):
            true_ids.append(int(batch[row, pos].item()))
            batch[row, pos] = mask_id

        logits = mlm_model(input_ids=batch, attention_mask=attn).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        for row, pos in enumerate(positions):
            lp = log_probs[row, pos, true_ids[row]].item()
            surprisals.append(-lp)

    if not surprisals:
        return 0.0
    return float(np.mean(surprisals))


def compute_semantics(lyrics: str) -> SemanticResponse:
    lines = content_lines(lyrics)

    # Too little signal to say anything meaningful -> neutral zeros.
    if len(lines) < 2:
        return SemanticResponse(
            coherence_score=0.0,
            semantic_surprisal_score=0.0,
            lexical_sophistication_score=0.0,
            theme_consistency_score=0.0,
            callback_score=0.0,
            metrics={"line_count": float(len(lines))},
        )

    emb = _embed_lines(lines)                    # (n, hidden), unit-normalised

    # 1. Coherence — mean cosine of ADJACENT line pairs.
    adj = [float(np.dot(emb[i], emb[i + 1])) for i in range(len(emb) - 1)]
    coherence_cos = float(np.mean(adj))

    # 4. Theme consistency — mean cosine of each line vs the verse centroid.
    centroid = emb.mean(axis=0)
    cnorm = np.linalg.norm(centroid)
    centroid = centroid / (cnorm if cnorm else 1e-9)
    theme_cos = float(np.mean([float(np.dot(centroid, e)) for e in emb]))

    # 3. Lexical sophistication — mean pairwise cosine DISTANCE (spread).
    #    High = lines roam across many semantic regions (rich); low = repetitive.
    sims = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)
    spread = float(1.0 - np.mean(sims[iu])) if len(iu[0]) else 0.0

    # 2. Semantic surprisal — masked-LM pseudo-surprisal in nats.
    surprisal = _pseudo_surprisal(lines)

    # 5. Callback / motif reuse — strongest similarity between DISTANT lines
    #    (|i-j| >= 4). Distinct from coherence (adjacent pairs): a hook, phrase,
    #    or image echoed far later in the verse. mean(top-k) so one strong echo
    #    registers without a single near-duplicate maxing it out.
    callback_cos = 0.0
    if len(emb) >= 5:
        distant = [
            float(sims[i, j])
            for i in range(len(emb))
            for j in range(i + 4, len(emb))
        ]
        if distant:
            distant.sort(reverse=True)
            topk = distant[: max(1, len(distant) // 10)]   # top ~10% of distant pairs
            callback_cos = float(np.mean(topk))

    # Score = percentile of each raw metric within the real-corpus distribution
    # for this verse's SCRIPT bucket (calibration.json). Script-aware by
    # construction, so Devanagari no longer saturates and EN separation holds.
    # Falls back to fixed anchors if calibration data is absent.
    # NOTE: surprisal AND spread both reward randomness (word-salad maxes them),
    # so they only signal skill *in combination with* coherence/theme -- do not
    # read either in isolation. This is why coherence/theme are the load-bearing
    # discriminators and why these axes aren't folded into total_score yet.
    bucket = _script_bucket(" ".join(lines))
    return SemanticResponse(
        coherence_score=_calibrated_score(bucket, "coherence_cosine", coherence_cos),
        semantic_surprisal_score=_calibrated_score(bucket, "mean_surprisal_nats", surprisal),
        lexical_sophistication_score=_calibrated_score(bucket, "pairwise_spread", spread),
        theme_consistency_score=_calibrated_score(bucket, "theme_cosine", theme_cos),
        callback_score=_calibrated_score(bucket, "callback_cosine", callback_cos),
        metrics={
            "line_count": float(len(lines)),
            "script_bucket": bucket,
            "coherence_cosine": round(coherence_cos, 4),
            "theme_cosine": round(theme_cos, 4),
            "pairwise_spread": round(spread, 4),
            "mean_surprisal_nats": round(surprisal, 4),
            "callback_cosine": round(callback_cos, 4),
        },
    )


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "embed_model": EMBED_MODEL_ID, "mlm_model": MLM_MODEL_ID}


@app.post("/semantic", response_model=SemanticResponse)
def run_semantic(req: SemanticRequest) -> SemanticResponse:
    if not req.lyrics or not req.lyrics.strip():
        raise HTTPException(status_code=400, detail="lyrics field is empty")
    try:
        return compute_semantics(req.lyrics)
    except Exception as exc:
        logger.exception("Semantic scoring failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"semantic scoring error: {exc}")
