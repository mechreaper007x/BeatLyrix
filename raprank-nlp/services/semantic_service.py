"""
Async HTTP client for the raprank-semantic HF Space (Hindi BERT / MuRIL).

Returns four label-free meaning-based axes (coherence, semantic surprisal,
lexical sophistication, theme consistency) that the phonetic/lexicon scorers
can't capture. This is an ADDITIVE feature layer -- the whole scoring pipeline
must never fail because the semantic service is slow or down, so every error
path returns None and the caller simply omits the semantic fields.

Set the SEMANTIC_API_URL env var to override the default HF Space URL (useful
for local development against a locally-running semantic service).
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

SEMANTIC_BASE_URL: str = os.getenv(
    "SEMANTIC_API_URL",
    "https://mechreaper007x-raprank-semantic.hf.space",   # live HF Space
).rstrip("/")

# Short timeout: this is an optional enrichment axis, not on the critical path.
# A cold HF Space may exceed this on the first hit -- that's acceptable, the
# pipeline degrades gracefully and the next request warms it up.
_TIMEOUT_S = 60.0


async def analyze_semantics(lyrics: str) -> dict | None:
    """
    POST lyrics to raprank-semantic and return the parsed JSON, or None on any
    failure (network error, non-2xx, timeout, malformed body).

    Expected response shape:
        {
            "coherence_score": float,               # 0-100
            "semantic_surprisal_score": float,      # 0-100
            "lexical_sophistication_score": float,  # 0-100
            "theme_consistency_score": float,       # 0-100
            "metrics": { ... raw cosine/surprisal values ... }
        }
    """
    text = (lyrics or "").strip()
    if not text:
        return None

    url = f"{SEMANTIC_BASE_URL}/semantic"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            response = await client.post(url, json={"lyrics": text})
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Semantic service unavailable (%s) -- skipping semantic axes: %s", url, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Semantic service returned unexpected payload type: %s", type(data))
        return None

    logger.info(
        "Semantic scores: coherence=%.1f surprisal=%.1f lexsoph=%.1f theme=%.1f",
        data.get("coherence_score", 0.0),
        data.get("semantic_surprisal_score", 0.0),
        data.get("lexical_sophistication_score", 0.0),
        data.get("theme_consistency_score", 0.0),
    )
    return data
