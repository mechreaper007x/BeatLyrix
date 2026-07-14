"""
Synthetic lyrics generator, STYLE-anchored on a real Indian rap reference
song (from corpus/data/, the 302-song corpus already scraped by
corpus/scrape_corpus.py) instead of numeric targets alone.

Same numeric tier targets as corpus/synthetic/generate.py (from
corpus.synthetic.tier_profiles), same local validation loop
(score_against_tier), same accept/reject/retry logic and output format --
the only difference is the prompt also shows the model one real reference
verse from an artist already known to sit in that tier (via
corpus/real_corpus/indian_tiers.py's expected_profile-derived labels) and
instructs it to study the TECHNIQUE (rhyme/wordplay density, flow) without
copying or paraphrasing any of it.

NOTE on where this runs: the model here is `minimax-m3:cloud` via a local
Ollama daemon -- the `:cloud` tag means Ollama proxies the request to
MiniMax's/Ollama's remote inference servers, so the reference lyric text
(real, third-party lyrics) DOES leave this machine on every call. This is
a deliberate, explicit choice made after being told the cloud/local
distinction (see conversation) -- if that changes, swap MODEL below for a
fully local Ollama model (e.g. gaalibot) instead.

Hinglish-only, matching corpus/synthetic/generate.py -- pure "hi" and pure
"en" generation modes were dropped entirely.

Usage (pilot batch, resumable like corpus/synthetic/generate.py):
    python -m corpus.synthetic.generate_with_reference --count 60 --resume
    python -m corpus.synthetic.generate_with_reference --tier elite --count 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402

from corpus.real_corpus.indian_tiers import artist_tier_map  # noqa: E402
from corpus.synthetic.generate import (  # noqa: E402
    MAX_ATTEMPTS_MIXED,
    OUT_DIR,
    _LANG_LABEL,
    _TOPICS,
    _has_literary_diction,
    _has_nostalgia_opening,
    _has_prose_sentences,
    _has_stacked_nature_metaphors,
    score_against_tier,
)
from corpus.synthetic.tier_profiles import TIER_NAMES, build_targets  # noqa: E402
from services.language_utils import content_lines  # noqa: E402

OLLAMA_HOST = "http://localhost:11434"
MODEL = "minimax-m3:cloud"
INDIAN_CORPUS_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REFERENCE_MAX_CHARS = 1200


def _has_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def _pick_reference(candidates: list[tuple[str, str]]) -> tuple[str, str]:
    """Pick a reference song, preferring Devanagari-script Hindi references so the
    model isn't shown a Romanized-Hindi reference while being told to output
    Devanagari -- most of corpus/data/ is itself Romanized (~73%), so without this the
    reference "feel" actively pulls generation back toward Romanized Hindi."""
    devanagari_candidates = [c for c in candidates if _has_devanagari(c[1])]
    if devanagari_candidates:
        return random.choice(devanagari_candidates)
    return random.choice(candidates)


def load_reference_pool() -> dict[str, list[tuple[str, str]]]:
    """tier -> [(artist, lyrics), ...] from corpus/data/, tier-labeled via
    corpus/real_corpus/indian_tiers.py against corpus/artists.py priors."""
    tiers = artist_tier_map()
    pool: dict[str, list[tuple[str, str]]] = {t: [] for t in TIER_NAMES}
    if not INDIAN_CORPUS_DATA_DIR.exists():
        return pool
    for artist_dir in INDIAN_CORPUS_DATA_DIR.iterdir():
        if not artist_dir.is_dir():
            continue
        for path in artist_dir.glob("*.json"):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tier = tiers.get(rec.get("artist"))
            lyrics = rec.get("lyrics")
            if tier and lyrics:
                pool[tier].append((rec["artist"], lyrics))
    return pool


# Romanized-Urdu ghazal register (izafat constructions like "zakhm-e-nazara",
# "bam-e-falak", "hawa-e-shab" -- a Persian/Urdu grammatical construction, not
# Hinglish rap) plus explicit ghazal-lexicon words. generate.py's own gates
# only look for English literary diction or Devanagari patterns, so this
# register is otherwise invisible to them -- needed to catch e.g. EPR's
# ADR/ABR, a Bhatiyali-folk/ghazal fusion piece, not a rap song at all.
_IZAFAT = re.compile(r"\b\w+-e-\w+\b", re.IGNORECASE)
_GHAZAL_LEXICON = re.compile(
    r"\b(?:mushaira|ghazal|saaqi|aashiqana|shayari|nazm)\w*\b", re.IGNORECASE
)
_GHAZAL_MIN_IZAFAT = 4
_GHAZAL_MIN_LEXICON_HITS = 2


def _has_ghazal_register(lyrics: str) -> bool:
    """4+ izafat constructions, or 2+ ghazal-lexicon words, marks
    Romanized-Urdu ghazal/shayari register rather than rap -- a single
    incidental word (e.g. a rapper name-dropping "shayari" once) is normal
    Hindi rap vocabulary, not evidence of ghazal register on its own."""
    if len(_IZAFAT.findall(lyrics)) >= _GHAZAL_MIN_IZAFAT:
        return True
    return len(_GHAZAL_LEXICON.findall(lyrics)) >= _GHAZAL_MIN_LEXICON_HITS


def _trips_poetry_register(lyrics: str) -> bool:
    """Register-only subset of run_content_gates()'s checks, applied to REAL
    reference songs rather than generated output -- repetition/anaphora
    detectors are a *generation* concern (an LLM defaulting to templated
    patterns), not a real-song disqualifier, so only the poetry-vs-rap
    register checks are reused here. Catches non-rap contamination in
    corpus/data/ like EPR's ADR/ABR (a Bhatiyali-folk/ghazal fusion piece)."""
    return (
        _has_literary_diction(lyrics)
        or _has_stacked_nature_metaphors(lyrics)
        or _has_nostalgia_opening(lyrics)
        or _has_prose_sentences(lyrics)
        or _has_ghazal_register(lyrics)
    )


def load_artist_reference_pool() -> dict[str, list[str]]:
    """artist display name -> [real lyrics], from corpus/data/{slug}/*.json,
    filtered to drop songs that trip the poetry-register gates also used on
    generated output."""
    pool: dict[str, list[str]] = {}
    if not INDIAN_CORPUS_DATA_DIR.exists():
        return pool
    for artist_dir in INDIAN_CORPUS_DATA_DIR.iterdir():
        if not artist_dir.is_dir():
            continue
        for path in artist_dir.glob("*.json"):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            artist = rec.get("artist")
            lyrics = rec.get("lyrics")
            if not artist or not lyrics:
                continue
            if _trips_poetry_register(lyrics):
                continue
            pool.setdefault(artist, []).append(lyrics)
    return pool


REFERENCE_PROMPT_TEMPLATE = """You are an expert Hindi/English/Hinglish rap lyricist. Write an ORIGINAL 16-24 line
verse about: {topic}.

{craft_block}

Below is a REFERENCE verse by a real artist, shown ONLY so you can feel the TEXTURE and density of this tier of
rap -- how it flows, how tightly it rhymes -- NOT its words, story, or topic.

--- REFERENCE (feel for the tier only) ---
{reference_lyrics}
--- END REFERENCE ---

STRICT RULES about the reference:
1. Do NOT copy, quote, paraphrase, or reuse any phrase of 3 or more consecutive words from it.
2. Do NOT write about the same topic, story, or subject as it.
3. Do NOT name-drop the reference artist or mention that a reference was used.
4. It is only a tier "feel" -- your output must be 100% original content about "{topic}".

Language: write in {language_label}. {lang_block}
Think through your rhyme scheme first, then write. Apply the craft above at the density its tier calls for --
show the technique through the actual words, never by naming it. Do NOT write technical terms like "internal rhyme",
"multisyllabic", "assonance", "metaphor", "flow", or "score" in the lyrics themselves.
Output ONLY the finished verse lines, one bar per line -- no title, no explanation, no markdown, no analysis,
no mention of the reference or the craft rules."""


_LANG_BLOCK = (
    "Genuinely code-switch -- flip between Devanagari-script Hindi and English within and across lines "
    "throughout the verse, not just a token Hindi/English word dropped in, and not segregated into an "
    "all-Hindi half and an all-English half. ALL Hindi words/phrases/lines MUST be written in actual "
    "Devanagari script, e.g. \"मेरे नाम की गूँज यहाँ हर तरफ है\" (a generic illustrative phrase, not "
    "part of the verse itself) -- NEVER Romanized Hindi like \"mere naam ki goonj yahaan har taraf hai\". "
    "English words stay in Latin script as normal."
)


def build_reference_prompt(tier: str, topic_idx: int, reference_lyrics: str) -> str:
    from corpus.synthetic.rap_craft import craft_block

    topic = _TOPICS[topic_idx % len(_TOPICS)]
    return REFERENCE_PROMPT_TEMPLATE.format(
        craft_block=craft_block(tier),
        language_label=_LANG_LABEL,
        lang_block=_LANG_BLOCK,
        topic=topic,
        reference_lyrics=reference_lyrics[:REFERENCE_MAX_CHARS],
    )


async def call_ollama(client: httpx.AsyncClient, prompt: str) -> str:
    resp = await client.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.9},
        },
        # measured ~118s for a full reference+targets prompt (the "thinking"
        # mode this cloud model uses is slow) -- comfortably above that.
        timeout=240.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


async def generate_one(client: httpx.AsyncClient, tier: str, reference_pool: dict,
                        topic_idx: int = 0) -> dict | None:
    candidates = reference_pool.get(tier) or []
    if not candidates:
        print(f"    ! no real reference songs available for tier={tier}, skipping")
        return None

    for attempt in range(1, MAX_ATTEMPTS_MIXED + 1):
        ref_artist, ref_lyrics = _pick_reference(candidates)
        prompt = build_reference_prompt(tier, topic_idx + attempt, ref_lyrics)
        try:
            lyrics = await call_ollama(client, prompt)
        except Exception as exc:
            detail = str(exc) or repr(exc)
            print(f"    ! generation call failed (attempt {attempt}): {type(exc).__name__}: {detail}")
            # Back off before retrying -- the 300-song batch showed long stretches of
            # "Server disconnected"/connection-refused from the Ollama cloud proxy;
            # retrying instantly just re-hits the same overloaded/down endpoint.
            await asyncio.sleep(min(5.0 * attempt, 20.0))
            continue

        lyrics = re.sub(r"^```[a-z]*\n?|```$", "", lyrics, flags=re.MULTILINE).strip()
        if len(content_lines(lyrics)) < 6:
            continue

        accepted, actual, expected = score_against_tier(tier, lyrics)
        if accepted:
            return {
                "id": str(uuid.uuid4()),
                "tier": tier,
                "language": "mixed",
                "target_profile": build_targets(tier),
                "actual_scores": actual,
                "expected_scores": expected,
                "attempts": attempt,
                "generator": MODEL,
                "reference_artist_tier": tier,  # which tier's reference informed style, not the text itself
                "lyrics": lyrics,
            }
        print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), drifted from {tier} targets")
    return None


def _plan_batch(count: int) -> list[str]:
    """Spread `count` samples evenly across tiers -- Hinglish-only, so there's
    no language dimension left to cross against."""
    return [TIER_NAMES[i % len(TIER_NAMES)] for i in range(count)]


async def run(count: int, tier_filter: str | None, resume: bool) -> int:
    reference_pool = load_reference_pool()
    for tier in TIER_NAMES:
        print(f"  reference pool: {tier:11} {len(reference_pool.get(tier, []))} real songs available")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = _plan_batch(count)
    if tier_filter:
        plan = [t for t in plan if t == tier_filter]

    accepted_total = rejected_total = 0
    async with httpx.AsyncClient() as client:
        for i, tier in enumerate(plan, 1):
            tier_dir = OUT_DIR / tier / "mixed"
            tier_dir.mkdir(parents=True, exist_ok=True)
            existing = len(list(tier_dir.glob("*.json")))
            if resume and existing > i // len(TIER_NAMES):
                continue

            record = await generate_one(client, tier, reference_pool, topic_idx=i)
            if record is None:
                rejected_total += 1
                print(f"  [{i}/{len(plan)}] {tier}/mixed -> DROPPED after {MAX_ATTEMPTS_MIXED} attempts")
                continue

            path = tier_dir / f"{record['id']}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            accepted_total += 1
            print(f"  [{i}/{len(plan)}] {tier}/mixed -> accepted in {record['attempts']} attempt(s) ({path.name})")

    print(f"\nDONE: {accepted_total} accepted, {rejected_total} dropped, out of {len(plan)} planned")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate reference-anchored synthetic Hinglish lyrics via minimax-m3:cloud")
    ap.add_argument("--count", type=int, default=30, help="total samples to attempt across tiers")
    ap.add_argument("--tier", choices=TIER_NAMES, help="only this tier")
    ap.add_argument("--resume", action="store_true", help="skip tier slots that already have enough samples")
    args = ap.parse_args()

    return asyncio.run(run(args.count, args.tier, args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
