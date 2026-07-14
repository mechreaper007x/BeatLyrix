"""
Synthetic lyrics generator, STYLE-anchored on REAL, CONSENTED reference lyrics
(corpus/consented_seeds/ -- the project owner's own writing + named
collaborators who agreed to this use), NOT the scraped/copyrighted corpus.

Mirrors corpus/synthetic/generate_with_reference.py exactly (same numeric tier
targets, same score_against_tier validation loop, same accept/reject/retry
logic, same output schema/location) -- the only differences: (1) the reference
pool comes from consented_seeds/ instead of corpus/data/, with tier assigned by
the seed owner rather than derived from corpus/real_corpus/indian_tiers.py, and
(2) generation runs on Mistral (mistral-medium-latest, matching the reliability
fix already applied to main.py's calls -- Ministral 3B hallucinates too much
for this) instead of Ollama.

Even with consent, the STRICT RULES anti-copy contract still applies: consent
covers *using* the seed as an input, not license to just reword it -- the goal
is a genuinely new verse that only borrows the seed's texture/density.

Hinglish-only, matching corpus/synthetic/generate.py -- pure "hi" and pure
"en" generation modes were dropped entirely.

Usage (resumable, like the other two generators):
    MISTRAL_API_KEY=xxxx python -m corpus.synthetic.generate_from_consented --count 60 --resume
    MISTRAL_API_KEY=xxxx python -m corpus.synthetic.generate_from_consented --tier elite --count 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpus.synthetic.generate import (  # noqa: E402
    MAX_ATTEMPTS_MIXED,
    OUT_DIR,
    _LANG_LABEL,
    _TOPICS,
    score_against_tier,
)
from corpus.synthetic.tier_profiles import TIER_NAMES, build_targets  # noqa: E402
from services.language_utils import content_lines  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

SEEDS_DIR = Path(__file__).resolve().parents[1] / "consented_seeds"
REFERENCE_MAX_CHARS = 1200
GENERATION_MODEL = "mistral-medium-latest"


def load_consented_pool() -> dict[str, list[tuple[str, str]]]:
    """tier -> [(seed_id, lyrics), ...] from corpus/consented_seeds/*.json."""
    pool: dict[str, list[tuple[str, str]]] = {t: [] for t in TIER_NAMES}
    if not SEEDS_DIR.exists():
        return pool
    for path in SEEDS_DIR.glob("*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not rec.get("consented"):
            continue
        tier = rec.get("tier")
        lyrics = rec.get("lyrics")
        seed_id = rec.get("seed_id", path.stem)
        if tier in TIER_NAMES and lyrics:
            pool[tier].append((seed_id, lyrics))
    return pool


# Identical anti-copy contract to generate_with_reference.py's
# REFERENCE_PROMPT_TEMPLATE -- consent covers using the seed as input, not a
# license to just reword it. The goal is a genuinely new verse that only
# borrows the seed's texture/density.
REFERENCE_PROMPT_TEMPLATE = """You are an expert Hindi/English/Hinglish rap lyricist. Write an ORIGINAL 16-24 line
verse about: {topic}.

{craft_block}

Below is a REFERENCE verse, shown ONLY so you can feel the TEXTURE and density of this tier of rap -- how it
flows, how tightly it rhymes -- NOT its words, story, or topic.

--- REFERENCE (feel for the tier only) ---
{reference_lyrics}
--- END REFERENCE ---

STRICT RULES about the reference:
1. Do NOT copy, quote, paraphrase, or reuse any phrase of 3 or more consecutive words from it.
2. Do NOT write about the same topic, story, or subject as it.
3. Do NOT name-drop or mention that a reference was used.
4. It is only a tier "feel" -- your output must be 100% original content about "{topic}".

Language: write in {language_label}.
Think through your rhyme scheme first, then write. Apply the craft above at the density its tier calls for --
show the technique through the actual words, never by naming it. Do NOT write technical terms like "internal rhyme",
"multisyllabic", "assonance", "metaphor", "flow", or "score" in the lyrics themselves.
Output ONLY the finished verse lines, one bar per line -- no title, no explanation, no markdown, no analysis,
no mention of the reference or the craft rules."""


def build_reference_prompt(tier: str, topic_idx: int, reference_lyrics: str) -> str:
    from corpus.synthetic.rap_craft import craft_block

    topic = _TOPICS[topic_idx % len(_TOPICS)]
    return REFERENCE_PROMPT_TEMPLATE.format(
        craft_block=craft_block(tier),
        language_label=_LANG_LABEL,
        topic=topic,
        reference_lyrics=reference_lyrics[:REFERENCE_MAX_CHARS],
    )


async def generate_one(client, tier: str, consented_pool: dict, topic_idx: int = 0) -> dict | None:
    candidates = consented_pool.get(tier) or []
    if not candidates:
        print(f"    ! no consented seeds available for tier={tier}, skipping")
        return None

    for attempt in range(1, MAX_ATTEMPTS_MIXED + 1):
        seed_id, ref_lyrics = random.choice(candidates)
        prompt = build_reference_prompt(tier, topic_idx + attempt, ref_lyrics)
        try:
            response = await client.chat.complete_async(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            lyrics = response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"    ! generation call failed (attempt {attempt}): {exc}")
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
                "generator": GENERATION_MODEL,
                "seed_id": seed_id,          # provenance slug only, never the raw seed text
                "consented": True,
                "lyrics": lyrics,
            }
        print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), drifted from {tier} targets")
    return None


def _plan_batch(count: int) -> list[str]:
    """Spread `count` samples evenly across tiers -- Hinglish-only, so there's
    no language dimension left to weight/shuffle."""
    return [TIER_NAMES[i % len(TIER_NAMES)] for i in range(count)]


async def run(count: int, tier_filter: str | None, resume: bool) -> int:
    from mistralai import Mistral

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: set MISTRAL_API_KEY", file=sys.stderr)
        return 2
    client = Mistral(api_key=api_key)

    consented_pool = load_consented_pool()
    for tier in TIER_NAMES:
        n = len(consented_pool.get(tier, []))
        print(f"  consented pool: {tier:11} {n} seed(s) available")
    if not any(consented_pool.values()):
        print(
            f"ERROR: no consented seeds found in {SEEDS_DIR} -- add seed JSON files "
            f"first (see corpus/consented_seeds/README.md)",
            file=sys.stderr,
        )
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = _plan_batch(count)
    if tier_filter:
        plan = [t for t in plan if t == tier_filter]

    accepted_total = rejected_total = 0
    for i, tier in enumerate(plan, 1):
        tier_dir = OUT_DIR / tier / "mixed"
        tier_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(tier_dir.glob("*.json")))
        if resume and existing > i // len(TIER_NAMES):
            continue

        record = await generate_one(client, tier, consented_pool, topic_idx=i)
        if record is None:
            rejected_total += 1
            print(f"  [{i}/{len(plan)}] {tier}/mixed -> DROPPED after {MAX_ATTEMPTS_MIXED} attempts")
            continue

        path = tier_dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        accepted_total += 1
        print(f"  [{i}/{len(plan)}] {tier}/mixed -> accepted in {record['attempts']} attempt(s) ({path.name})")
        time.sleep(0.3)  # gentle pacing against Mistral rate limits

    print(f"\nDONE: {accepted_total} accepted, {rejected_total} dropped, out of {len(plan)} planned")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate consented-seed-anchored synthetic Hinglish lyrics via Mistral")
    ap.add_argument("--count", type=int, default=60, help="total samples to attempt across tiers")
    ap.add_argument("--tier", choices=TIER_NAMES, help="only this tier")
    ap.add_argument("--resume", action="store_true", help="skip tier slots that already have enough samples")
    args = ap.parse_args()

    return asyncio.run(run(args.count, args.tier, args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
