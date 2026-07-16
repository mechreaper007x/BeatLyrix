"""
Synthetic lyrics generator, STYLE-anchored on a real ENGLISH rap reference
song, sibling to generate_with_reference.py (which does the same thing for
Hinglish). Same numeric tier targets (corpus.synthetic.tier_profiles), same
local validation loop (score_against_tier, now called with lang="en" so the
english_ratio/codeswitch_density targets are English-appropriate instead of
the Hinglish "moderate code-switch" defaults), same anti-poetry/anti-artifact
content gates (generate.py's run_content_gates) -- the only difference from
generate_with_reference.py is the reference pool and the prompt's language
instructions: no code-switching, no Devanagari-script requirements.

Reference pool = corpus/real_corpus/data/ (English, tier baked into each
record from real_artists.py's REAL_ARTIST_TIERS) union kaggle_rap_data/'s 11
artists (tier from local_real_model/kaggle_artist_tiers.py, produced by
local_real_model/derive_english_tiers.py -- see that script's docstring for
why those tiers are critical-consensus, not measured-density, labels).

Anonymous tier-anchoring only, per explicit user decision: the prompt shows
the model a real reference verse purely to convey the TIER's texture/density
("study the technique, not the words"), never the reference artist's name --
no persona branding (contrast with generate_persona.py, which is reserved for
the Hindi/mixed artist roster in corpus/artists.py).

kaggle_rap_data/ and corpus/real_corpus/data/ are both scraped/copyrighted
and gitignored, local-only -- but this script's OUTPUT (new, original,
LLM-generated lyrics) is not derived text and is committed normally into the
live corpus/synthetic_data/ corpus, same as every other synthetic generator.

Usage (pilot batch, resumable like generate.py/generate_with_reference.py):
    GEMINI_API_KEY=xxxx python -m corpus.synthetic.generate_english_reference --count 30 --resume
    GEMINI_API_KEY=xxxx python -m corpus.synthetic.generate_english_reference --tier elite --count 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LOCAL_REAL_MODEL_DIR = Path(__file__).resolve().parents[3] / "local_real_model"
sys.path.insert(0, str(LOCAL_REAL_MODEL_DIR))

from corpus.synthetic.generate import (  # noqa: E402
    GEMINI_API_KEY,
    MAX_ATTEMPTS_MIXED,
    OUT_DIR,
    _TOPICS,
    _gemini_generate,
    _line_uniqueness_ratio,
    _strip_leaked_metrics_footer,
    _strip_meta_annotation_lines,
    run_content_gates,
    score_against_tier,
)
from corpus.synthetic.generate_with_reference import (  # noqa: E402
    REFERENCE_MAX_CHARS,
    REFERENCE_PROMPT_TEMPLATE,
)
from corpus.synthetic.tier_profiles import TIER_NAMES, build_targets  # noqa: E402
from corpus.synthetic import refine as refine_mod  # noqa: E402
from corpus.synthetic import rhyme_families  # noqa: E402
from services.language_utils import content_lines  # noqa: E402

from derive_english_tiers import KAGGLE_CSV_PATH, clean_kaggle_lyrics  # noqa: E402
from kaggle_artist_tiers import KAGGLE_ARTIST_TIERS  # noqa: E402

REAL_CORPUS_DATA_DIR = Path(__file__).resolve().parents[1] / "real_corpus" / "data"
_LANG_LABEL_EN = "English"
_LANG_BLOCK_EN = "Write entirely in English -- no code-switching, no non-English words or script."

# How many rhyme families to inject per elite/mid prompt (commercial wants
# simple single-syllable rhymes, so no multisyllabic scaffolding there).
_SCAFFOLD_FAMILIES = {"elite": 4, "mid": 2, "commercial": 0}
# Revision rounds after the initial draft (see corpus.synthetic.refine).
_REFINE_ROUNDS = 2

# A/B toggles: setting BEATLYRIX_NO_SCAFFOLD/BEATLYRIX_NO_REFINE=1 reproduces the
# old blind-rejection loop (no rhyme scaffolding, no critique-revise) through
# this same code path, so the pilot compares only those two features. Default
# off -> the new loop runs.
_SCAFFOLD_ON = os.getenv("BEATLYRIX_NO_SCAFFOLD", "") not in ("1", "true", "True")
_REFINE_ON = False  # A/B pilot showed net negative; scaffold-only is the quality lever


def _clean_lyrics(raw: str) -> str:
    """The generator's shared post-processing -- strip code fences, markdown
    bold, stray wrapping quotes, and leaked meta/metrics lines. Used for both
    the first draft and every refine revision so scoring sees the same shape."""
    lyrics = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
    lyrics = re.sub(r"\*\*(.+?)\*\*", r"\1", lyrics)
    lyrics = re.sub(r'^"|"$', "", lyrics.strip(), flags=re.MULTILINE).strip()
    lyrics = _strip_meta_annotation_lines(lyrics)
    lyrics = _strip_leaked_metrics_footer(lyrics)
    return lyrics


def _scaffold_block(tier: str, rng: random.Random) -> str:
    """Rhyme building-block prompt fragment for this tier, or '' when the tier
    calls for no multisyllabic scaffolding."""
    k = _SCAFFOLD_FAMILIES.get(tier, 0)
    if k <= 0 or not _SCAFFOLD_ON:
        return ""
    return rhyme_families.build_scaffold_block("en", k, rng)



def load_real_corpus_pool() -> dict[str, list[tuple[str, str]]]:
    """tier -> [(artist, lyrics), ...] from corpus/real_corpus/data/, tier
    already baked into each record by corpus/real_corpus/fetch_real_corpus.py."""
    pool: dict[str, list[tuple[str, str]]] = {t: [] for t in TIER_NAMES}
    if not REAL_CORPUS_DATA_DIR.exists():
        return pool
    for artist_dir in REAL_CORPUS_DATA_DIR.iterdir():
        if not artist_dir.is_dir():
            continue
        for path in artist_dir.glob("*.json"):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tier, lyrics, artist = rec.get("tier"), rec.get("lyrics"), rec.get("artist")
            if tier in pool and lyrics and artist:
                pool[tier].append((artist, lyrics))
    return pool


def load_kaggle_pool() -> dict[str, list[tuple[str, str]]]:
    """tier -> [(artist, lyrics), ...] from kaggle_rap_data/, tier from
    local_real_model/kaggle_artist_tiers.py (Part A's output)."""
    import pandas as pd

    pool: dict[str, list[tuple[str, str]]] = {t: [] for t in TIER_NAMES}
    if not KAGGLE_CSV_PATH.exists():
        return pool
    df = pd.read_csv(KAGGLE_CSV_PATH)
    for artist, group in df.groupby("artist"):
        tier = KAGGLE_ARTIST_TIERS.get(str(artist))
        if tier not in pool:
            continue
        for raw in group["artist_verses"].tolist():
            lyrics = clean_kaggle_lyrics(str(raw))
            if len(lyrics) >= 50:
                pool[tier].append((str(artist), lyrics))
    return pool


def load_english_reference_pool() -> dict[str, list[tuple[str, str]]]:
    pool: dict[str, list[tuple[str, str]]] = {t: [] for t in TIER_NAMES}
    for source_pool in (load_real_corpus_pool(), load_kaggle_pool()):
        for tier in TIER_NAMES:
            pool[tier].extend(source_pool[tier])
    return pool


def build_prompt(tier: str, topic_idx: int, reference_lyrics: str,
                 scaffold_block: str = "") -> str:
    from corpus.synthetic.rap_craft import craft_block

    topic = _TOPICS[topic_idx % len(_TOPICS)]
    # REFERENCE_PROMPT_TEMPLATE has no scaffold placeholder, so the rhyme
    # building blocks ride along inside craft_block -- they sit right below the
    # tier's craft guidance, which is where the model is already being told how
    # densely to rhyme for this tier.
    craft = craft_block(tier)
    if scaffold_block:
        craft = f"{craft}\n\n{scaffold_block}"
    return REFERENCE_PROMPT_TEMPLATE.format(
        craft_block=craft,
        language_label=_LANG_LABEL_EN,
        lang_block=_LANG_BLOCK_EN,
        topic=topic,
        reference_lyrics=reference_lyrics[:REFERENCE_MAX_CHARS],
    )


async def generate_one(tier: str, reference_pool: dict, topic_idx: int = 0) -> dict | None:
    candidates = reference_pool.get(tier) or []
    if not candidates:
        print(f"    ! no real English reference songs available for tier={tier}, skipping")
        return None

    # Deterministic-per-call RNG so scaffold sampling doesn't touch the global
    # random stream the reference/topic choices use.
    rng = random.Random(f"{tier}:{topic_idx}")

    for attempt in range(1, MAX_ATTEMPTS_MIXED + 1):
        ref_artist, ref_lyrics = random.choice(candidates)
        scaffold = _scaffold_block(tier, rng)
        prompt = build_prompt(tier, topic_idx + attempt, ref_lyrics, scaffold)
        try:
            raw = await _gemini_generate(prompt, temperature=0.9)
        except Exception as exc:
            detail = str(exc) or repr(exc)
            print(f"    ! generation call failed (attempt {attempt}): {type(exc).__name__}: {detail}")
            await asyncio.sleep(min(5.0 * attempt, 20.0))
            continue

        lyrics = _clean_lyrics(raw)
        if len(content_lines(lyrics)) < 6:
            continue

        uniq_ratio = _line_uniqueness_ratio(lyrics)
        if uniq_ratio < 0.5:
            print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), "
                  f"line_uniqueness={uniq_ratio:.2f} < 0.5 (degenerate repetition)")
            continue

        # Critique-revise: use the per-axis gap from score_against_tier as a
        # reward signal and nudge the weak devices instead of discarding a
        # near-miss draft. Returns the strongest iteration seen (accepted or
        # not) so the content gates below still get the best candidate. The
        # _REFINE_ON toggle lets the A/B pilot run the old blind-rejection arm.
        if _REFINE_ON:
            lyrics, accepted, actual, expected = await refine_mod.refine(
                tier=tier,
                lang="en",
                draft=lyrics,
                gen=lambda p: _gemini_generate(p, temperature=0.9),
                clean=_clean_lyrics,
                lang_label=_LANG_LABEL_EN,
                lang_block=_LANG_BLOCK_EN,
                scaffold_fn=lambda: _scaffold_block(tier, rng),
                max_rounds=_REFINE_ROUNDS,
                log=print,
            )

        # Re-check the cheap shape gates on the (possibly revised) best draft.
        if len(content_lines(lyrics)) < 6:
            continue
        uniq_ratio = _line_uniqueness_ratio(lyrics)
        if uniq_ratio < 0.5:
            print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), "
                  f"line_uniqueness={uniq_ratio:.2f} < 0.5 after refine")
            continue

        # client=None: run_content_gates' only follow-up-edit repair path
        # (_repair_anaphora) degrades gracefully to "repair failed" when its
        # Mistral call target is None -- no Mistral client is wired up here
        # since this generator is Gemini-only, and that's an acceptable
        # rejection rather than a crash.
        lyrics, gate_reason = await run_content_gates(None, lyrics, attempt, MAX_ATTEMPTS_MIXED)
        if gate_reason:
            print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), {gate_reason}")
            continue

        # Content gates may have edited lines; re-score so the stored actual/
        # accept reflect exactly what we keep.
        accepted, actual, expected = score_against_tier(tier, lyrics, lang="en")
        if accepted:
            return {
                "id": str(uuid.uuid4()),
                "tier": tier,
                "language": "en",
                "target_profile": build_targets(tier),
                "actual_scores": actual,
                "expected_scores": expected,
                "attempts": attempt,
                "generator": "gemini-en-reference-refine" if _REFINE_ON else "gemini-en-reference",
                "reference_artist_tier": tier,  # which tier's reference informed style, not the text itself
                "lyrics": lyrics,
            }
        print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), drifted from {tier} targets")
    return None


def _plan_batch(count: int) -> list[str]:
    return [TIER_NAMES[i % len(TIER_NAMES)] for i in range(count)]


async def run(count: int, tier_filter: str | None, resume: bool) -> int:
    if not GEMINI_API_KEY:
        print("ERROR: set GEMINI_API_KEY", file=sys.stderr)
        return 2

    reference_pool = load_english_reference_pool()
    for tier in TIER_NAMES:
        print(f"  reference pool: {tier:11} {len(reference_pool.get(tier, []))} real English songs available")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = _plan_batch(count)
    if tier_filter:
        plan = [t for t in plan if t == tier_filter]

    accepted_total = rejected_total = 0
    for i, tier in enumerate(plan, 1):
        tier_dir = OUT_DIR / tier / "en"
        tier_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(tier_dir.glob("*.json")))
        if resume and existing > i // len(TIER_NAMES):
            continue

        record = await generate_one(tier, reference_pool, topic_idx=i)
        if record is None:
            rejected_total += 1
            print(f"  [{i}/{len(plan)}] {tier}/en -> DROPPED after {MAX_ATTEMPTS_MIXED} attempts")
            continue

        path = tier_dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        accepted_total += 1
        print(f"  [{i}/{len(plan)}] {tier}/en -> accepted in {record['attempts']} attempt(s) ({path.name})")

    print(f"\nDONE: {accepted_total} accepted, {rejected_total} dropped, out of {len(plan)} planned")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate reference-anchored synthetic English lyrics via Gemini, "
                     "seeded from corpus/real_corpus/data/ + kaggle_rap_data/"
    )
    ap.add_argument("--count", type=int, default=30, help="total samples to attempt across tiers")
    ap.add_argument("--tier", choices=TIER_NAMES, help="only this tier")
    ap.add_argument("--resume", action="store_true", help="skip tier slots that already have enough samples")
    args = ap.parse_args()

    return asyncio.run(run(args.count, args.tier, args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
