"""
Persona-anchored Hinglish lyrics generator -- ghostwrites a verse in the
EXPLICIT style of one named real artist (e.g. KR$NA, Raftaar), instead of
only numeric targets (corpus/synthetic/generate.py) or an anonymous
tier-only reference (corpus/synthetic/generate_with_reference.py).

This wires up an approach already manually validated: prompting Gemma-4-31b
with explicit persona framing ("considering the persona of KR$NA and
explained it how it all works each element") produced genuinely good rap on
the first try, where the 20-numeric-axis prompt in generate.py was hitting
Devanagari-ratio failures, translation-gloss artifacts, and pure-language
drift with a 0% acceptance rate.

Primary generation runs on Gemma-4 via generate.py's _gemini_generate()
(GEMINI_API_KEY required) -- the narrow repair-pass edit calls
(intraline-mix repair, anaphora repair, inside run_content_gates()) stay on
Mistral (MISTRAL_API_KEY required), matching generate.py exactly.

NOTE on exposure: like generate_with_reference.py, this sends a real
(capped, ~1200-char) excerpt of scraped lyric text to a third-party cloud
API (Gemini) on every call. Same category of exposure already flagged and
accepted for minimax-m3:cloud in generate_with_reference.py, just a
different endpoint.

Usage (fast manual-comparison mode -- per the project's standing rule,
manually read every accepted sample against a real song from
corpus/data/{artist}/ before trusting gate-pass numbers alone):
    GEMINI_API_KEY=xxxx MISTRAL_API_KEY=xxxx python -m corpus.synthetic.generate_persona --artist "KR$NA" --tier elite --count 5
    GEMINI_API_KEY=xxxx MISTRAL_API_KEY=xxxx python -m corpus.synthetic.generate_persona --artist Raftaar --tier elite --count 5

Usage (production batch mode -- one persona artist per tier slot, drawn from
corpus/real_corpus/indian_tiers.py's artist_tier_map(), same output
location/schema as generate.py; concurrent workers, actual throughput still
capped by generate.py's shared GEMINI_RPM rate limiter regardless of
--concurrency):
    GEMINI_API_KEY=xxxx MISTRAL_API_KEY=xxxx python -m corpus.synthetic.generate_persona --count 300 --concurrency 6 --resume
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

from corpus.artists import unique_artists  # noqa: E402
from corpus.real_corpus.indian_tiers import artist_tier_map  # noqa: E402
from corpus.synthetic.generate import (  # noqa: E402
    GEMINI_API_KEY,
    MAX_ATTEMPTS_MIXED,
    OUT_DIR,
    _ANCHORS,
    _LANG_LABEL,
    _MIN_DEVANAGARI_RATIO,
    _TOPICS,
    _devanagari_ratio,
    _gemini_generate,
    _intraline_mix_fraction,
    _line_uniqueness_ratio,
    _repair_intraline_mix,
    _strip_leaked_metrics_footer,
    _strip_meta_annotation_lines,
    run_content_gates,
    score_against_tier,
)
from corpus.synthetic.generate_with_reference import load_artist_reference_pool  # noqa: E402
from corpus.synthetic.rap_craft import craft_block  # noqa: E402
from corpus.synthetic.tier_profiles import TIER_NAMES, build_targets  # noqa: E402
from services.language_utils import content_lines  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

REFERENCE_MAX_CHARS = 1200
SCRATCH_DIR = Path(__file__).resolve().parents[2] / "scratch" / "persona_test"

_ARTISTS_BY_NAME = {a.name: a for a in unique_artists()}


def _persona_traits(profile: dict) -> str:
    """Turn artists.py's expected_profile 0-1 priors into plain-language
    persona traits -- never shown as raw numbers to the model, since a
    "multisyllabic=0.9" style hint reads as a numeric target again, the
    exact dilution problem this persona approach is meant to fix."""
    multisyl = profile.get("multisyllabic", 0.5)
    wordplay = profile.get("wordplay", 0.5)
    commercial = profile.get("commercial", 0.5)

    if multisyl >= 0.75:
        rhyme_trait = "dense, technical multisyllabic rhyme schemes"
    elif multisyl >= 0.45:
        rhyme_trait = "clean, well-constructed rhyme schemes"
    else:
        rhyme_trait = "simple, easy-to-follow rhymes"

    if wordplay >= 0.75:
        wordplay_trait = "layered wordplay and clever double meanings"
    elif wordplay >= 0.45:
        wordplay_trait = "occasional sharp wordplay"
    else:
        wordplay_trait = "direct, plainspoken bars over intricate wordplay"

    if commercial <= 0.3:
        vibe_trait = "an underground, lyricist-first reputation over chart-chasing hooks"
    elif commercial <= 0.6:
        vibe_trait = "a balance of technical skill and mainstream appeal"
    else:
        vibe_trait = "a catchy, hook-driven, radio-friendly sound"

    return f"known for {rhyme_trait}, {wordplay_trait}, and {vibe_trait}"


PERSONA_PROMPT_TEMPLATE = """You are ghostwriting a verse in the exact style of {artist}, a Hinglish rapper {traits}.

LANGUAGE (read this first, it governs everything below): you must write in {language_label}. Genuinely code-switch --
flip between Devanagari-script Hindi and English WITHIN and across lines throughout the verse, not just a token
Hindi/English word dropped in, and not segregated into an all-Hindi half and an all-English half. This is a HINDI
verse with English mixed in, NOT an English verse -- most lines should contain Hindi. ALL Hindi words/phrases/lines
MUST be written in actual Devanagari script, e.g. "मेरे नाम की गूँज यहाँ हर तरफ है" (a generic illustrative phrase,
not part of the verse itself) -- NEVER Romanized Hindi like "mere naam ki goonj yahaan har taraf hai". English words
stay in Latin script as normal. At least 1 out of every 4 lines must individually contain BOTH Devanagari and
Latin-script words in that SAME line.

{craft_block}

Below is a REFERENCE verse by {artist}, shown ONLY so you can feel THEIR voice, flow, and technique -- NOT their
words, story, or topic.

--- REFERENCE (feel for {artist}'s style only) ---
{reference_lyrics}
--- END REFERENCE ---

STRICT RULES about the reference:
1. Do NOT copy, quote, paraphrase, or reuse any phrase of 3 or more consecutive words from it.
2. Do NOT write about the same topic, story, or subject as it.
3. Do NOT name-drop {artist} or mention that a reference was used.
4. It is only a style "feel" -- your output must be 100% original content about "{topic}".

Write an ORIGINAL 16-24 line verse about: {topic}. Build the verse's structure and imagery around this specific,
concrete detail so it can't fall back on a generic template: {anchor}.

REMINDER before you write: this must be a genuinely Hindi(Devanagari)+English code-switched verse, not an English
verse with a stray Hindi word -- most lines need real Devanagari Hindi in them, mixed mid-line with English, the way
{artist} actually raps. Think through your rhyme scheme first, then write. Apply the craft above at the density
{artist} would actually use -- show the technique through the actual words, never by naming it. Do NOT write
technical terms like "internal rhyme", "multisyllabic", "assonance", "metaphor", "flow", or "score" in the lyrics
themselves. Output ONLY the finished verse lines, one bar per line -- no title, no explanation, no markdown, no
analysis, no mention of the reference, the craft rules, or {artist}'s name."""


def build_persona_prompt(artist: str, tier: str, topic_idx: int, reference_lyrics: str) -> str:
    profile = _ARTISTS_BY_NAME[artist].expected_profile if artist in _ARTISTS_BY_NAME else {}
    topic = _TOPICS[topic_idx % len(_TOPICS)]
    anchor = _ANCHORS[(topic_idx * 7 + 3) % len(_ANCHORS)]
    return PERSONA_PROMPT_TEMPLATE.format(
        artist=artist,
        traits=_persona_traits(profile),
        craft_block=craft_block(tier),
        language_label=_LANG_LABEL,
        topic=topic,
        anchor=anchor,
        reference_lyrics=reference_lyrics[:REFERENCE_MAX_CHARS],
    )


async def generate_one_persona(client, artist: str, tier: str, reference_pool: dict,
                                topic_idx: int = 0) -> dict | None:
    """Mirrors generate.py::generate_one()'s validation pipeline exactly
    (strip -> devanagari gate -> intraline-mix gate+repair -> uniqueness gate
    -> run_content_gates -> score_against_tier -> language-balance gate) so
    persona-anchored output is held to the identical bar as numeric-only
    generation, just with a persona-framed prompt instead."""
    candidates = reference_pool.get(artist) or []
    if not candidates:
        print(f"    ! no real reference songs available for artist={artist}, skipping")
        return None

    for attempt in range(1, MAX_ATTEMPTS_MIXED + 1):
        reference_lyrics = random.choice(candidates)
        prompt = build_persona_prompt(artist, tier, topic_idx + attempt, reference_lyrics)
        try:
            lyrics = await _gemini_generate(prompt, temperature=0.9)
        except Exception as exc:
            detail = str(exc) or repr(exc)
            print(f"    ! generation call failed (attempt {attempt}): {type(exc).__name__}: {detail}")
            await asyncio.sleep(min(5.0 * attempt, 20.0))
            continue

        lyrics = re.sub(r"^```[a-z]*\n?|```$", "", lyrics, flags=re.MULTILINE).strip()
        lyrics = re.sub(r"\*\*(.+?)\*\*", r"\1", lyrics)
        lyrics = re.sub(r'^"|"$', "", lyrics.strip(), flags=re.MULTILINE).strip()
        lyrics = _strip_meta_annotation_lines(lyrics)
        lyrics = _strip_leaked_metrics_footer(lyrics)
        if len(content_lines(lyrics)) < 6:
            continue

        deva_ratio = _devanagari_ratio(lyrics)
        if deva_ratio < _MIN_DEVANAGARI_RATIO:
            print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), "
                  f"devanagari_ratio={deva_ratio:.2f} < {_MIN_DEVANAGARI_RATIO} (Romanized, not Devanagari)")
            continue

        mix_frac = _intraline_mix_fraction(lyrics)
        if mix_frac < 0.2:
            repaired = await _repair_intraline_mix(client, lyrics)
            repaired_mix_frac = _intraline_mix_fraction(repaired) if repaired else 0.0
            if repaired and repaired_mix_frac >= 0.2 and len(content_lines(repaired)) >= 6:
                print(f"    ~ repaired via follow-up edit (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), "
                      f"intraline_mix_fraction {mix_frac:.2f} -> {repaired_mix_frac:.2f}")
                lyrics = repaired
            else:
                print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), "
                      f"intraline_mix_fraction={mix_frac:.2f} < 0.2 "
                      f"(lines are segregated mono-lingual, not genuinely code-switched within a line); "
                      f"repair attempt also failed (repaired={repaired_mix_frac:.2f})")
                continue

        uniq_ratio = _line_uniqueness_ratio(lyrics)
        if uniq_ratio < 0.5:
            print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), "
                  f"line_uniqueness={uniq_ratio:.2f} < 0.5 (degenerate repetition)")
            continue

        lyrics, gate_reason = await run_content_gates(client, lyrics, attempt, MAX_ATTEMPTS_MIXED)
        if gate_reason:
            print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), {gate_reason}")
            if os.environ.get("DEBUG_DUMP_LYRICS"):
                print("    ---- rejected lyrics dump ----")
                print(lyrics)
                print("    -------------------------------")
            continue

        accepted, actual, expected = score_against_tier(tier, lyrics)
        # NOTE: no english_ratio/codeswitch_density post-hoc gate here -- that
        # metric filters out Devanagari words before computing the ratio, so it
        # reads ~100% for any genuinely bilingual verse (see generate.py for the
        # full explanation). devanagari_ratio and intraline_mix_fraction above
        # already correctly verify genuine code-switching.
        if accepted:
            return {
                "id": str(uuid.uuid4()),
                "tier": tier,
                "language": "mixed",
                "target_profile": build_targets(tier),
                "actual_scores": actual,
                "expected_scores": expected,
                "attempts": attempt,
                "generator": "gemma-4-persona",
                "persona_artist": artist,  # provenance only -- the reference text itself is never persisted
                "lyrics": lyrics,
            }
        print(f"    ~ rejected (attempt {attempt}/{MAX_ATTEMPTS_MIXED}), drifted from {tier} targets")
    return None


async def run_artist_test(client, artist: str, tier: str, count: int) -> int:
    """Fast manual-comparison mode (plan step 5): a handful of samples for
    ONE named artist, written next to nothing but their own id -- read them
    against corpus/data/{artist}/ yourself, gate-pass numbers alone don't
    count as validation."""
    artist_meta = _ARTISTS_BY_NAME.get(artist)
    if artist_meta is not None and artist_meta.primary_language != "mixed":
        print(
            f"ERROR: {artist}'s real corpus is primarily {artist_meta.primary_language!r}, not genuinely "
            f"code-switched Hinglish -- this persona template forces intra-line Hindi/English mixing, which "
            f"would not sound like {artist}'s actual voice. Skipping (see corpus/artists.py primary_language).",
            file=sys.stderr,
        )
        return 2

    reference_pool = load_artist_reference_pool()
    candidates = reference_pool.get(artist) or []
    print(f"  reference pool: {artist} has {len(candidates)} real song(s) available (post poetry-register filter)")
    if not candidates:
        print(
            f"ERROR: no usable real songs found for artist={artist!r} in corpus/data/ -- "
            f"check the name matches corpus/artists.py exactly (e.g. 'KR$NA', not 'Krsna')",
            file=sys.stderr,
        )
        return 2

    out_dir = SCRATCH_DIR / artist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-artist offset so independent --artist runs for different artists
    # don't all land on the same (topic, anchor) pair for their Nth sample --
    # confirmed in practice: Panther/Raga/Vichaar's first sample each landed on
    # the identical "loyalty and betrayal" + "torn poster on a wall" combo
    # because topic_idx always started at 1 regardless of artist.
    artist_offset = sum(ord(c) for c in artist) % len(_TOPICS)

    accepted_total = rejected_total = 0
    for i in range(1, count + 1):
        record = await generate_one_persona(client, artist, tier, reference_pool, topic_idx=artist_offset + i)
        if record is None:
            rejected_total += 1
            print(f"  [{i}/{count}] {artist}/{tier} -> DROPPED after {MAX_ATTEMPTS_MIXED} attempts")
            continue
        path = out_dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        accepted_total += 1
        print(f"  [{i}/{count}] {artist}/{tier} -> accepted in {record['attempts']} attempt(s) ({path.name})")

    print(f"\nDONE: {accepted_total} accepted, {rejected_total} dropped, out of {count} planned")
    print(f"Wrote accepted samples to {out_dir} -- read them against real {artist} songs in "
          f"corpus/data/ before trusting these gate-pass numbers alone.")
    return 0


def _artists_for_tier(tier: str) -> list[str]:
    tiers = artist_tier_map()
    return [name for name, t in tiers.items() if t == tier]


def _plan_batch(count: int) -> list[str]:
    """Spread `count` samples evenly across tiers -- Hinglish-only, so
    there's no language dimension left to cross against."""
    return [TIER_NAMES[i % len(TIER_NAMES)] for i in range(count)]


async def _persona_worker(sem: asyncio.Semaphore, client, idx: int, total: int, tier: str, artist: str,
                           reference_pool: dict, tier_dir: Path) -> dict | None:
    async with sem:
        artist_offset = sum(ord(c) for c in artist) % len(_TOPICS)
        record = await generate_one_persona(client, artist, tier, reference_pool, topic_idx=artist_offset + idx)
        if record is None:
            print(f"  [{idx}/{total}] {tier}/{artist} -> DROPPED after {MAX_ATTEMPTS_MIXED} attempts")
            return None
        path = tier_dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{idx}/{total}] {tier}/mixed ({artist}) -> accepted in {record['attempts']} attempt(s) ({path.name})")
        return record


async def run_batch(client, count: int, tier_filter: str | None, resume: bool, concurrency: int = 6) -> int:
    reference_pool = load_artist_reference_pool()
    # Persona generation forces genuine intra-line Hindi/English code-switching --
    # artists whose real corpus isn't primarily "mixed" (e.g. Hanumankind is
    # all-English, CarryMinati/King/Yashraj are all-Hindi) would be forced into a
    # voice that doesn't match how they actually rap, so they're excluded here.
    tier_artists = {
        t: [a for a in _artists_for_tier(t)
            if _ARTISTS_BY_NAME.get(a) is None or _ARTISTS_BY_NAME[a].primary_language == "mixed"]
        for t in TIER_NAMES
    }
    for t in TIER_NAMES:
        n = len([a for a in tier_artists[t] if reference_pool.get(a)])
        print(f"  tier {t:11} {n} persona artist(s) available")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = _plan_batch(count)
    if tier_filter:
        plan = [t for t in plan if t == tier_filter]

    # Resolve artist + output dir for every planned slot up front (not inside
    # the worker) so concurrent workers never race on the same
    # tier_dir.glob("*.json") existing-count check.
    slots: list[tuple[str, str, Path]] = []
    skipped_no_artist = 0
    per_tier_seen: dict[str, int] = {}
    for tier in plan:
        tier_dir = OUT_DIR / tier / "mixed"
        tier_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(tier_dir.glob("*.json")))
        per_tier_seen[tier] = per_tier_seen.get(tier, 0) + 1
        if resume and per_tier_seen[tier] <= existing:
            continue

        candidates = [a for a in tier_artists.get(tier, []) if reference_pool.get(a)]
        if not candidates:
            print(f"  {tier} -> no persona artists available, skipping a planned slot")
            skipped_no_artist += 1
            continue
        artist = random.choice(candidates)
        slots.append((tier, artist, tier_dir))

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _persona_worker(sem, client, i, len(slots), tier, artist, reference_pool, tier_dir)
        for i, (tier, artist, tier_dir) in enumerate(slots, 1)
    ]
    results = await asyncio.gather(*tasks)

    accepted_total = sum(1 for r in results if r is not None)
    rejected_total = sum(1 for r in results if r is None) + skipped_no_artist
    print(f"\nDONE: {accepted_total} accepted, {rejected_total} dropped, out of {len(plan)} planned")
    return 0


def main() -> int:
    from mistralai import Mistral

    ap = argparse.ArgumentParser(description="Generate persona-anchored synthetic Hinglish lyrics via Gemma-4")
    ap.add_argument("--artist", help="fast manual-comparison mode: generate for ONE named real artist "
                                      "(exact name from corpus/artists.py, e.g. 'KR$NA')")
    ap.add_argument("--tier", choices=TIER_NAMES, help="tier for --artist mode (defaults to that artist's derived "
                                                         "tier); tier filter in batch mode")
    ap.add_argument("--count", type=int, default=5, help="samples to attempt (default 5)")
    ap.add_argument("--resume", action="store_true", help="batch mode only: skip tier slots that already have "
                                                            "enough samples")
    ap.add_argument("--concurrency", type=int, default=6, help="batch mode only: max in-flight Gemini calls "
                                                                 "(actual throughput still capped by GEMINI_RPM)")
    args = ap.parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: set GEMINI_API_KEY (primary generation runs on Gemma-4)", file=sys.stderr)
        return 2
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        print("ERROR: set MISTRAL_API_KEY (repair-pass edit calls run on Mistral)", file=sys.stderr)
        return 2
    client = Mistral(api_key=mistral_api_key)

    if args.artist:
        if args.artist not in _ARTISTS_BY_NAME:
            print(f"ERROR: unknown artist {args.artist!r}, expected one of "
                  f"{sorted(_ARTISTS_BY_NAME)}", file=sys.stderr)
            return 2
        tier = args.tier or artist_tier_map().get(args.artist, "mid")
        return asyncio.run(run_artist_test(client, args.artist, tier, args.count))

    return asyncio.run(run_batch(client, args.count, args.tier, args.resume, args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
