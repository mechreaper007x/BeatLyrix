"""
Collaborative Hinglish rap lyric generator -- combines Mistral (for drafting) 
and Gemma-4-31b-it (for deep technical refinement and polishing) to produce 
premium, persona-based synthetic lyrics.

Usage:
    python -m corpus.synthetic.generate_collaborative --count 6 --concurrency 2
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
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpus.artists import unique_artists
from corpus.real_corpus.indian_tiers import artist_tier_map
from corpus.synthetic.generate import (
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
from corpus.synthetic.generate_persona import _persona_traits, load_artist_reference_pool
from corpus.synthetic.rap_craft import craft_block
from corpus.synthetic.tier_profiles import TIER_NAMES, build_targets
from services.language_utils import content_lines

load_dotenv()

_ARTISTS_BY_NAME = {a.name: a for a in unique_artists()}

MISTRAL_DRAFT_TEMPLATE = """You are ghostwriting a rough draft of a rap verse in the style of {artist}, a Hinglish rapper {traits}.

Write in {language_label}. Genuinely code-switch -- flip between Devanagari script for Hindi and Latin script for English words within and across lines.
Topic: {topic}
Anchor detail: {anchor}

Write about 16-24 lines. Just write the rough bars representing their flow and style. Output ONLY the lines of the verse. Do not include titles, intro tags, or explanation."""

GEMINI_REFINEMENT_TEMPLATE = """You are an elite rap doctor and lyric supervisor. Your job is to take a rough draft of a Hinglish rap verse written in the style of {artist} ({traits}) and refine it so that it hits these targets:

{craft_block}

Rough Draft:
---
{draft_lyrics}
---

Refinement Instructions:
{complexity_instructions}
2. Keep the same topic ("{topic}") and core ideas, but rewrite the lines to make them punchier and more artist-appropriate.
3. Strict Language rules:
   - Use {language_label}. All Hindi words/phrases MUST be written in actual Devanagari script (e.g. "मैं हूँ"), never romanized (like "main hoon"). English words stay in Latin.
   - At least 1 out of every 4 lines must contain BOTH Devanagari and Latin words in the SAME line.
4. Output ONLY the final polished verse lines (16-24 lines), one bar per line. Do NOT output any titles, markdown tags, self-analysis, or explanation."""


def complexity_guidelines(tier: str) -> str:
    if tier == "elite":
        return """1. Elevate the technical complexity of the rough draft:
   - Enhance internal rhymes, multisyllabic rhymes, rhyme chains, and clever wordplay.
   - Refine the rhythm, meter, and alliteration."""
    elif tier == "mid":
        return """1. Keep technical complexity moderate:
   - Use internal rhymes in roughly half the lines (the other half should have none).
   - Include 2-3 clear multisyllabic rhymes, but do not stack them or chain them.
   - Keep wordplay/metaphors simple and light, not overly layered."""
    else:  # commercial
        return """1. Keep technical complexity extremely low:
   - Use simple single-syllable end rhymes.
   - Almost NO internal rhymes, and ZERO multisyllabic rhymes.
   - Heavy repetition: repeat a catchy hook phrase or refrain 3+ times.
   - Keep lines short and easy to chant."""


async def generate_collaborative_one(
    client, artist: str, tier: str, reference_pool: dict, topic_idx: int = 0
) -> dict | None:
    profile = _ARTISTS_BY_NAME[artist].expected_profile if artist in _ARTISTS_BY_NAME else {}
    traits = _persona_traits(profile)
    topic = _TOPICS[topic_idx % len(_TOPICS)]
    anchor = _ANCHORS[(topic_idx * 7 + 3) % len(_ANCHORS)]
    
    candidates = reference_pool.get(artist) or [""]
    reference_lyrics = random.choice(candidates)
    
    for attempt in range(1, MAX_ATTEMPTS_MIXED + 1):
        # 1. Draft Phase: Mistral complete
        draft_prompt = MISTRAL_DRAFT_TEMPLATE.format(
            artist=artist,
            traits=traits,
            language_label=_LANG_LABEL,
            topic=topic,
            anchor=anchor
        )
        
        try:
            print(f"    [*] Attempt {attempt}: Drafting with Mistral...")
            response = await asyncio.wait_for(
                client.chat.complete_async(
                    model="open-mistral-nemo",
                    messages=[{"role": "user", "content": draft_prompt}],
                    temperature=0.9,
                ),
                timeout=45.0
            )
            draft_lyrics = response.choices[0].message.content.strip()
        except Exception as exc:
            exc_str = str(exc)
            print(f"    ! Mistral draft failed: {type(exc).__name__}: {exc}")
            # Exponential backoff on rate limit
            if "429" in exc_str or "capacity" in exc_str.lower() or "rate" in exc_str.lower():
                wait = min(30.0 * (2 ** (attempt - 1)), 120.0)
                print(f"    ~ rate limited, waiting {wait:.0f}s before retry...")
                await asyncio.sleep(wait)
            elif "TimeoutError" in type(exc).__name__ or "timeout" in exc_str.lower():
                print(f"    ~ timeout, waiting 10s before retry...")
                await asyncio.sleep(10.0)
            else:
                await asyncio.sleep(3.0)
            continue
            
        # 2. Refinement Phase: Gemma-4 polish
        refinement_prompt = GEMINI_REFINEMENT_TEMPLATE.format(
            artist=artist,
            traits=traits,
            craft_block=craft_block(tier),
            complexity_instructions=complexity_guidelines(tier),
            draft_lyrics=draft_lyrics,
            topic=topic,
            language_label=_LANG_LABEL
        )
        
        try:
            print(f"    [*] Attempt {attempt}: Refining with Gemma 4...")
            lyrics = await _gemini_generate(refinement_prompt, temperature=0.8)
        except Exception as exc:
            print(f"    ! Gemma 4 refinement failed: {type(exc).__name__}: {exc}")
            await asyncio.sleep(5.0)
            continue
        
        # Brief cooldown after each successful API round-trip to avoid rate limits
        await asyncio.sleep(2.0)
            
        # Process and Validate
        lyrics = re.sub(r"^```[a-z]*\n?|```$", "", lyrics, flags=re.MULTILINE).strip()
        lyrics = re.sub(r"\*\*(.+?)\*\*", r"\1", lyrics)
        lyrics = re.sub(r'^"|"$', "", lyrics.strip(), flags=re.MULTILINE).strip()
        lyrics = _strip_meta_annotation_lines(lyrics)
        lyrics = _strip_leaked_metrics_footer(lyrics)
        
        if len(content_lines(lyrics)) < 6:
            continue
            
        deva_ratio = _devanagari_ratio(lyrics)
        if deva_ratio < _MIN_DEVANAGARI_RATIO:
            print(f"    ~ rejected deva_ratio={deva_ratio:.2f} (under {_MIN_DEVANAGARI_RATIO})")
            continue
            
        # Mix/Code-switching validation
        mix_fraction = _intraline_mix_fraction(lyrics)
        if mix_fraction < 0.10:
            print(f"    [*] Attempting intraline-mix repair on mix_fraction={mix_fraction:.2f}...")
            lyrics = await _repair_intraline_mix(client, lyrics)
            mix_fraction = _intraline_mix_fraction(lyrics)
            
        if mix_fraction < 0.10:
            print(f"    ~ rejected mix_fraction={mix_fraction:.2f} (under 0.10)")
            continue
            
        # Uniqueness — commercial hooks are naturally repetitive, so lower threshold
        uniq_threshold = 0.50 if tier == "commercial" else 0.70
        uniq = _line_uniqueness_ratio(lyrics)
        if uniq < uniq_threshold:
            print(f"    ~ rejected uniqueness={uniq:.2f} (under {uniq_threshold})")
            continue
            
        # Content Gates (poetry, openers, etc.)
        lyrics, gate_err = await run_content_gates(client, lyrics, attempt, MAX_ATTEMPTS_MIXED)
        if gate_err:
            print(f"    ~ rejected by content gate: {gate_err}")
            continue
            
        # Score against target tier
        score_ok, metrics, expected = score_against_tier(tier, lyrics)
        if not score_ok:
            print(f"    ~ rejected on metrics (failed to match profile targets for tier={tier})")
            continue
            
        # Build dataset record
        record_id = f"collab_{record_id_suffix(artist)}_{topic_idx}_{attempt}"
        record = {
            "id": record_id,
            "tier": tier,
            "language": "mixed",
            "target_profile": build_targets(tier),
            "actual_scores": metrics,
            "expected_scores": expected,
            "attempts": attempt,
            "generator": "gemma-4-mistral-collab",
            "persona_artist": artist,
            "lyrics": lyrics,
        }
        return record
        
    return None


def record_id_suffix(artist: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", artist).lower()[:10]


async def run(count: int, concurrency: int, resume: bool):
    from mistralai import Mistral
    
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        print("ERROR: set MISTRAL_API_KEY", file=sys.stderr)
        return 2
    client = Mistral(api_key=mistral_key)
    
    # Load reference pool for styling
    print("[*] Loading reference pool...")
    ref_pool = load_artist_reference_pool()
    
    # Decide which artist maps to which tier
    # Drew from real_corpus.indian_tiers.artist_tier_map()
    from corpus.real_corpus.indian_tiers import artist_tier_map
    atm = artist_tier_map()
    
    # Collect matching artists for tiers
    tier_artists = {t: [] for t in TIER_NAMES}
    for artist_name, tier_name in atm.items():
        if tier_name in tier_artists:
            tier_artists[tier_name].append(artist_name)
            
    # Fallback to unique artists if maps are empty
    for t in TIER_NAMES:
        if not tier_artists[t]:
            tier_artists[t] = list(_ARTISTS_BY_NAME.keys())
            
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plan = []
    for i in range(count):
        tier = TIER_NAMES[i % len(TIER_NAMES)]
        artist = random.choice(tier_artists[tier])
        plan.append((tier, artist))
        
    if resume:
        filtered = []
        for tier, artist in plan:
            tier_dir = OUT_DIR / tier / "mixed"
            # Only count collab_* files, NOT all existing synthetic files
            existing = len(list(tier_dir.glob("collab_*.json"))) if tier_dir.exists() else 0
            if existing >= count // len(TIER_NAMES):
                continue
            filtered.append((tier, artist))
        plan = filtered
        
    sem = asyncio.Semaphore(concurrency)
    
    async def _worker(sem, idx, total, tier, artist):
        async with sem:
            print(f"[*] [{idx}/{total}] Starting collab generation for tier={tier}, artist={artist}...")
            record = await generate_collaborative_one(client, artist, tier, ref_pool, topic_idx=idx)
            if record is None:
                print(f"  [-] [{idx}/{total}] {tier}/mixed -> DROPPED")
                return None
            path = OUT_DIR / tier / "mixed" / f"{record['id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [+] [{idx}/{total}] {tier}/mixed -> accepted ({path.name})")
            return record
            
    tasks = [
        _worker(sem, i, len(plan), tier, artist)
        for i, (tier, artist) in enumerate(plan, 1)
    ]
    results = await asyncio.gather(*tasks)
    
    accepted = sum(1 for r in results if r is not None)
    print(f"\nDONE: {accepted} accepted out of {len(plan)} attempted.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Collaborative Mistral/Gemma 4 lyric generator")
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    
    asyncio.run(run(args.count, args.concurrency, args.resume))


if __name__ == "__main__":
    main()
