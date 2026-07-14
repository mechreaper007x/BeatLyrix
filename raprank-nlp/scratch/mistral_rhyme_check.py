import sys
import io
import os
import json
import asyncio
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from services.tests.conftest import load_corpus
from services import rhyme_service
from config import scoring_config
from mistralai import Mistral


def full_breakdown(lyrics):
    lines = rhyme_service.content_lines(lyrics)
    unique_lines, seen_lines = [], set()
    import re
    for line in lines:
        norm = re.sub(r"[^\wऀ-ॿ]", "", line).strip().lower()
        if norm not in seen_lines:
            seen_lines.add(norm)
            unique_lines.append(line)
    lines = unique_lines

    score, pairs, multisyl_count, internal_score, chain_score, compound_count, holorime_count = rhyme_service.calculate(lyrics)

    indexed_words = [(i, rhyme_service._last_word(lines[i])) for i in range(len(lines))]
    indexed_words = [(i, w) for i, w in indexed_words if w]
    div_mult = rhyme_service._diversity_multiplier(indexed_words)

    return {
        "final_score": score,
        "num_lines": len(lines),
        "rhyme_pairs": [f"{p.word_a}/{p.word_b} (multi={p.is_multisyllabic})" for p in pairs],
        "num_pairs": len(pairs),
        "multisyl_count": multisyl_count,
        "internal_score": internal_score,
        "chain_score": chain_score,
        "compound_count": compound_count,
        "holorime_count": holorime_count,
        "diversity_multiplier": round(div_mult, 3),
    }

TARGETS = [
    "tsumyoki/dont-even-text",
    "paradox/glitch",
    "tsumyoki/idk",
    "king/iiconic",
    "karma/airplane-mode",
]


async def mistral_rhyme_read(client, lyrics):
    prompt = f"""You are a rap lyric analyst. Read these lyrics and identify the ACTUAL rhyme scheme
a human listener would perceive when the song is performed aloud (including slant/near rhyme,
assonance-based rhyme, and multisyllabic rhyme -- not just exact spelling matches).

Lyrics:
{lyrics}

Respond with JSON only:
{{
  "rhyme_density": "none|sparse|moderate|dense",
  "rhyme_pairs_found": ["word1/word2", ...],
  "one_line_verdict": "..."
}}"""
    resp = await client.chat.complete_async(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(resp.choices[0].message.content)


async def main():
    corpus = load_corpus()
    by_id = {}
    for t in corpus:
        path = t.get("_path", "").replace("\\", "/")
        parts = path.split("/")
        artist = parts[-2] if len(parts) >= 2 else "unknown"
        title = parts[-1].replace(".json", "")
        by_id[f"{artist}/{title}"] = t["lyrics"]

    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    results = []
    for song_id in TARGETS:
        lyrics = by_id.get(song_id)
        if not lyrics:
            print(f"MISSING: {song_id}")
            continue

        breakdown = full_breakdown(lyrics)

        mistral_read = await mistral_rhyme_read(client, lyrics)

        result = {
            "song_id": song_id,
            "breakdown": breakdown,
            "mistral_read": mistral_read,
        }
        results.append(result)

        print(f"\n{'='*70}")
        print(f"{song_id}")
        print(f"{'='*70}")
        print(f"RULE-BASED FINAL SCORE: {breakdown['final_score']}")
        print(json.dumps(breakdown, indent=2, ensure_ascii=False))
        print(f"MISTRAL READ:")
        print(f"  density: {mistral_read.get('rhyme_density')}")
        print(f"  pairs: {mistral_read.get('rhyme_pairs_found')}")
        print(f"  verdict: {mistral_read.get('one_line_verdict')}")

    out_path = Path(__file__).parent / "mistral_rhyme_check_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


asyncio.run(main())
