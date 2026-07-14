import sys
import io
import os
import re
import json
import asyncio
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from services.tests.conftest import load_corpus
from services import rhyme_service
from mistralai import Mistral

OUT_PATH = Path(__file__).parent / "mistral_rhyme_check_full_results.jsonl"
CONCURRENCY = 8


def full_breakdown(lyrics):
    lines = rhyme_service.content_lines(lyrics)
    unique_lines, seen_lines = [], set()
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
        "num_pairs": len(pairs),
        "multisyl_count": multisyl_count,
        "internal_score": internal_score,
        "chain_score": chain_score,
        "compound_count": compound_count,
        "holorime_count": holorime_count,
        "diversity_multiplier": round(div_mult, 3),
    }


async def mistral_rhyme_read(client, lyrics, sem):
    prompt = f"""You are a rap lyric analyst. Read these lyrics and identify the ACTUAL rhyme scheme
a human listener would perceive when the song is performed aloud (including slant/near rhyme,
assonance-based rhyme, and multisyllabic rhyme -- not just exact spelling matches).

Lyrics:
{lyrics}

Respond with JSON only:
{{
  "rhyme_density": "none|sparse|moderate|dense",
  "num_rhyme_pairs_found": <int>,
  "one_line_verdict": "..."
}}"""
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.complete_async(
                    model="mistral-medium-latest",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as exc:
                if attempt == 2:
                    return {"error": str(exc)}
                await asyncio.sleep(2 * (attempt + 1))


async def process_one(client, sem, song_id, lyrics, done_ids):
    if song_id in done_ids:
        return None
    breakdown = full_breakdown(lyrics)
    mistral_read = await mistral_rhyme_read(client, lyrics, sem)
    row = {"song_id": song_id, "breakdown": breakdown, "mistral_read": mistral_read}
    return row


async def main():
    corpus = load_corpus()
    songs = []
    for t in corpus:
        path = t.get("_path", "").replace("\\", "/")
        parts = path.split("/")
        artist = parts[-2] if len(parts) >= 2 else "unknown"
        title = parts[-1].replace(".json", "")
        songs.append((f"{artist}/{title}", t["lyrics"]))

    done_ids = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["song_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} songs already done")

    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    sem = asyncio.Semaphore(CONCURRENCY)

    todo = [(sid, lyr) for sid, lyr in songs if sid not in done_ids]
    print(f"Total songs: {len(songs)}, remaining: {len(todo)}")

    completed = 0
    lock = asyncio.Lock()

    async def worker(song_id, lyrics):
        nonlocal completed
        row = await process_one(client, sem, song_id, lyrics, done_ids)
        if row is not None:
            async with lock:
                with open(OUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 10 == 0:
                    print(f"  {completed}/{len(todo)} done")

    await asyncio.gather(*(worker(sid, lyr) for sid, lyr in todo))
    print(f"Done. Wrote results to {OUT_PATH}")


asyncio.run(main())
