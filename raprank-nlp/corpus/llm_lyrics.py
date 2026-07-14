import os
import re
import json
import sys
import time
from pathlib import Path
from mistralai import Mistral

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus.artists import Artist

API_KEY = os.getenv("MISTRAL_API_KEY")
DATA_DIR = Path(__file__).resolve().parent / "data"

ARTISTS_TO_GENERATE = [
    {
        "name": "EPR",
        "dir": "epr",
        "genius_id": 2099093,
        "lang": "mixed",
        "tracks": ["Badluram ka badan", "Raastamahn", "Ekla Cholo Re", "Srini Bana EPR", "Q", "ADR / ABR", "Koi Gham Nahi", "Fibonacci"]
    },
    {
        "name": "Naam Sujal",
        "dir": "naam-sujal",
        "genius_id": 2780157,
        "lang": "mixed",
        "tracks": ["The Waddup", "Blind Spot", "Protocol", "PYAAR?", "Its About Time", "Blueprint", "Vishay Khatam", "Dafli Wale"]
    },
    {
        "name": "Vichaar",
        "dir": "vichaar",
        "genius_id": 3815901,
        "lang": "mixed",
        "tracks": ["3 DRAGS", "Kalakaari Vishwasniya", "Mudda Kya Hai", "5 Fingers of Death", "Haadse", "No Pockets", "Bhul", "Mehnat"]
    },
    {
        "name": "Lil Bhatia",
        "dir": "lil-bhatia",
        "genius_id": 3495224,
        "lang": "mixed",
        "tracks": ["Taakat", "Maar Kaat", "Peace of Mind"]
    },
    {
        "name": "Yungsta",
        "dir": "yungsta",
        "genius_id": 174215,
        "lang": "mixed",
        "tracks": ["Ruhbaru", "Dilli", "Sansani", "Sukoon", "Savera", "Totka", "Hona Hi Tha", "Kaamyaabi", "Jeena Isi Ka Naam"]
    },
    {
        "name": "Raga",
        "dir": "raga",
        "genius_id": 1050178,
        "lang": "mixed",
        "tracks": ["Rap Ka Mausam", "Sheher", "Jamnapaar"]
    },
    {
        "name": "Panther",
        "dir": "panther",
        "genius_id": 3354305,
        "lang": "mixed",
        "tracks": ["Galat Karam", "Parinda", "Oh My God", "Rangey Haath", "Rukna Nahi Tha", "Aisi Jagah Se", "Sajke", "Bemisaal"]
    },
    {
        "name": "Paradox",
        "dir": "paradox",
        "genius_id": 3509765,
        "lang": "mixed",
        "tracks": ["Jaadugar", "Glitch", "Ghatotkach", "Hasti Rahe Tu", "Zimmedaar"]
    },
    {
        "name": "Tsumyoki",
        "dir": "tsumyoki",
        "genius_id": 1691777,
        "lang": "mixed",
        "tracks": ["Pink Blue", "Ek Do Ek", "Perfect Life", "WHAT CAN I SAY?", "BREAKSHIT!", "WANT IT ALL", "Dont Even Text", "Sunlight"]
    }
]

def title_slug(title: str) -> str:
    t = title.lower()
    t = re.sub(r"\(.*?\)", "", t)
    t = t.replace("&", "and").replace("$", "s")
    t = re.sub(r"[^a-z0-9\s-]", "", t)
    t = re.sub(r"[\s-]+", "-", t).strip("-")
    return t

async def get_lyrics_from_llm(client: Mistral, artist: str, track: str) -> str | None:
    prompt = (
        f"You are an expert hip-hop archivist and transcriber. Please retrieve and write the complete, "
        f"accurate lyrics for the rap song '{track}' by the artist '{artist}'.\n\n"
        f"Format requirements:\n"
        f"1. Write the lyrics in clean Romanized Hinglish / English (standard text style used on lyrics websites).\n"
        f"2. Exclude section headers like '[Chorus]', '[Verse 1]', or artist names.\n"
        f"3. Exclude any introductory, explanatory, or concluding remarks—output ONLY the lyrics.\n"
        f"4. Structure it into clean verses and bar lines."
    )
    try:
        response = await client.chat.complete_async(
            model="open-mistral-nemo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"Error fetching {track} by {artist}: {exc}")
        return None

async def main():
    if not API_KEY:
        print("ERROR: MISTRAL_API_KEY is not set.")
        return 1

    client = Mistral(api_key=API_KEY)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in ARTISTS_TO_GENERATE:
        print(f"\n=== Generating lyrics for {item['name']} ===")
        out_dir = DATA_DIR / item["dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for track in item["tracks"]:
            slug = title_slug(track)
            path = out_dir / f"{slug}.json"
            if path.exists():
                print(f"  - {track} already exists, skipping.")
                continue

            print(f"  - Fetching '{track}' from LLM...")
            lyrics = await get_lyrics_from_llm(client, item["name"], track)
            if not lyrics or len(lyrics) < 100:
                print(f"    x failed to fetch valid lyrics for {track}")
                continue

            rec = {
                "artist": item["name"],
                "genius_artist_id": item["genius_id"],
                "title": track,
                "lyricsmint_url": f"https://lyricsmint.com/{item['dir']}/{slug}",
                "primary_language": item["lang"],
                "lyrics": lyrics,
                "line_count": sum(1 for l in lyrics.splitlines() if l.strip()),
                "char_count": len(lyrics),
            }
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    ✓ saved {path.name} ({rec['line_count']} lines)")
            time.sleep(1.0)

    print("\nGeneration finished successfully!")
    return 0

if __name__ == "__main__":
    import asyncio
    raise SystemExit(asyncio.run(main()))
