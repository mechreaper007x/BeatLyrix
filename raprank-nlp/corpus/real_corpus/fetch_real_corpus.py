"""
Downloads + parses the real English rap corpus (Cropinky/rap_lyrics_english
on Hugging Face) for use as a second training source for
services/bayesian_scoring_service.py.

Local-only, git-ignored (corpus/ is already excluded in .gitignore) --
lyric text is fetched at run-time and never bundled or served to end users,
same handling as corpus/data/ (the existing scraped 302-song corpus).

Only per-artist song files are pulled (songs/<category>/<Artist>.txt); the
top-level songs/<category>.txt files are just concatenations of the same
artists and would double-count every song.

Usage:
    python -m corpus.real_corpus.fetch_real_corpus
    python -m corpus.real_corpus.fetch_real_corpus --resume
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import hf_hub_download  # noqa: E402

from corpus.real_corpus.real_artists import REAL_ARTIST_TIERS  # noqa: E402

REPO_ID = "Cropinky/rap_lyrics_english"
DATA_DIR = Path(__file__).resolve().parent / "data"

# a song boundary is a line of 10+ asterisks
_SONG_SEP = re.compile(r"^\*{10,}\s*$", re.MULTILINE)
_MIN_SONG_CHARS = 200


def parse_songs(raw: str) -> list[tuple[str, str]]:
    """Splits one artist's raw dump into (title, lyrics) pairs."""
    blocks = _SONG_SEP.split(raw)
    out = []
    for block in blocks:
        block = block.strip()
        if len(block) < _MIN_SONG_CHARS:
            continue
        lines = block.splitlines()
        title = lines[0].strip() or "untitled"
        lyrics = "\n".join(lines[1:]).strip()
        if len(lyrics) < _MIN_SONG_CHARS:
            continue
        out.append((title, lyrics))
    return out


def artist_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def fetch_artist(repo_path: str, name: str, tier: str, resume: bool) -> int:
    out_dir = DATA_DIR / artist_slug(name)
    if resume and out_dir.exists() and any(out_dir.glob("*.json")):
        print(f"  {name:24} skipped (resume)")
        return 0

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"songs/{repo_path}.txt",
    )
    raw = Path(local_path).read_text(encoding="utf-8", errors="replace")
    songs = parse_songs(raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, (title, lyrics) in enumerate(songs):
        rec = {
            "artist": name,
            "tier": tier,
            "title": title,
            "source": f"hf:{REPO_ID}/songs/{repo_path}.txt",
            "lyrics": lyrics,
            "line_count": sum(1 for l in lyrics.splitlines() if l.strip()),
            "char_count": len(lyrics),
        }
        path = out_dir / f"{i:03d}-{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:60] or 'untitled'}.json"
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        saved += 1
    print(f"  {name:24} tier={tier:11} saved={saved}")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch + parse the real English rap corpus (HF)")
    ap.add_argument("--resume", action="store_true", help="skip artists already fetched")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for repo_path, (name, tier) in REAL_ARTIST_TIERS.items():
        total += fetch_artist(repo_path, name, tier, args.resume)

    print(f"\nTOTAL: {total} songs across {len(REAL_ARTIST_TIERS)} artists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
