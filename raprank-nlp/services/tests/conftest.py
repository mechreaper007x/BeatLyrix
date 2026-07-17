"""
Shared pytest fixtures for the RapRank NLP suite.

Adds the project root to sys.path so `services` / `config` / `corpus` import
cleanly, and exposes the scraped lyrics corpus (if present) to the
corpus-driven robustness and calibration tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

CORPUS_DIR = ROOT / "corpus" / "data"
CONSENTED_DIR = ROOT / "corpus" / "consented_seeds"


def load_corpus() -> list[dict]:
    """Load every scraped track JSON. Empty list if the corpus hasn't been scraped."""
    tracks: list[dict] = []
    if not CORPUS_DIR.exists():
        return tracks
    for artist_dir in sorted(CORPUS_DIR.iterdir()):
        if not artist_dir.is_dir():
            continue
        for f in sorted(artist_dir.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("lyrics") and rec.get("line_count", 0) >= 8:
                rec["_path"] = str(f.relative_to(ROOT))
                tracks.append(rec)
    return tracks


def load_consented_seeds() -> list[dict]:
    """Load consented seed lyrics, normalized to load_corpus()'s record shape
    (lyrics/artist/tier/_path) so both pools can be concatenated directly.
    Empty list if the directory hasn't been created."""
    seeds: list[dict] = []
    if not CONSENTED_DIR.exists():
        return seeds
    for f in sorted(CONSENTED_DIR.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("consented") is not True or not rec.get("lyrics"):
            continue
        seeds.append({
            "lyrics": rec["lyrics"],
            "artist": rec.get("relationship", "consented"),
            "tier": rec.get("tier"),
            "_path": str(f.relative_to(ROOT)),
        })
    return seeds


_CORPUS_CACHE: list[dict] | None = None


def corpus() -> list[dict]:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        _CORPUS_CACHE = load_corpus()
    return _CORPUS_CACHE


@pytest.fixture(scope="session")
def corpus_tracks() -> list[dict]:
    return corpus()


def pytest_generate_tests(metafunc):
    """Parametrize any test that requests `track` over the whole corpus."""
    if "track" in metafunc.fixturenames:
        tracks = corpus()
        ids = [f"{t['artist']}:{t['title']}"[:60] for t in tracks]
        metafunc.parametrize("track", tracks, ids=ids or ["<no-corpus>"])
