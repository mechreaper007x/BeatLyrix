"""
Real Indian rap corpus -> quality-tier map, derived from corpus/artists.py's
existing `expected_profile` priors (multisyllabic/wordplay/commercial, each
0-1, already hand-set per artist for test_calibration.py's relative-ordering
assertions). This is the SAME artist-level-label limitation as the English
real-corpus module (corpus/real_corpus/real_artists.py) -- no per-song
ground truth exists for real lyrics -- but it is domain-matched: this
product scores Hindi/Hinglish Indian rap, and this corpus (corpus/data/,
302 songs from corpus/scrape_corpus.py) is exactly that, unlike the English
corpus which is a different language/rhyme-system/culture entirely.

composite = mean(multisyllabic, wordplay) -- the two priors most aligned
with "quality tier" as this product defines it (vs. flow/production, which
expected_profile doesn't capture).

    elite:      composite >= 0.80 and commercial <= 0.30
    commercial: commercial >= 0.50
    mid:        everything else
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpus.artists import Artist, unique_artists  # noqa: E402


def tier_from_profile(profile: dict) -> str | None:
    if not profile:
        return None
    composite = (profile.get("multisyllabic", 0.0) + profile.get("wordplay", 0.0)) / 2
    commercial = profile.get("commercial", 0.0)
    if commercial >= 0.50:
        return "commercial"
    if composite >= 0.80 and commercial <= 0.30:
        return "elite"
    return "mid"


def artist_tier_map() -> dict[str, str]:
    """artist display name -> tier, for every artist with an expected_profile."""
    out: dict[str, str] = {}
    for a in unique_artists():
        tier = tier_from_profile(a.expected_profile)
        if tier:
            out[a.name] = tier
    return out
