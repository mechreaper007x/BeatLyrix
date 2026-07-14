"""
Assigns elite/mid/commercial tiers to the 11 Kaggle rap-lyrics-for-nlp
artists (kaggle_rap_data/).

Two derive-from-lyrics attempts were tried first, per explicit user
direction to trust measured density over reputation (after two reputation
guesses were wrong this session: CarryMinati's real lyrics measuring weaker
than his own generated commercial persona; Hanumankind's expected_profile
under-tiering him despite being real-world high tier):
  1. Curved 0-100 axis scores (services.bayesian_scoring_service.
     _axis_scores_from_lyrics) -- rejected: those curves are fit to
     Hindi/Hinglish syllable-per-line statistics, so English lyrics land in
     the wrong part of the curve (Lauryn Hill scored bottom-of-45, below
     every "commercial" Kaggle artist, purely from that mismatch).
  2. Raw pre-curve density ratios, z-score standardized within the English
     population only -- fixed the worst Hindi-calibration artifact, but the
     composite (mean of rhyme/wordplay/vocabulary/syllable density) still
     ranked Nicki Minaj above J. Cole/Kendrick Lamar and Nas below several
     "mid"-reputation Wu-Tang-affiliated acts, on a very tight spread
     (-0.41 to +0.31) where small noise flips the tier. Conclusion: these
     four density axes just aren't a reliable proxy for perceived skill
     tier in English hip-hop the way they are for the Hindi/Hinglish corpus
     this pipeline was built around.

Per explicit user decision, KAGGLE_ARTIST_TIERS below is now the
critical-consensus tier (same style/sourcing as real_artists.py's
REAL_ARTIST_TIERS), NOT the measured composite. The measured composite is
still computed and printed as a documented QA signal -- large disagreements
are flagged for a human to sanity-check, not used to override the label.

Lives outside raprank-nlp/ and reads/writes only local, gitignored data
(corpus/real_corpus/data/ is itself gitignored; kaggle_rap_data/ is
gitignored at the repo root) -- consistent with local_real_model/'s existing
scraped-lyrics-never-committed pattern.

Usage (run with raprank-nlp's venv active):
    python local_real_model/derive_english_tiers.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

NLP_ROOT = Path(__file__).resolve().parent.parent / "raprank-nlp"
sys.path.insert(0, str(NLP_ROOT))

from services import (  # noqa: E402
    rhyme_service as rh, syllable_service as sy,
    vocabulary_service as vo, wordplay_service as wp,
)

REAL_CORPUS_DATA_DIR = NLP_ROOT / "corpus" / "real_corpus" / "data"
KAGGLE_CSV_PATH = (
    Path(__file__).resolve().parent.parent / "kaggle_rap_data" / "extracted" / "lyrics_raw.csv"
)
OUT_PATH = Path(__file__).resolve().parent / "kaggle_artist_tiers.py"

MAX_TRACKS_PER_ARTIST = 150  # mirrors REAL_CORPUS_MAX_PER_ARTIST
RAW_METRICS = ("syllable_avg", "rhyme_end_ratio", "wordplay_density", "vocab_msttr")

# Critical-consensus tiers for the 11 Kaggle artists (XXL/Rolling Stone/
# Complex "greatest lyricists" style rankings, same sourcing convention as
# real_artists.py's REAL_ARTIST_TIERS) -- the source of truth for
# KAGGLE_ARTIST_TIERS, per explicit user decision after measured density
# proved unreliable for English (see module docstring).
KAGGLE_REPUTATION_TIERS: dict[str, str] = {
    "Nas": "elite",
    "Rapsody": "elite",
    "Kendrick Lamar": "elite",
    "Eminem": "elite",
    "2Pac": "elite",
    "J. Cole": "elite",
    "Dave": "mid",
    "Skepta": "mid",
    "Drake": "commercial",
    "Nicki Minaj": "commercial",
    "Future": "commercial",
}

# Genius-scraping artifacts that show up as their own lines in the Kaggle
# CSV's artist_verses column and would otherwise pollute density scoring.
_JUNK_LINE = re.compile(
    r"^\s*(you might also like|embed|get tickets|see .* live|"
    r"\d+embed)\s*\$?\s*$",
    re.IGNORECASE,
)


def clean_kaggle_lyrics(raw: str) -> str:
    lines = [ln for ln in raw.splitlines() if not _JUNK_LINE.match(ln)]
    return "\n".join(lines)


def load_real_corpus_tracks() -> dict[str, list[str]]:
    """artist display name -> list of lyric strings, capped per artist."""
    by_artist: dict[str, list[str]] = {}
    if not REAL_CORPUS_DATA_DIR.exists():
        return by_artist
    for artist_dir in sorted(REAL_CORPUS_DATA_DIR.iterdir()):
        if not artist_dir.is_dir():
            continue
        lyrics_list: list[str] = []
        for path in sorted(artist_dir.glob("*.json")):
            if len(lyrics_list) >= MAX_TRACKS_PER_ARTIST:
                break
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("lyrics") and rec.get("artist"):
                lyrics_list.append(rec["lyrics"])
                artist_name = rec["artist"]
        if lyrics_list:
            by_artist.setdefault(artist_name, []).extend(lyrics_list)
    return by_artist


def load_kaggle_tracks() -> dict[str, list[str]]:
    import pandas as pd

    if not KAGGLE_CSV_PATH.exists():
        raise FileNotFoundError(f"Kaggle CSV not found at {KAGGLE_CSV_PATH}")
    df = pd.read_csv(KAGGLE_CSV_PATH)
    by_artist: dict[str, list[str]] = {}
    for artist, group in df.groupby("artist"):
        cleaned = [clean_kaggle_lyrics(str(v)) for v in group["artist_verses"].tolist()]
        by_artist[str(artist)] = [c for c in cleaned if len(c) >= 50]
    return by_artist


def raw_metrics_for_lyrics(lyrics: str) -> dict[str, float]:
    """RAW pre-curve density metrics, not the 0-100 curved scores.

    The curved scores (services.bayesian_scoring_service._axis_scores_from_lyrics)
    run through scoring_config piecewise curves fit to Hindi/Hinglish
    syllable-per-line statistics -- comparing English lyrics on that curve
    conflates two different languages' distributions (confirmed: Lauryn
    Hill's real, lower syllables-per-line delivery scored near the bottom of
    all 45 artists on the curved scale, below every "commercial"-labeled
    Kaggle artist, purely because the curve's breakpoints assume Hindi/
    Hinglish density norms). Using raw ratios and standardizing within the
    English population avoids importing that mismatch.
    """
    _, avg_syl_per_line, _, _ = sy.calculate(lyrics)
    *_, rhyme_debug = rh.calculate(lyrics, debug=True)
    _, wp_meta = wp.calculate(lyrics)
    _, msttr = vo.calculate(lyrics)
    return {
        "syllable_avg": avg_syl_per_line,
        "rhyme_end_ratio": rhyme_debug["end_rhyme_ratio"],
        "wordplay_density": wp_meta["total_density"],
        "vocab_msttr": msttr,
    }


def main() -> int:
    print("Scoring corpus/real_corpus/data/ (existing 30 reputation-tiered artists)...")
    real_tracks = load_real_corpus_tracks()
    print(f"  {len(real_tracks)} artists, {sum(len(v) for v in real_tracks.values())} tracks")

    print("Scoring kaggle_rap_data/ (11 untiered artists)...")
    kaggle_tracks = load_kaggle_tracks()
    print(f"  {len(kaggle_tracks)} artists, {sum(len(v) for v in kaggle_tracks.values())} tracks")

    all_artists = {**real_tracks, **kaggle_tracks}
    print("\nScoring raw pre-curve metrics for every track...")
    track_metrics: dict[str, list[dict[str, float]]] = {
        artist: [raw_metrics_for_lyrics(l) for l in lyrics_list]
        for artist, lyrics_list in all_artists.items()
    }

    # Standardize each raw metric (z-score) across the FULL population of
    # tracks so metrics on different scales (syllables/line ~5-15,
    # msttr/ratios ~0-1) contribute comparably, without going through the
    # Hindi-calibrated 0-100 curves.
    pooled = {m: [tm[m] for tracks in track_metrics.values() for tm in tracks] for m in RAW_METRICS}
    mean = {m: statistics.mean(pooled[m]) for m in RAW_METRICS}
    stdev = {m: statistics.pstdev(pooled[m]) or 1.0 for m in RAW_METRICS}

    def zscore_composite(tm: dict[str, float]) -> float:
        return statistics.mean((tm[m] - mean[m]) / stdev[m] for m in RAW_METRICS)

    artist_composite: dict[str, float] = {}
    for artist, tracks in track_metrics.items():
        per_track = [zscore_composite(tm) for tm in tracks]
        artist_composite[artist] = statistics.median(per_track)
        print(f"  {artist:24} n={len(per_track):4}  composite={artist_composite[artist]:6.2f}")

    values = sorted(artist_composite.values())
    n = len(values)
    p33 = values[int(n * 0.33)]
    p66 = values[int(n * 0.66)]
    print(f"\nTercile cutoffs across all {n} artists: p33={p33:.2f}  p66={p66:.2f}")

    def tier_for_composite(c: float) -> str:
        if c >= p66:
            return "elite"
        if c >= p33:
            return "mid"
        return "commercial"

    print("\nFinal tiers for the 11 Kaggle artists (critical-consensus, per user decision):")
    kaggle_tiers: dict[str, str] = {}
    agree = disagree = 0
    for artist in sorted(kaggle_tracks):
        measured = tier_for_composite(artist_composite[artist])
        final = KAGGLE_REPUTATION_TIERS[artist]
        kaggle_tiers[artist] = final
        flag = ""
        if measured == final:
            agree += 1
        else:
            disagree += 1
            flag = f"  (QA flag: measured density suggests '{measured}' instead)"
        print(f"  {artist:16} composite={artist_composite[artist]:6.2f}  -> {final}{flag}")

    print(f"\nMeasured-vs-reputation agreement: {agree}/{agree + disagree}"
          " (disagreements are expected -- density axes don't reliably track"
          " English skill tier, see module docstring; not auto-corrected).")

    OUT_PATH.write_text(
        '"""Auto-generated by derive_english_tiers.py -- KAGGLE_ARTIST_TIERS is the\n'
        "critical-consensus tier (same sourcing convention as real_artists.py's\n"
        "REAL_ARTIST_TIERS), not the measured density composite -- see that script's\n"
        'module docstring for why the measured-density approach was rejected.\n'
        'Regenerate by re-running that script; do not hand-edit.\n"""\n\n'
        f"KAGGLE_ARTIST_TIERS: dict[str, str] = {json.dumps(kaggle_tiers, indent=4)}\n",
        encoding="utf-8",
    )
    print(f"\nSaved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
