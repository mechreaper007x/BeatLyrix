"""
Per-song element signatures + per-artist specialty (z-score) analysis.

Read-only over the git-ignored corpus. Computes an element vector for every
song (existing scorers + a few cheap heuristic axes), then standardises each
axis across the corpus and reports, per artist, which elements they over-index
on (their specialty) and each song's most-prominent elements (its fingerprint).

    python -m corpus.analysis.signature            # full report
    python -m corpus.analysis.signature KR$NA      # focus one artist
"""
from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.tests.conftest import corpus  # noqa: E402
from services import (  # noqa: E402
    alliteration_service as al, assonance_service as aso,
    consonance_service as cons, onomatopoeia_service as ono,
    rhyme_service as rh, syllable_service as sy,
    vocabulary_service as vo, wordplay_service as wp,
)
from services.language_utils import (  # noqa: E402
    content_lines, clean_word, is_hindi_word, get_multilingual_stopwords,
)

_VOWELS = set("aeiou")


# ── cheap heuristic axes (prototypes; promoted to services/ once validated) ──
def _english_ratio(lyrics: str) -> float:
    """Fraction of content tokens that are plausibly English (ASCII, in CMU)."""
    import pronouncing
    stops = get_multilingual_stopwords()
    en = tot = 0
    for line in content_lines(lyrics):
        for raw in line.split():
            w = clean_word(raw)
            if not w or len(w) < 2 or w in stops or is_hindi_word(w):
                continue
            tot += 1
            if pronouncing.phones_for_word(w.lower()):
                en += 1
    return en / tot if tot else 0.0


# Code-switch / repetition / cadence raw heuristics now live in
# services/prosody_service.py (promoted to scored axes) -- import them so the
# offline analysis and the live endpoint share one source of truth.
from services.prosody_service import (  # noqa: E402
    codeswitch_ratio as _codeswitch,
    repetition_ratio as _repetition,
    cadence_var_raw as _cadence_var,
)


def signature(lyrics: str) -> dict:
    s = sy.calculate(lyrics)
    r = rh.calculate(lyrics)
    a = al.calculate(lyrics)
    aso_score, _ = aso.calculate(lyrics)
    cons_score, _ = cons.calculate(lyrics)
    ono_score, _ = ono.calculate(lyrics)
    v = vo.calculate(lyrics)
    w, meta = wp.calculate(lyrics)
    nlines = max(len(content_lines(lyrics)), 1)
    return {
        "syl_density": s[0],
        "syl_weight": s[2],
        "rhyme": r[0],
        "internal": r[3],
        "chain": r[4],
        "multi_dens": 100.0 * r[2] / nlines,
        "compound_dens": 100.0 * r[5] / nlines,
        "holorime_dens": 100.0 * r[6] / nlines,
        "alliteration": a[0],
        "assonance": aso_score,
        "consonance": cons_score,
        "onomatopoeia": ono_score,
        "vocab": v[0],
        "wordplay": w,
        "simile": 100.0 * meta["simile_count"] / nlines,
        "metaphor": 100.0 * meta["metaphor_count"] / nlines,
        "pun": 100.0 * meta["puns_count"] / nlines,
        "entendre": 100.0 * meta["double_entendres_count"] / nlines,
        "english": 100.0 * _english_ratio(lyrics),
        "codeswitch": 100.0 * _codeswitch(lyrics),
        "repetition": 100.0 * _repetition(lyrics),
        "cadence_var": 100.0 * _cadence_var(lyrics),
    }


AXES = ["syl_density", "syl_weight", "rhyme", "internal", "chain", "multi_dens",
        "compound_dens", "holorime_dens", "alliteration", "assonance",
        "consonance", "onomatopoeia", "vocab", "wordplay", "simile", "metaphor",
        "pun", "entendre", "english", "codeswitch", "repetition", "cadence_var"]


def main() -> int:
    focus = sys.argv[1] if len(sys.argv) > 1 else None
    tracks = corpus()
    if not tracks:
        print("No corpus. Run the scraper first.")
        return 1

    sigs = [(t, signature(t["lyrics"])) for t in tracks]

    # corpus mean/std per axis for z-scoring
    mean = {ax: st.mean(s[ax] for _, s in sigs) for ax in AXES}
    std = {ax: (st.pstdev(s[ax] for _, s in sigs) or 1.0) for ax in AXES}

    by: dict[str, list] = {}
    for t, s in sigs:
        by.setdefault(t["artist"], []).append(s)

    print(f"# Signature analysis — {len(tracks)} songs, {len(by)} artists\n")
    print("## Per-artist specialty (top z-score axes vs corpus)\n")
    for artist in sorted(by):
        rows = by[artist]
        z = {ax: (st.mean(s[ax] for s in rows) - mean[ax]) / std[ax] for ax in AXES}
        top = sorted(z.items(), key=lambda kv: -kv[1])[:4]
        if focus and artist.lower() != focus.lower():
            continue
        tag = " ".join(f"{ax}(+{zz:.2f})" for ax, zz in top if zz > 0)
        print(f"  {artist:14} n={len(rows):3}  ->  {tag or '(no positive z-axis)'}")

    if focus:
        rows = by.get(focus) or by.get(next((a for a in by if a.lower() == focus.lower()), ""), [])
        if rows:
            print(f"\n## {focus}: mean per axis\n")
            for ax in AXES:
                m = st.mean(s[ax] for s in rows)
                z = (m - mean[ax]) / std[ax]
                print(f"  {ax:13} {m:6.1f}   z={z:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
