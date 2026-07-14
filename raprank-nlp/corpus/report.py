"""
Generate a corpus + scoring summary for the RapRank NLP calibration corpus.

    python -m corpus.report            # prints markdown table to stdout

Reads the scraped corpus under corpus/data/ and runs every local scorer over
it, printing per-artist means and global distribution stats. Used to sanity
check calibration after changing the scorers.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.tests.conftest import corpus  # noqa: E402
from services import (  # noqa: E402
    alliteration_service as al, rhyme_service as rh, syllable_service as sy,
    vocabulary_service as vo, wordplay_service as wp,
)
from config import scoring_config  # noqa: E402

METRICS = {
    "syllable": lambda l: sy.calculate(l)[0],
    "rhyme": lambda l: rh.calculate(l)[0],
    "allit": lambda l: al.calculate(l)[0],
    "vocab": lambda l: vo.calculate(l)[0],
    "wordplay": lambda l: wp.calculate(l)[0],
}


def main() -> int:
    tracks = corpus()
    if not tracks:
        print("No corpus found. Run: python -m corpus.scrape_corpus")
        return 1

    by_artist: dict[str, list] = {}
    for t in tracks:
        by_artist.setdefault(t["artist"], []).append(t)

    scores = {m: [] for m in METRICS}
    scores["total"] = []
    
    per_artist = {a: {m: [] for m in METRICS} for a in by_artist}
    for a in per_artist:
        per_artist[a]["total"] = []

    for t in tracks:
        metric_vals = {}
        for m, fn in METRICS.items():
            v = fn(t["lyrics"])
            scores[m].append(v)
            per_artist[t["artist"]][m].append(v)
            metric_vals[m] = v
            
        # Get syllable weight (complex word ratio)
        _, _, syllable_weight, _ = sy.calculate(t["lyrics"])
        
        # Calculate overall Total Score using config weights
        w = scoring_config.MAIN_WEIGHTS["TEXT_ONLY"]
        total = (
            metric_vals["rhyme"] * w["rhyme"] +
            metric_vals["syllable"] * w["syllable"] +
            metric_vals["allit"] * w["alliteration"] +
            metric_vals["vocab"] * w["vocabulary"] +
            metric_vals["wordplay"] * w["wordplay"] +
            syllable_weight * w["syllable_weight"]
        )
        scores["total"].append(total)
        per_artist[t["artist"]]["total"].append(total)

    print(f"# Corpus: {len(tracks)} tracks across {len(by_artist)} artists\n")
    print("## Per-artist mean scores\n")
    cols = list(METRICS.keys()) + ["total"]
    hdr = "| Artist | n | " + " | ".join(cols) + " |"
    print(hdr)
    print("|" + "---|" * (len(cols) + 2))
    for a in sorted(by_artist, key=lambda x: -st.mean(per_artist[x]["total"])):
        row = [a, str(len(by_artist[a]))]
        row += [f"{st.mean(per_artist[a][m]):.1f}" for m in cols]
        print("| " + " | ".join(row) + " |")

    print("\n## Global distribution\n")
    print("| Metric | mean | stdev | %==0 | %>=99.5 |")
    print("|---|---|---|---|---|")
    for m, vals in scores.items():
        z = sum(1 for v in vals if v <= 0.01) / len(vals) * 100
        hi = sum(1 for v in vals if v >= 99.5) / len(vals) * 100
        print(f"| {m} | {st.mean(vals):.1f} | {st.pstdev(vals):.1f} | {z:.0f}% | {hi:.0f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
