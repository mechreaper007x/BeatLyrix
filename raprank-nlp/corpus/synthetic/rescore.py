# -*- coding: utf-8 -*-
"""
Re-score every synthetic sample's actual_scores with the current scorer
(DHH phoneme rhyme keys + recalibrated ELITE_TARGETS), so the whole corpus
is on one measurement scale before the scoring heads are retrained.

Lyrics are NOT touched -- only actual_scores/expected_scores/accepted are
recomputed. The original values are preserved under _prev_scores the first
time so the shift stays auditable. Re-runs are idempotent.

Usage:
    python -m corpus.synthetic.rescore            # re-score + report
    python -m corpus.synthetic.rescore --dry-run  # report only, no writes
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_NLP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_NLP_ROOT))

from corpus.synthetic.generate import score_against_tier  # noqa: E402
from corpus.synthetic.tier_profiles import TIER_NAMES  # noqa: E402

SYN_DIR = _NLP_ROOT / "corpus" / "synthetic_data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stats = defaultdict(lambda: {"n": 0, "flipped": 0, "delta": defaultdict(float)})
    for tier in TIER_NAMES:
        for lang_dir in (SYN_DIR / tier).iterdir() if (SYN_DIR / tier).exists() else []:
            lang = lang_dir.name  # en | mixed | hi
            for path in sorted(lang_dir.glob("*.json")):
                try:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                lyrics = rec.get("lyrics")
                if not lyrics:
                    continue
                accepted, actual, expected = score_against_tier(
                    tier, lyrics, lang="en" if lang == "en" else "mixed")
                key = f"{tier}/{lang}"
                s = stats[key]
                s["n"] += 1
                old = rec.get("actual_scores", {})
                for ax, v in actual.items():
                    if ax in old:
                        s["delta"][ax] += v - old[ax]
                if not accepted:
                    s["flipped"] += 1
                if not args.dry_run:
                    if "_prev_scores" not in rec:
                        rec["_prev_scores"] = rec.get("actual_scores")
                    rec["actual_scores"] = actual
                    rec["expected_scores"] = expected
                    rec["accepted_on_rescore"] = accepted
                    path.write_text(
                        json.dumps(rec, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"{'tier/lang':18} {'n':>4} {'now-off-tier':>12}   biggest mean deltas")
    for key in sorted(stats):
        s = stats[key]
        if not s["n"]:
            continue
        deltas = sorted(s["delta"].items(), key=lambda kv: abs(kv[1]) / s["n"], reverse=True)
        top = ", ".join(f"{ax} {v / s['n']:+.1f}" for ax, v in deltas[:3])
        print(f"{key:18} {s['n']:>4} {s['flipped']:>12}   {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
