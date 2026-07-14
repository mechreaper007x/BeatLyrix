"""
Full-corpus scoring table: every song scored on every LOCAL axis (signature +
prosody + LLPC total) joined with the cached SEMANTIC raw metrics -- no network
or LLM calls. Writes a complete CSV and prints a per-artist rollup.

    python -m corpus.score_all            # CSV + rollup
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.tests.conftest import load_corpus
from corpus.analysis.signature import signature
from services import prosody_service
from services.lyrical_compiler import compile_lyrics

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "corpus" / "_full_scores.csv"
SEM_FILES = [
    ROOT.parent / "raprank-semantic" / "calibration_raw.jsonl",           # newer (+callback)
    ROOT.parent / "raprank-semantic" / "calibration_raw_precallback.jsonl.bak",  # original 302
]

TECH_COLS = [
    "rhyme", "internal", "chain", "multi_dens", "compound_dens", "holorime_dens",
    "syl_density", "syl_weight", "alliteration", "assonance", "consonance",
    "onomatopoeia", "vocab", "wordplay", "simile", "metaphor", "pun", "entendre",
    "english", "codeswitch", "repetition", "cadence_var",
]
SEM_COLS = ["coherence_cosine", "theme_cosine", "pairwise_spread", "mean_surprisal_nats", "callback_cosine"]


def load_semantic() -> dict[str, dict]:
    """id -> metrics, preferring the newer file (with callback) but backfilling."""
    out: dict[str, dict] = {}
    for f in reversed(SEM_FILES):  # load older first, newer overwrites
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "_id" in r:
                out.setdefault(r["_id"], {}).update(r)
    return out


def sid(path_str: str) -> str:
    p = Path(path_str)
    return f"{p.parent.name}/{p.stem}"


def main() -> int:
    tracks = load_corpus()
    sem = load_semantic()
    rows = []
    for i, t in enumerate(tracks, 1):
        lyr = t["lyrics"]
        sig = signature(lyr)
        pros = prosody_service.calculate(lyr)
        comp = compile_lyrics(lyr)
        s = sem.get(sid(t["_path"]), {})
        row = {
            "artist": t["artist"],
            "title": t["title"],
            "lines": t.get("line_count", 0),
            "words": len(lyr.split()),
            "total_LQI": comp["lyrical_score"],
            "codeswitch_score": pros["codeswitch_score"],
            "repetition_score": pros["repetition_score"],
            "cadence_text_score": pros["cadence_text_score"],
        }
        for c in TECH_COLS:
            row[c] = round(float(sig.get(c, 0.0)), 2)
        for c in SEM_COLS:
            row[c] = round(float(s[c]), 4) if c in s else ""
        rows.append(row)
        if i % 50 == 0:
            print(f"  scored {i}/{len(tracks)} ...", file=sys.stderr)

    cols = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} songs x {len(cols)} cols -> {OUT_CSV}", file=sys.stderr)

    # Per-artist rollup (means of the headline axes)
    by_artist: dict[str, list] = {}
    for r in rows:
        by_artist.setdefault(r["artist"], []).append(r)
    head = ["artist", "n", "total_LQI", "rhyme", "wordplay", "vocab", "codeswitch_score",
            "coherence_cosine", "callback_cosine"]
    print("\t".join(head))
    for artist in sorted(by_artist, key=lambda a: -st.mean(x["total_LQI"] for x in by_artist[a])):
        g = by_artist[artist]
        def m(c):
            vals = [x[c] for x in g if isinstance(x[c], (int, float))]
            return round(st.mean(vals), 1) if vals else 0.0
        print("\t".join(str(x) for x in [
            artist, len(g), m("total_LQI"), m("rhyme"), m("wordplay"), m("vocab"),
            m("codeswitch_score"), round(m("coherence_cosine"), 3), round(m("callback_cosine"), 3),
        ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
