"""Per-track rap-element table for the real DHH corpus: v19 tier + 10 explicit axes.
Writes full CSV + prints a condensed per-artist table (top-2 longest tracks each)."""
import csv
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus.real_corpus.indian_tiers import artist_tier_map
from services import dpst_quality_service as dpst
from services.bayesian_scoring_service import _axis_scores_from_lyrics

ROOT = Path(__file__).resolve().parent / "corpus" / "data"
AXES = ("rhyme", "syllable", "alliteration", "vocabulary", "wordplay",
        "assonance", "consonance", "onomatopoeia", "compound_dens", "holorime_dens")

tier_map = artist_tier_map()
files = sorted(glob.glob(str(ROOT / "*" / "*.json")))
files = [f for f in files if not Path(f).name.startswith("_")]

bundle = dpst.load()

rows = []
t0 = time.time()
for i, f in enumerate(files):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    artist = d.get("artist", "")
    lyrics = d.get("lyrics", "")
    if not lyrics.strip() or artist not in tier_map:
        continue
    try:
        sig = _axis_scores_from_lyrics(lyrics)
        out = dpst.predict(bundle, lyrics)
    except Exception as e:
        print(f"ERROR {Path(f).name}: {e}", flush=True)
        continue
    rows.append({
        "artist": artist,
        "title": d.get("title", Path(f).stem),
        "true_tier": tier_map[artist],
        "pred_tier": out["tier"],
        "conf": out["confidence"],
        "chars": len(lyrics),
        **{ax: round(float(sig.get(ax, 0.0)), 1) for ax in AXES},
    })
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

print(f"\n{len(rows)} tracks scored in {time.time()-t0:.0f}s")

# Full CSV
csv_path = Path(__file__).resolve().parent / "dhh_rap_elements.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"Full table -> {csv_path}\n")

# Condensed: top-2 longest tracks per artist
by_artist = defaultdict(list)
for r in rows:
    by_artist[r["artist"]].append(r)

order = {"elite": 0, "mid": 1, "commercial": 2}
hdr = (f"{'artist':<18} {'song':<26} {'true':<5} {'pred':<5} {'conf':>5} "
       f"{'rhym':>5} {'syll':>5} {'alli':>5} {'vocb':>5} {'wrdp':>5} "
       f"{'asso':>5} {'cons':>5} {'onom':>5} {'cmpd':>5} {'holo':>5}")
print(hdr)
print("-" * len(hdr))
prev_tier = None
for artist in sorted(by_artist, key=lambda a: (order[by_artist[a][0]["true_tier"]], a)):
    picks = sorted(by_artist[artist], key=lambda r: -r["chars"])[:2]
    if picks[0]["true_tier"] != prev_tier:
        prev_tier = picks[0]["true_tier"]
        print(f"== {prev_tier.upper()} ==")
    for r in picks:
        print(f"{r['artist'][:17]:<18} {r['title'][:25]:<26} "
              f"{r['true_tier'][:4]:<5} {r['pred_tier'][:4]:<5} {r['conf']:>5.2f} "
              + "".join(f"{r[ax]:>5.0f} " for ax in AXES))
