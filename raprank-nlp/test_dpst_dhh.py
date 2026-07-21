"""Eval DPST v17 on the REAL DHH corpus (corpus/data), tiers from indian_tiers.artist_tier_map()."""
import glob
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus.real_corpus.indian_tiers import artist_tier_map
from services import dpst_quality_service as dpst

ROOT = Path(__file__).resolve().parent / "corpus" / "data"
TIERS = ("elite", "mid", "commercial")

tier_map = artist_tier_map()  # display name -> tier
files = sorted(glob.glob(str(ROOT / "*" / "*.json")))
files = [f for f in files if not Path(f).name.startswith("_")]
print(f"Found {len(files)} DHH tracks\n", flush=True)

bundle = dpst.load()

results = []  # (true, pred, artist, conf)
skipped = Counter()
t_start = time.time()
for i, f in enumerate(files):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        skipped["unreadable"] += 1
        continue
    artist = d.get("artist", "")
    tier = tier_map.get(artist)
    if tier is None:
        skipped[f"no-tier:{artist}"] += 1
        continue
    lyrics = d.get("lyrics", "")
    if not lyrics.strip():
        skipped["empty"] += 1
        continue
    try:
        out = dpst.predict(bundle, lyrics)
    except Exception as e:
        print(f"ERROR {Path(f).name}: {e}", flush=True)
        continue
    results.append((tier, out["tier"], artist, out["confidence"]))
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(files)} ({time.time()-t_start:.0f}s)", flush=True)

print(f"\nScored {len(results)} tracks in {time.time()-t_start:.0f}s; skipped: {dict(skipped)}\n")

n = len(results)
acc = sum(t == p for t, p, *_ in results) / n
print(f"== REAL DHH CORPUS: {n} tracks, accuracy {acc:.1%} ==")
cm = Counter((t, p) for t, p, *_ in results)
hdr = "true/pred"
print(f"{hdr:<14}" + "".join(f"{t:>12}" for t in TIERS) + f"{'recall':>10}")
for t in TIERS:
    row_n = sum(cm[(t, p)] for p in TIERS)
    if row_n:
        print(f"{t:<14}" + "".join(f"{cm[(t, p)]:>12}" for p in TIERS) + f"{cm[(t,t)]/row_n:>9.1%}")

pred_dist = Counter(p for _, p, *_ in results)
print("\nPrediction distribution:", dict(pred_dist))
print(f"Mean confidence: {sum(r[3] for r in results)/n:.3f}")

print("\nPer-artist (true tier | pred majority | acc | n):")
by_a = defaultdict(list)
for t, p, a, _ in results:
    by_a[a].append((t, p))
order = {"elite": 0, "mid": 1, "commercial": 2}
for a in sorted(by_a, key=lambda a: (order[by_a[a][0][0]], a)):
    rows = by_a[a]
    maj = Counter(p for _, p in rows).most_common(1)[0][0]
    a_acc = sum(t == p for t, p in rows) / len(rows)
    print(f"  {a:<22} {rows[0][0]:<11} -> {maj:<11} {a_acc:>4.0%}  (n={len(rows)})")
