"""Eval DPST v17 on the REAL corpus (corpus/real_corpus/data): stratified sample per artist."""
import glob
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services import dpst_quality_service as dpst

ROOT = Path(__file__).resolve().parent / "corpus" / "real_corpus" / "data"
TIERS = ("elite", "mid", "commercial")
PER_ARTIST = 15
random.seed(42)

by_artist = {}
for adir in sorted(ROOT.iterdir()):
    if not adir.is_dir():
        continue
    fs = sorted(glob.glob(str(adir / "*.json")))
    if fs:
        by_artist[adir.name] = random.sample(fs, min(PER_ARTIST, len(fs)))

files = [f for fs in by_artist.values() for f in fs]
print(f"Sampled {len(files)} tracks from {len(by_artist)} artists ({PER_ARTIST}/artist)\n", flush=True)

bundle = dpst.load()

results = []  # (true, pred, artist, conf)
t_start = time.time()
for i, f in enumerate(files):
    try:
        d = json.load(open(f, encoding="utf-8"))
        if d.get("tier") not in TIERS or not d.get("lyrics", "").strip():
            continue
        out = dpst.predict(bundle, d["lyrics"])
    except Exception as e:
        print(f"ERROR {Path(f).name}: {e}", flush=True)
        continue
    results.append((d["tier"], out["tier"], d["artist"], out["confidence"]))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(files)} ({time.time()-t_start:.0f}s)", flush=True)

print(f"\nScored {len(results)} real tracks in {time.time()-t_start:.0f}s\n")

n = len(results)
acc = sum(t == p for t, p, *_ in results) / n
print(f"== REAL CORPUS (english): {n} tracks, accuracy {acc:.1%} ==")
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

# per-artist breakdown
print("\nPer-artist (true tier | pred majority | acc):")
by_a = defaultdict(list)
for t, p, a, _ in results:
    by_a[a].append((t, p))
for a in sorted(by_a, key=lambda a: by_a[a][0][0]):
    rows = by_a[a]
    maj = Counter(p for _, p in rows).most_common(1)[0][0]
    a_acc = sum(t == p for t, p in rows) / len(rows)
    print(f"  {a:<28} {rows[0][0]:<11} -> {maj:<11} {a_acc:.0%}")
