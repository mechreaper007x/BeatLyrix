"""Full eval: DPST v17 vs ALL lyrics in corpus/synthetic_data (synthetic + scraped)."""
import glob
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services import dpst_quality_service as dpst

ROOT = Path(__file__).resolve().parent / "corpus" / "synthetic_data"
TIERS = ("elite", "mid", "commercial")

files = sorted(glob.glob(str(ROOT / "*" / "*" / "*.json")))
files = [f for f in files if not Path(f).name.startswith("_")]
print(f"Found {len(files)} lyric files\n", flush=True)

bundle = dpst.load()

results = []
t_start = time.time()
for i, f in enumerate(files):
    p = Path(f)
    true_tier = p.parent.parent.name
    lang = p.parent.name
    if true_tier not in TIERS:
        continue
    try:
        d = json.load(open(f, encoding="utf-8"))
        lyrics = d["lyrics"] if isinstance(d, dict) else str(d)
        if not lyrics.strip():
            continue
        out = dpst.predict(bundle, lyrics)
    except Exception as e:
        print(f"ERROR {p.name}: {e}", flush=True)
        continue
    is_real = p.name.startswith("scraped_")
    results.append((true_tier, out["tier"], lang, is_real, out["confidence"]))
    if (i + 1) % 50 == 0:
        el = time.time() - t_start
        print(f"  {i+1}/{len(files)} done ({el:.0f}s elapsed)", flush=True)

print(f"\nScored {len(results)} tracks in {time.time()-t_start:.0f}s\n")

def report(rows, label):
    if not rows:
        return
    n = len(rows)
    acc = sum(t == p for t, p, *_ in rows) / n
    print(f"== {label}: {n} tracks, accuracy {acc:.1%} ==")
    # confusion matrix
    cm = Counter((t, p) for t, p, *_ in rows)
    hdr = "true/pred"
    print(f"{hdr:<14}" + "".join(f"{t:>12}" for t in TIERS) + f"{'recall':>10}")
    for t in TIERS:
        row_n = sum(cm[(t, p)] for p in TIERS)
        if row_n == 0:
            continue
        rec = cm[(t, t)] / row_n
        print(f"{t:<14}" + "".join(f"{cm[(t, p)]:>12}" for p in TIERS) + f"{rec:>9.1%}")
    print()

report(results, "ALL")
report([r for r in results if not r[3]], "SYNTHETIC only")
report([r for r in results if r[3]], "REAL (scraped) only")

by_lang = defaultdict(list)
for r in results:
    by_lang[r[2]].append(r)
for lang in sorted(by_lang):
    report(by_lang[lang], f"lang={lang}")

# prediction distribution
pred_dist = Counter(p for _, p, *_ in results)
print("Prediction distribution:", dict(pred_dist))
avg_conf = sum(r[4] for r in results) / len(results)
print(f"Mean confidence: {avg_conf:.3f}")
