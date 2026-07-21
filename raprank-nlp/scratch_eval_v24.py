import json, sys, glob, hashlib
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, ".")
import services.dpst_quality_service as dpst
dpst.DPST_MODEL_PATH = dpst._LOCAL_MODEL_DIR / "dhh_classifier_v24_real.pt"
dpst.DPST_META_PATH  = dpst._LOCAL_MODEL_DIR / "dpst_model_meta_v24.json"
print("weights md5:", hashlib.md5(open(dpst.DPST_MODEL_PATH,'rb').read()).hexdigest()[:12])

from corpus.real_corpus.indian_tiers import artist_tier_map
tier_map = artist_tier_map()
TIERS = ("elite","mid","commercial")
bundle = dpst.load()

res = []
by_artist = defaultdict(list)
for f in sorted(glob.glob("corpus/data/*/*.json")):
    if Path(f).name.startswith("_"): continue
    try: d = json.load(open(f, encoding="utf-8"))
    except: continue
    tier = tier_map.get(d.get("artist"))
    if not tier or not d.get("lyrics","").strip(): continue
    p = dpst.predict(bundle, d["lyrics"])
    res.append((tier, p["tier"]))
    by_artist[d["artist"]].append((tier, p["tier"], p["confidence"]))

n=len(res); acc=sum(t==p for t,p in res)/n; cm=Counter(res)
print(f"\n== v24 REAL-ONLY | REAL DHH CORPUS: {n} tracks, accuracy {acc:.1%} ==")
hdr="true/pred"
print(f"{hdr:<14}" + "".join(f"{t:>12}" for t in TIERS) + f"{'recall':>10}")
for t in TIERS:
    row=sum(cm[(t,x)] for x in TIERS)
    if row: print(f"{t:<14}" + "".join(f"{cm[(t,x)]:>12}" for x in TIERS) + f"{cm[(t,t)]/row:>9.1%}")
print("pred dist:", dict(Counter(p for _,p in res)))
print(f"mean conf: {sum(c for a in by_artist for _,_,c in by_artist[a])/n:.3f}")

order={"elite":0,"mid":1,"commercial":2}
print("\nPer-artist (true | majority pred | acc | n):")
prev=None
for a in sorted(by_artist, key=lambda a:(order[by_artist[a][0][0]], a)):
    rows=by_artist[a]; t=rows[0][0]
    if t!=prev: prev=t; print(f"-- {t.upper()} --")
    maj=Counter(p for _,p,_ in rows).most_common(1)[0][0]
    a_acc=sum(tt==p for tt,p,_ in rows)/len(rows)
    print(f"  {a:<20} -> {maj:<11} {a_acc:>4.0%}  (n={len(rows)})")
