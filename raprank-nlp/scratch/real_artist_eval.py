import sys
import io
import collections
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.tests.conftest import load_corpus
from services import rf_quality_service as rf
from services import gmm_style_service as gm

corpus = load_corpus()
print(f"Real corpus size: {len(corpus)} songs")

bundle = rf.load()
gmm_bundle = gm.load()
gmm_labels = gmm_bundle.get("_labels") or gm.label_clusters(gmm_bundle)
gmm_song_cluster = {
    sid: gmm_labels[str(c)]["label"]
    for sid, c in zip(gmm_bundle["kept_ids"], gmm_bundle["assignments"])
}

by_artist_tier = collections.defaultdict(collections.Counter)
by_artist_cluster = collections.defaultdict(collections.Counter)
rows = []

for t in corpus:
    lyrics = t["lyrics"]
    path = t.get("_path", "").replace("\\", "/")
    parts = path.split("/")
    artist = parts[-2] if len(parts) >= 2 else "unknown"
    title = parts[-1].replace(".json", "")
    song_id = f"{artist}/{title}"

    pred = rf.predict_tier(bundle, lyrics)
    by_artist_tier[artist][pred["tier"]] += 1

    cluster_label = gmm_song_cluster.get(song_id, "N/A (not in GMM training set)")
    if cluster_label != "N/A (not in GMM training set)":
        by_artist_cluster[artist][cluster_label] += 1

    rows.append({
        "artist": artist,
        "title": title,
        "rf_tier": pred["tier"],
        "rf_conf": pred["confidence"],
        "gmm_cluster": cluster_label,
    })

out_path = Path(__file__).parent / "real_artist_eval_results.json"
out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== RF predicted tier distribution per artist ===")
for artist, counter in sorted(by_artist_tier.items()):
    total = sum(counter.values())
    print(f"{artist:20} n={total:3}  " + "  ".join(f"{k}:{v}" for k, v in counter.most_common()))

print("\n=== GMM cluster distribution per artist ===")
for artist, counter in sorted(by_artist_cluster.items()):
    total = sum(counter.values())
    print(f"{artist:20} n={total:3}")
    for label, cnt in counter.most_common():
        print(f"    {label:45} {cnt}")

overall_tier = collections.Counter(r["rf_tier"] for r in rows)
print(f"\n=== Overall RF tier distribution across {len(rows)} real songs ===")
for tier, cnt in overall_tier.most_common():
    print(f"  {tier:12} {cnt:4}  ({cnt/len(rows):.1%})")

print(f"\nFull per-song results written to {out_path}")
