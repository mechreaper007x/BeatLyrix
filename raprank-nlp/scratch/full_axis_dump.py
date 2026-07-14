import sys
import io
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.tests.conftest import load_corpus
from corpus.analysis.signature import signature, AXES
from services import rf_quality_service as rf

corpus = load_corpus()
bundle = rf.load()

rows = []
for t in corpus:
    lyrics = t["lyrics"]
    path = t.get("_path", "").replace("\\", "/")
    parts = path.split("/")
    artist = parts[-2] if len(parts) >= 2 else "unknown"
    title = parts[-1].replace(".json", "")

    sig = signature(lyrics)
    pred = rf.predict_tier(bundle, lyrics)

    row = {"artist": artist, "title": title}
    row.update({k: round(v, 2) for k, v in sig.items()})
    row["rf_tier"] = pred["tier"]
    row["rf_confidence"] = pred["confidence"]
    rows.append(row)

out_path = Path(__file__).parent / "full_axis_dump.json"
out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

csv_path = Path(__file__).parent / "full_axis_dump.csv"
cols = ["artist", "title"] + list(AXES) + ["rf_tier", "rf_confidence"]
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    f.write(",".join(cols) + "\n")
    for row in rows:
        f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")

print(f"Wrote {len(rows)} rows to:")
print(f"  {out_path}")
print(f"  {csv_path}")
