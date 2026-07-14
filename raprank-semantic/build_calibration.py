"""
Offline calibration builder for raprank-semantic (run once, locally).

Collects the RAW semantic metrics (coherence_cosine, theme_cosine,
pairwise_spread, mean_surprisal_nats) for every song in the domain corpus by
POSTing each to the LIVE Space, buckets them by script (deva / latin), and
writes calibration.json -- per-(bucket, metric) sorted reference arrays that the
service turns into percentiles at runtime.

The corpus is used ONLY as an unlabeled feature-distribution reference (never as
quality labels, never redistributed). Lyrics are sent transiently to the Space
exactly as a normal scoring request would.

Resumable: every fetched result is appended to calibration_raw.jsonl; rerunning
skips songs already recorded, so an interrupted run just continues.

Usage:
    python build_calibration.py \
        --corpus ../raprank-nlp/corpus/data \
        --url https://mechreaper007x-raprank-semantic.hf.space
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

METRICS = ("coherence_cosine", "theme_cosine", "pairwise_spread", "mean_surprisal_nats", "callback_cosine")

HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "calibration_raw.jsonl"
OUT_PATH = HERE / "calibration.json"

# Same Devanagari block + threshold the service uses to pick a bucket.
_DEVA_START, _DEVA_END = 0x0900, 0x097F
_DEVA_RATIO_THRESHOLD = 0.20


def script_bucket(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "latin"
    deva = sum(1 for c in letters if _DEVA_START <= ord(c) <= _DEVA_END)
    return "deva" if (deva / len(letters)) >= _DEVA_RATIO_THRESHOLD else "latin"


def load_corpus(corpus_dir: Path) -> list[dict]:
    """Every scraped track JSON with enough lines. Mirrors
    raprank-nlp/tests/conftest.py::load_corpus."""
    tracks: list[dict] = []
    if not corpus_dir.exists():
        sys.exit(f"corpus dir not found: {corpus_dir}")
    for artist_dir in sorted(corpus_dir.iterdir()):
        if not artist_dir.is_dir():
            continue
        for f in sorted(artist_dir.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("lyrics") and rec.get("line_count", 0) >= 8:
                rec["_id"] = f"{artist_dir.name}/{f.stem}"
                tracks.append(rec)
    return tracks


def load_done() -> dict[str, dict]:
    """Read already-fetched results from the resume log, keyed by song id."""
    done: dict[str, dict] = {}
    if RAW_PATH.exists():
        for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[rec["_id"]] = rec
            except Exception:
                continue
    return done


def fetch_metrics(client: httpx.Client, url: str, lyrics: str) -> dict | None:
    for attempt in range(4):
        try:
            resp = client.post(f"{url}/semantic", json={"lyrics": lyrics}, timeout=180.0)
            resp.raise_for_status()
            return resp.json().get("metrics", {})
        except Exception as exc:
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt + 1}/4 after {wait}s ({exc})")
            time.sleep(wait)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE.parent / "raprank-nlp" / "corpus" / "data"))
    ap.add_argument("--url", default="https://mechreaper007x-raprank-semantic.hf.space")
    args = ap.parse_args()

    corpus = load_corpus(Path(args.corpus).resolve())
    done = load_done()
    print(f"{len(corpus)} songs in corpus; {len(done)} already fetched.")

    with httpx.Client() as client, RAW_PATH.open("a", encoding="utf-8") as log:
        for i, song in enumerate(corpus, 1):
            sid = song["_id"]
            if sid in done:
                continue
            bucket = script_bucket(song["lyrics"])
            print(f"[{i}/{len(corpus)}] {sid} ({bucket}) ...")
            metrics = fetch_metrics(client, args.url.rstrip("/"), song["lyrics"])
            if not metrics:
                print(f"    SKIP (failed after retries): {sid}")
                continue
            rec = {"_id": sid, "bucket": bucket, **{m: metrics.get(m) for m in METRICS}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            done[sid] = rec

    # ── Aggregate into per-(bucket, metric) sorted reference arrays ──────────
    calibration: dict = {"deva": {}, "latin": {}, "_meta": {}}
    for bucket in ("deva", "latin"):
        rows = [r for r in done.values() if r.get("bucket") == bucket]
        calibration["_meta"][bucket] = len(rows)
        for m in METRICS:
            vals = sorted(float(r[m]) for r in rows if isinstance(r.get(m), (int, float)))
            calibration[bucket][m] = vals

    OUT_PATH.write_text(json.dumps(calibration), encoding="utf-8")
    print(
        f"\nWrote {OUT_PATH.name}: "
        f"deva={calibration['_meta']['deva']} latin={calibration['_meta']['latin']} songs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
