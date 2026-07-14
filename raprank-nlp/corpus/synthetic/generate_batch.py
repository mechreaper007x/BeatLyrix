"""
Concurrent driver for corpus.synthetic.generate -- same generator, same
accept/reject/validation logic, just fans out multiple in-flight generation
calls instead of awaiting them one at a time so a 300-song batch takes
~1.5-2hrs instead of ~10hrs.

NOTE: this used to wrap generate_with_reference.py's reference-anchored
generator (which fed a real, copyrighted lyric excerpt from corpus/data/ into
every prompt, sent to a third-party cloud inference API on every call). That
approach has been dropped entirely -- this driver now only wraps
corpus.synthetic.generate's numeric-target-only generator, which never sends
any real lyric text anywhere. Every prompt is built purely from the numeric
tier/axis targets in corpus.synthetic.tier_profiles plus a topic label.

Hinglish-only, matching corpus/synthetic/generate.py -- pure "hi" and pure
"en" generation modes were dropped entirely.

Usage:
    python -m corpus.synthetic.generate_batch --count 300 --concurrency 6 --resume
    python -m corpus.synthetic.generate_batch --count 12 --concurrency 4  # smoke batch
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpus.synthetic.generate import MAX_ATTEMPTS_MIXED, generate_one  # noqa: E402
from corpus.synthetic.tier_profiles import TIER_NAMES  # noqa: E402

# Deliberately separate from corpus/synthetic/generate.py's own OUT_DIR
# (corpus/synthetic_data/), which holds the classification-training corpus
# used to label RF/SVM/Bayesian tier models. This batch feeds the
# clustering-algorithm pipeline instead -- a different purpose, different
# destination, so the two datasets never get mixed together on disk.
OUT_DIR = Path(__file__).resolve().parents[1] / "synthetic_data_clustering"


def _plan_batch(count: int) -> list[str]:
    """Spread `count` samples evenly across tiers -- Hinglish-only, so there's
    no language dimension left to cross against."""
    return [TIER_NAMES[i % len(TIER_NAMES)] for i in range(count)]


async def _worker(sem: asyncio.Semaphore, client, idx: int, total: int, tier: str) -> dict | None:
    async with sem:
        record = await generate_one(client, tier, topic_idx=idx)
        if record is None:
            print(f"  [{idx}/{total}] {tier}/mixed -> DROPPED after {MAX_ATTEMPTS_MIXED} attempts")
            return None
        path = OUT_DIR / tier / "mixed" / f"{record['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{idx}/{total}] {tier}/mixed -> accepted in {record['attempts']} attempt(s) ({path.name})")
        return record


async def run(count: int, concurrency: int, tier_filter: str | None, resume: bool) -> int:
    from mistralai import Mistral

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: set MISTRAL_API_KEY", file=sys.stderr)
        return 2
    client = Mistral(api_key=api_key)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = _plan_batch(count)
    if tier_filter:
        plan = [t for t in plan if t == tier_filter]

    if resume:
        # Skip slots whose tier already has >= its expected share of
        # accepted samples on disk.
        filtered = []
        seen_counts: dict[str, int] = {}
        for tier in plan:
            tier_dir = OUT_DIR / tier / "mixed"
            existing = len(list(tier_dir.glob("*.json"))) if tier_dir.exists() else 0
            seen_counts[tier] = seen_counts.get(tier, 0) + 1
            if seen_counts[tier] <= existing:
                continue
            filtered.append(tier)
        plan = filtered

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _worker(sem, client, i, len(plan), tier)
        for i, tier in enumerate(plan, 1)
    ]
    results = await asyncio.gather(*tasks)

    accepted_total = sum(1 for r in results if r is not None)
    rejected_total = sum(1 for r in results if r is None)
    print(f"\nDONE: {accepted_total} accepted, {rejected_total} dropped, out of {len(plan)} planned")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Concurrent numeric-target-only synthetic Hinglish lyrics generation")
    ap.add_argument("--count", type=int, default=300, help="total samples to attempt across tiers")
    ap.add_argument("--concurrency", type=int, default=6, help="max in-flight Mistral calls")
    ap.add_argument("--tier", choices=TIER_NAMES, help="only this tier")
    ap.add_argument("--resume", action="store_true", help="skip tier slots that already have enough samples")
    args = ap.parse_args()

    return asyncio.run(run(args.count, args.concurrency, args.tier, args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
