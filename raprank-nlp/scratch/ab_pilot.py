"""
A/B pilot: old blind-rejection loop vs new refine+rhyme-scaffold loop.

Drives corpus.synthetic.generate_english_reference.generate_one directly
(Gemini-backed) for both arms, flipping the module's _REFINE_ON/_SCAFFOLD_ON
toggles in-process between arms so the ONLY difference is the two features
under test -- same prompts, same reference pool, same scorer, same tiers.

Nothing is written into the live corpus/synthetic_data/ corpus: accepted
records are held in memory only, their actual_scores aggregated to a per-arm,
per-tier, per-axis mean, and the delta (new - old) reported. The raw per-sample
scores are dumped to scratch/ab_pilot_result.json (gitignored) for inspection.
This measures quality lift before committing to a full re-scale.

The A/B is English-only because that arm uses Gemini (GEMINI_API_KEY); the
Hinglish sibling needs a local Ollama daemon. The refine loop and scaffold code
are shared between both generators, so the English lift is representative.

Usage:
    GEMINI_API_KEY=xxx python scratch/ab_pilot.py --per-tier 15
    GEMINI_API_KEY=xxx python scratch/ab_pilot.py --per-tier 15 --tiers elite mid
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus.synthetic import generate_english_reference as gen  # noqa: E402
from corpus.synthetic.generate import GEMINI_API_KEY  # noqa: E402

# Axes the two features are meant to move (the ones models undershoot); reported
# first and called out in the summary. The full axis set is still aggregated.
_FOCUS_AXES = [
    "rhyme_internal_density",
    "rhyme_multisyllabic_density",
    "rhyme_chain_density",
    "rhyme_compound_density",
    "wordplay_simile_density",
    "wordplay_metaphor_density",
    "wordplay_pun_density",
    "wordplay_entendre_density",
]


# Incremental persistence: every accepted record is appended here immediately,
# so a mid-run crash never loses completed samples and --resume can pick up.
_RECORDS_PATH = Path(__file__).resolve().parent / "ab_pilot2_records.jsonl"


def _load_existing_records() -> list[dict]:
    records: list[dict] = []
    if _RECORDS_PATH.exists():
        for line in _RECORDS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _persist_record(rec: dict) -> None:
    slim = {"_arm": rec["_arm"], "tier": rec["tier"], "attempts": rec["attempts"],
            "actual_scores": rec["actual_scores"]}
    with _RECORDS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(slim, ensure_ascii=False) + "\n")


async def _run_arm(arm: str, scaffold_on: bool, refine_on: bool,
                   tiers: list[str], per_tier: int,
                   reference_pool: dict, existing: list[dict]) -> list[dict]:
    """Generate up to per_tier accepted samples per tier for one arm.
    Sets the module toggles for this arm before generating. Samples already
    persisted from a previous (crashed) run count toward the quota."""
    gen._SCAFFOLD_ON = scaffold_on
    gen._REFINE_ON = refine_on
    print(f"\n=== ARM: {arm}  (scaffold={scaffold_on}, refine={refine_on}) ===")

    records: list[dict] = []
    for tier in tiers:
        done = sum(1 for r in existing if r["_arm"] == arm and r["tier"] == tier)
        got = done
        if done:
            print(f"  [{arm}] {tier}: resuming with {done}/{per_tier} already persisted")
        # Give a few extra topic slots of headroom since some attempts drop.
        # Offset topic_idx by resumed count so reruns draw fresh topics.
        for topic_idx in range(done, per_tier * 2):
            if got >= per_tier:
                break
            try:
                rec = await gen.generate_one(tier, reference_pool, topic_idx=topic_idx)
            except Exception:
                # One bad sample must not kill the whole pilot again.
                import traceback
                print(f"  [{arm}] {tier}: sample crashed, skipping --")
                traceback.print_exc()
                continue
            if rec is None:
                continue
            rec["_arm"] = arm
            _persist_record(rec)
            records.append(rec)
            got += 1
            print(f"  [{arm}] {tier}: {got}/{per_tier} accepted")
        if got < per_tier:
            print(f"  [{arm}] {tier}: WARNING only {got}/{per_tier} accepted")
    return records


def _aggregate(records: list[dict]) -> dict:
    """arm -> tier -> axis -> mean actual score across accepted samples."""
    sums: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts: dict = defaultdict(lambda: defaultdict(int))
    for rec in records:
        arm, tier = rec["_arm"], rec["tier"]
        counts[arm][tier] += 1
        for axis, val in rec["actual_scores"].items():
            sums[arm][tier][axis] += val
    means: dict = {}
    for arm in sums:
        means[arm] = {}
        for tier in sums[arm]:
            n = counts[arm][tier] or 1
            means[arm][tier] = {ax: s / n for ax, s in sums[arm][tier].items()}
    return means, counts


def _report(means: dict, counts: dict, tiers: list[str],
            base: str, treat: str) -> None:
    print("\n" + "=" * 72)
    print(f"A/B RESULT -- mean actual axis score, {treat} vs {base} "
          f"(delta = {treat} - {base})")
    print("=" * 72)
    for tier in tiers:
        old = means.get(base, {}).get(tier, {})
        new = means.get(treat, {}).get(tier, {})
        n_old = counts.get(base, {}).get(tier, 0)
        n_new = counts.get(treat, {}).get(tier, 0)
        print(f"\n--- {tier}  ({base} n={n_old}, {treat} n={n_new}) ---")
        axes = _FOCUS_AXES + sorted(set(old) | set(new) - set(_FOCUS_AXES))
        seen = set()
        for axis in axes:
            if axis in seen or axis not in old and axis not in new:
                continue
            seen.add(axis)
            o, nw = old.get(axis, 0.0), new.get(axis, 0.0)
            delta = nw - o
            focus = "*" if axis in _FOCUS_AXES else " "
            arrow = "up" if delta > 0.5 else ("dn" if delta < -0.5 else "  ")
            print(f"  {focus} {axis:34} {base} {o:6.1f}  {treat} {nw:6.1f}  d {delta:+6.1f} {arrow}")

    # Focus-axis summary: mean delta across focus axes, per tier.
    print(f"\n--- FOCUS-AXIS mean delta (rhyme + wordplay, {treat} - {base}) ---")
    for tier in tiers:
        old = means.get(base, {}).get(tier, {})
        new = means.get(treat, {}).get(tier, {})
        deltas = [new.get(a, 0.0) - old.get(a, 0.0)
                  for a in _FOCUS_AXES if a in old or a in new]
        if deltas:
            print(f"  {tier:12} mean focus delta {sum(deltas)/len(deltas):+6.2f}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=8)
    ap.add_argument("--tiers", nargs="+", default=["elite", "mid", "commercial"])
    args = ap.parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: set GEMINI_API_KEY", file=sys.stderr)
        return 1

    reference_pool = gen.load_english_reference_pool()
    for tier in args.tiers:
        print(f"  reference pool: {tier:11} {len(reference_pool.get(tier, []))} songs")

    existing = _load_existing_records()
    if existing:
        print(f"  resume: {len(existing)} records already persisted in {_RECORDS_PATH.name}")

    # Round 2: the old blind-rejection baseline is already characterized, so
    # this pilot isolates the refine loop's marginal value on top of the proven
    # scaffold. Both arms inject the scaffold; only the treatment arm also runs
    # the (now focus-weighted) refine loop.
    base_records = await _run_arm("scaffold", scaffold_on=True, refine_on=False,
                                  tiers=args.tiers, per_tier=args.per_tier,
                                  reference_pool=reference_pool, existing=existing)
    treat_records = await _run_arm("scaffold+refine", scaffold_on=True, refine_on=True,
                                   tiers=args.tiers, per_tier=args.per_tier,
                                   reference_pool=reference_pool, existing=existing)
    all_records = existing + base_records + treat_records

    means, counts = _aggregate(all_records)
    _report(means, counts, args.tiers, base="scaffold", treat="scaffold+refine")

    # Dump raw per-sample scores (NOT lyrics) for inspection; gitignored path.
    out = Path(__file__).resolve().parent / "ab_pilot2_result.json"
    dump = [{"arm": r["_arm"], "tier": r["tier"], "attempts": r["attempts"],
             "actual_scores": r["actual_scores"]} for r in all_records]
    out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRaw per-sample scores -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
