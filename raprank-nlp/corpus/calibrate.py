"""
ECDF-based calibration tool for scoring_config.py constants.

Computes the empirical distribution (percentiles, mean, std, saturation
fraction) of a metric's RAW pre-curve value across the full scraped corpus,
plus optional spot-checks against known gold-standard tracks. This replaces
ad-hoc inline `python -c` percentile scripts with a single reusable,
metric-agnostic tool -- every recalibration pass (alliteration, rhyme,
wordplay, etc.) should go through this rather than re-deriving percentiles
by hand.

Each metric function must return the RAW density/ratio a service computes
BEFORE it is passed through evaluate_piecewise_curve or an ELITE_TARGETS
linear-cap -- percentiles of an already-curved 0-100 score just re-derive
the curve's own shape and tell you nothing about where to place thresholds.
Services expose this via a `debug=True` kwarg on calculate() that returns
the raw value as an extra tuple element.

Usage:
    python -m corpus.calibrate --metric allit.density
    python -m corpus.calibrate --metric allit.density --spotcheck
    python -m corpus.calibrate --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.tests.conftest import corpus  # noqa: E402
from services import alliteration_service as al  # noqa: E402
from services import rhyme_service as rh  # noqa: E402
from services import assonance_service as asn  # noqa: E402
from services import consonance_service as cns  # noqa: E402
from services import onomatopoeia_service as ono  # noqa: E402
from services import vocabulary_service as voc  # noqa: E402
from services import syllable_service as syl  # noqa: E402
from services import wordplay_service as wp  # noqa: E402
from services import prosody_service as pros  # noqa: E402
from services.lyrical_compiler import compile_lyrics  # noqa: E402

PERCENTILES = [5, 10, 25, 50, 75, 90, 95, 99]

# Known gold-standard / reference tracks for qualitative sanity-checking a
# fitted curve, keyed by (artist_dir, title_slug) matching corpus/data/.
SPOTCHECK_TRACKS = [
    ("kr-na", "vyanjan"),       # community-celebrated technical rhyme showcase
    ("carryminati", "yalgaar"),  # commercial/mainstream reference point
]


def _raw_alliteration_density(lyrics: str) -> float:
    _, _, raw_density = al.calculate(lyrics, debug=True)
    return raw_density


def _raw_assonance_density(lyrics: str) -> float:
    _, _, raw_density = asn.calculate(lyrics, debug=True)
    return raw_density


def _raw_consonance_density(lyrics: str) -> float:
    _, _, raw_density = cns.calculate(lyrics, debug=True)
    return raw_density


def _raw_onomatopoeia_density(lyrics: str) -> float:
    _, _, raw_density = ono.calculate(lyrics, debug=True)
    return raw_density


def _rhyme_raw(key: str):
    def _fn(lyrics: str) -> float:
        *_, ratios = rh.calculate(lyrics, debug=True)
        return ratios[key]
    return _fn


# Dispatch table: metric name -> function(lyrics: str) -> float raw value.
# Extend this per-phase as each service grows a debug=True raw-value path.
METRICS = {
    "allit.density": _raw_alliteration_density,
    "assonance.density": _raw_assonance_density,
    "consonance.density": _raw_consonance_density,
    "onomatopoeia.density": _raw_onomatopoeia_density,
    "vocabulary.msttr": lambda lyrics: voc.calculate(lyrics)[1],
    "syllable.avg_per_line": lambda lyrics: syl.calculate(lyrics)[1],
    "syllable.weight_ratio": lambda lyrics: syl.calculate(lyrics)[3],
    "wordplay.simile_density": lambda lyrics: wp.calculate(lyrics)[1]["simile_density"],
    "wordplay.metaphor_density": lambda lyrics: wp.calculate(lyrics)[1]["metaphor_density"],
    "wordplay.pun_density": lambda lyrics: wp.calculate(lyrics)[1]["pun_density"],
    "wordplay.entendre_density": lambda lyrics: wp.calculate(lyrics)[1]["entendre_density"],
    "wordplay.total_density": lambda lyrics: wp.calculate(lyrics)[1]["total_density"],
    "prosody.codeswitch": lambda lyrics: pros.calculate(lyrics)["raw"]["codeswitch"],
    "prosody.repetition": lambda lyrics: pros.calculate(lyrics)["raw"]["repetition"],
    "prosody.cadence_var": lambda lyrics: pros.calculate(lyrics)["raw"]["cadence_var"],
    "rhyme.end_rhyme_ratio": _rhyme_raw("end_rhyme_ratio"),
    "rhyme.internal_ratio": _rhyme_raw("internal_ratio"),
    "rhyme.multisyl_ratio": _rhyme_raw("multisyl_ratio"),
    "rhyme.chain_ratio": _rhyme_raw("chain_ratio"),
    "rhyme.compound_ratio": _rhyme_raw("compound_ratio"),
    "rhyme.holorime_ratio": _rhyme_raw("holorime_ratio"),
    "llpc.rhyme_density": lambda lyrics: compile_lyrics(lyrics)["rhyme_density"],
    "llpc.normalized_entropy": lambda lyrics: compile_lyrics(lyrics)["normalized_entropy"],
    "llpc.rhyme_complexity": lambda lyrics: compile_lyrics(lyrics)["rhyme_complexity"],
}


def _find_track(artist_dir: str, title_slug: str) -> dict | None:
    for t in corpus():
        if t.get("_path", "").replace("\\", "/").startswith(f"corpus/data/{artist_dir}/"):
            if title_slug in t.get("_path", "").lower():
                return t
    return None


def run(metric_name: str, spotcheck: bool) -> int:
    if metric_name not in METRICS:
        print(f"Unknown metric '{metric_name}'. Available: {', '.join(sorted(METRICS))}")
        return 1

    fn = METRICS[metric_name]
    tracks = corpus()
    if not tracks:
        print("No corpus found. Run: python -m corpus.scrape_corpus")
        return 1

    values = np.array([fn(t["lyrics"]) for t in tracks], dtype=float)

    print(f"# ECDF for '{metric_name}' -- {len(values)} tracks\n")
    print(f"mean={values.mean():.4f}  std={values.std():.4f}  "
          f"frac>=1.0={(values >= 1.0).mean():.2%}")
    print("\npercentile | value")
    print("-----------|------")
    for p in PERCENTILES:
        print(f"{p:>10} | {np.percentile(values, p):.4f}")

    if spotcheck:
        print("\n# Spot-checks")
        for artist_dir, title_slug in SPOTCHECK_TRACKS:
            track = _find_track(artist_dir, title_slug)
            if track is None:
                print(f"  {artist_dir}/{title_slug}: NOT FOUND in corpus")
                continue
            raw = fn(track["lyrics"])
            pct = (values < raw).mean() * 100
            print(f"  {track['artist']} - {track['title']}: raw={raw:.4f} "
                  f"(~{pct:.0f}th percentile)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", help="Metric name, e.g. allit.density")
    parser.add_argument("--spotcheck", action="store_true",
                         help="Also print raw values for known gold-standard tracks")
    parser.add_argument("--list", action="store_true", help="List available metrics")
    args = parser.parse_args()

    if args.list or not args.metric:
        print("Available metrics:")
        for name in sorted(METRICS):
            print(f"  {name}")
        return 0

    return run(args.metric, args.spotcheck)


if __name__ == "__main__":
    raise SystemExit(main())
