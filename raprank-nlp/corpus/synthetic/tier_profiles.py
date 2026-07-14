"""
Numeric tier targets for synthetic lyrics generation.

Every target here is a scalar against a threshold/target constant that
*already exists* in config/scoring_config.py (RHYME.ELITE_TARGETS,
WORDPLAY.ELITE_TARGETS, SOUND.*_THRESHOLDS, SYLLABLE.*_THRESHOLDS,
VOCABULARY.CURVE_THRESHOLDS) — no new statistical extraction step, and no
real lyric text is read or referenced anywhere in this module.

    elite      = 100% of the existing "elite" constant
    mid        = ~50%
    commercial = ~20%

Run `python -m corpus.synthetic.tier_profiles` to sanity-check these tiers
against corpus/artists.py's existing 0-1 `expected_profile` priors (numbers
only, never lyric text).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import scoring_config as cfg  # noqa: E402

TIER_NAMES: tuple[str, ...] = ("elite", "mid", "commercial")
TIER_SCALARS: dict[str, float] = {"elite": 1.0, "mid": 0.5, "commercial": 0.2}

# axis_name -> raw "elite" value, sourced directly from scoring_config.py.
# Units match whatever the underlying service measures (density fractions,
# syllables/line, or MSTTR ratio) -- see the comment on each constant's
# origin in scoring_config.py for exact meaning.
_ELITE_RAW: dict[str, float] = {
    "rhyme_internal_density": cfg.RHYME["ELITE_TARGETS"]["internal_density"],
    "rhyme_multisyllabic_density": cfg.RHYME["ELITE_TARGETS"]["multisyllabic_density"],
    "rhyme_chain_density": cfg.RHYME["ELITE_TARGETS"]["chain_density"],
    "rhyme_compound_density": cfg.RHYME["ELITE_TARGETS"]["compound_density"],
    "rhyme_holorime_density": cfg.RHYME["ELITE_TARGETS"]["holorime_density"],
    "assonance_density": cfg.SOUND["ASSONANCE_THRESHOLDS"][-1],
    "consonance_density": cfg.SOUND["CONSONANCE_THRESHOLDS"][-1],
    "onomatopoeia_density": cfg.SOUND["ONOMATOPOEIA_THRESHOLDS"][-1],
    "syllable_density_per_line": cfg.SYLLABLE["DENSITY_THRESHOLDS"][-1],
    "syllable_complex_word_ratio": cfg.SYLLABLE["WEIGHT_THRESHOLDS"][-1],
    "vocabulary_msttr": cfg.VOCABULARY["CURVE_THRESHOLDS"][-1],
    "wordplay_simile_density": cfg.WORDPLAY["ELITE_TARGETS"]["simile"],
    "wordplay_metaphor_density": cfg.WORDPLAY["ELITE_TARGETS"]["metaphor"],
    "wordplay_pun_density": cfg.WORDPLAY["ELITE_TARGETS"]["pun"],
    "wordplay_entendre_density": cfg.WORDPLAY["ELITE_TARGETS"]["entendre"],
}


def build_targets(tier: str) -> dict[str, float]:
    """Numeric generation targets for a tier -- the only thing the prompt sees."""
    if tier not in TIER_SCALARS:
        raise ValueError(f"unknown tier {tier!r}, expected one of {TIER_NAMES}")
    scalar = TIER_SCALARS[tier]
    return {axis: round(raw * scalar, 4) for axis, raw in _ELITE_RAW.items()}


# How each axis turns a raw target into an expected 0-100 measurement, so the
# validation loop in generate.py can compare against the *exact* formula the
# corresponding service already uses (scoring_config.py is the single source
# of truth for both generation targets and scoring, so this is a real check,
# not an approximation):
#
#   "linear_ratio"    rhyme sub-scores are `min(ratio/elite_target*100, 100)`
#                     (see services/rhyme_service.py) -- expected score is
#                     just the tier scalar, e.g. mid -> 50.0.
#   "density_direct"  wordplay device densities are read straight off
#                     corpus/analysis/signature.py as count/lines*100 -- the
#                     "expected score" is simply target_density*100.
#   "curve"           assonance/consonance/onomatopoeia/syllable/vocabulary
#                     scores run the raw measurement through the same
#                     evaluate_piecewise_curve() thresholds/scores already in
#                     scoring_config.py.
AXIS_KIND: dict[str, str] = {
    "rhyme_internal_density": "linear_ratio",
    "rhyme_multisyllabic_density": "linear_ratio",
    "rhyme_chain_density": "linear_ratio",
    "rhyme_compound_density": "linear_ratio",
    "rhyme_holorime_density": "linear_ratio",
    "wordplay_simile_density": "density_direct",
    "wordplay_metaphor_density": "density_direct",
    "wordplay_pun_density": "density_direct",
    "wordplay_entendre_density": "density_direct",
    "assonance_density": "curve",
    "consonance_density": "curve",
    "onomatopoeia_density": "curve",
    "syllable_density_per_line": "curve",
    "syllable_complex_word_ratio": "curve",
    "vocabulary_msttr": "curve",
}

# axis_name -> (thresholds, scores) for "curve" axes, straight from scoring_config.py.
_CURVES: dict[str, tuple[list[float], list[float]]] = {
    "assonance_density": (cfg.SOUND["ASSONANCE_THRESHOLDS"], cfg.SOUND["ASSONANCE_SCORES"]),
    "consonance_density": (cfg.SOUND["CONSONANCE_THRESHOLDS"], cfg.SOUND["CONSONANCE_SCORES"]),
    "onomatopoeia_density": (cfg.SOUND["ONOMATOPOEIA_THRESHOLDS"], cfg.SOUND["ONOMATOPOEIA_SCORES"]),
    "syllable_density_per_line": (cfg.SYLLABLE["DENSITY_THRESHOLDS"], cfg.SYLLABLE["DENSITY_SCORES"]),
    "syllable_complex_word_ratio": (cfg.SYLLABLE["WEIGHT_THRESHOLDS"], cfg.SYLLABLE["WEIGHT_SCORES"]),
    "vocabulary_msttr": (cfg.VOCABULARY["CURVE_THRESHOLDS"], cfg.VOCABULARY["CURVE_SCORES"]),
}


def expected_axis_score(tier: str, axis: str) -> float:
    """The 0-100 measurement a perfectly-on-target sample would produce for this axis."""
    kind = AXIS_KIND[axis]
    target = build_targets(tier)[axis]
    if kind == "linear_ratio":
        return min(TIER_SCALARS[tier] * 100.0, 100.0)
    if kind == "density_direct":
        return target * 100.0
    thresholds, scores = _CURVES[axis]
    return cfg.evaluate_piecewise_curve(target, thresholds, scores)


# ── Axes previously computed post-hoc but never targeted during generation ──
# alliteration, english/codeswitch, repetition, and cadence_var don't fit the
# same "elite raw value * tier scalar" model as the rhyme/wordplay/sound
# devices above -- they aren't skill-tier dials (more repetition or more
# code-switching isn't "more elite"), they're style/language dials driven by
# the `lang` slot and a universally-desirable moderate-variety band. Modeled
# separately so build_targets()/expected_axis_score() above stay a clean,
# tier-only model instead of being stretched to fit axes that don't belong.

# alliteration IS a tier-scaled technical device (elite writers stack it more
# skillfully/densely), so it reuses the tier-scalar model -- just as a
# "score_direct" axis (alliteration_service.calculate() already returns a
# calibrated 0-100 score, no separate raw-value curve to re-apply).
_ALLITERATION_ELITE_SCORE = 90.0

# english / codeswitch are two views of language composition. Generation was
# Hinglish-only for a long time (pure "hi" and pure "en" modes were dropped),
# hence the single "mixed" target below -- generate_english_reference.py
# (corpus/synthetic/) reintroduces a pure "en" mode for the Kaggle-seeded
# English reference generator, so this is now keyed by lang instead of a
# single fixed dict.
_LANGUAGE_RATIO_TARGETS_BY_LANG: dict[str, dict[str, float]] = {
    "mixed": {"english_ratio": 0.45, "codeswitch_density": 0.45},
    # Pure English: almost all content words are English, and there's no
    # language to switch INTO, so codeswitch_density should measure near-zero
    # rather than the Hinglish "moderate" target.
    "en": {"english_ratio": 0.95, "codeswitch_density": 0.05},
}

# repetition / cadence_var: a moderate, universally-desirable band regardless
# of tier or language (near-zero repetition reads as robotic, near-1.0 reads
# as spammy; near-zero cadence variance reads as monotone delivery) -- so a
# fixed mid-curve target is used at every tier/lang rather than scaling to
# "more is more elite."
_UNIVERSAL_PROSODY_TARGETS = {
    "repetition_density": cfg.PROSODY["REPETITION_THRESHOLDS"][1],   # mid rung of the curve
    "cadence_var_density": cfg.PROSODY["CADENCE_THRESHOLDS"][1],
}

EXTRA_AXIS_KIND: dict[str, str] = {
    "alliteration_density": "score_direct",
    "english_ratio": "curve_identity",     # already 0-1, no piecewise curve -- compared as a plain ratio *100
    "codeswitch_density": "curve",
    "repetition_density": "curve",
    "cadence_var_density": "curve",
}

_EXTRA_CURVES: dict[str, tuple[list[float], list[float]]] = {
    "codeswitch_density": (cfg.PROSODY["CODESWITCH_THRESHOLDS"], cfg.PROSODY["CODESWITCH_SCORES"]),
    "repetition_density": (cfg.PROSODY["REPETITION_THRESHOLDS"], cfg.PROSODY["REPETITION_SCORES"]),
    "cadence_var_density": (cfg.PROSODY["CADENCE_THRESHOLDS"], cfg.PROSODY["CADENCE_SCORES"]),
}


def build_extra_targets(tier: str, lang: str = "mixed") -> dict[str, float]:
    """Raw generation targets for the axes not covered by build_targets()."""
    if tier not in TIER_SCALARS:
        raise ValueError(f"unknown tier {tier!r}, expected one of {TIER_NAMES}")
    if lang not in _LANGUAGE_RATIO_TARGETS_BY_LANG:
        raise ValueError(f"unknown lang {lang!r}, expected one of {list(_LANGUAGE_RATIO_TARGETS_BY_LANG)}")
    targets = dict(_UNIVERSAL_PROSODY_TARGETS)
    targets.update(_LANGUAGE_RATIO_TARGETS_BY_LANG[lang])
    targets["alliteration_density"] = round(_ALLITERATION_ELITE_SCORE * TIER_SCALARS[tier], 4)
    return targets


def expected_extra_axis_score(tier: str, axis: str, lang: str = "mixed") -> float:
    """The 0-100 measurement a perfectly-on-target sample would produce for this axis."""
    kind = EXTRA_AXIS_KIND[axis]
    target = build_extra_targets(tier, lang)[axis]
    if kind == "score_direct":
        return target
    if kind == "curve_identity":
        return target * 100.0
    thresholds, scores = _EXTRA_CURVES[axis]
    return cfg.evaluate_piecewise_curve(target, thresholds, scores)


def _tier_for_prior(value: float) -> str:
    """Bin a 0-1 artists.py prior into the tier it's closest to, for cross-checking."""
    if value >= 0.75:
        return "elite"
    if value >= 0.40:
        return "mid"
    return "commercial"


def cross_check() -> list[str]:
    """
    Sanity-check tier scalars against corpus/artists.py's existing
    expected_profile priors -- numbers only, never lyric text. Returns a list
    of human-readable lines; does not raise, since these priors are coarse
    and only used for plausibility, not as ground truth.
    """
    from corpus.artists import unique_artists

    lines = []
    for a in unique_artists():
        prof = a.expected_profile
        if not prof:
            continue
        multisyl = prof.get("multisyllabic")
        commercial = prof.get("commercial")
        bits = []
        if multisyl is not None:
            bits.append(f"multisyllabic={multisyl:.2f}->{_tier_for_prior(multisyl)}")
        if commercial is not None:
            # commercial prior is inverted: high commercial ~ low artistic tier
            bits.append(f"commercial={commercial:.2f}->{_tier_for_prior(1.0 - commercial)}")
        if bits:
            lines.append(f"  {a.name:14} " + "  ".join(bits))
    return lines


def main() -> int:
    print("## Tier targets (numeric, no lyric text)\n")
    for tier in TIER_NAMES:
        print(f"  {tier}:")
        for axis, val in build_targets(tier).items():
            print(f"    {axis:30} {val:<10} expected_score={expected_axis_score(tier, axis):.1f}")
        print()

    print("\n## Cross-check vs. corpus/artists.py expected_profile priors\n")
    for line in cross_check():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
