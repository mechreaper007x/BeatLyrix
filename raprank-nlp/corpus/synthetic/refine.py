"""
Critique-revise loop for the reference-anchored synthetic generators.

The old loop was blind rejection sampling: generate -> score_against_tier ->
if it drifts from tier, THROW IT AWAY and regenerate from scratch. That
discards the most valuable signal available -- the per-axis gap between what
the draft measured and what the tier wants. This module reuses that same
gap (score_against_tier already returns actual + expected for every axis) as
a reward signal and feeds it back as a TARGETED revision instruction: keep the
theme and the lines that work, fix the specific weak devices. Re-measure,
iterate a couple of rounds, and keep the best-scoring version seen -- so a
near-miss draft gets nudged over the line instead of being wasted.

This is generator-agnostic: callers pass in their own async `gen(prompt)`
(Gemini for the English/mixed reference generators), the `tier`/`lang`, and a
scorer. Nothing here calls an LLM API directly.
"""
from __future__ import annotations

from corpus.synthetic.generate import score_against_tier

# Human-readable, craft-language repair guidance per axis -- what to DO to raise
# a weak axis, phrased the way rap_craft.py teaches technique (never "raise the
# score", always the concrete move). Only the axes an LLM can actually act on
# are listed; curve/style axes it can't titrate are intentionally omitted.
_AXIS_FIX: dict[str, str] = {
    "rhyme_internal_density":
        "add INTERNAL rhymes -- make words rhyme WITHIN a line, not only at the end.",
    "rhyme_multisyllabic_density":
        "rhyme 2-3 SYLLABLES together as a unit (e.g. 'celebrated / decorated'), "
        "not just the final syllable -- lean on the rhyme building blocks above.",
    "rhyme_chain_density":
        "sustain the SAME rhyme sound across 3-5 lines in a row before switching it.",
    "rhyme_compound_density":
        "land a compound/mosaic rhyme -- a multi-word phrase rhyming with another "
        "('hold me / goalie').",
    "assonance_density":
        "repeat the same VOWEL sound across several words in a line.",
    "consonance_density":
        "repeat a CONSONANT sound (not the first letter) across nearby words.",
    "syllable_density_per_line":
        "pack MORE syllables per line -- longer, denser bars.",
    "syllable_complex_word_ratio":
        "use longer, multi-syllable words instead of only short everyday ones.",
    "vocabulary_msttr":
        "widen the vocabulary -- stop repeating the same words, vary word choice.",
    "wordplay_simile_density":
        "add a concrete simile ('X like Y') -- sparingly, not abstract/poetic.",
    "wordplay_metaphor_density":
        "add an implicit metaphor -- call one thing another without 'like'.",
    "wordplay_pun_density":
        "work in a pun -- play on a word that means or sounds like two things.",
    "wordplay_entendre_density":
        "land a double entendre -- a line carrying two meanings at once.",
    "alliteration_density":
        "stack alliteration -- nearby words starting with the same sound.",
}

# Axes where OVERSHOOT (too dense for the tier) is the failure, so the fix is to
# pull back rather than add. At mid/commercial, over-hitting elite devices is
# the tier-confusing miss; guidance flips to "simplify".
_AXIS_REDUCE: dict[str, str] = {
    "rhyme_internal_density": "use FEWER internal rhymes -- let more lines rhyme only at the end.",
    "rhyme_multisyllabic_density": "simplify -- fewer multisyllabic rhymes, more single-syllable end rhymes.",
    "rhyme_chain_density": "switch the rhyme sound more often instead of long chains.",
    "syllable_density_per_line": "shorten the lines -- fewer syllables per bar, punchier.",
    "syllable_complex_word_ratio": "use simpler, shorter everyday words.",
    "vocabulary_msttr": "it's fine to repeat words/hooks -- less lexical variety.",
    "alliteration_density": "ease off the alliteration.",
}


# Axes the scaffold+refine features exist to move (the rhyme/wordplay devices
# LLMs undershoot). fitness() weights these higher so a revision round spends
# its effort closing the gaps that matter instead of defending axes already at
# ceiling -- the A/B pilot showed big per-axis wins here (elite multisyllabic
# +12, chain +20) washing out to a ~+1.4 net under equal weighting.
_FOCUS_AXES = frozenset({
    "rhyme_internal_density",
    "rhyme_multisyllabic_density",
    "rhyme_chain_density",
    "rhyme_compound_density",
    "wordplay_simile_density",
    "wordplay_metaphor_density",
    "wordplay_pun_density",
    "wordplay_entendre_density",
})
_FOCUS_WEIGHT = 3.0


def fitness(actual: dict, expected: dict, tier: str) -> float:
    """Scalar quality of a draft against its tier: weighted mean per-axis
    closeness in [0,1], where 1.0 == every axis exactly on target. The
    rhyme/wordplay focus axes count _FOCUS_WEIGHT-fold so refine optimizes
    toward the gaps the scaffold targets, not the already-saturated ones. Used
    to KEEP the best iteration; the accept/reject boolean still comes from
    score_against_tier."""
    from corpus.synthetic.generate import _ONE_SIDED_BELOW_ELITE

    if not actual:
        return 0.0
    total = 0.0
    weight_sum = 0.0
    for axis in actual:
        w = _FOCUS_WEIGHT if axis in _FOCUS_AXES else 1.0
        diff = actual[axis] - expected[axis]
        # Below elite, undershooting a rare all-or-nothing device is not a miss.
        if tier != "elite" and axis in _ONE_SIDED_BELOW_ELITE and diff < 0:
            total += w * 1.0
            weight_sum += w
            continue
        # Normalize the gap against a full 0-100 span; clamp to [0,1].
        closeness = max(0.0, 1.0 - abs(diff) / 100.0)
        total += w * closeness
        weight_sum += w
    return total / weight_sum if weight_sum else 0.0


def weakest_axes(actual: dict, expected: dict, tier: str, top_k: int = 4) -> list[tuple[str, str]]:
    """The axes furthest off-target that the model can actually act on, each
    paired with a concrete craft instruction. Returns [(axis, instruction)]
    ordered worst-first, capped at top_k so the revision prompt stays focused."""
    from corpus.synthetic.generate import _ONE_SIDED_BELOW_ELITE

    scored: list[tuple[float, str, str]] = []
    for axis in actual:
        diff = actual[axis] - expected[axis]
        if tier != "elite" and axis in _ONE_SIDED_BELOW_ELITE and diff < 0:
            continue  # legitimate below-elite undershoot, not a defect
        gap = abs(diff)
        if gap < 12.0:
            continue  # close enough; don't nag the model about near-hits
        if diff < 0 and axis in _AXIS_FIX:
            scored.append((gap, axis, _AXIS_FIX[axis]))
        elif diff > 0 and axis in _AXIS_REDUCE:
            scored.append((gap, axis, _AXIS_REDUCE[axis]))
    scored.sort(reverse=True)
    return [(axis, instr) for _, axis, instr in scored[:top_k]]


def build_revision_prompt(tier: str, lang_label: str, lang_block: str,
                          draft: str, fixes: list[tuple[str, str]],
                          scaffold_block: str = "") -> str:
    """A revision prompt: keep what works in the draft, apply targeted fixes to
    the specific weak devices. Deliberately preserves theme/content so revision
    IMPROVES a draft rather than replacing it."""
    fix_lines = "\n".join(f"  {i+1}. {instr}" for i, (_axis, instr) in enumerate(fixes))
    scaffold = f"\n{scaffold_block}\n" if scaffold_block else ""
    return f"""You wrote this {tier}-tier rap verse. It is close, but a few techniques need
strengthening. REVISE it -- keep the same topic, story, and the lines that
already work; only rewrite what's needed to fix these specific issues:

{fix_lines}
{scaffold}
--- YOUR DRAFT ---
{draft}
--- END DRAFT ---

Language: write in {lang_label}. {lang_block}
Output ONLY the revised verse, one bar per line -- no title, no explanation,
no markdown, no analysis, no technical terms like "internal rhyme" or "score"
in the lyrics themselves. Keep it the same length (16-24 lines)."""


async def refine(
    *,
    tier: str,
    lang: str,
    draft: str,
    gen,
    clean,
    lang_label: str,
    lang_block: str,
    scaffold_fn=None,
    max_rounds: int = 2,
    log=lambda *_: None,
) -> tuple[str, bool, dict, dict]:
    """Iteratively improve a draft toward its tier.

    Args:
        tier, lang: target tier and language ('en' or 'mixed').
        draft: the initial generated verse.
        gen:   async (prompt) -> raw model text.
        clean: (raw) -> cleaned lyrics (the generator's own post-processing).
        lang_label, lang_block: language instructions for the revision prompt.
        scaffold_fn: optional () -> str rhyme-building-block block to include.
        max_rounds: revision passes after the initial draft.

    Returns (best_lyrics, accepted, actual, expected) for the best iteration
    seen -- so even a non-accepted result is the strongest draft, ready for the
    caller's content gates. score_against_tier is the single source of truth
    for both the reward and the final accept decision.
    """
    accepted, actual, expected = score_against_tier(tier, draft, lang=lang)
    best = (fitness(actual, expected, tier), draft, accepted, actual, expected)
    if accepted:
        return draft, True, actual, expected

    for rnd in range(1, max_rounds + 1):
        fixes = weakest_axes(actual, expected, tier)
        if not fixes:
            break  # nothing actionable left to improve
        scaffold_block = scaffold_fn() if scaffold_fn else ""
        prompt = build_revision_prompt(tier, lang_label, lang_block, best[1], fixes, scaffold_block)
        try:
            revised = clean(await gen(prompt))
        except Exception as exc:
            log(f"      revise round {rnd} call failed: {type(exc).__name__}: {exc}")
            break
        r_accepted, r_actual, r_expected = score_against_tier(tier, revised, lang=lang)
        r_fit = fitness(r_actual, r_expected, tier)
        weak_names = ", ".join(a for a, _ in fixes)
        log(f"      revise round {rnd}: fitness {best[0]:.2f} -> {r_fit:.2f} "
            f"(targeted: {weak_names})")
        if r_fit > best[0]:
            best = (r_fit, revised, r_accepted, r_actual, r_expected)
            actual, expected = r_actual, r_expected
        if r_accepted:
            return revised, True, r_actual, r_expected

    return best[1], best[2], best[3], best[4]
