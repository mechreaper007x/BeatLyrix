"""
Prosody / structural rap-element axes, promoted from the prototype heuristics
that previously lived only in corpus/analysis/signature.py (offline z-scores).
Now real scored axes surfaced in /analyze:

  - codeswitch_score      -- lines mixing English + non-English words (Hinglish
                             code-switching is central to Indian rap)
  - repetition_score      -- anaphora: consecutive lines sharing their first word
  - cadence_text_score    -- delivery-variety proxy: stdev of words-per-line
                             (text-only; distinct from the audio cadence in
                             flow_service.py, which is null without audio)

Raw-fraction helpers are the single source of truth -- corpus/analysis/signature.py
imports them so the offline analysis and the live endpoint can never diverge.
All local, no network, no LLM.
"""
from __future__ import annotations

import statistics as st

from services.language_utils import (
    clean_word,
    content_lines,
    get_multilingual_stopwords,
    is_hindi_word,
)
from config import scoring_config


# ── Raw heuristics (0.0-1.0 fractions) -- shared with signature.py ───────────
def codeswitch_ratio(lyrics: str) -> float:
    """Fraction of lines that mix an English-dict word with a non-English word."""
    import pronouncing
    stops = get_multilingual_stopwords()
    mix = n = 0
    for line in content_lines(lyrics):
        has_en = has_other = False
        for raw in line.split():
            w = clean_word(raw)
            if not w or len(w) < 2 or w in stops:
                continue
            if not is_hindi_word(w) and pronouncing.phones_for_word(w.lower()):
                has_en = True
            else:
                has_other = True
        n += 1
        if has_en and has_other:
            mix += 1
    return mix / n if n else 0.0


def repetition_ratio(lyrics: str) -> float:
    """Anaphora proxy: fraction of consecutive line pairs sharing the first word."""
    firsts = []
    for l in content_lines(lyrics):
        toks = [clean_word(w) for w in l.split()]
        toks = [t for t in toks if t]
        firsts.append(toks[0] if toks else "")
    if len(firsts) < 2:
        return 0.0
    same = sum(1 for a, b in zip(firsts, firsts[1:]) if a and a == b)
    return same / (len(firsts) - 1)


def cadence_var_raw(lyrics: str) -> float:
    """Stdev of words-per-line (delivery-variety proxy), normalised to ~0-1."""
    lens = [len([w for w in l.split() if w]) for l in content_lines(lyrics)]
    lens = [x for x in lens if x]
    if len(lens) < 2:
        return 0.0
    return min(st.pstdev(lens) / 4.0, 1.0)


# ── Public scored axes ───────────────────────────────────────────────────────
def calculate(lyrics: str) -> dict:
    """Return the three prosody axes as calibrated 0-100 scores plus raw values."""
    p = scoring_config.PROSODY
    cs = codeswitch_ratio(lyrics)
    rep = repetition_ratio(lyrics)
    cad = cadence_var_raw(lyrics)
    return {
        "codeswitch_score": round(
            scoring_config.evaluate_piecewise_curve(cs, p["CODESWITCH_THRESHOLDS"], p["CODESWITCH_SCORES"]), 2
        ),
        "repetition_score": round(
            scoring_config.evaluate_piecewise_curve(rep, p["REPETITION_THRESHOLDS"], p["REPETITION_SCORES"]), 2
        ),
        "cadence_text_score": round(
            scoring_config.evaluate_piecewise_curve(cad, p["CADENCE_THRESHOLDS"], p["CADENCE_SCORES"]), 2
        ),
        "raw": {"codeswitch": round(cs, 4), "repetition": round(rep, 4), "cadence_var": round(cad, 4)},
    }
