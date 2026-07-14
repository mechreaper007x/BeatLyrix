"""
Onomatopoeia / ad-lib density detection.

Vocalized sound-effect interjections ("woo", "skrrt", "brrr", elongated
"ayyy") are a real stylistic axis distinct from ordinary filler words (which
`language_utils.get_multilingual_stopwords` already treats as noise for other
detectors). Unlike `content_lines`, which strips parenthetical ad-libs and
section headers before scoring rhyme/wordplay/etc., this detector scores the
*raw* lines -- ad-lib density is itself the signal here, not something to
discard before scoring.

A word counts as an ad-lib when it is either:
  1. In the known onomatopoeia/interjection lexicon (config-driven), or
  2. An "elongated" interjection: any single letter repeats 3+ times in a row
     within the word (e.g. "ayyy", "yooo", "brrrr") -- catches ad-libbed
     stylization of any base word without needing to enumerate every spelling.
"""
from __future__ import annotations

import re

from config import scoring_config

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _raw_lines(lyrics: str) -> list[str]:
    """Non-empty, non-section-header lines, WITHOUT ad-lib stripping."""
    if not lyrics.strip():
        return []
    lines = []
    for line in lyrics.strip().split("\n"):
        stripped = line.strip()
        if stripped and not (stripped.startswith("[") and stripped.endswith("]")):
            lines.append(stripped)
    return lines


def _is_elongated(word: str) -> bool:
    min_repeat = scoring_config.SOUND["ONOMATOPOEIA_ELONGATION_MIN_REPEAT"]
    pattern = r"(.)\1{" + str(min_repeat - 1) + r",}"
    return bool(re.search(pattern, word.lower()))


def _is_adlib(word: str) -> bool:
    w = word.lower()
    if w in scoring_config.SOUND["ONOMATOPOEIA_WORDS"]:
        return True
    return _is_elongated(w)


def calculate(lyrics: str, debug: bool = False):
    """
    Returns (onomatopoeia_score 0-100, list of detected ad-lib hits).
    With debug=True, returns (score, details, raw_density) where raw_density
    is the pre-curve density -- used by corpus/calibrate.py to fit curve
    constants from the corpus's actual empirical distribution instead of the
    already-curved 0-100 score.
    """
    lines = _raw_lines(lyrics)
    if not lines:
        return (0.0, [], 0.0) if debug else (0.0, [])

    details: list[str] = []
    hits = 0
    for line in lines:
        for word in _WORD_RE.findall(line):
            if _is_adlib(word):
                hits += 1
                details.append(word)

    density = hits / len(lines)
    score = scoring_config.evaluate_piecewise_curve(
        density,
        scoring_config.SOUND["ONOMATOPOEIA_THRESHOLDS"],
        scoring_config.SOUND["ONOMATOPOEIA_SCORES"],
    )
    if debug:
        return round(score, 2), details, density
    return round(score, 2), details
