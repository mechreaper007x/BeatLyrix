"""
Shared helpers for sound-device detectors (alliteration, assonance, consonance).

All three services detect a repeated sound across content words and previously
scoped that detection to a single line, making repeated hooks/refrains spread
across consecutive lines invisible regardless of density tuning. This module
provides the cross-line clustering primitive (mirroring the line-window scan
`rhyme_service.py` already uses for end rhymes) so each service shares one
clustering implementation instead of three drifting copies.
"""
from __future__ import annotations

# Words that are grammatically "function words" by NLTK/general convention
# (and so are stripped by get_multilingual_stopwords(), which is shared by 7
# services including non-sound ones like vocabulary/syllable/wordplay) but
# carry real perceptible phonetic weight in Hindi/Hinglish rap hooks -- common
# past-tense/aux verb forms used as content, not connectives (e.g. "Dekh kaun
# aaya wapas" repeated as a hook -- "kaun"/"aaya" are stopwords but are the
# words carrying the line's actual sound). Only sound-device detectors should
# treat these as content; editing the shared stopword list itself would touch
# 7 services' unrelated density calculations.
PHONETIC_CONTENT_OVERRIDES = {
    "kaun", "aaya", "aayi", "aaye", "gaya", "gayi", "gaye",
    "kaha", "raha", "rahi", "rahe", "diya", "liya", "hua", "hui", "hue",
}


def sound_stopwords(base_stopwords: set[str]) -> set[str]:
    """base_stopwords minus the phonetic-content overrides, for sound-device use only."""
    return base_stopwords - PHONETIC_CONTENT_OVERRIDES


def cluster_by_line_proximity(
    occurrences: list[tuple[int, str]],
    window_lines: int,
) -> list[list[tuple[int, str]]]:
    """
    Group (line_idx, word) occurrences of the same sound into clusters where
    consecutive occurrences are at most `window_lines` lines apart.

    occurrences must be sorted by line_idx ascending. Mirrors the contiguous-
    run grouping alliteration_service used for same-line word-index gaps, but
    keyed on line-index gaps so a sound repeated across nearby lines (e.g. a
    hook's first word repeated every line) clusters into one group instead of
    vanishing at each line boundary.
    """
    clusters: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for line_idx, word in occurrences:
        if current and line_idx - current[-1][0] > window_lines:
            clusters.append(current)
            current = []
        current.append((line_idx, word))
    if current:
        clusters.append(current)
    return clusters
