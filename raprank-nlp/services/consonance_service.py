"""
Consonance detection — repeated NON-ONSET consonant sound across content words.

Consonance is the repetition of consonant sounds in the interior or at the end
of nearby words ("bla**nk**", "thi**nk**", "ju**nk**"). It deliberately ignores
each word's first sound, because onset repetition is *alliteration* and is scored
by alliteration_service — counting onsets here would double-credit the same
device. Vowel repetition is handled by assonance_service.

Approach mirrors alliteration_service / assonance_service:
  1. English words → CMU consonant phonemes, dropping the first phoneme.
  2. Hindi (Devanagari) → transliterated, then spelled-consonant fallback.
  3. Romanized Hinglish → spelled consonants (digraphs collapsed: bh→b, etc.),
     dropping the onset.
  4. Non-onset consonant occurrences are collected across the whole song, then
     clustered by line proximity (config.SOUND["WINDOW_SIZE_LINES"]) so a
     consonant motif spread across nearby lines fires, not just words sharing
     a consonant within a single line; a cluster fires when it has
     >= MIN_WORDS_PER_GROUP distinct words. Density over valid lines drives a
     piecewise curve.
"""
from __future__ import annotations

import re

import pronouncing

from services.language_utils import (
    clean_word, content_lines, devanagari_to_roman, is_hindi_word,
    get_multilingual_stopwords,
)
from services.sound_device_utils import sound_stopwords, cluster_by_line_proximity
from config import scoring_config

_CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
# Collapse aspirated / cross-script digraphs to a single consonant identity so
# Hinglish spelling variants align (matches rhyme/alliteration normalization).
_DIGRAPHS = ("kh", "gh", "ch", "jh", "th", "dh", "ph", "bh", "sh", "chh")


def _collapse_digraphs(word: str) -> str:
    w = word.lower()
    for dg in sorted(_DIGRAPHS, key=len, reverse=True):
        w = w.replace(dg, dg[0])
    return w


def _noninitial_consonants(word: str) -> set[str]:
    """Set of consonant sounds in *word* excluding its onset."""
    rom = devanagari_to_roman(word) if is_hindi_word(word) else word
    rom_clean = re.sub(r"[^a-zA-Z]", "", rom)
    if not rom_clean:
        return set()

    phones = pronouncing.phones_for_word(rom_clean.lower())
    if phones:
        cons = [re.sub(r"\d", "", p).lower()
                for p in phones[0].split() if not any(c.isdigit() for c in p)]
        return set(cons[1:]) if len(cons) > 1 else set()

    letters = _collapse_digraphs(rom_clean)
    cons = [c for c in letters if c in _CONSONANTS]
    return set(cons[1:]) if len(cons) > 1 else set()


def calculate(lyrics: str, debug: bool = False):
    """
    Returns (consonance_score 0-100, list of detected consonant-repetition groups).
    With debug=True, returns (score, details, raw_density) where raw_density
    is the pre-curve density -- used by corpus/calibrate.py to fit curve
    constants from the corpus's actual empirical distribution instead of the
    already-curved 0-100 score.
    """
    cfg = scoring_config.SOUND
    stops = sound_stopwords(get_multilingual_stopwords())

    line_entries: list[tuple[int, list[str]]] = []
    for line_idx, line in enumerate(content_lines(lyrics)):
        words = [clean_word(w) for w in line.split()]
        words = [w for w in words
                 if w and len(w) >= cfg["MIN_WORD_LEN"] and w.lower() not in stops]
        if words:
            line_entries.append((line_idx, words))

    valid_lines = len(line_entries)
    if valid_lines == 0:
        return (0.0, [], 0.0) if debug else (0.0, [])

    consonant_occurrences: dict[str, list[tuple[int, str]]] = {}
    for line_idx, words in line_entries:
        for w in words:
            for c in _noninitial_consonants(w):
                consonant_occurrences.setdefault(c, []).append((line_idx, w.lower()))

    window = cfg["WINDOW_SIZE_LINES"]
    details: list[str] = []
    total_weight = 0.0

    for c, occurrences in consonant_occurrences.items():
        for cluster in cluster_by_line_proximity(occurrences, window):
            ws = {w for _, w in cluster}
            if len(ws) < cfg["MIN_WORDS_PER_GROUP"]:
                continue
            details.append(f"{c.upper()}: " + " - ".join(sorted(ws)))
            total_weight += min(len(ws) - 2, cfg["MAX_GROUP_WEIGHT"])

    density = total_weight / valid_lines
    score = scoring_config.evaluate_piecewise_curve(
        density, cfg["CONSONANCE_THRESHOLDS"], cfg["CONSONANCE_SCORES"]
    )
    if debug:
        return round(score, 2), details, density
    return round(score, 2), details
