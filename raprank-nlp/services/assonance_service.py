"""
Assonance detection — repeated vowel *nucleus* across content words in a line.

Assonance is the vowel counterpart of alliteration: the same stressed vowel
sound recurring across nearby words ("the rain in Sp**ai**n"), independent of the
consonants around it. It is a distinct axis from rhyme (which anchors on line
*ends*) and from alliteration (which anchors on word *onsets*).

Approach mirrors alliteration_service:
  1. English words → CMU vowel phonemes (stress-stripped); the dominant vowel
     sound is the last stressed vowel of the word.
  2. Hindi (Devanagari) → transliterated, then spelled-vowel fallback.
  3. Romanized Hinglish → spelled-vowel fallback (aa→a, ee→i, oo→u).
  4. Vowel-nucleus occurrences are collected across the whole song, then
     clustered by line proximity (config.SOUND["WINDOW_SIZE_LINES"]) so a
     vowel motif spread across nearby lines fires, not just words sharing a
     nucleus within a single line; a cluster fires when it has
     >= MIN_WORDS_PER_GROUP distinct words. Density over valid lines drives a
     piecewise 0-100 curve.
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

_VOWELS = "aeiou"


def _normalize_hinglish_vowels(word: str) -> str:
    word = word.lower()
    word = word.replace("aa", "a").replace("ee", "i").replace("oo", "u")
    word = word.replace("y", "i")
    return word


def _nucleus(word: str) -> str | None:
    """Dominant vowel sound of *word*: full CMU phoneme of its last *primary*-
    stressed vowel (falling back to its last vowel if none is primary-stressed),
    else the last spelled vowel letter. Returns a phoneme/letter key or None.

    Uses the full ARPAbet vowel symbol (e.g. "EY", "AA", "AH", "AO", "AW"), not
    just its first letter -- collapsing to one letter would merge phonetically
    distinct vowels (father/but/off/how would all become "a") and create
    false-positive assonance on ordinary text.
    """
    rom = devanagari_to_roman(word) if is_hindi_word(word) else word
    rom_clean = re.sub(r"[^a-zA-Z]", "", rom)
    if not rom_clean:
        return None

    phones = pronouncing.phones_for_word(rom_clean.lower())
    if phones:
        vowel_phones = [p for p in phones[0].split() if any(c.isdigit() for c in p)]
        if vowel_phones:
            primary = [p for p in vowel_phones if p.endswith("1")]
            chosen = primary[-1] if primary else vowel_phones[-1]
            return re.sub(r"\d", "", chosen).lower()

    spelled = [c for c in _normalize_hinglish_vowels(rom_clean) if c in _VOWELS]
    return spelled[-1] if spelled else None


def calculate(lyrics: str, debug: bool = False):
    """
    Returns (assonance_score 0-100, list of detected vowel-repetition groups).
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

    vowel_occurrences: dict[str, list[tuple[int, str]]] = {}
    for line_idx, words in line_entries:
        for w in words:
            v = _nucleus(w)
            if v:
                vowel_occurrences.setdefault(v, []).append((line_idx, w))

    window = cfg["WINDOW_SIZE_LINES"]
    details: list[str] = []
    total_weight = 0.0

    for v, occurrences in vowel_occurrences.items():
        for cluster in cluster_by_line_proximity(occurrences, window):
            distinct = {w.lower() for _, w in cluster}
            if len(distinct) < cfg["MIN_WORDS_PER_GROUP"]:
                continue
            details.append(f"{v.upper()}: " + " - ".join(sorted(distinct)))
            total_weight += min(len(distinct) - 2, cfg["MAX_GROUP_WEIGHT"])

    density = total_weight / valid_lines
    score = scoring_config.evaluate_piecewise_curve(
        density, cfg["ASSONANCE_THRESHOLDS"], cfg["ASSONANCE_SCORES"]
    )
    if debug:
        return round(score, 2), details, density
    return round(score, 2), details
