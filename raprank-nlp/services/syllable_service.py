"""
Syllable density scoring.

English  → pyphen (Hunspell hyphenation dictionary for en_US)
Hindi    → Devanagari matra / consonant nucleus counting
Mixed    → per-word dispatch
"""
from __future__ import annotations

import pyphen

from services.language_utils import is_hindi_word, clean_word, content_lines

# ── English hyphenation dictionary ───────────────────────────────────────────
_dic_en = pyphen.Pyphen(lang="en_US")

# ── Devanagari Unicode codepoints ─────────────────────────────────────────────
# Independent vowels
_DEVA_VOWELS = set(
    "\u0905\u0906\u0907\u0908\u0909\u090A\u090B"   # a ā i ī u ū ṛ
    "\u090E\u090F\u0910"                             # short e, e, ai
    "\u0912\u0913\u0914"                             # short o, o, au
)
# Dependent vowel matras (attached to consonants)
_DEVA_MATRAS = set(
    "\u093E\u093F\u0940"   # ā i ī
    "\u0941\u0942\u0943"   # u ū ṛ
    "\u0946\u0947\u0948"   # short e, e, ai
    "\u094A\u094B\u094C"   # short o, o, au
)
_HALANT = "\u094D"  # virama — cancels inherent vowel of preceding consonant


def _count_hindi(word: str) -> int:
    """
    Each vowel nucleus = 1 syllable.
    Independent vowel OR matra    → +1
    Consonant NOT followed by halant → +1 (inherent 'a' is present)
    """
    chars = list(word)
    count = 0
    for i, ch in enumerate(chars):
        if ch in _DEVA_VOWELS or ch in _DEVA_MATRAS:
            count += 1
        elif "\u0900" <= ch <= "\u0939" or "\u0958" <= ch <= "\u095F":
            # It's a consonant — count if NOT followed by halant
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt != _HALANT:
                count += 1
    return max(count, 1)


def _count_english(word: str) -> int:
    hyphenated = _dic_en.inserted(word)
    return max(hyphenated.count("-") + 1, 1)


def count_syllables(word: str) -> int:
    """Dispatch to Hindi or English counter based on script."""
    return _count_hindi(word) if is_hindi_word(word) else _count_english(word)


def calculate(lyrics: str) -> tuple[float, float]:
    """
    Returns (syllable_score 0-100, avg_syllables_per_word).
    """
    total_syllables = 0
    total_words = 0

    for line in content_lines(lyrics):
        for raw_word in line.split():
            word = clean_word(raw_word)
            if not word:
                continue
            total_syllables += count_syllables(word)
            total_words += 1

    if total_words == 0:
        return 0.0, 0.0

    avg = total_syllables / total_words

    # Piecewise linear scoring curve
    if avg < 1.1:
        score = 30.0
    elif avg < 1.5:
        score = 30.0 + ((avg - 1.1) / 0.4) * 20.0
    elif avg < 2.0:
        score = 50.0 + ((avg - 1.5) / 0.5) * 20.0
    elif avg < 2.5:
        score = 70.0 + ((avg - 2.0) / 0.5) * 15.0
    elif avg < 3.0:
        score = 85.0 + ((avg - 2.5) / 0.5) * 10.0
    else:
        score = min(95.0 + (avg - 3.0) * 5.0, 100.0)

    return round(score, 2), round(avg, 2)
