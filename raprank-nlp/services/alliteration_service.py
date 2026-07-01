"""
Alliteration detection — Hindi and English aware.

Alliteration = two or more consecutive words in the same line
that start with the same phonetic unit:
  • English/Roman: first Latin character (lowercased)
  • Devanagari:    first Devanagari consonant/vowel character

Score is based on alliterative pair density across all content lines.
"""
from __future__ import annotations

import re

from services.language_utils import is_hindi_word, is_devanagari_char, content_lines


def _first_sound(word: str) -> str | None:
    """
    Return the sound-unit that determines alliteration for *word*.

    For Hindi (Devanagari), returns the first Devanagari character.
    For English/Roman, returns the first Latin letter (lowercase).
    Returns None if the word is empty or purely numeric.
    """
    word = word.strip()
    if not word:
        return None

    if is_hindi_word(word):
        for ch in word:
            if is_devanagari_char(ch):
                return ch
        return None
    else:
        # Latin: first alphabetic character
        for ch in word.lower():
            if ch.isalpha():
                return ch
        return None


def _clean(word: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F]", "", word).strip()


def calculate(lyrics: str) -> tuple[float, list[str]]:
    """
    Returns (alliteration_score 0-100, list of alliterative pair strings).
    """
    lines = content_lines(lyrics)
    pairs: list[str] = []

    for line in lines:
        words = [_clean(w) for w in line.split()]
        words = [w for w in words if w]

        for i in range(len(words) - 1):
            s1 = _first_sound(words[i])
            s2 = _first_sound(words[i + 1])
            if s1 and s2 and s1 == s2:
                pairs.append(f"{words[i]} – {words[i + 1]}")

    valid_lines = len(lines)
    if valid_lines == 0:
        return 0.0, []

    density = len(pairs) / valid_lines

    if density == 0.0:
        score = 0.0
    elif density < 0.1:
        score = (density / 0.1) * 40.0
    elif density < 0.2:
        score = 40.0 + ((density - 0.1) / 0.1) * 25.0
    elif density < 0.3:
        score = 65.0 + ((density - 0.2) / 0.1) * 15.0
    elif density < 0.5:
        score = 80.0 + ((density - 0.3) / 0.2) * 12.0
    else:
        score = min(92.0 + (density - 0.5) * 10.0, 100.0)

    return round(score, 2), pairs
