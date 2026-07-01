"""
Language detection and script-aware text utilities.

Hindi (Devanagari) Unicode block: U+0900 – U+097F
"""
from __future__ import annotations

import re

# Devanagari Unicode range
_DEVA_START = 0x0900
_DEVA_END   = 0x097F


def is_devanagari_char(ch: str) -> bool:
    return _DEVA_START <= ord(ch) <= _DEVA_END


def is_hindi_word(word: str) -> bool:
    """True if any character in *word* is Devanagari."""
    return any(is_devanagari_char(c) for c in word)


def detect_language(text: str) -> str:
    """
    Classify the dominant script in *text*.

    Returns:
        'hi'    — >60 % of words are Devanagari
        'mixed' — 20–60 % Devanagari (code-switching / Hinglish)
        'en'    — <20 % Devanagari
    """
    words = text.split()
    if not words:
        return "en"
    hindi_count = sum(1 for w in words if is_hindi_word(w))
    ratio = hindi_count / len(words)
    if ratio > 0.6:
        return "hi"
    if ratio > 0.2:
        return "mixed"
    return "en"


def clean_word(word: str) -> str:
    """
    Strip punctuation from a word while preserving Devanagari characters.
    Lowercases Latin characters only (Devanagari has no case).
    """
    # Keep word characters + Devanagari Unicode block
    cleaned = re.sub(r"[^\w\u0900-\u097F]", "", word).strip()
    if not is_hindi_word(cleaned):
        cleaned = cleaned.lower()
    return cleaned


def content_lines(lyrics: str) -> list[str]:
    """Return non-empty, non-header lyric lines."""
    result = []
    for line in lyrics.strip().split("\n"):
        stripped = line.strip()
        if stripped and not (stripped.startswith("[") and stripped.endswith("]")):
            result.append(stripped)
    return result
