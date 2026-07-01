"""
Alliteration detection — phonetic, Hindi, and English aware.

Features:
  1. Uses CMU Pronouncing Dictionary to resolve English words to starting phoneme.
  2. Normalizes Hinglish/transliterated consonant clusters (e.g. kh/k -> k).
  3. Excludes stop words and repeated identical words.
  4. Requires at least 3 distinct words in the same line sharing the same sound.
"""
from __future__ import annotations

import re
import logging

import pronouncing

from services.language_utils import is_hindi_word, clean_word, content_lines, devanagari_to_roman
from services.vocabulary_service import _STOP_WORDS

logger = logging.getLogger(__name__)


def _first_sound(word: str) -> str | None:
    """
    Return the base sound character or phoneme for the start of the word.
    Handles English pronunciation lookup and Devanagari transliteration.
    """
    word = word.strip().lower()
    if not word:
        return None

    # If Hindi Devanagari, convert to Roman first to unify spelling
    if is_hindi_word(word):
        word = devanagari_to_roman(word).lower()

    # Clean non-alphanumeric at start
    word = re.sub(r"^[^a-z0-9]+", "", word)
    if not word:
        return None

    # Try English pronouncing dict lookup
    phones = pronouncing.phones_for_word(word)
    if phones:
        first_phone = phones[0].split()[0]
        # Strip numbers (stress markings on vowels)
        return re.sub(r"\d+", "", first_phone)

    # For Hinglish/transliterated words, take the first 1 or 2 characters
    if len(word) >= 2 and word.startswith(("kh", "gh", "ch", "jh", "th", "dh", "ph", "bh", "sh")):
        return word[:2]

    ch = word[0]
    return ch if ch.isalpha() else None


def _normalize_hinglish_sound(sound: str) -> str:
    """Normalize aspirated and transliterated consonant clusters to a base phoneme."""
    mapping = {
        "kh": "k",
        "gh": "g",
        "ch": "c",
        "jh": "j",
        "th": "t",
        "dh": "d",
        "ph": "p",
        "bh": "b",
        "sh": "s"
    }
    return mapping.get(sound, sound)


def _clean(word: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F]", "", word).strip()


def calculate(lyrics: str) -> tuple[float, list[str]]:
    """
    Returns (alliteration_score 0-100, list of detected alliterative groups).
    """
    lines = content_lines(lyrics)
    allit_details: list[str] = []
    total_allit_weight = 0.0
    valid_lines_count = 0

    for line in lines:
        words = [_clean(w) for w in line.split()]
        # Filter out empty words, stop words, and single-letter words
        words = [w for w in words if w and len(w) > 1 and w.lower() not in _STOP_WORDS]
        if len(words) < 3:
            continue

        valid_lines_count += 1
        line_sounds = []
        for w in words:
            snd = _first_sound(w)
            if snd:
                snd = _normalize_hinglish_sound(snd)
            line_sounds.append(snd)

        # Count sound occurrences in the line
        sound_counts: dict[str, list[int]] = {}
        for idx, snd in enumerate(line_sounds):
            if snd:
                sound_counts.setdefault(snd, []).append(idx)

        line_matches = []
        for snd, idxs in sound_counts.items():
            # Require at least 3 words in the line to share the sound
            if len(idxs) >= 3:
                # Exclude lines that are just the same word repeated
                unique_words = set(words[idx].lower() for idx in idxs)
                if len(unique_words) >= 3:
                    matched_words = [words[idx] for idx in idxs]
                    match_key = f"{snd.upper()}: " + " - ".join(matched_words)
                    line_matches.append(match_key)
                    # Weight proportional to length of chain
                    line_weight = len(idxs) - 2
                    total_allit_weight += min(line_weight, 3.0)  # cap weight per line

        if line_matches:
            allit_details.extend(line_matches)

    if valid_lines_count == 0:
        return 0.0, []

    density = total_allit_weight / valid_lines_count

    # Calibrate: since we require 3+ words, any density > 0.05 (1 chain every 20 lines) is good.
    # 0.15 (1 chain every 6 lines) is elite.
    if density == 0.0:
        score = 0.0
    elif density < 0.05:
        score = (density / 0.05) * 40.0
    elif density < 0.15:
        score = 40.0 + ((density - 0.05) / 0.10) * 45.0  # up to 85
    else:
        score = min(85.0 + ((density - 0.15) / 0.15) * 15.0, 100.0)

    return round(score, 2), allit_details
