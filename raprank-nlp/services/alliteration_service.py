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

from services.language_utils import is_hindi_word, clean_word, content_lines, devanagari_to_roman, get_multilingual_stopwords

logger = logging.getLogger(__name__)


def _first_sound(word: str) -> str | None:
    """
    Return the base sound character or phoneme for the start of the word.
    Handles Devanagari consonants directly, falls back to CMU phonetic mapping.
    """
    if not word:
        return None
    if is_hindi_word(word):
        # Return first character (Devanagari consonant)
        return word[0]

    # English phonetic lookup
    phones_list = pronouncing.phones_for_word(word)
    if phones_list:
        # Extract the first phoneme (consonant or vowel stress)
        first_phone = phones_list[0].split()[0]
        # Strip trailing numbers (stress markers)
        return re.sub(r"\d", "", first_phone)
    
    # Fallback to character start
    return word[0].lower()


def _normalize_hinglish_sound(sound: str) -> str:
    """Consolidate Hinglish consonant sounds (e.g., kh/k -> k)."""
    # Devanagari mapping
    deva_map = {
        'ख': 'क', 'घ': 'ग', 'छ': 'च', 'झ': 'ज', 'ठ': 'ट', 'ढ': 'ड',
        'थ': 'त', 'ध': 'द', 'फ': 'प', 'भ': 'ब', 'श': 'स', 'ष': 's'
    }
    if sound in deva_map:
        return deva_map[sound]
    # English/Roman phonetic mapping (lowercased)
    snd = sound.lower()
    if snd.startswith("kh"): return "k"
    if snd.startswith("gh"): return "g"
    if snd.startswith("ch"): return "c"
    if snd.startswith("jh"): return "j"
    if snd.startswith("th"): return "t"
    if snd.startswith("dh"): return "d"
    if snd.startswith("ph"): return "p"
    if snd.startswith("bh"): return "b"
    if snd.startswith("sh"): return "s"
    return snd


def _clean(word: str) -> str:
    # Preserve Devanagari chars during punctuation stripping
    return re.sub(r"[^\w\u0900-\u097F]", "", word).strip()


def calculate(lyrics: str) -> tuple[float, list[str]]:
    """
    Returns (alliteration_score 0-100, list of detected alliterative groups).
    """
    lines = content_lines(lyrics)
    allit_details: list[str] = []
    total_allit_weight = 0.0
    valid_lines_count = 0
    stop_words = get_multilingual_stopwords()

    for line in lines:
        words = [_clean(w) for w in line.split()]
        # Filter out empty words, stop words, and single-letter words
        words = [w for w in words if w and len(w) > 1 and w.lower() not in stop_words]
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
