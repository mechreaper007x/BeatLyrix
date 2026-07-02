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
    if any(0x0900 <= ord(c) <= 0x097F for c in word):
        # Return first character (Devanagari consonant) directly
        return word[0]

    # English phonetic lookup
    phones_list = pronouncing.phones_for_word(word)
    if phones_list:
        # Extract the first phoneme (consonant or vowel stress)
        first_phone = phones_list[0].split()[0]
        # Strip trailing numbers (stress markers)
        return re.sub(r"\d", "", first_phone)
    
    # Fallback for Romanized Hinglish: check for common consonant clusters
    w_lower = word.lower()
    for cluster in ["kh", "gh", "ch", "jh", "th", "dh", "ph", "bh", "sh", "gy"]:
        if w_lower.startswith(cluster):
            return cluster
            
    # Fallback to character start
    return word[0].lower()


def _normalize_hinglish_sound(sound: str) -> str:
    """Consolidate sound mapping for case-insensitivity and cross-script alignment."""
    snd = sound.lower()
    # Map Devanagari starting characters to Romanized equivalents to support cross-script matching
    deva_consonants = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
        'ळ': 'l', 'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy'
    }
    if snd in deva_consonants:
        return deva_consonants[snd]
    return snd


def _clean(word: str) -> str:
    # Preserve Devanagari chars during punctuation stripping
    return re.sub(r"[^\w\u0900-\u097F]", "", word).strip()


def _get_contiguous_chains(items: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Find contiguous subchains of matching words where they are adjacent in the filtered list (gap of 1)."""
    chains = []
    current_chain = []
    for filt_idx, w in items:
        if not current_chain:
            current_chain.append((filt_idx, w))
        else:
            prev_idx, prev_w = current_chain[-1]
            if filt_idx - prev_idx == 1:
                current_chain.append((filt_idx, w))
            else:
                if len(current_chain) >= 3:
                    chains.append(current_chain)
                current_chain = [(filt_idx, w)]
    if len(current_chain) >= 3:
        chains.append(current_chain)
    return chains


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
        # Skip empty words but preserve their original indices in the line
        words_indexed = [(orig_idx, w) for orig_idx, w in enumerate(words) if w]
        if not words_indexed:
            continue
            
        # Filter out stop words and single-letter words for sound grouping
        filtered_words = [w for _, w in words_indexed if len(w) > 1 and w.lower() not in stop_words]
                
        if len(filtered_words) < 3:
            continue

        valid_lines_count += 1
        sound_groups = {}
        for filt_idx, w in enumerate(filtered_words):
            snd = _first_sound(w)
            if snd:
                snd = _normalize_hinglish_sound(snd)
                sound_groups.setdefault(snd, []).append((filt_idx, w))

        line_matches = []
        for snd, items in sound_groups.items():
            # Check for contiguous chains (adjacent in filtered list)
            chains = _get_contiguous_chains(items)
            for chain in chains:
                unique_words = set(w.lower() for _, w in chain)
                if len(unique_words) >= 3:
                    matched_words = [w for _, w in chain]
                    match_key = f"{snd.upper()}: " + " - ".join(matched_words)
                    line_matches.append(match_key)
                    line_weight = len(chain) - 2
                    total_allit_weight += min(line_weight, 3.0)

        if line_matches:
            allit_details.extend(line_matches)

    if valid_lines_count == 0:
        return 0.0, []

    density = total_allit_weight / valid_lines_count

    # Calibrate: since we require 3+ adjacent words, any density > 0.05 (1 chain every 20 lines) is good.
    # We use a continuous curve so there are no sudden jumps from 0% to 40%.
    if density < 0.05:
        score = 0.0
    elif density < 0.15:
        score = ((density - 0.05) / 0.10) * 50.0  # Scales 0% to 50%
    elif density < 0.30:
        score = 50.0 + ((density - 0.15) / 0.15) * 35.0  # Scales 50% to 85%
    else:
        score = min(85.0 + ((density - 0.30) / 0.30) * 15.0, 100.0)

    return round(score, 2), allit_details
