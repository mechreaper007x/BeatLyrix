"""
Alliteration detection — phonetic, Hindi, and English aware.

Features:
  1. Uses CMU Pronouncing Dictionary to resolve English words to starting phoneme.
  2. Normalizes Hinglish/transliterated consonant clusters (e.g. kh/k -> k).
  3. Excludes stop words (except phonetically-salient overrides, see
     sound_device_utils.PHONETIC_CONTENT_OVERRIDES) and repeated identical words.
  4. Clusters onset-sound occurrences across the whole song within a line-proximity
     window (config.scoring_config.ALLITERATION["WINDOW_SIZE_LINES"]), not just
     within a single line -- a repeated hook spread over consecutive lines is
     alliteration too, and was previously invisible to this detector.
"""
from __future__ import annotations

import re
import logging

import pronouncing

from services.language_utils import content_lines, get_multilingual_stopwords
from services.sound_device_utils import sound_stopwords, cluster_by_line_proximity
from config import scoring_config

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
        'ळ': 'l', 'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy',
        # Independent (word-initial) vowel letters -- previously missing, so a
        # vowel-initial Devanagari word (e.g. "आया") could never be recognized
        # as alliterating with its Romanized/English counterpart (e.g. "aaya",
        # or an English CMU vowel-onset word), unlike every consonant above.
        # Long/short pairs collapsed the same way rhyme_service.normalize_hinglish
        # does (aa->a, ee->i, oo->u) so both scripts land on one shared symbol.
        'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u',
        'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    }
    if snd in deva_consonants:
        return deva_consonants[snd]
    return snd


def _clean(word: str) -> str:
    # Preserve Devanagari chars during punctuation stripping
    return re.sub(r"[^\w\u0900-\u097F]", "", word).strip()


def calculate(lyrics: str, debug: bool = False):
    """
    Returns (alliteration_score 0-100, list of detected alliterative groups).
    With debug=True, returns (score, details, raw_density) where raw_density
    is the pre-curve density (before the DENSITY_NORM divide and the
    piecewise-curve mapping) -- used by corpus/calibrate.py to fit curve
    constants from the corpus's actual empirical distribution instead of the
    already-curved 0-100 score.

    Onset-sound occurrences are collected across the whole song, then
    clustered by line proximity (config.ALLITERATION["WINDOW_SIZE_LINES"]) so
    a repeated hook spread over several lines is detected, not just words
    sharing a sound within a single line.
    """
    cfg = scoring_config.ALLITERATION
    lines = content_lines(lyrics)
    stop_words = sound_stopwords(get_multilingual_stopwords())

    window = cfg["WINDOW_SIZE_LINES"]

    # One entry per line with >=1 surviving content word; a line can
    # contribute to a cross-line cluster even if it can't self-fire alone.
    #
    # `is_recurrence` flags a line whose content-word text exactly repeats an
    # earlier line's, more than `window` lines after that earlier line -- a
    # structural chorus/hook block recurring later in the song, as opposed to
    # that same block's own internal repeats (which sit inside the window and
    # are already credited once via MAX_GROUP_WEIGHT below). Without this, a
    # hook repeated verbatim in two separate chorus sections re-earns the
    # per-cluster cap independently each time it recurs -- e.g. "Dekh kaun
    # aaya wapas" printed as two separate chorus blocks lets one 4-word hook
    # dominate a song's density via structural repetition, not phonetic
    # craft. Recurrent lines still count toward valid_lines_count (they're
    # still real lines occupying the song) but contribute no new sound
    # occurrences.
    line_entries: list[tuple[int, list[str], bool]] = []
    last_seen_at: dict[str, int] = {}
    for line_idx, line in enumerate(lines):
        words = [_clean(w) for w in line.split()]
        filtered = [w for w in words if w and len(w) >= cfg["MIN_WORD_LEN"] and w.lower() not in stop_words]
        if not filtered:
            continue

        key = " ".join(w.lower() for w in filtered)
        prev_idx = last_seen_at.get(key)
        is_recurrence = prev_idx is not None and (line_idx - prev_idx) > window
        if not is_recurrence:
            last_seen_at[key] = line_idx

        line_entries.append((line_idx, filtered, is_recurrence))

    valid_lines_count = len(line_entries)
    if valid_lines_count == 0:
        return (0.0, [], 0.0) if debug else (0.0, [])

    sound_occurrences: dict[str, list[tuple[int, str]]] = {}
    for line_idx, filtered, is_recurrence in line_entries:
        if is_recurrence:
            continue
        for w in filtered:
            snd = _first_sound(w)
            if snd:
                snd = _normalize_hinglish_sound(snd)
                sound_occurrences.setdefault(snd, []).append((line_idx, w))
    allit_details: list[str] = []
    total_weight = 0.0

    for snd, occurrences in sound_occurrences.items():
        for cluster in cluster_by_line_proximity(occurrences, window):
            words = [w for _, w in cluster]
            if len(words) < cfg["MIN_OCCURRENCES_PER_GROUP"]:
                continue

            # Dedupe near-identical inflections of the same root (e.g.
            # kar/karke/karta/karna) so one verb repeated three ways doesn't
            # masquerade as three distinct alliterating word choices -- but
            # don't truncate so hard that unrelated words collide (first 5
            # chars, only for words long enough that a shared prefix means
            # something).
            unique_words = {w.lower()[:5] if len(w) > 5 else w.lower() for w in words}

            if len(unique_words) >= 2:
                allit_details.append(f"{snd.upper()}: " + " - ".join(words))
            else:
                allit_details.append(f"{snd.upper()} (repeated): {words[0].lower()}")

            # Variety component: distinct words beyond the first, full weight
            # -- variety is the more skilled device. Repetition component:
            # occurrences beyond one-per-word, weighted down -- but still
            # credited, since a word repeated close together (mid-line or
            # across lines, e.g. "talve laal ... talve laal") is alliteration
            # too, not just line-initial hooks.
            variety_weight = len(unique_words) - 1
            repetition_weight = (len(words) - len(unique_words)) * cfg["REPEATED_WORD_WEIGHT_SCALE"]
            total_weight += min(variety_weight + repetition_weight, cfg["MAX_GROUP_WEIGHT"])

    raw_density = total_weight / valid_lines_count
    density = raw_density / cfg["DENSITY_NORM"]
    score = scoring_config.evaluate_piecewise_curve(density, cfg["THRESHOLDS"], cfg["SCORES"])
    if debug:
        return round(score, 2), allit_details, raw_density
    return round(score, 2), allit_details
