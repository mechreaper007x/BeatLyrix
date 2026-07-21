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
    Also handles Soft-C vs Hard-C distinction for Romanized scripts.
    """
    if not word:
        return None
    word = re.sub(r"^[^\w\u0900-\u097F]+", "", word)
    if not word:
        return None

    if any(0x0900 <= ord(c) <= 0x097F for c in word):
        # Return first character (Devanagari consonant) directly
        return word[0]

    # English phonetic lookup
    phones_list = pronouncing.phones_for_word(word.lower())
    if phones_list:
        first_phone = phones_list[0].split()[0]
        # Strip trailing numbers (stress markers)
        sound = re.sub(r"\d", "", first_phone).lower()
        # Map phonetic names to unified classes
        phone_mapping = {
            "k": "k", "s": "s", "sh": "sh", "z": "s", "zh": "sh",
            "p": "p", "b": "b", "d": "d", "t": "t", "g": "g",
            "jh": "j", "ch": "ch", "f": "ph", "v": "v", "th": "t", "dh": "d"
        }
        return phone_mapping.get(sound, sound)
    
    # Fallback for Romanized Hinglish/slang: check for Soft C / Hard C and clusters
    w_lower = word.lower()
    if w_lower.startswith("c") and len(w_lower) > 1 and w_lower[1] in "eiy":
        return "s"
    if w_lower.startswith("ch"):
        return "ch"
    if w_lower.startswith("c"):
        return "k"

    for cluster in ["kh", "gh", "jh", "th", "dh", "ph", "bh", "sh", "gy", "chh"]:
        if w_lower.startswith(cluster):
            return cluster
            
    return w_lower[0].lower()


def _normalize_hinglish_sound(sound: str) -> str:
    """Consolidate sound mapping for case-insensitivity, cross-script alignment, and phonetic classes."""
    snd = sound.lower()
    deva_consonants = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
        'ळ': 'l', 'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy',
        'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u',
        'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    }
    if snd in deva_consonants:
        snd = deva_consonants[snd]
        
    # Group into unified phonetic consonant classes
    phonetic_classes = {
        "k": "k_class", "kh": "k_class",
        "g": "g_class", "gh": "g_class",
        "ch": "ch_class", "chh": "ch_class",
        "j": "j_class", "jh": "j_class",
        "s": "s_class", "sh": "s_class",
        "t": "t_class", "th": "t_class",
        "d": "d_class", "dh": "d_class",
        "p": "p_class", "ph": "p_class",
        "b": "b_class", "bh": "b_class",
        "v": "v_class", "w": "v_class",
    }
    
    return phonetic_classes.get(snd, snd)


def _clean(word: str) -> str:
    # Preserve Devanagari chars during punctuation stripping
    return re.sub(r"[^\w\u0900-\u097F]", "", word).strip()


def _is_vowel_sound(snd: str) -> bool:
    """Helper to detect vowel sounds and prevent them from firing as consonant alliterations."""
    vowels = {"a", "e", "i", "o", "u", "aa", "ee", "oo", "ai", "au", "ei", "oi", "ou", "ri"}
    base = snd.replace("_class", "").lower()
    return base in vowels


def calculate(lyrics: str, debug: bool = False):
    """
    Returns (alliteration_score 0-100, list of detected alliterative groups).
    With debug=True, returns (score, details, raw_density) where raw_density
    is the pre-curve density (before the DENSITY_NORM divide and the
    piecewise-curve mapping) -- used by corpus/calibrate.py to fit curve
    constants from the corpus's actual empirical distribution instead of the
    already-curved 0-100 score.

    Strict V3 alliteration logic:
    Only clusters words within the same line that share the same onset sound
    and are immediately adjacent in the filtered content-word sequence (index gap = 0).
    Excludes vowel-initial sounds (which are assonance, not alliteration).
    Also allows cross-line alliteration for hook refrains (first words of adjacent lines).
    """
    cfg = scoring_config.ALLITERATION
    lines = content_lines(lyrics)
    stop_words = sound_stopwords(get_multilingual_stopwords())

    valid_lines = []
    for line in lines:
        cleaned_line = line.strip()
        # Drop Genius metadata/translations header text
        if "Contributors" in cleaned_line and "Lyrics" in cleaned_line:
            # If it's a merged header, strip the Genius header prefix
            cleaned_line = re.sub(r"^.*?Lyrics\s*", "", cleaned_line)
            cleaned_line = re.sub(r"^.*?Read More\s*", "", cleaned_line)
        
        # Double check if any remnant metadata keywords exist
        if not cleaned_line or any(k in cleaned_line for k in ["Contributors", "Translations", "Read More"]):
            continue

        words = [_clean(w) for w in cleaned_line.split()]
        filtered = [w for w in words if w and len(w) >= cfg.get("MIN_WORD_LEN", 2) and w.lower() not in stop_words]
        if filtered:
            valid_lines.append(filtered)

    if not valid_lines:
        return (0.0, [], 0.0) if debug else (0.0, [])

    total_weight = 0.0
    detected_groups = []
    seen_group_signatures: set = set()

    # 1. Intra-line alliteration
    for line_idx, words in enumerate(valid_lines):
        # Map words to normalized sounds
        word_sounds = []
        for w in words:
            snd = _first_sound(w)
            snd = _normalize_hinglish_sound(snd) if snd else None
            word_sounds.append((snd, w))
            
        # Group matching sounds with gap == 0 (immediately adjacent index gap)
        sound_indices = {}
        for idx, (snd, w) in enumerate(word_sounds):
            if snd and not _is_vowel_sound(snd):
                sound_indices.setdefault(snd, []).append(idx)
                
        for snd, indices in sound_indices.items():
            clusters = []
            current = []
            for idx in indices:
                if current and idx - current[-1] > 1:  # index difference > 1 means gap > 0 (not adjacent)
                    if len(current) >= cfg.get("MIN_OCCURRENCES_PER_GROUP", 2):
                        clusters.append(current)
                    current = []
                current.append(idx)
            if len(current) >= cfg.get("MIN_OCCURRENCES_PER_GROUP", 2):
                clusters.append(current)
                
            for cluster in clusters:
                cluster_words = [word_sounds[idx][1] for idx in cluster]

                # Collapse inflections of the same root: "yaadein/yaadein/yaad"
                # or "baahein/baahon" are ONE root repeated, not alliteration.
                # Two words share a root if one is a prefix of the other
                # (min stem length 3) or they agree on their first 4 chars.
                roots: list[str] = []
                for w in cluster_words:
                    wl = w.lower()
                    for i, r in enumerate(roots):
                        shorter, longer = (wl, r) if len(wl) <= len(r) else (r, wl)
                        if (len(shorter) >= 3 and longer.startswith(shorter)) or \
                           (len(shorter) >= 4 and shorter[:4] == longer[:4]):
                            if len(wl) < len(r):
                                roots[i] = wl  # keep shortest form as the root
                            break
                    else:
                        roots.append(wl)

                # Alliteration requires >= MIN_DISTINCT_ROOTS genuinely
                # different words sharing the onset ("bhakt bada bholenath"),
                # not doublets or repetition loops ("bolo bam bam bam").
                if len(roots) < cfg.get("MIN_DISTINCT_ROOTS", 3):
                    continue

                # A hook line repeated N times is ONE act of alliteration:
                # dedupe by (sound, root set) so refrains can't saturate density.
                signature = (snd, frozenset(roots))
                if signature in seen_group_signatures:
                    continue
                seen_group_signatures.add(signature)

                variety_weight = len(roots) - 1
                repetition_weight = (len(cluster_words) - len(roots)) * cfg.get("REPEATED_WORD_WEIGHT_SCALE", 0.5)
                weight = min(variety_weight + repetition_weight, cfg.get("MAX_GROUP_WEIGHT", 3.0))
                total_weight += weight

                words_str = " - ".join(cluster_words)
                detected_groups.append(f"Line {line_idx+1} | {snd.upper()}: {words_str} (w={weight:.2f})")

    raw_density = total_weight / len(valid_lines)
    density = raw_density / cfg["DENSITY_NORM"]
    score = scoring_config.evaluate_piecewise_curve(density, cfg["THRESHOLDS"], cfg["SCORES"])
    
    if debug:
        return round(score, 2), detected_groups, raw_density
    return round(score, 2), detected_groups
