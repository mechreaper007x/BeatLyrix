"""
Syllable density and syllable weight scoring.

English  → pyphen (Hunspell hyphenation dictionary for en_US)
Hindi    → Devanagari matra / consonant nucleus counting
Mixed    → per-word dispatch
"""
from __future__ import annotations

import pyphen
import pronouncing

from services.language_utils import is_hindi_word, clean_word, content_lines, get_multilingual_stopwords
from config import scoring_config

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


# Combining signs that modify a sound but do NOT add a vowel nucleus. Previously
# mis-counted as consonants (each adding a phantom syllable):
#   U+0901 chandrabindu, U+0902 anusvara, U+0903 visarga (nasalisation/aspiration),
#   U+093C nukta (consonant variant marker), U+093D avagraha.
_DEVA_NON_SYLLABIC = {"ँ", "ं", "ः", "़", "ऽ"}


def _count_hindi(word: str) -> int:
    """
    Each vowel nucleus = 1 syllable.
    Independent vowel OR matra    → +1
    Consonant NOT followed by halant or matra → +1 (inherent 'a' is present)
    Implements Hindi schwa deletion: final consonants do not count as a new syllable if the word has other syllables.

    Combining signs (nukta, anusvara, chandrabindu, visarga, avagraha) are
    stripped first: they never introduce a new vowel nucleus, and treating them
    as consonants inflated counts (e.g. zindagi 5->3, zameen 4->2).
    """
    chars = [c for c in word if c not in _DEVA_NON_SYLLABIC]
    
    # Find the index of the last Devanagari character that is a consonant
    last_consonant_idx = -1
    for idx, ch in enumerate(chars):
        if "\u0900" <= ch <= "\u0939" or "\u0958" <= ch <= "\u095F":
            last_consonant_idx = idx

    count = 0
    for i, ch in enumerate(chars):
        if ch in _DEVA_VOWELS or ch in _DEVA_MATRAS:
            count += 1
        elif "\u0900" <= ch <= "\u0939" or "\u0958" <= ch <= "\u095F":
            # Schwa deletion on final consonant of a word of length > 1
            if i == last_consonant_idx and len(word) > 1:
                continue
                
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt != _HALANT and nxt not in _DEVA_MATRAS:
                count += 1
    return max(count, 1)


def _count_vowel_groups(w_lower: str) -> int:
    """Count vowel-nucleus groups, treating intervocalic/consonant-context 'y'
    as a vowel and deducting a silent trailing 'e'."""
    vowels = set("aeiou")
    vowel_groups = 0
    in_vowel = False
    for idx, char in enumerate(w_lower):
        is_y_vowel = False
        if char == "y":
            prev_is_vowel = (idx > 0 and w_lower[idx - 1] in vowels)
            next_is_vowel = (idx + 1 < len(w_lower) and w_lower[idx + 1] in vowels)
            if not prev_is_vowel and not next_is_vowel:
                is_y_vowel = True
        if char in vowels or is_y_vowel:
            if not in_vowel:
                vowel_groups += 1
                in_vowel = True
        else:
            in_vowel = False
    # Deduct 1 for silent trailing 'e' (but not 'le' endings like 'table')
    if w_lower.endswith("e") and len(w_lower) > 3 and w_lower[-2] not in vowels:
        if not w_lower.endswith("le"):
            vowel_groups = max(vowel_groups - 1, 1)
    return vowel_groups


def _count_english(word: str) -> int:
    """Syllable count for a Latin-script word.

    Real English words: trust the CMU pronouncing dictionary (counts vowel
    phonemes), which correctly models elision — 'every'->2, 'business'->2 — that
    letter-based methods miss.

    Words absent from CMU are almost always Romanized Hinglish/slang. There,
    pyphen (en_US hyphenation) silently UNDER-counts (e.g. 'zindagi'->2, should
    be 3), so we take the max of pyphen and vowel-group counting, which rarely
    over-counts transliterated Hindi.
    """
    w_lower = word.lower()
    phones = pronouncing.phones_for_word(w_lower)
    if phones:
        return max(pronouncing.syllable_count(phones[0]), 1)

    pyphen_count = _dic_en.inserted(w_lower).count("-") + 1
    vowel_groups = _count_vowel_groups(w_lower)
    return max(pyphen_count, vowel_groups, 1)


def count_syllables(word: str) -> int:
    """Dispatch to Hindi or English counter based on script."""
    return _count_hindi(word) if is_hindi_word(word) else _count_english(word)


def calculate(lyrics: str) -> tuple[float, float, float, float]:
    """
    Returns:
       syllable_score        (0-100)
       avg_syllables_per_line
       syllable_weight_score (0-100)
       syllable_weight_ratio
    """
    total_syllables = 0
    total_lines = 0
    
    total_words = 0
    complex_words = 0

    min_words = scoring_config.SYLLABLE["MIN_WORDS_FOR_DENSITY"]
    complex_threshold = scoring_config.SYLLABLE["COMPLEX_WORD_SYLLABLES"]
    stop_words = get_multilingual_stopwords()

    for line in content_lines(lyrics):
        line_words = [clean_word(w) for w in line.split()]
        line_words = [w for w in line_words if w]
        
        # EXCLUDE ad-lib / short lines from syllable density calculations
        if len(line_words) >= min_words:
            line_syllables = 0
            for w in line_words:
                syl_count = count_syllables(w)
                line_syllables += syl_count
                
                # Syllable Weight: ratio of complex words to simple words, excluding stop words
                if w.lower() not in stop_words:
                    total_words += 1
                    if syl_count >= complex_threshold:
                        complex_words += 1
                        
            total_syllables += line_syllables
            total_lines += 1

    if total_lines == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_syllables_per_line = total_syllables / total_lines

    # Dynamic Syllable Density Score (0-100)
    score = scoring_config.evaluate_piecewise_curve(
        avg_syllables_per_line,
        scoring_config.SYLLABLE["DENSITY_THRESHOLDS"],
        scoring_config.SYLLABLE["DENSITY_SCORES"]
    )

    # Dynamic Syllable Weight Score (0-100)
    weight_ratio = complex_words / max(total_words, 1)
    syllable_weight_score = scoring_config.evaluate_piecewise_curve(
        weight_ratio,
        scoring_config.SYLLABLE["WEIGHT_THRESHOLDS"],
        scoring_config.SYLLABLE["WEIGHT_SCORES"]
    )

    return round(score, 2), round(avg_syllables_per_line, 2), round(syllable_weight_score, 2), round(weight_ratio, 4)
