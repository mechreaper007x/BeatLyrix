"""
End-rhyme, internal rhyme, and multi-syllabic rhyme chain detection.

English  → CMU Pronouncing Dictionary (via `pronouncing`)
Hindi    → Devanagari suffix matching (last 3 chars)
Mixed    → per-word dispatch

Rhyme scheme window: checks up to 4 lines ahead (handles AABB, ABAB, ABCB, AAAA).
"""
from __future__ import annotations

from typing import Optional, List, Tuple

import pronouncing
import re

from models.schemas import RhymeMatch
from services.language_utils import is_hindi_word, clean_word, content_lines, devanagari_to_roman
from config import scoring_config


# ── English helpers ───────────────────────────────────────────────────────────

def _phones(word: str) -> Optional[list[str]]:
    """Return the first CMU phoneme sequence for *word*, or None."""
    result = pronouncing.phones_for_word(word.lower())
    return result[0].split() if result else None


def _rhyme_key_en(word: str) -> Optional[tuple]:
    """
    Rhyme key = phonemes from the last stressed vowel onward.
    e.g. 'nation' → ('EY1', 'SH', 'AH0', 'N')
    """
    phones = _phones(word)
    if not phones:
        return None
    last_v = max(
        (i for i, p in enumerate(phones) if any(c.isdigit() for c in p)),
        default=-1,
    )
    if last_v == -1:
        return None
    return tuple(phones[last_v:])


def _multi_rhyme_key_en(word: str, n: int = 2) -> Optional[tuple]:
    """
    Multi-syllabic rhyme key = phonemes from the *n*-th-to-last stressed vowel.
    """
    phones = _phones(word)
    if not phones:
        return None
    vowel_idxs = [i for i, p in enumerate(phones) if any(c.isdigit() for c in p)]
    if len(vowel_idxs) < n:
        return None
    key = tuple(phones[vowel_idxs[-n]:])
    # Must span at least 3 phonemes to count as multi-syllabic
    return key if len(key) >= 3 else None


# ── Hindi helpers ─────────────────────────────────────────────────────────────

def _rhyme_key_hi(word: str) -> Optional[str]:
    """Last N Devanagari characters from config form the rhyme suffix."""
    length = scoring_config.RHYME["DEVA_SUFFIX_LENGTH"]
    if len(word) < 2:
        return None
    return word[-length:] if len(word) >= length else word[-2:]


def _multi_rhyme_key_hi(word: str) -> Optional[str]:
    """Last N characters from config for multi-syllabic Hindi rhyme."""
    length = scoring_config.RHYME["DEVA_MULTI_SUFFIX_LENGTH"]
    return word[-length:] if len(word) >= length else None


# ── Hinglish helpers ──────────────────────────────────────────────────────────

def normalize_hinglish(word: str) -> str:
    word = word.lower()
    word = word.replace("aa", "a")
    word = word.replace("ee", "i")
    word = word.replace("oo", "u")
    word = word.replace("bh", "b")
    word = word.replace("dh", "d")
    word = word.replace("kh", "k")
    word = word.replace("gh", "g")
    word = word.replace("ph", "f")
    word = word.replace("th", "t")
    word = word.replace("ch", "c")
    word = word.replace("sh", "s")
    return word


def _rhyme_key_hinglish(word: str) -> Optional[str]:
    word = normalize_hinglish(word)
    if len(word) < 1:
        return None
    
    vowels = "aeiouy"
    vowel_idxs = []
    i = 0
    while i < len(word):
        if word[i] in vowels:
            vowel_idxs.append(i)
            while i < len(word) and word[i] in vowels:
                i += 1
        else:
            i += 1
            
    if not vowel_idxs:
        return None
        
    start_idx = vowel_idxs[-1]
    
    # If the word ends in a vowel group (e.g. "bade", "uske"), include the preceding consonant
    if start_idx == len(word) - 1 or (start_idx < len(word) - 1 and all(c in vowels for c in word[start_idx:])):
        consonant_idx = start_idx - 1
        while consonant_idx >= 0 and word[consonant_idx] in vowels:
            consonant_idx -= 1
        if consonant_idx >= 0:
            return word[consonant_idx:]
            
    return word[start_idx:]


def _multi_rhyme_key_hinglish(word: str) -> Optional[str]:
    word = normalize_hinglish(word)
    if len(word) < 2:
        return None
    
    vowels = "aeiouy"
    vowel_idxs = []
    i = 0
    while i < len(word):
        if word[i] in vowels:
            vowel_idxs.append(i)
            while i < len(word) and word[i] in vowels:
                i += 1
        else:
            i += 1
            
    if len(vowel_idxs) < 2:
        return None
        
    start_idx = vowel_idxs[-2]
    
    # If the word ends in a vowel group, include the preceding consonant
    if vowel_idxs[-1] == len(word) - 1 or (vowel_idxs[-1] < len(word) - 1 and all(c in vowels for c in word[vowel_idxs[-1]:])):
        consonant_idx = start_idx - 1
        while consonant_idx >= 0 and word[consonant_idx] in vowels:
            consonant_idx -= 1
        if consonant_idx >= 0:
            return word[consonant_idx:]
            
    return word[start_idx:]


def _normalize_hinglish_vowels(word: str) -> str:
    word = word.lower()
    word = word.replace("aa", "a")
    word = word.replace("ee", "i")
    word = word.replace("oo", "u")
    word = word.replace("y", "i")
    return word


def _get_line_vowels(line: str) -> list[str]:
    words = [clean_word(w) for w in line.split()]
    words = [w for w in words if w]
    if not words:
        return []
    last_words = words[-2:] if len(words) >= 2 else words[-1:]
    
    line_vowels = []
    for w in last_words:
        rom = devanagari_to_roman(w) if is_hindi_word(w) else w
        rom_clean = re.sub(r"[^\w]", "", rom)
        
        phones = pronouncing.phones_for_word(rom_clean.lower())
        if phones:
            vowel_phones = []
            for p in phones[0].split():
                if any(c.isdigit() for c in p):
                    vowel_phones.append(re.sub(r"\d", "", p).lower())
            line_vowels.extend(vowel_phones)
        else:
            norm = _normalize_hinglish_vowels(rom_clean)
            spelling_vowels = [c for c in norm if c in "aeiou"]
            line_vowels.extend(spelling_vowels)
            
    return line_vowels


def _get_word_vowels(word: str) -> list[str]:
    rom = devanagari_to_roman(word) if is_hindi_word(word) else word
    rom_clean = re.sub(r"[^\w]", "", rom)
    
    phones = pronouncing.phones_for_word(rom_clean.lower())
    vowels = []
    if phones:
        for p in phones[0].split():
            if any(c.isdigit() for c in p):
                vowels.append(re.sub(r"\d", "", p).lower())
    else:
        norm = _normalize_hinglish_vowels(rom_clean)
        vowels = [c for c in norm if c in "aeiou"]
    return vowels


def _get_longest_common_vowel_suffix(v1: list[str], v2: list[str]) -> list[str]:
    suffix = []
    for a, b in zip(reversed(v1), reversed(v2)):
        if a == b:
            suffix.append(a)
        else:
            break
    return list(reversed(suffix))


# ── Word extraction ───────────────────────────────────────────────────────

def _last_word(line: str) -> Optional[str]:
    tokens = [clean_word(w) for w in line.strip().split()]
    tokens = [t for t in tokens if t]
    return tokens[-1] if tokens else None


def _get_rhyme_key(word: str) -> Optional[tuple | str]:
    """Helper to get a rhyme key for any word (English or Hindi)."""
    rom = devanagari_to_roman(word) if is_hindi_word(word) else word
    key = _rhyme_key_en(rom)
    if key is None:
        key = _rhyme_key_hinglish(rom)
    return key


def _get_multi_rhyme_key(word: str) -> Optional[tuple | str]:
    """Helper to get a multisyllabic rhyme key."""
    rom = devanagari_to_roman(word) if is_hindi_word(word) else word
    key = _multi_rhyme_key_en(rom)
    if key is None:
        key = _multi_rhyme_key_hinglish(rom)
    return key


def detect_internal_rhymes(lines: List[str]) -> Tuple[int, List[str]]:
    """
    Detect rhyming words inside the same line.
    Returns (internal_rhyme_count, list of rhyming word pairs).
    """
    pairs = []
    for line in lines:
        words = [clean_word(w) for w in line.split()]
        # Skip stop words and very short words
        words = [w for w in words if w and len(w) > 2]
        
        # Check all pairs in the same line
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                if w1.lower() == w2.lower():
                    continue # identical words are not rhymes
                    
                key1 = _get_rhyme_key(w1)
                key2 = _get_rhyme_key(w2)
                
                if key1 and key2 and key1 == key2:
                    pair_str = f"{w1} – {w2}"
                    if pair_str not in pairs:
                        pairs.append(pair_str)
                        
    return len(pairs), pairs


def detect_chain_rhymes(indexed_words: List[Tuple[int, str]]) -> Tuple[int, List[List[Tuple[int, str]]]]:
    """
    Detect sequences of 3 or more consecutive lines that carry the same end-rhyme sound.
    Returns (chain_lines_count, list of chains).
    """
    if not indexed_words:
        return 0, []

    # Map each line's last word to its rhyme key
    keys = []
    for idx, word in indexed_words:
        key = _get_rhyme_key(word)
        keys.append((idx, key, word))

    chains: List[List[Tuple[int, str]]] = []
    current_chain = []

    for idx, key, word in keys:
        if not key:
            if len(current_chain) >= 3:
                chains.append(current_chain)
            current_chain = []
            continue

        if not current_chain:
            current_chain.append((idx, word, key))
        else:
            prev_idx, prev_word, prev_key = current_chain[-1]
            # A chain can have a gap of at most 1 line (e.g. idx - prev_idx <= 2)
            if key == prev_key and idx - prev_idx <= 2:
                current_chain.append((idx, word, key))
            else:
                if len(current_chain) >= 3:
                    chains.append(current_chain)
                current_chain = [(idx, word, key)]

    if len(current_chain) >= 3:
        chains.append(current_chain)

    # Clean chains to only return line_number and word
    formatted_chains = [[(item[0], item[1]) for item in c] for c in chains]
    chain_lines_count = sum(len(c) for c in formatted_chains)
    
    return chain_lines_count, formatted_chains


# ── Main calculation ───────────────────────────────────────────────────────────

def calculate(lyrics: str) -> tuple[float, list[RhymeMatch], int, float, float]:
    """
    Detect end rhymes, internal rhymes, multisyllabic rhymes, and chain rhymes.

    Returns:
        rhyme_score         (overall 0-100)
        rhyme_pairs         list of RhymeMatch
        multisyl_count      number of multi-syllabic rhyme pairs
        internal_score      0-100 sub-score
        chain_score         0-100 sub-score
    """
    lines = content_lines(lyrics)
    if len(lines) < 2:
        return 0.0, [], 0, 0.0, 0.0

    indexed_words = [(i, _last_word(lines[i])) for i in range(len(lines))]
    indexed_words = [(i, w) for i, w in indexed_words if w]

    # 1. End Rhymes & Multisyllabic
    rhyme_pairs: list[RhymeMatch] = []
    seen: set[tuple[int, int]] = set()

    for a in range(len(indexed_words)):
        idx_a, word_a = indexed_words[a]
        window = scoring_config.RHYME["WINDOW_SIZE_LINES"]
        for b in range(a + 1, min(a + window, len(indexed_words))):
            idx_b, word_b = indexed_words[b]
            if word_a == word_b:
                continue

            rom_a = devanagari_to_roman(word_a) if is_hindi_word(word_a) else word_a
            rom_b = devanagari_to_roman(word_b) if is_hindi_word(word_b) else word_b

            key_a = _rhyme_key_en(rom_a)
            key_b = _rhyme_key_en(rom_b)
            if key_a is None or key_b is None:
                key_a = _rhyme_key_hinglish(rom_a)
                key_b = _rhyme_key_hinglish(rom_b)

            is_end_rhyme = bool(key_a and key_b and key_a == key_b)
            
            # Robust multisyllabic rhyme detection across last 2 words of the line
            is_multi = False
            v1 = _get_line_vowels(lines[idx_a])
            v2 = _get_line_vowels(lines[idx_b])
            common_suffix = _get_longest_common_vowel_suffix(v1, v2)
            if len(common_suffix) >= 3:
                is_multi = True

            # Slant Rhyme Matcher: 2-syllable vowel pattern match for Hinglish/Hindi
            is_slant_rhyme = False
            if not is_end_rhyme and not is_multi:
                vowels_a = _get_word_vowels(word_a)
                vowels_b = _get_word_vowels(word_b)
                if len(vowels_a) >= 2 and len(vowels_b) >= 2:
                    if vowels_a[-2:] == vowels_b[-2:]:
                        is_slant_rhyme = True

            if is_end_rhyme or is_multi or is_slant_rhyme:
                pair = (idx_a, idx_b)
                if pair not in seen:
                    seen.add(pair)
                    rhyme_pairs.append(
                        RhymeMatch(
                            line_a=idx_a,
                            line_b=idx_b,
                            word_a=word_a,
                            word_b=word_b,
                            is_multisyllabic=is_multi or is_slant_rhyme,
                        )
                    )

    multisyl_count = sum(1 for r in rhyme_pairs if r.is_multisyllabic)

    # Compute End Rhyme Score (0-100)
    rhyme_ratio = len(rhyme_pairs) / max(len(lines) - 1, 1)
    end_rhyme_score = scoring_config.evaluate_piecewise_curve(
        rhyme_ratio,
        scoring_config.RHYME["CURVE_THRESHOLDS"],
        scoring_config.RHYME["CURVE_SCORES"]
    )

    # 2. Internal Rhymes
    internal_count, internal_pairs = detect_internal_rhymes(lines)
    internal_ratio = internal_count / max(len(lines), 1)
    internal_score = min((internal_ratio / scoring_config.RHYME["ELITE_TARGETS"]["internal_density"]) * 100.0, 100.0)

    # 3. Multisyllabic Rhymes Score
    multisyl_ratio = multisyl_count / max(len(lines) - 1, 1)
    multisyl_score = min((multisyl_ratio / scoring_config.RHYME["ELITE_TARGETS"]["multisyllabic_density"]) * 100.0, 100.0)

    # 4. Chain Rhymes
    chain_lines, chains = detect_chain_rhymes(indexed_words)
    chain_ratio = chain_lines / max(len(lines), 1)
    chain_score = min((chain_ratio / scoring_config.RHYME["ELITE_TARGETS"]["chain_density"]) * 100.0, 100.0)

    # Combined Rhyme Score dynamically weighted
    weights = scoring_config.RHYME["WEIGHTS"]
    rhyme_score = (
        end_rhyme_score * weights["end_rhyme"] +
        internal_score * weights["internal"] +
        multisyl_score * weights["multisyllabic"] +
        chain_score * weights["chains"]
    )

    return round(rhyme_score, 2), rhyme_pairs, multisyl_count, round(internal_score, 2), round(chain_score, 2)
