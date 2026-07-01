"""
End-rhyme and multi-syllabic rhyme detection.

English  → CMU Pronouncing Dictionary (via `pronouncing`)
Hindi    → Devanagari suffix matching (last 3 chars)
Mixed    → per-word dispatch

Rhyme scheme window: checks up to 4 lines ahead (handles AABB, ABAB, ABCB, AAAA).
"""
from __future__ import annotations

from typing import Optional

import pronouncing

from models.schemas import RhymeMatch
from services.language_utils import is_hindi_word, clean_word, content_lines


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
    """Last 3 Devanagari characters form the rhyme suffix."""
    if len(word) < 2:
        return None
    return word[-3:] if len(word) >= 3 else word[-2:]


def _multi_rhyme_key_hi(word: str) -> Optional[str]:
    """Last 5 characters for multi-syllabic Hindi rhyme."""
    return word[-5:] if len(word) >= 5 else None


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
            
    if not vowel_idxs:
        return None
        
    if len(vowel_idxs) >= 2:
        start_idx = vowel_idxs[-2]
    else:
        start_idx = vowel_idxs[-1]
        
    return word[start_idx:]


# ── Last-word extraction ───────────────────────────────────────────────────────

def _last_word(line: str) -> Optional[str]:
    tokens = [clean_word(w) for w in line.strip().split()]
    tokens = [t for t in tokens if t]
    return tokens[-1] if tokens else None


# ── Main calculation ───────────────────────────────────────────────────────────

def calculate(lyrics: str) -> tuple[float, list[RhymeMatch], int]:
    """
    Detect end rhymes between lines within a 4-line window.

    Returns:
        rhyme_score      (0-100)
        rhyme_pairs      list of RhymeMatch
        multisyl_count   number of multi-syllabic rhyme pairs
    """
    lines = content_lines(lyrics)
    if len(lines) < 2:
        return 0.0, [], 0

    indexed_words = [(i, _last_word(lines[i])) for i in range(len(lines))]
    indexed_words = [(i, w) for i, w in indexed_words if w]

    rhyme_pairs: list[RhymeMatch] = []
    seen: set[tuple[int, int]] = set()

    for a in range(len(indexed_words)):
        idx_a, word_a = indexed_words[a]
        for b in range(a + 1, min(a + 5, len(indexed_words))):
            idx_b, word_b = indexed_words[b]
            if word_a == word_b:
                continue  # identical words don't count as rhymes

            hi_a, hi_b = is_hindi_word(word_a), is_hindi_word(word_b)

            if hi_a or hi_b:
                # Use Devanagari suffix matching
                key_a = _rhyme_key_hi(word_a)
                key_b = _rhyme_key_hi(word_b)
                multi_a = _multi_rhyme_key_hi(word_a)
                multi_b = _multi_rhyme_key_hi(word_b)
                is_multi = bool(
                    multi_a and multi_b and multi_a == multi_b
                    and multi_a != key_a  # truly longer match
                )
            else:
                key_a = _rhyme_key_en(word_a)
                key_b = _rhyme_key_en(word_b)
                multi_a = _multi_rhyme_key_en(word_a)
                multi_b = _multi_rhyme_key_en(word_b)
                
                is_hinglish_a = (key_a is None)
                is_hinglish_b = (key_b is None)
                
                if is_hinglish_a or is_hinglish_b:
                    key_a = _rhyme_key_hinglish(word_a)
                    key_b = _rhyme_key_hinglish(word_b)
                    
                    def count_vowels_hinglish(suffix: Optional[str]) -> int:
                        if not suffix:
                            return 0
                        vowels = "aeiouy"
                        count = 0
                        i = 0
                        while i < len(suffix):
                            if suffix[i] in vowels:
                                count += 1
                                while i < len(suffix) and suffix[i] in vowels:
                                    i += 1
                            else:
                                i += 1
                        return count
                        
                    is_multi = bool(
                        key_a and key_b and key_a == key_b and count_vowels_hinglish(key_a) >= 2
                    )
                else:
                    is_multi = bool(
                        multi_a and multi_b and multi_a == multi_b
                        and len(multi_a) > 2
                    )

            if key_a and key_b and key_a == key_b:
                pair = (idx_a, idx_b)
                if pair not in seen:
                    seen.add(pair)
                    rhyme_pairs.append(
                        RhymeMatch(
                            line_a=idx_a,
                            line_b=idx_b,
                            word_a=word_a,
                            word_b=word_b,
                            is_multisyllabic=is_multi,
                        )
                    )

    multisyl_count = sum(1 for r in rhyme_pairs if r.is_multisyllabic)

    # Rhyme density: matched pairs / possible adjacent pairs
    rhyme_ratio = len(rhyme_pairs) / max(len(lines) - 1, 1)

    if rhyme_ratio == 0:
        score = 0.0
    elif rhyme_ratio < 0.2:
        score = (rhyme_ratio / 0.2) * 40.0
    elif rhyme_ratio < 0.4:
        score = 40.0 + ((rhyme_ratio - 0.2) / 0.2) * 25.0
    elif rhyme_ratio < 0.6:
        score = 65.0 + ((rhyme_ratio - 0.4) / 0.2) * 20.0
    elif rhyme_ratio < 0.8:
        score = 85.0 + ((rhyme_ratio - 0.6) / 0.2) * 10.0
    else:
        score = min(95.0 + (rhyme_ratio - 0.8) * 25.0, 100.0)

    # Multi-syllabic bonus (up to +10)
    if multisyl_count:
        score = min(score + min(multisyl_count * 2.0, 10.0), 100.0)

    return round(score, 2), rhyme_pairs, multisyl_count
