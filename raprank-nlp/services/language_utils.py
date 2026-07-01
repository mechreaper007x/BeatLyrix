"""
Language detection and script-aware text utilities.

Hindi (Devanagari) Unicode block: U+0900 – U+097F
"""
from __future__ import annotations

import re
import nltk
from nltk.corpus import stopwords

# Devanagari Unicode range
_DEVA_START = 0x0900
_DEVA_END   = 0x097F

_STOP_WORDS_CACHE: set[str] | None = None


def get_multilingual_stopwords() -> set[str]:
    """
    Dynamically loads NLTK English and Devanagari Hindi stopwords,
    transliterates Devanagari stopwords to Roman Hinglish,
    creates spelling variations, and returns the merged cache.
    """
    global _STOP_WORDS_CACHE
    if _STOP_WORDS_CACHE is not None:
        return _STOP_WORDS_CACHE

    try:
        nltk.download('stopwords', quiet=True)
        en_stops = set(stopwords.words('english'))
        hi_stops_deva = set(stopwords.words('hindi'))
    except Exception:
        en_stops = set()
        hi_stops_deva = set()

    hi_stops_roman = set()
    for word in hi_stops_deva:
        roman = devanagari_to_roman(word)
        if roman:
            hi_stops_roman.add(roman)
            # Add phonetic spelling variations: aa -> a, ee -> i, oo -> u
            normalized = roman.replace("aa", "a").replace("ee", "i").replace("oo", "u")
            hi_stops_roman.add(normalized)
            # Add 'h' endings: to -> toh, wo -> woh
            if roman.endswith("o"):
                hi_stops_roman.add(roman + "h")
            if normalized.endswith("o"):
                hi_stops_roman.add(normalized + "h")

    # Merge English, Devanagari, and Romanized Hinglish sets
    _STOP_WORDS_CACHE = en_stops.union(hi_stops_deva).union(hi_stops_roman)
    return _STOP_WORDS_CACHE


def is_devanagari_char(ch: str) -> bool:
    return _DEVA_START <= ord(ch) <= _DEVA_END


def is_hindi_word(word: str) -> bool:
    """True if any character in *word* is Devanagari."""
    return any(is_devanagari_char(c) for c in word)


def detect_language(text: str) -> str:
    """
    Classify the dominant script/language in *text*.

    Returns:
        'hi'    — >15 % of words are Devanagari or Romanized Hinglish filler words
        'en'    — otherwise
    """
    words = [w.lower().strip(",.?!()[]{}*\"'") for w in text.split()]
    if not words:
        return "en"
    
    # Devanagari words
    devanagari_count = sum(1 for w in words if is_hindi_word(w))
    
    # Romanised Hinglish stop words from dynamic set
    stops = get_multilingual_stopwords()
    romanized_hinglish_count = sum(1 for w in words if w in stops and not is_hindi_word(w))
    
    total_indicative = devanagari_count + romanized_hinglish_count
    ratio = total_indicative / len(words)
    
    if ratio > 0.15:
        return "hi"
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


def devanagari_to_roman(word: str) -> str:
    """
    Transliterate Devanagari (Hindi) words to consistent Romanized Latin text.
    Handles vowels, consonants, dependent vowel matras, and half-letters (virama).
    """
    consonants = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
        'ळ': 'l', 'क़': 'q', 'ख़': 'kh', 'ग़': 'g', 'ज़': 'z', 'फ़': 'f', 'ड़': 'd', 'ढ़': 'dh',
    }
    vowels = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
        'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    }
    matras = {
        'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
        'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ः': 'h',
    }
    
    res = []
    i = 0
    n = len(word)
    while i < n:
        char = word[i]
        if char in consonants:
            base = consonants[char]
            if i + 1 < n and word[i+1] == '्': # virama (half letter)
                res.append(base)
                i += 2
            elif i + 1 < n and word[i+1] in matras:
                matra = matras[word[i+1]]
                res.append(base + matra)
                i += 2
            else:
                # Schwa deletion for final consonants
                if i + 1 == n:
                    res.append(base)
                else:
                    res.append(base + 'a')
                i += 1
        elif char in vowels:
            res.append(vowels[char])
            i += 1
        elif char in matras:
            res.append(matras[char])
            i += 1
        else:
            res.append(char)
            i += 1
            
    return "".join(res)

