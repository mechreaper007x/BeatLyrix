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

    en_stops = set()
    try:
        nltk.download('stopwords', quiet=True)
        en_stops = set(stopwords.words('english'))
    except Exception:
        pass
    
    if not en_stops:
        en_stops = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
            "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
            "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        }

    # Add common English conversational words, fillers, and modal verbs to prevent false-positive puns
    en_stops.update({
        "uh", "yuh", "yeah", "yo", "like", "go", "get", "got", "do", "did", "does", "make", "made", "take", "took", 
        "come", "came", "see", "saw", "know", "say", "said", "tell", "told", "give", "gave", "find", "found", 
        "think", "thought", "look", "looked", "want", "wanted", "put", "let", "us", "would", "could", "should", 
        "will", "can", "may", "might", "must", "shall"
    })

    hi_stops_deva = set()
    try:
        hi_stops_deva = set(stopwords.words('hindi'))
    except Exception:
        pass
        
    if not hi_stops_deva:
        hi_stops_deva = {
            "अंदर", "अत", "अदि", "अप", "अपना", "अपने", "अपनी", "अब", "अभी", "अलबत्ता", "अस", "अस्त", "अह", "आदि", "आप", 
            "इत्यादि", "इन", "इन्हीं", "इन्हें", "इन्हों", "इस", "इसी", "इसे", "उन", "उन्हीं", "उन्हें", "उन्हों", "उस", 
            "उसी", "उसे", "एक", "एवं", "एस", "ऐसे", "ऐसे ही", "ओर", "और", "कइ", "कई", "कर", "करता", "करते", "करना", 
            "करने", "करें", "कहते", "कहा", "का", "काफ़ी", "कि", "किन", "किन्होंने", "किन्हें", "किन्हों", "किया", "किर", 
            "किस", "किसने", "किसे", "की", "के", "केवल", "को", "कोइ", "कोई", "कोन", "कोनसा", "कौन", "कौनसा", "गया", "घर", 
            "जब", "जहाँ", "जा", "जाता", "जाते", "जाना", "जाने", "जो", "तो", "था", "थी", "थे", "दबारा", "दिया", "दुसरा", 
            "दूसरे", "दो", "द्वारा", "न", "नही", "नहीं", "ना", "ने", "पर", "पहले", "पूरा", "पे", "फिर", "बनी", "बही", 
            "बहुत", "बाद", "बाला", "बिलकुल", "भी", "भितर", "मगर", "मानो", "मे", "में", "यदि", "यह", "यहाँ", "यही", "या", 
            "यिह", "ये", "रखें", "रहा", "रहे", "रही", "लरका", "लोग", "लोगों", "व", "वरन", "वर्ग", "वह", "वहाँ", "वही", 
            "वाले", "वुह", "वे", "सकता", "सकते", "सकती", "सबसे", "सभ", "सभी", "समय", "समान", "तरह", "सा", "सामने", 
            "साल", "साभ", "सारे", "से", "सो", "संग", "ही", "हुआ", "हुए", "हुई", "है", "हैं", "हो", "होता", "होते", "होती", 
            "होना", "होने", "तू", "तुम", "तेरा", "तुम्हारी", "तुम्हारे", "तुम्हारा", "मेरा", "मेरे", "मेरी", "मुझे", "मुझको"
        }

    # Add extra Hindi verbs / grammar particles to Devanagari stop words
    hi_stops_deva.update({
        "करके", "करना", "करता", "करती", "करते", "करो", "करें", "करेंगे", "करूँगा", "कहा", "कहे", "कह", "कहना", 
        "रहना", "रहा", "रही", "रहे", "रहो", "रहें", "रहेंगे", "रहूँगा", "जाना", "गया", "गई", "गए", "जा", "जाओ", "जाएँ",
        "आना", "आया", "आई", "आए", "आ", "आओ", "आएँ", "देना", "दिया", "दी", "दिए", "दे", "दो", "दें", "लेना", "लिया",
        "ली", "लिए", "ले", "लो", "लें"
    })

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

    # Manually append common romanized / Hinglish pronouns, conjunctions and particles
    extra_romanized = {
        "mein", "main", "maine", "mujhe", "mujhko", "mujh", "mera", "mere", "meri", "tu", "tujhe", "tujhko", "tujh", "tera", "tere", "teri",
        "tum", "tumhe", "tumhara", "tumhare", "tumhari", "apna", "apne", "apni", "aap", "aapka", "aapke", "aapki",
        "woh", "usne", "usse", "uska", "uske", "uski", "us", "unhone", "unhe", "unka", "unke", "unki", "yeh", "isne", "isse",
        "iska", "iske", "iski", "is", "inhone", "inhe", "inka", "inke", "inki", "ko", "se", "ka", "ke", "ki", "par", "pe",
        "ne", "bhi", "hi", "to", "toh", "aur", "ya", "tha", "thi", "the", "hai", "hain", "hoon", "ho", "hoge", "hogi",
        "hoga", "kar", "karna", "karta", "karti", "karte", "kiya", "kiye", "kiyi", "de", "dena", "diya", "diye",
        "le", "lena", "liya", "liye", "ja", "jana", "gaya", "gayi", "gaye", "aa", "aana", "aaya", "aayi", "aaye",
        "jis", "kis"
    }

    # Merge English, Devanagari, and Romanized Hinglish sets
    _STOP_WORDS_CACHE = en_stops.union(hi_stops_deva).union(hi_stops_roman).union(extra_romanized)
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
    """Return non-empty, non-header lyric lines, stripping parenthetical ad-libs."""
    result = []
    for line in lyrics.strip().split("\n"):
        stripped = line.strip()
        if stripped and not (stripped.startswith("[") and stripped.endswith("]")):
            # Strip out parenthetical ad-libs (e.g. "(Woo)", "(Yeah)", "(Brr)")
            cleaned_line = re.sub(r"\s*\(.*?\)\s*", " ", stripped).strip()
            if cleaned_line:
                result.append(cleaned_line)
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

