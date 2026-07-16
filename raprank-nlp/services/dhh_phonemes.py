# -*- coding: utf-8 -*-
"""
DHH phoneme engine -- from-scratch Hindi/Hinglish grapheme-to-phoneme with
rhyme keys, built for rap lyrics (see docs/dhh_dictionary/PLAN.md).

Why this exists: English rhyme detection rides on CMUdict phonemes, but the
Hindi/Hinglish side keyed rhymes off SPELLING (rhyme_service.normalize_hinglish
+ last-N-Devanagari-chars), which cannot know that घर is pronounced "ghar"
(never "ghara") or that hoon/hun are the same word. This module gives both
scripts a real phoneme layer; rhyme keys are derived from pronunciation.

Design choices (all driven by the gold set in
services/tests/data/dhh_rhyme_gold.py):
- Compact phoneme set, only fine enough to distinguish rhymes -- not IPA.
  The 'a' vs 'aa' length contrast is KEPT (kal/kaal don't rhyme); the i/ii
  and u/uu contrasts are MERGED because corpus Latin spelling can't encode
  them reliably (hoon/hun) and no rhyme judgment in the gold set needs them.
- Retroflex consonants map onto dentals: Latin spelling writes both as t/d,
  so keeping the contrast would make cross-script matching impossible.
- Anusvara / chandrabindu -> the consonant 'n' in all positions. This makes
  ज़िंदगी == zindagi and हूँ == hoon by construction, and vowel-nasality
  (जहाँ vs जहान) is not a distinction rap rhyme relies on.
- Schwa deletion implemented directly (final always when droppable; medial
  in VCaCV contexts, right-to-left) -- the single biggest correctness win
  over the old spelling keyer. Geminates collapse AFTER schwa deletion so
  the cluster correctly blocks medial deletion (mohabbat keeps its 'a').
- Romanized Hinglish goes roman -> phonemes DIRECTLY (not via a Devanagari
  round-trip): corpus spellings are too noisy for faithful back-
  transliteration, but they map fine onto the phoneme set itself. Where a
  common corpus spelling under-writes vowel length (pyar for pyaar), a small
  variant table canonicalizes it -- grown over time by Phase 4's coverage
  report, exactly as the plan's "small disambiguation table" prescribes.

Public API:
    deva_to_phones(word)      Devanagari -> (phonemes,)
    hinglish_to_phones(word)  romanized  -> (phonemes,)
    to_phones(word)           routes by script
    rhyme_key(phones)         final rhyme unit            (single-syllable)
    multi_rhyme_key(phones)   from 2nd-to-last nucleus    (multisyllabic)
"""
from __future__ import annotations

import re
from functools import lru_cache

# ── Phoneme inventory ────────────────────────────────────────────────────────
# Vowels:     a aa i u e ai o au         (i/ii and u/uu merged; see header)
# Consonants: k kh g gh c ch j jh t th d dh n p ph b bh m
#             y r l w sh s h f z ng ny

_VOWELS = {"a", "aa", "i", "u", "e", "ai", "o", "au"}


def _is_vowel(p: str) -> bool:
    return p in _VOWELS


# ── Devanagari tables ────────────────────────────────────────────────────────

_DEVA_INDEP_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ॲ": "a", "ऑ": "o",
}

_DEVA_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "i", "ु": "u",
    "ू": "u", "ृ": "ri", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "a",
}

# Retroflex -> dental and nukta -> nearest roman-representable sound: Latin
# spelling can't encode these contrasts, so the phoneme layer doesn't either.
_DEVA_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "w",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # nukta forms
    "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "dh", "फ़": "f",
    "ळ": "l", "य़": "y",
}

_VIRAMA = "्"
_ANUSVARA = "ं"
_CHANDRABINDU = "ँ"
_NUKTA = "़"
_VISARGA = "ः"

_NUKTA_MAP = {"क": "k", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f"}


def _deva_syllabify(word: str) -> list[str]:
    """Devanagari -> phonemes with inherent schwas still present. Every bare
    consonant is followed by 'a' (the inherent vowel) unless killed by a
    virama or replaced by a matra."""
    phones: list[str] = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in _DEVA_CONSONANTS:
            if i + 1 < len(chars) and chars[i + 1] == _NUKTA:
                phones.append(_NUKTA_MAP.get(ch, _DEVA_CONSONANTS[ch]))
                i += 2
            else:
                phones.append(_DEVA_CONSONANTS[ch])
                i += 1
            if i < len(chars) and chars[i] == _VIRAMA:
                i += 1  # conjunct: no vowel
            elif i < len(chars) and chars[i] in _DEVA_MATRAS:
                phones.append(_DEVA_MATRAS[chars[i]])
                i += 1
            else:
                phones.append("a")  # inherent schwa (deleted later)
        elif ch in _DEVA_INDEP_VOWELS:
            phones.append(_DEVA_INDEP_VOWELS[ch])
            i += 1
        elif ch in (_ANUSVARA, _CHANDRABINDU):
            # Homorganic nasal before a consonant (ज़िंदगी = zin-da-gi) and
            # plain 'n' word-finally (मैं = main, हूँ = hoon): always 'n',
            # matching how the corpus romanizes nasality.
            phones.append("n")
            i += 1
        elif ch == _VISARGA:
            phones.append("h")
            i += 1
        else:
            i += 1  # ZWJ/ZWNJ, danda, digits, stray marks...
    return phones


def _delete_schwas(phones: list[str]) -> list[str]:
    """Hindi schwa deletion on the flat phoneme list. Final inherent schwa
    drops whenever the word can stand without it; medial schwas drop
    right-to-left in VC_CV contexts (sapanaa -> sapnaa). Geminates are still
    doubled at this point, so clusters correctly block deletion."""
    if not phones:
        return phones
    out = list(phones)

    # Final inherent schwa (घर syllabifies to gh-a-r-a; the trailing 'a' goes)
    if (
        len(out) >= 3
        and out[-1] == "a"
        and not _is_vowel(out[-2])
        and any(_is_vowel(p) for p in out[:-2])
    ):
        out = out[:-1]

    # Medial: delete 'a' in VOWEL CONS _ CONS VOWEL, right to left
    i = len(out) - 3
    while i >= 2:
        if (
            out[i] == "a"
            and not _is_vowel(out[i - 1])
            and _is_vowel(out[i - 2])
            and not _is_vowel(out[i + 1])
            and i + 2 < len(out)
            and _is_vowel(out[i + 2])
        ):
            del out[i]
            i -= 1
        i -= 1
    return out


def _canonicalize(phones: list[str]) -> list[str]:
    """Collapse what neither script side needs: geminate consonants
    (mohabbat -> mohabat, izzat -> izat) and repeated identical vowels."""
    out: list[str] = []
    for p in phones:
        if out and p == out[-1]:
            continue
        out.append(p)
    return out


@lru_cache(maxsize=8192)
def deva_to_phones(word: str) -> tuple[str, ...]:
    """Devanagari word -> phoneme tuple (schwa deletion applied)."""
    word = word.strip()
    if not word:
        return ()
    return tuple(_canonicalize(_delete_schwas(_deva_syllabify(word))))


# ── Romanized Hinglish -> phonemes ───────────────────────────────────────────

# Corpus spellings that under-write vowel length or use ad-hoc vowels; each
# maps to the canonical romanization whose tokenization matches the
# Devanagari pronunciation. Grown by Phase 4's coverage report.
_ROM_VARIANTS = {
    "pyar": "pyaar",
    "yar": "yaar",
    "sham": "shaam",
    "ankh": "aankh",
    "pani": "paani",
    "kahani": "kahaani",
    "jawani": "jawaani",
    "jahan": "jahaan",
    "yahan": "yahaan",
    "wahan": "wahaan",
    "kitab": "kitaab",
    "khwab": "khwaab",
    "jindagi": "zindagi",
    "najar": "nazar",
    "sheher": "shahar",
    "shehar": "shahar",
    "leher": "lahar",
    # 'deewana' spells its -waa- syllable short; canonicalize all variants
    "deewana": "deewaana",
    "diwana": "deewaana",
    "deewani": "deewaani",
    "diwani": "deewaani",
}

# Longest-match tokenizer units.
_ROM_UNITS = [
    ("chh", ("ch",)),
    ("kh", ("kh",)), ("gh", ("gh",)), ("jh", ("jh",)),
    ("th", ("th",)), ("dh", ("dh",)), ("ph", ("ph",)), ("bh", ("bh",)),
    ("sh", ("sh",)), ("ch", ("c",)),
    ("aa", ("aa",)), ("ee", ("i",)), ("oo", ("u",)),
    ("ai", ("ai",)), ("au", ("au",)), ("ei", ("e",)), ("ou", ("au",)),
    ("a", ("a",)), ("b", ("b",)), ("c", ("k",)), ("d", ("d",)), ("e", ("e",)),
    ("f", ("f",)), ("g", ("g",)), ("h", ("h",)), ("i", ("i",)), ("j", ("j",)),
    ("k", ("k",)), ("l", ("l",)), ("m", ("m",)), ("n", ("n",)), ("o", ("o",)),
    ("p", ("p",)), ("q", ("k",)), ("r", ("r",)), ("s", ("s",)), ("t", ("t",)),
    ("u", ("u",)), ("v", ("w",)), ("w", ("w",)), ("x", ("k", "s")), ("y", ("y",)),
    ("z", ("z",)),
]
_ROM_UNITS.sort(key=lambda kv: -len(kv[0]))


def _rom_tokenize(word: str) -> list[str]:
    phones: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        for unit, mapped in _ROM_UNITS:
            if word.startswith(unit, i):
                phones.extend(mapped)
                i += len(unit)
                break
        else:
            i += 1  # unknown char: skip
    return phones


@lru_cache(maxsize=8192)
def hinglish_to_phones(word: str) -> tuple[str, ...]:
    """Romanized Hinglish word -> phoneme tuple, collapsing spelling variants
    onto the Devanagari-derived pronunciation."""
    word = word.strip().lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return ()
    word = _ROM_VARIANTS.get(word, word)
    phones = _canonicalize(_rom_tokenize(word))
    # A pronounced word-final 'a' is always the long vowel: Hindi deletes true
    # final schwas, so "sona"/"apna" end in aa exactly like सोना/अपना.
    if phones and phones[-1] == "a":
        phones[-1] = "aa"
    return tuple(phones)


_DEVA_RE = re.compile(r"[ऀ-ॿ]")


def to_phones(word: str) -> tuple[str, ...]:
    """Route a word to the right phonemizer by script."""
    if _DEVA_RE.search(word):
        return deva_to_phones(word)
    return hinglish_to_phones(word)


# ── Rhyme keys ───────────────────────────────────────────────────────────────

# Aspiration is not rhyme-distinctive (haath/saath rhyme with laat): strip it
# in keys. sh stays (a real fricative, not aspirated s); c/ch merge.
_KEY_CONS = {
    "kh": "k", "gh": "g", "jh": "j", "th": "t", "dh": "d",
    "ph": "p", "bh": "b", "ch": "c",
}


def _key_norm(p: str) -> str:
    return p if _is_vowel(p) else _KEY_CONS.get(p, p)


def _nuclei_indices(phones) -> list[int]:
    return [i for i, p in enumerate(phones) if _is_vowel(p)]


def rhyme_key(phones):
    """The final rhyme unit, normalized.

    Consonant-final words rhyme on the last nucleus + coda (ghar/dar -> a r).
    Vowel-final words rhyme on the last TWO nuclei onward (sona/khona ->
    o n aa) because a bare final vowel is too weak a unit in Hindi -- this is
    how listeners actually judge sona/sena as a non-rhyme."""
    if not phones:
        return None
    idxs = _nuclei_indices(phones)
    if not idxs:
        return None
    if _is_vowel(phones[-1]):
        if len(idxs) >= 2:
            start = idxs[-2]
        else:
            start = max(0, idxs[-1] - 1)  # lone nucleus: include the onset
    else:
        start = idxs[-1]
    return tuple(_key_norm(p) for p in phones[start:])


def multi_rhyme_key(phones, n: int = 2):
    """From the n-th-to-last nucleus onward, normalized; None if fewer than n
    nuclei or the span is too short to be a real multisyllabic unit (mirrors
    the English key's minimum-span rule; two adjacent nuclei like -aai count)."""
    if not phones:
        return None
    idxs = _nuclei_indices(phones)
    if len(idxs) < n:
        return None
    tail = phones[idxs[-n]:]
    if len(tail) < 3 and not all(_is_vowel(p) for p in tail):
        return None
    return tuple(_key_norm(p) for p in tail)
