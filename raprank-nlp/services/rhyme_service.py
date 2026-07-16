"""
End-rhyme, internal rhyme, and multi-syllabic rhyme chain detection.

English  → CMU Pronouncing Dictionary (via `pronouncing`)
Hindi    → DHH phoneme engine (services/dhh_phonemes) — real G2P with schwa
           deletion; replaced the old Devanagari last-N-chars suffix keying
Hinglish → same DHH engine's romanized pipeline; replaced spelling heuristics
Mixed    → per-word dispatch

Rhyme scheme window: checks up to 4 lines ahead (handles AABB, ABAB, ABCB, AAAA).
"""
from __future__ import annotations

from typing import Optional, List, Tuple

import pronouncing
import re

from models.schemas import RhymeMatch
from services.language_utils import is_hindi_word, clean_word, content_lines, devanagari_to_roman
from services import dhh_phonemes
from config import scoring_config


# ── English helpers ───────────────────────────────────────────────────────────

def _phones(word: str) -> Optional[list[str]]:
    """Return the first CMU phoneme sequence for *word*, or None."""
    result = pronouncing.phones_for_word(word.lower())
    return result[0].split() if result else None


def _rhyme_key_en(word: str) -> Optional[tuple]:
    """
    Rhyme key = phonemes from the last primary/secondary stressed vowel onward.
    e.g. 'nation' → ('EY1', 'SH', 'AH0', 'N')
    """
    phones = _phones(word)
    if not phones:
        return None
    # Find the last vowel phone with primary/secondary stress (1 or 2)
    stressed = [i for i, p in enumerate(phones) if any(c in "12" for c in p)]
    if stressed:
        last_v = stressed[-1]
    else:
        # Fall back to any vowel (including stress 0)
        vowels = [i for i, p in enumerate(phones) if any(c.isdigit() for c in p)]
        last_v = vowels[-1] if vowels else -1
        
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
# Both scripts route through the DHH phoneme engine: rhyme keys come from
# PRONUNCIATION (schwa deletion, spelling-variant collapse), not spelling.
# घर/डर key as ('a','r'); hoon/hun key identically. The old last-N-chars and
# normalize_hinglish heuristics remain only for callers that need a plain
# letter-suffix (cross-script class matching below).

def _rhyme_key_hi(word: str) -> Optional[tuple]:
    """Phoneme rhyme key for a Devanagari word (nucleus+coda; vowel-final
    words extend to the previous nucleus -- see dhh_phonemes.rhyme_key)."""
    return dhh_phonemes.rhyme_key(dhh_phonemes.deva_to_phones(word))


def _multi_rhyme_key_hi(word: str) -> Optional[tuple]:
    """Phoneme multi-syllabic key (last two nuclei onward) for Devanagari."""
    return dhh_phonemes.multi_rhyme_key(dhh_phonemes.deva_to_phones(word))


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
    # De-duplicate repeated consonants (e.g., ll -> l, tt -> t, dd -> d)
    word = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+", r"\1", word)
    return word


def _rhyme_key_hinglish(word: str) -> Optional[tuple]:
    """Phoneme rhyme key for romanized Hinglish (spelling variants collapse
    inside dhh_phonemes; 'y' glide, vowel length, and schwa behaviour all
    handled at the phoneme layer instead of by string surgery here)."""
    return dhh_phonemes.rhyme_key(dhh_phonemes.hinglish_to_phones(word))


def _multi_rhyme_key_hinglish(word: str) -> Optional[tuple]:
    """Phoneme multi-syllabic key (last two nuclei onward) for Hinglish."""
    return dhh_phonemes.multi_rhyme_key(dhh_phonemes.hinglish_to_phones(word))


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


def _romanized_alpha(word: str) -> str:
    """Lowercase romanized letters only, for suffix comparison."""
    rom = devanagari_to_roman(word) if is_hindi_word(word) else word.lower()
    return re.sub(r"[^a-z]", "", rom)


def _is_trivial_suffix_rhyme(word_a: str, word_b: str) -> bool:
    """
    True when two words rhyme ONLY because they share a Hindi/Hinglish
    grammatical ending (e.g. rulaya/kamaya on -aya, sunani/jalani on -ani) while
    their stems differ. These conjugation rhymes are automatic and unskilled, so
    they must not be credited as multisyllabic or chain rhymes.
    """
    a, b = _romanized_alpha(word_a), _romanized_alpha(word_b)
    if not a or not b or a == b:
        return False
    for suf in sorted(scoring_config.RHYME["TRIVIAL_RHYME_SUFFIXES"], key=len, reverse=True):
        suf = suf.strip()
        if suf and a.endswith(suf) and b.endswith(suf):
            stem_a, stem_b = a[: -len(suf)], b[: -len(suf)]
            # Skilled rhyme would extend into the stem too (matching pre-suffix
            # consonant). If the stems' last letters differ, the rhyme is purely
            # the shared grammatical suffix -> trivial.
            if not stem_a or not stem_b or stem_a[-1] != stem_b[-1]:
                return True
    return False


def _ends_in_any_trivial_suffix(word: str) -> bool:
    """True if the word ends in any configured trivial grammatical suffix."""
    a = _romanized_alpha(word)
    if not a:
        return False
    for suf in scoring_config.RHYME["TRIVIAL_RHYME_SUFFIXES"]:
        suf = suf.strip()
        if suf and a.endswith(suf):
            return True
    return False


def _are_similar_consonants(c1: str, c2: str) -> bool:
    """True if two consonant sounds are phonetically similar/compatible for slant rhyming."""
    if c1 == c2:
        return True
    groups = [
        {"t", "d", "dh", "th", "z", "s", "sh"},
        {"p", "b", "f", "v"},
        {"k", "g", "kh", "gh"},
        {"m", "n", "ng"},
        {"r", "l"}
    ]
    for g in groups:
        if c1 in g and c2 in g:
            return True
    return False


def _get_final_consonant_sound(clean_word: str) -> str:
    """Extracts the final consonant or digraph sound from a cleaned word."""
    if len(clean_word) >= 2:
        tail2 = clean_word[-2:]
        if tail2 in {"kh", "gh", "ch", "jh", "th", "dh", "ph", "bh", "sh", "ng"}:
            return tail2
    return clean_word[-1] if clean_word else ""


_VOWEL_PHONES = {"ah", "ax", "ae", "ay", "ao", "aw", "eh", "ey", "er", "ih", "iy", "ow", "oy", "uh", "uw", "aa"}


def _is_vowel_phone(p: str) -> bool:
    p = re.sub(r"\d", "", p).lower()
    return p in _VOWEL_PHONES


def _get_phonetic_signature(word: str) -> Optional[dict]:
    rom = devanagari_to_roman(word) if is_hindi_word(word) else word
    phones = _phones(rom)
    if not phones:
        return None
    vowels = []
    for p in phones:
        if any(c.isdigit() for c in p) or re.sub(r"\d", "", p).lower() in _VOWEL_PHONES:
            vowels.append(re.sub(r"\d", "", p).lower())
    last_phone = phones[-1].lower()
    ends_vowel = _is_vowel_phone(last_phone)
    final_consonant = "" if ends_vowel else re.sub(r"\d", "", last_phone)
    return {
        "vowels": vowels,
        "ends_vowel": ends_vowel,
        "final_consonant": final_consonant
    }


def _get_spelling_signature(word: str) -> dict:
    clean = _romanized_alpha(word)
    vowels = _get_word_vowels(word)
    ends_vowel = clean[-1] in "aeiou" if clean else False
    final_consonant = "" if ends_vowel else _get_final_consonant_sound(clean)
    return {
        "vowels": vowels,
        "ends_vowel": ends_vowel,
        "final_consonant": final_consonant
    }




# ── Compound / mosaic rhyme ───────────────────────────────────────────────
# A compound rhyme is a single word's full sound rhyming with the combined
# sound of two or more words on the other side (e.g. one word vs. a two-word
# phrase where neither the phrase's last word alone, nor the whole word,
# match without the extra word's sound). Distinct from ordinary end/
# multisyllabic rhyme, which is anchored on a single last word each side.

def _last_n_words(line: str, n: int) -> List[str]:
    tokens = [clean_word(w) for w in line.strip().split()]
    tokens = [t for t in tokens if t]
    return tokens[-n:] if len(tokens) >= n else []


def _phrase_phones(words: List[str]) -> Optional[List[str]]:
    """Concatenated CMU phones (stress stripped) for an ordered word list, in
    order. None if any word has no CMU entry."""
    out: List[str] = []
    for w in words:
        rom = devanagari_to_roman(w) if is_hindi_word(w) else w
        p = pronouncing.phones_for_word(rom.lower())
        if not p:
            return None
        out.extend(re.sub(r"\d", "", ph).lower() for ph in p[0].split())
    return out


def _phrase_letters(words: List[str]) -> List[str]:
    """Concatenated normalized Hinglish letters for an ordered word list."""
    out: List[str] = []
    for w in words:
        rom = devanagari_to_roman(w) if is_hindi_word(w) else w
        letters = normalize_hinglish(re.sub(r"[^a-zA-Z]", "", rom))
        out.extend(list(letters))
    return out


def _phrase_signature(words: List[str]) -> Optional[Tuple[List[str], str]]:
    """(tokens, mode) for an ordered word list. mode is 'en' when every word
    resolves via CMU, else 'translit' (Hinglish letter fallback)."""
    if not words:
        return None
    phones = _phrase_phones(words)
    if phones is not None:
        return phones, "en"
    return _phrase_letters(words), "translit"


def _word_rhyme_tail(word: str) -> Optional[Tuple[List[str], str]]:
    """Rhyme-relevant tail of a single word: phones from its last primary-
    stressed vowel onward (English, onset dropped -- matching how ordinary
    rhyme keys work, e.g. `_rhyme_key_en`), or its Hinglish rhyme-key letters.
    """
    rom = devanagari_to_roman(word) if is_hindi_word(word) else word
    rom_clean = re.sub(r"[^a-zA-Z]", "", rom)
    if not rom_clean:
        return None

    phones = pronouncing.phones_for_word(rom_clean.lower())
    if phones:
        plist = phones[0].split()
        stressed = [i for i, p in enumerate(plist) if p.endswith("1")]
        vowel_idxs = [i for i, p in enumerate(plist) if any(c.isdigit() for c in p)]
        start = stressed[-1] if stressed else (vowel_idxs[-1] if vowel_idxs else None)
        if start is None:
            return None
        return [re.sub(r"\d", "", p).lower() for p in plist[start:]], "en"

    key = _rhyme_key_hinglish(rom_clean)
    return (list(key), "translit") if key else None


def detect_compound_rhymes(lines: List[str]) -> Tuple[int, List[str]]:
    """
    Detect compound/mosaic rhymes within the rhyme window: a line whose single
    last word's rhyme-relevant tail (from its last stressed vowel onward)
    matches the trailing sound of another line's last 2-3 words, where that
    match necessarily reaches back into the earlier word of the phrase (not
    just the phrase's own last word) -- otherwise it would just be an
    ordinary single-word end rhyme, already scored elsewhere.
    """
    window = scoring_config.RHYME["WINDOW_SIZE_LINES"]
    min_units = {"en": scoring_config.RHYME["COMPOUND_MIN_PHONES"],
                 "translit": scoring_config.RHYME["COMPOUND_MIN_LETTERS"]}

    per_line: List[dict] = []
    for line in lines:
        last1 = _last_n_words(line, 1)
        anchor = None
        full_last_word_len = 0
        if last1:
            tail = _word_rhyme_tail(last1[0])
            full_sig = _phrase_signature(last1)
            if tail and full_sig:
                anchor = (tail[0], tail[1], last1[0])
                full_last_word_len = len(full_sig[0])

        phrases: dict = {}
        for depth in (2, 3):
            words = _last_n_words(line, depth)
            if not words:
                continue
            sig = _phrase_signature(words)
            if sig:
                phrases[depth] = sig + (" ".join(words),)

        per_line.append({"anchor": anchor, "full_last_word_len": full_last_word_len,
                          "phrases": phrases})

    def _is_repeated_word_artifact(anchor_word: str, phrase_last_word: str) -> bool:
        """
        True when the "match" is just the phrase's last word literally
        reappearing inside the anchor word's own spelling (e.g. a compound
        word like "daydream" ending in the literal word "dream", matched
        against a phrase that also ends in "dream"). That is word reuse, not
        an independent phonetic coincidence across different words -- the
        same anti-inflation principle applied elsewhere to trivial-suffix and
        identity chain rhymes.
        """
        a, b = anchor_word.lower(), phrase_last_word.lower()
        return len(b) >= 3 and (a.endswith(b) or b.endswith(a))

    def _match(anchor, phrase_sig, long_last_word_len) -> bool:
        tok_s, mode_s, _ = anchor
        tok_l, mode_l, _ = phrase_sig
        if mode_s != mode_l:
            return False
        k = len(tok_s)
        if k < min_units[mode_s] or len(tok_l) < k:
            return False
        if tok_s != tok_l[-k:]:
            return False
        # must reach back beyond the long side's own last word alone
        return k > long_last_word_len

    pairs: List[str] = []
    seen: set[tuple[int, int]] = set()
    n = len(per_line)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            for a_idx, b_idx in ((i, j), (j, i)):
                a, b = per_line[a_idx], per_line[b_idx]
                if a["anchor"] is None:
                    continue
                for phrase_sig in b["phrases"].values():
                    if not _match(a["anchor"], phrase_sig, b["full_last_word_len"]):
                        continue
                    phrase_last_word = phrase_sig[2].split()[-1]
                    if _is_repeated_word_artifact(a["anchor"][2], phrase_last_word):
                        continue
                    key = (min(a_idx, b_idx), max(a_idx, b_idx))
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append(f"{a['anchor'][2]} ~ {phrase_sig[2]}")

    return len(pairs), pairs


def detect_holorimes(lines: List[str]) -> Tuple[int, List[str]]:
    """
    Detect holorime: two different multi-word phrases (same word count,
    N in {2,3}) at line-ends within the rhyme window whose FULL phonetic
    content matches end-to-end -- e.g. "ice cream" / "I scream" -- not just a
    shared tail (that's compound rhyme) or a single matched word (ordinary
    end rhyme).
    """
    window = scoring_config.RHYME["WINDOW_SIZE_LINES"]
    min_units = {"en": scoring_config.RHYME["HOLORIME_MIN_PHONES"],
                 "translit": scoring_config.RHYME["HOLORIME_MIN_LETTERS"]}

    def _normalized_word(w: str) -> str:
        """Script/spelling-invariant form of a word, for detecting when two
        phrases are really just the same underlying words (e.g. Devanagari vs.
        Roman transliteration of the same word, or "jave"/"jaave" long-vowel
        spelling variants) rather than genuinely different phrasings."""
        rom = devanagari_to_roman(w) if is_hindi_word(w) else w
        return normalize_hinglish(re.sub(r"[^a-zA-Z]", "", rom))

    def _is_same_underlying_words(words_i: List[str], words_j: List[str]) -> bool:
        if len(words_i) != len(words_j):
            return False
        return [_normalized_word(w) for w in words_i] == [_normalized_word(w) for w in words_j]

    per_line: List[dict] = []
    for line in lines:
        phrases: dict = {}
        for depth in (2, 3):
            words = _last_n_words(line, depth)
            if not words:
                continue
            sig = _phrase_signature(words)
            if sig:
                phrases[depth] = sig + (" ".join(words), words)
        per_line.append(phrases)

    pairs: List[str] = []
    seen: set[tuple[int, int]] = set()
    n = len(per_line)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            for depth, sig_i in per_line[i].items():
                sig_j = per_line[j].get(depth)
                if not sig_j:
                    continue
                tok_i, mode_i, phrase_i, words_i = sig_i
                tok_j, mode_j, phrase_j, words_j = sig_j
                if mode_i != mode_j or mode_i not in min_units:
                    continue
                if phrase_i.lower() == phrase_j.lower():
                    continue
                if len(tok_i) < min_units[mode_i] or tok_i != tok_j:
                    continue
                # Two lines transliterating/respelling the same underlying
                # words (e.g. Devanagari vs. Roman, or "jave" vs. "jaave") are
                # not a genuine holorime -- it's one phrase written two ways.
                if _is_same_underlying_words(words_i, words_j):
                    continue
                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(f"{phrase_i} ~ {phrase_j}")

    return len(pairs), pairs


# ── Cross-script (Hindi <-> English) rhyme matching ──────────────────────────
# `_rhyme_key_en` returns an ARPAbet-phoneme tuple (e.g. ('AO1', 'K')) and
# `_rhyme_key_hinglish` returns a letter-suffix string (e.g. 'aunk') -- two
# fundamentally different representations that can never be `==` equal. That
# silently drops Hindi/English code-switched rhymes, which are a hallmark
# technique in this corpus (KR$NA: "bhaunk"/"talk", "shaayar"/"desire"). Below
# is a coarser, script-agnostic key -- (vowel class, final-consonant class)
# from the last stressed/last vowel onward -- used ONLY as a fallback when one
# side is Devanagari and the other is Latin, so same-script matching keeps
# using the more precise exact keys above.

_ARPABET_VOWEL_CLASS = {
    "aa": "a", "ae": "a", "ah": "a", "ax": "a",
    "ao": "au", "aw": "au",
    "ay": "ai",
    "eh": "e", "ey": "e",
    "ih": "i", "iy": "ee",
    "ow": "o", "oy": "oi",
    "uh": "u", "uw": "oo",
}

_HINGLISH_VOWEL_CLASS = {
    "a": "a", "aa": "a", "i": "i", "ee": "ee", "u": "u", "oo": "oo",
    "e": "e", "ai": "ai", "o": "o", "au": "au",
}

_CONSONANT_CLASS_GROUPS = [
    ("t", {"t", "d", "dh", "th", "z", "s", "sh", "ch", "chh", "jh", "j"}),
    ("p", {"p", "b", "f", "v", "w"}),
    ("k", {"k", "g", "kh", "gh"}),
    ("m", {"m", "n", "ng", "ny"}),
    ("r", {"r", "l"}),
]

_CONSONANT_DIGRAPHS = {"kh", "gh", "chh", "jh", "ny", "th", "dh", "ph", "bh", "sh", "ng"}


def _consonant_class(sound: str) -> str:
    sound = sound.lower()
    for label, members in _CONSONANT_CLASS_GROUPS:
        if sound in members:
            return label
    return sound


def _final_consonant_chunk(tail: str) -> str:
    if len(tail) >= 2 and tail[-2:] in _CONSONANT_DIGRAPHS:
        return tail[-2:]
    return tail[-1] if tail else ""


def _cross_script_rhyme_key(word: str) -> Optional[tuple]:
    """
    Script-agnostic approximate rhyme key: (vowel_class, consonant_class) from
    the last stressed (English) or last vowel-nucleus (Hinglish) onward.
    """
    rom = devanagari_to_roman(word) if is_hindi_word(word) else word
    rom_clean = re.sub(r"[^a-zA-Z]", "", rom)
    # Guard against 1-2 letter filler/function words (e.g. "ना"/"ah") producing
    # trivial coincidental matches -- mirrors the length>2 filter already used
    # for internal-rhyme word candidates elsewhere in this module.
    if len(rom_clean) < 3:
        return None

    phones = pronouncing.phones_for_word(rom_clean.lower())
    if phones:
        plist = phones[0].split()
        stressed = [i for i, p in enumerate(plist) if p.endswith("1")]
        vowel_idxs = [i for i, p in enumerate(plist) if any(c.isdigit() for c in p)]
        start = stressed[-1] if stressed else (vowel_idxs[-1] if vowel_idxs else None)
        if start is None:
            return None
        vowel_raw = re.sub(r"\d", "", plist[start]).lower()
        vowel_class = _ARPABET_VOWEL_CLASS.get(vowel_raw, vowel_raw)
        tail_phones = [re.sub(r"\d", "", p).lower() for p in plist[start + 1:]]
        cons_class = _consonant_class(tail_phones[-1]) if tail_phones else ""
        return (vowel_class, cons_class)

    lower = rom_clean.lower()
    vowels = "aeiou"
    vowel_runs: list[tuple[int, int]] = []
    i = 0
    while i < len(lower):
        if lower[i] in vowels:
            start_i = i
            while i < len(lower) and lower[i] in vowels:
                i += 1
            vowel_runs.append((start_i, i))
        else:
            i += 1
    if not vowel_runs:
        return None
    last_start, last_end = vowel_runs[-1]
    vowel_raw = lower[last_start:last_end]
    vowel_class = _HINGLISH_VOWEL_CLASS.get(vowel_raw, vowel_raw)
    tail = lower[last_end:]
    cons_class = _consonant_class(_final_consonant_chunk(tail)) if tail else ""
    return (vowel_class, cons_class)


def _is_cross_script_pair(word_a: str, word_b: str) -> bool:
    return is_hindi_word(word_a) != is_hindi_word(word_b)


def _cross_script_keys_match(word_a: str, word_b: str) -> bool:
    key_a = _cross_script_rhyme_key(word_a)
    key_b = _cross_script_rhyme_key(word_b)
    return bool(key_a and key_b and key_a == key_b)


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

                matched = bool(key1 and key2 and key1 == key2)
                if not matched and _is_cross_script_pair(w1, w2):
                    matched = _cross_script_keys_match(w1, w2)

                if matched and not _is_trivial_suffix_rhyme(w1, w2):
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
            # A chain can have a gap of at most 1 line (e.g. idx - prev_idx <= 2).
            # Links that are only a shared grammatical suffix (conjugation rhyme)
            # or the *same word* repeated (identity rhyme, e.g. ending every line
            # with "hai") do not extend an *elite* chain — otherwise unskilled
            # repetition trivially maxes the chain score.
            same_sound = key == prev_key or (
                _is_cross_script_pair(prev_word, word)
                and _cross_script_keys_match(prev_word, word)
            )
            if (same_sound and idx - prev_idx <= 2
                    and prev_word.lower() != word.lower()
                    and not _is_trivial_suffix_rhyme(prev_word, word)):
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

def calculate(lyrics: str, debug: bool = False):
    """
    Detect end rhymes, internal rhymes, multisyllabic rhymes, chain rhymes,
    compound/mosaic rhymes, and holorimes.

    Returns:
        rhyme_score         (overall 0-100)
        rhyme_pairs         list of RhymeMatch
        multisyl_count      number of multi-syllabic rhyme pairs
        internal_score      0-100 sub-score
        chain_score         0-100 sub-score
        compound_count      number of compound/mosaic rhyme pairs
        holorime_count      number of holorime pairs

    With debug=True, an 8th element is appended: a dict of the 6 RAW
    pre-curve ratios (end_rhyme_ratio, internal_ratio, multisyl_ratio,
    chain_ratio, compound_ratio, holorime_ratio) -- used by
    corpus/calibrate.py to fit ELITE_TARGETS/CURVE_THRESHOLDS from the
    corpus's actual empirical distribution instead of the already-curved
    0-100 sub-scores.
    """
    lines = content_lines(lyrics)
    # De-duplicate identical lines to prevent repeated hooks/choruses from inflating the rhyme score
    unique_lines = []
    seen_lines = set()
    for line in lines:
        normalized_line = re.sub(r"[^\w\u0900-\u097F]", "", line).strip().lower()
        if normalized_line not in seen_lines:
            seen_lines.add(normalized_line)
            unique_lines.append(line)
    lines = unique_lines

    if len(lines) < 2:
        empty = (0.0, [], 0, 0.0, 0.0, 0, 0)
        if debug:
            return empty + ({
                "end_rhyme_ratio": 0.0, "internal_ratio": 0.0, "multisyl_ratio": 0.0,
                "chain_ratio": 0.0, "compound_ratio": 0.0, "holorime_ratio": 0.0,
            },)
        return empty

    indexed_words = [(i, _last_word(lines[i])) for i in range(len(lines))]
    indexed_words = [(i, w) for i, w in indexed_words if w]

    # 1. End Rhymes & Multisyllabic
    rhyme_pairs: list[RhymeMatch] = []
    seen: set[tuple[int, int]] = set()
    trivial_pair_count = 0

    for a in range(len(indexed_words)):
        idx_a, word_a = indexed_words[a]
        window = scoring_config.RHYME["WINDOW_SIZE_LINES"]
        for b in range(a + 1, min(a + window, len(indexed_words))):
            idx_b, word_b = indexed_words[b]
            is_identity = (word_a == word_b)
            actual_word_a = word_a
            actual_word_b = word_b

            if is_identity:
                tokens_a = [clean_word(w) for w in lines[idx_a].strip().split()]
                tokens_a = [t for t in tokens_a if t]
                tokens_b = [clean_word(w) for w in lines[idx_b].strip().split()]
                tokens_b = [t for t in tokens_b if t]
                if len(tokens_a) >= 2 and len(tokens_b) >= 2:
                    actual_word_a = tokens_a[-2]
                    actual_word_b = tokens_b[-2]
                    # If the second-to-last word is ALSO identical (e.g. both
                    # lines end in the literal 2-word phrase "kya hai"), this
                    # is phrase repetition, not a rhyme -- comparing the two
                    # occurrences of the same word to itself would trivially
                    # "rhyme" and get credited as multisyllabic/skilled, which
                    # rewards repeating a hook phrase rather than actual
                    # technique. Skip the pair entirely.
                    if actual_word_a.lower() == actual_word_b.lower():
                        continue
                else:
                    continue

            rom_a = devanagari_to_roman(actual_word_a) if is_hindi_word(actual_word_a) else actual_word_a
            rom_b = devanagari_to_roman(actual_word_b) if is_hindi_word(actual_word_b) else actual_word_b

            key_a = _rhyme_key_en(rom_a)
            key_b = _rhyme_key_en(rom_b)
            if key_a is None or key_b is None:
                key_a = _rhyme_key_hinglish(rom_a)
                key_b = _rhyme_key_hinglish(rom_b)

            is_end_rhyme = bool(key_a and key_b and key_a == key_b)

            # Cross-script fallback: exact keys above can never match across
            # Hindi/English (different representations), so try the coarser
            # vowel+consonant-class key when the pair spans both scripts.
            if not is_end_rhyme and _is_cross_script_pair(actual_word_a, actual_word_b):
                is_end_rhyme = _cross_script_keys_match(actual_word_a, actual_word_b)

            # A rhyme that is only a shared grammatical suffix (conjugation rhyme)
            # is unskilled: it may count as a basic end rhyme but must never be
            # promoted to multisyllabic or slant.
            trivial = _is_trivial_suffix_rhyme(actual_word_a, actual_word_b)

            # Slant Rhyme Matcher: 2-syllable vowel pattern match using phonetics (where available) or spelling
            is_slant_rhyme = False
            if not is_end_rhyme and not trivial:
                # Do not count as slant rhyme if both words end in different trivial suffixes
                # (e.g. karta and gaya both end in different trivial suffixes but share vowels).
                if not (_ends_in_any_trivial_suffix(actual_word_a) and _ends_in_any_trivial_suffix(actual_word_b)):
                    sig_a = _get_phonetic_signature(actual_word_a) or _get_spelling_signature(actual_word_a)
                    sig_b = _get_phonetic_signature(actual_word_b) or _get_spelling_signature(actual_word_b)
                    
                    if sig_a["ends_vowel"] == sig_b["ends_vowel"]:
                        ok_consonant = True
                        if not sig_a["ends_vowel"]:
                            c1 = sig_a["final_consonant"]
                            c2 = sig_b["final_consonant"]
                            ok_consonant = _are_similar_consonants(c1, c2)
                        
                        if ok_consonant:
                            v1 = sig_a["vowels"]
                            v2 = sig_b["vowels"]
                            if len(v1) >= 2 and len(v2) >= 2:
                                if v1[-2:] == v2[-2:]:
                                    is_slant_rhyme = True

            # Robust multisyllabic rhyme detection across last 2 words of the line
            is_multi = False
            if not trivial and (is_end_rhyme or is_slant_rhyme):
                v1 = _get_line_vowels(lines[idx_a])
                v2 = _get_line_vowels(lines[idx_b])
                common_suffix = _get_longest_common_vowel_suffix(v1, v2)
                if len(common_suffix) >= 3:
                    is_multi = True

            if is_end_rhyme or is_multi or is_slant_rhyme:
                pair = (idx_a, idx_b)
                if pair not in seen:
                    seen.add(pair)
                    if trivial and not (is_multi or is_slant_rhyme):
                        trivial_pair_count += 1
                    rhyme_pairs.append(
                        RhymeMatch(
                            line_a=idx_a,
                            line_b=idx_b,
                            word_a=actual_word_a,
                            word_b=actual_word_b,
                            is_multisyllabic=is_multi or is_slant_rhyme,
                        )
                    )

    multisyl_count = sum(1 for r in rhyme_pairs if r.is_multisyllabic)

    # Compute End Rhyme Score (0-100). Trivial conjugation rhymes still count but
    # at a reduced weight so filler-heavy verses can't inflate the ratio.
    trivial_w = scoring_config.RHYME["TRIVIAL_END_RHYME_WEIGHT"]
    weighted_pairs = (len(rhyme_pairs) - trivial_pair_count) + trivial_pair_count * trivial_w
    rhyme_ratio = weighted_pairs / max(len(lines) - 1, 1)
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

    # 5. Compound / Mosaic Rhymes
    compound_count, _compound_pairs = detect_compound_rhymes(lines)
    compound_ratio = compound_count / max(len(lines) - 1, 1)
    compound_score = min(
        (compound_ratio / scoring_config.RHYME["ELITE_TARGETS"]["compound_density"]) * 100.0, 100.0
    )

    # 6. Holorime ("perfect" multi-word rhyme)
    holorime_count, _holorime_pairs = detect_holorimes(lines)
    holorime_ratio = holorime_count / max(len(lines) - 1, 1)
    holorime_score = min(
        (holorime_ratio / scoring_config.RHYME["ELITE_TARGETS"]["holorime_density"]) * 100.0, 100.0
    )

    # Combined Rhyme Score dynamically weighted
    weights = scoring_config.RHYME["WEIGHTS"]
    rhyme_score = (
        end_rhyme_score * weights["end_rhyme"] +
        internal_score * weights["internal"] +
        multisyl_score * weights["multisyllabic"] +
        chain_score * weights["chains"] +
        compound_score * weights["compound"] +
        holorime_score * weights["holorime"]
    )

    # Rhyme-scheme diversity penalty: a song that hammers the same 1-2 rhyme
    # sounds every line is monotonous, not lyrical. Scale the rhyme score down
    # toward MIN_MULT when the distinct-key ratio falls below FLOOR.
    rhyme_score *= _diversity_multiplier(indexed_words)

    result = (round(rhyme_score, 2), rhyme_pairs, multisyl_count, round(internal_score, 2),
              round(chain_score, 2), compound_count, holorime_count)
    if debug:
        result = result + ({
            "end_rhyme_ratio": rhyme_ratio,
            "internal_ratio": internal_ratio,
            "multisyl_ratio": multisyl_ratio,
            "chain_ratio": chain_ratio,
            "compound_ratio": compound_ratio,
            "holorime_ratio": holorime_ratio,
        },)
    return result


def _diversity_multiplier(indexed_words: List[Tuple[int, str]]) -> float:
    """Fraction of distinct end-rhyme sounds -> a multiplier in [MIN_MULT, 1.0]."""
    cfg = scoring_config.RHYME
    keys = [str(_get_rhyme_key(w)) for _, w in indexed_words if _get_rhyme_key(w) is not None]
    if len(keys) < cfg["DIVERSITY_MIN_LINES"]:
        return 1.0
    diversity = len(set(keys)) / len(keys)
    floor, ceil, min_mult = cfg["DIVERSITY_FLOOR"], cfg["DIVERSITY_CEIL"], cfg["DIVERSITY_MIN_MULT"]
    if diversity >= ceil:
        return 1.0
    if diversity <= floor:
        return min_mult
    return min_mult + (diversity - floor) / (ceil - floor) * (1.0 - min_mult)
