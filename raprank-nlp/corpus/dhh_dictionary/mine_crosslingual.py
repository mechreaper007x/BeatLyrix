# -*- coding: utf-8 -*-
"""
Cross-lingual homophone miner -- exhaustively derive Hindi/Hinglish <-> English
near-homophone pairs (jagannath/juggernaut, haram/"hai Raam") from the full
52K DHH dictionary x CMUdict, via a shared coarse phoneme skeleton.

This is DERIVED DATA AT SCALE, not a hand-curated list: every DHH entry is
skeletonized, every common English word (wordfreq top-N with a CMUdict
pronunciation) is skeletonized through an ARPAbet->DHH bridge, and collisions
are mined three ways:

  exact   -- identical coarse skeletons (word <-> word)
  near    -- skeleton edit distance 1 (deletion/substitution), same syllable
             count +-1 (jagannath/juggernaut class)
  resegment -- a Hindi word's skeleton == concatenation of TWO common English
             word skeletons (gorej = god+rage class; the mondegreen seed)

Output: corpus/dhh_dictionary/crosslingual_homophones.json -- derived
phonetic facts only (no lyric text), committable.

Usage:
    python -m corpus.dhh_dictionary.mine_crosslingual
    python -m corpus.dhh_dictionary.mine_crosslingual --en-top 50000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_NLP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_NLP_ROOT))

import pronouncing  # noqa: E402

from services import dhh_phonemes as dhh  # noqa: E402
from services.language_utils import is_hindi_word  # noqa: E402

DICT_PATH = _NLP_ROOT / "corpus" / "dhh_dictionary" / "dhh_dictionary.json"
OUT_PATH = _NLP_ROOT / "corpus" / "dhh_dictionary" / "crosslingual_homophones.json"

# ── ARPAbet -> DHH coarse bridge ─────────────────────────────────────────────
# Both alphabets reduce to the same coarse skeleton: vowel-quality classes
# (length dropped -- Hinglish spelling can't encode it and near-homophones
# don't preserve it) + consonant classes (aspiration/retroflex dropped).

_ARPA_VOWEL = {
    "AA": "a", "AE": "a", "AH": "a", "AX": "a",
    "AO": "o", "AW": "au", "AY": "ai",
    "EH": "e", "EY": "e", "ER": "ar",   # ER unpacks to vowel+r
    "IH": "i", "IY": "i",
    "OW": "o", "OY": "ai",
    "UH": "u", "UW": "u",
}
_ARPA_CONS = {
    "B": "b", "CH": "c", "D": "d", "DH": "d", "F": "f", "G": "g",
    "HH": "h", "JH": "j", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "n", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t",
    "TH": "t", "V": "v", "W": "v", "Y": "y", "Z": "z", "ZH": "j",
}

# DHH phone -> coarse class (drop aspiration, length, w/v distinction)
_DHH_COARSE = {
    "aa": "a", "ai": "ai", "au": "au",
    "kh": "k", "gh": "g", "ch": "c", "jh": "j",
    "th": "t", "dh": "d", "ph": "f", "bh": "b",
    "w": "v", "ny": "n", "ng": "n", "ri": "ri",
}

_VOWEL_CLASSES = {"a", "e", "i", "o", "u", "ai", "au"}

# Consonant equivalence classes for NEAR matching only (exact matching keeps
# the finer skeleton). A Hindi ear hears b/v/w as one family (सम्बन्ध/someone),
# t/d as one, s/z/sh as one -- mirrors rhyme_service._CONSONANT_CLASS_GROUPS.
_NEAR_CLASS = {
    "b": "B", "v": "B", "p": "B", "f": "B",
    "t": "T", "d": "T",
    "s": "S", "z": "S", "sh": "S",
    "c": "C", "j": "C",
    "k": "K", "g": "K",
    "m": "M", "n": "M",
    "r": "R", "l": "R",
    "y": "Y", "h": "H",
}


def _near_class_skel(skel: tuple) -> tuple:
    """Map a skeleton onto coarse consonant classes (vowels unchanged)."""
    return tuple(p if p in _VOWEL_CLASSES else _NEAR_CLASS.get(p, p) for p in skel)


def en_skeleton(word: str) -> tuple | None:
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return None
    out: list[str] = []
    for p in phones[0].split():
        base = "".join(c for c in p if not c.isdigit())
        if base in _ARPA_VOWEL:
            v = _ARPA_VOWEL[base]
            if v == "ar":
                out.extend(("a", "r"))
            else:
                out.append(v)
        elif base in _ARPA_CONS:
            out.append(_ARPA_CONS[base])
    return _dedupe(out)


def dhh_skeleton(phones) -> tuple | None:
    if not phones:
        return None
    out = [_DHH_COARSE.get(p, p) for p in phones]
    if "ri" in out:  # rare; unpack to r+i for comparability
        flat: list[str] = []
        for p in out:
            flat.extend(("r", "i") if p == "ri" else (p,))
        out = flat
    return _dedupe(out)


def _dedupe(seq) -> tuple:
    out: list[str] = []
    for p in seq:
        if out and p == out[-1]:
            continue
        out.append(p)
    return tuple(out)


def _syllables(skel: tuple) -> int:
    return sum(1 for p in skel if p in _VOWEL_CLASSES)


def _del_variants(skel: tuple, max_deletions: int):
    """All variants of *skel* with up to max_deletions phones deleted --
    shared hashing trick: two sequences are within k deletions/substitutions
    iff they share a variant. k scales with skeleton length so long pairs
    like सम्बन्ध/someone (7 phones, v~b class + trailing-d fade) still match."""
    yield skel
    if max_deletions >= 1:
        for i in range(len(skel)):
            v1 = skel[:i] + skel[i + 1:]
            yield v1
            if max_deletions >= 2:
                for j in range(len(v1)):
                    yield v1[:j] + v1[j + 1:]


def _max_edits(skel: tuple) -> int:
    """Allowed edit distance by length: 1 for short skeletons, 2 for 6+."""
    return 2 if len(skel) >= 6 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-top", type=int, default=50000,
                    help="mine against this many top-frequency English words")
    ap.add_argument("--min-syllables", type=int, default=2,
                    help="minimum syllables for exact/near pairs (1-syllable "
                         "collisions are mostly trivial)")
    ap.add_argument("--max-near", type=int, default=50000,
                    help="keep only the top-scored near pairs")
    args = ap.parse_args()

    print("loading DHH dictionary...")
    ddict = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    dhh_by_skel: dict[tuple, list[str]] = defaultdict(list)
    for word, entry in ddict.items():
        skel = dhh_skeleton(entry["phones"])
        if skel:
            dhh_by_skel[skel].append(word)
    print(f"  {len(ddict)} words -> {len(dhh_by_skel)} distinct skeletons")

    print(f"loading top {args.en_top} English words with CMU phonemes...")
    from wordfreq import top_n_list
    en_by_skel: dict[tuple, list[str]] = defaultdict(list)
    en_count = 0
    for w in top_n_list("en", args.en_top):
        if not w.isalpha() or len(w) < 3:
            continue
        skel = en_skeleton(w)
        if skel:
            en_by_skel[skel].append(w)
            en_count += 1
    print(f"  {en_count} English words -> {len(en_by_skel)} skeletons")

    # ── exact collisions ────────────────────────────────────────────────────
    exact: list[dict] = []
    for skel, en_words in en_by_skel.items():
        if _syllables(skel) < args.min_syllables:
            continue
        dh_words = dhh_by_skel.get(skel)
        if not dh_words:
            continue
        # only pair with actual Hindi-side words (Devanagari, or Latin words
        # that are corpus-attested Hinglish, i.e. not also common English)
        en_set = set(en_words)
        dh_side = [w for w in dh_words if is_hindi_word(w) or w not in en_set]
        if not dh_side:
            continue
        exact.append({
            "skeleton": list(skel),
            "hindi": dh_side[:12],
            "english": en_words[:12],
        })

    # ── near collisions (class-skeleton, length-scaled edit distance) ───────
    print("mining near matches (consonant classes, edit distance 1-2)...")
    from wordfreq import zipf_frequency

    # index English by CLASS skeleton; deletion variants capped at each
    # skeleton's own allowance
    en_class_index: dict[tuple, set[tuple]] = defaultdict(set)
    en_class_words: dict[tuple, list[str]] = defaultdict(list)
    for skel, words in en_by_skel.items():
        if _syllables(skel) < args.min_syllables:
            continue
        cskel = _near_class_skel(skel)
        en_class_words[cskel].extend(words)
        for v in _del_variants(cskel, _max_edits(cskel)):
            en_class_index[v].add(cskel)

    near: list[dict] = []
    seen_pairs: set[tuple] = set()
    for dskel, dh_words in dhh_by_skel.items():
        if _syllables(dskel) < args.min_syllables:
            continue
        dcskel = _near_class_skel(dskel)
        candidates: set[tuple] = set()
        for v in _del_variants(dcskel, _max_edits(dcskel)):
            candidates |= en_class_index.get(v, set())
        for ecskel in candidates:
            if abs(_syllables(ecskel) - _syllables(dcskel)) > 1:
                continue
            key = (dcskel, ecskel)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            en_words = en_class_words[ecskel]
            en_set = set(en_words)
            dh_side = [w for w in dh_words
                       if is_hindi_word(w) or w not in en_set]
            if not dh_side:
                continue
            # Rank: both halves must be words people instantly recognize.
            # English side: best zipf. Hindi side: graded by REAL-WORLD Hindi
            # frequency (zipf 'hi'), not just rap-corpus attestation -- a
            # binary corpus bonus buried common Hindi like सम्बन्ध (freq-list
            # word) below Devanagari-spelled English loans. Corpus attestation
            # still adds a small nudge; identical class skeletons rank higher
            # than edit-2 pairs.
            en_best = max(en_words, key=lambda w: zipf_frequency(w, "en"))
            en_zipf = zipf_frequency(en_best, "en")
            hi_zipf = max(zipf_frequency(w, "hi") for w in dh_side)
            dh_freq = max((ddict[w]["freq"] for w in dh_side if w in ddict), default=0)
            exact_class = dcskel == ecskel
            score = (en_zipf + hi_zipf
                     + (0.5 if dh_freq > 0 else 0.0)
                     + (1.5 if exact_class else 0.0))
            near.append({
                "hindi": dh_side[:8],
                "english": sorted(en_words, key=lambda w: -zipf_frequency(w, "en"))[:8],
                "hindi_skeleton": list(dskel),
                "class_match": exact_class,
                "score": round(score, 2),
            })
    near.sort(key=lambda r: -r["score"])
    if len(near) > args.max_near:
        print(f"  ranked {len(near)} near pairs; keeping top {args.max_near}")
        near = near[:args.max_near]

    # ── re-segmentation (Hindi word == two English words) ───────────────────
    print("mining re-segmentation pairs (mondegreen seeds)...")
    common_en_by_skel = {s: ws for s, ws in en_by_skel.items() if len(s) >= 2}
    reseg: list[dict] = []
    for dskel, dh_words in dhh_by_skel.items():
        if _syllables(dskel) < 2 or len(dskel) < 4:
            continue
        dh_side = [w for w in dh_words if is_hindi_word(w)]
        if not dh_side:
            continue
        for cut in range(2, len(dskel) - 1):
            left, right = dskel[:cut], dskel[cut:]
            lw = common_en_by_skel.get(left)
            rw = common_en_by_skel.get(right)
            if lw and rw:
                reseg.append({
                    "hindi": dh_side[:6],
                    "english_phrase": [lw[:4], rw[:4]],
                    "skeleton": list(dskel),
                })
                break  # one segmentation per word is enough

    out = {
        "meta": {
            "dhh_entries": len(ddict),
            "english_words": en_count,
            "exact": len(exact),
            "near": len(near),
            "resegment": len(reseg),
        },
        "exact": exact,
        "near": near,
        "resegment": reseg,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\nexact word<->word homophones:  {len(exact)} skeleton groups")
    print(f"near (edit-1) pairs:           {len(near)}")
    print(f"re-segmentation (mondegreen):  {len(reseg)}")
    print(f"-> {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
