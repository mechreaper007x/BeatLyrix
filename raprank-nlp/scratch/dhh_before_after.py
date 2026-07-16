# -*- coding: utf-8 -*-
"""Before/after report: old spelling-heuristic rhyme keys vs the new DHH
phoneme engine, scored against the gold set. Run once for the Phase 4
verification report; keep for regression comparison after future changes."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import dhh_phonemes as dhh
from services.language_utils import is_hindi_word, devanagari_to_roman
from services.tests.data.dhh_rhyme_gold import (
    GOLD_MULTI_NEGATIVE, GOLD_MULTI_POSITIVE, GOLD_NEGATIVE, GOLD_POSITIVE,
)


# ── The OLD heuristics, verbatim (removed from rhyme_service.py) ─────────────

def _old_normalize(word):
    word = word.lower()
    for a, b in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("bh", "b"), ("dh", "d"),
                 ("kh", "k"), ("gh", "g"), ("ph", "f"), ("th", "t"), ("ch", "c"),
                 ("sh", "s")):
        word = word.replace(a, b)
    return re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+", r"\1", word)


def _old_vowel_idxs(word):
    vowels = "aeiou"
    idxs, i = [], 0
    while i < len(word):
        if word[i] in vowels:
            idxs.append(i)
            while i < len(word) and word[i] in vowels:
                i += 1
        else:
            i += 1
    return idxs


def _old_key(word, multi=False):
    if is_hindi_word(word):
        # old Devanagari path: last-N-chars suffix (config: 3 single / 4 multi)
        n = 4 if multi else 3
        if multi:
            return word[-n:] if len(word) >= n else None
        return word[-n:] if len(word) >= n else (word[-2:] if len(word) >= 2 else None)
    word = _old_normalize(word)
    vowels = "aeiou"
    idxs = _old_vowel_idxs(word)
    need = 2 if multi else 1
    if len(idxs) < need:
        return None
    start = idxs[-2] if multi else idxs[-1]
    last = idxs[-1]
    if last == len(word) - 1 or (last < len(word) - 1 and all(c in vowels for c in word[last:])):
        ci = start - 1
        while ci >= 0 and word[ci] in vowels:
            ci -= 1
        if ci >= 0:
            return word[ci:]
    return word[start:]


def _new_key(word, multi=False):
    phones = dhh.to_phones(word)
    if not phones:
        return None
    return dhh.multi_rhyme_key(phones) if multi else dhh.rhyme_key(phones)


def _score(pairs, want_equal, multi):
    old_ok = new_ok = 0
    old_fail_new_ok = []
    for a, b in pairs:
        ko = (_old_key(a, multi), _old_key(b, multi))
        kn = (_new_key(a, multi), _new_key(b, multi))
        o = (ko[0] is not None and ko[0] == ko[1]) == want_equal
        n = (kn[0] is not None and kn[0] == kn[1]) == want_equal
        old_ok += o
        new_ok += n
        if n and not o:
            old_fail_new_ok.append(f"{a}~{b}")
    return old_ok, new_ok, old_fail_new_ok


def main():
    sections = [
        ("positive (must rhyme)", GOLD_POSITIVE, True, False),
        ("negative (must NOT rhyme)", GOLD_NEGATIVE, False, False),
        ("multi positive", GOLD_MULTI_POSITIVE, True, True),
        ("multi negative", GOLD_MULTI_NEGATIVE, False, True),
    ]
    t_old = t_new = t_n = 0
    print(f"{'section':28} {'old':>7} {'new':>7}")
    for name, pairs, want, multi in sections:
        o, n, fixed = _score(pairs, want, multi)
        t_old += o; t_new += n; t_n += len(pairs)
        print(f"{name:28} {o:>3}/{len(pairs):<3} {n:>3}/{len(pairs):<3}"
              + (f"   fixed: {', '.join(fixed[:4])}{'...' if len(fixed) > 4 else ''}" if fixed else ""))
    print("-" * 60)
    print(f"{'TOTAL':28} {t_old:>3}/{t_n:<3} {t_new:>3}/{t_n:<3}"
          f"   ({100*t_old/t_n:.0f}% -> {100*t_new/t_n:.0f}%)")


if __name__ == "__main__":
    main()
