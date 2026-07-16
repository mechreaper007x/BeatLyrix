# -*- coding: utf-8 -*-
"""
Build the DHH pronunciation dictionary -- derived phonetic facts mined from
the local real corpus (Phase 4 of docs/dhh_dictionary/PLAN.md).

For every unique content word in the Hindi/Hinglish corpus, record:
    word -> {phones, rhyme_key, multi_key}
computed by services/dhh_phonemes (schwa deletion, spelling-variant collapse).

The output JSON contains ONLY derived facts about pronunciation -- no lyric
lines, no verse text, no artist attribution -- so it is copyright-clean and
committable (raprank-nlp/corpus/data stays gitignored; this file does not).
Rebuildable and idempotent: rerun after mining new artists to grow coverage
toward the ~10k-entry target.

Usage:
    python -m corpus.dhh_dictionary.build_dictionary
    python -m corpus.dhh_dictionary.build_dictionary --out path.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_NLP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_NLP_ROOT))

from corpus.synthetic.rhyme_families import _iter_corpus_lyrics  # noqa: E402
from services import dhh_phonemes  # noqa: E402
from services.language_utils import clean_word, is_hindi_word  # noqa: E402

DEFAULT_OUT = _NLP_ROOT / "corpus" / "dhh_dictionary" / "dhh_dictionary.json"

# Latin-script tokens shorter than this are mostly English stop words / ad-lib
# noise ("a", "yo", "uh"); Devanagari has no such floor (घर is 2 chars).
_MIN_LATIN_LEN = 3
_WORD_RE = re.compile(r"[\wऀ-ॿ']+")


def _iter_extra_lyrics():
    """Additional local Hindi/Hinglish lyric sources beyond corpus/data: the
    synthetic corpus (our own generated verses -- their vocabulary is real
    DHH vocabulary) and the consented seed verses. Only word-level phonetic
    facts are extracted; the texts themselves stay local, as always."""
    roots = [
        _NLP_ROOT / "corpus" / "synthetic_data",
        _NLP_ROOT / "corpus" / "consented_seeds",
    ]
    patterns = ["*/mixed/*.json", "*/hi/*.json", "**/*.json"]
    seen: set[Path] = set()
    for root, pat in ((r, p) for r in roots for p in patterns):
        if not root.exists():
            continue
        for path in root.glob(pat):
            if path in seen or path.name.startswith("_"):
                continue
            seen.add(path)
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            lyrics = rec.get("lyrics") if isinstance(rec, dict) else None
            if lyrics:
                yield lyrics


def _iter_words():
    """Yield cleaned candidate words from every local DHH lyric source."""
    import itertools
    for lyrics in itertools.chain(_iter_corpus_lyrics("mixed"), _iter_extra_lyrics()):
        for raw in _WORD_RE.findall(lyrics):
            w = clean_word(raw)
            if not w:
                continue
            if is_hindi_word(w):
                yield w
            else:
                w = w.lower()
                if len(w) >= _MIN_LATIN_LEN and w.isalpha():
                    yield w


# How many top-frequency conversational Hindi words to fold in (wordfreq's
# 'hi' list, built from subtitles/news/web). These are a backstop for
# user-submitted lyrics whose vocabulary the rap corpus hasn't seen yet;
# corpus words always outrank them (freq counts add on top of rank weight).
_WORDFREQ_TOP_N = 25000


def _iter_wordfreq_words():
    """Top conversational Hindi words (Devanagari only -- the Latin tokens in
    wordfreq's hi list are English bleed-through, not romanized Hindi)."""
    try:
        from wordfreq import top_n_list
    except ImportError:
        print("  (wordfreq not installed; skipping frequency-list source)")
        return
    for w in top_n_list("hi", _WORDFREQ_TOP_N):
        w = clean_word(w)
        if w and is_hindi_word(w):
            yield w


def build(out_path: Path) -> dict:
    counts = Counter(_iter_words())
    corpus_words = len(counts)
    # Frequency-list words join at freq 0: present (usable for lookup and
    # rhyme grouping) but never outranking corpus-attested vocabulary.
    freq_added = 0
    for w in _iter_wordfreq_words():
        if w not in counts:
            counts[w] = 0
            freq_added += 1
    entries: dict[str, dict] = {}
    failed: list[str] = []
    for word, freq in counts.most_common():
        phones = dhh_phonemes.to_phones(word)
        if not phones:
            failed.append(word)
            continue
        rkey = dhh_phonemes.rhyme_key(phones)
        mkey = dhh_phonemes.multi_rhyme_key(phones)
        entries[word] = {
            "phones": list(phones),
            "rhyme_key": list(rkey) if rkey else None,
            "multi_key": list(mkey) if mkey else None,
            "freq": freq,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )

    total = len(counts)
    print(f"unique words seen:   {total} "
          f"({corpus_words} corpus + {freq_added} frequency-list)")
    print(f"phonemized:          {len(entries)} ({100.0 * len(entries) / max(total, 1):.1f}%)")
    print(f"failed to phonemize: {len(failed)}")
    if failed:
        print("  e.g.:", ", ".join(failed[:15]))
    with_multi = sum(1 for e in entries.values() if e["multi_key"])
    print(f"with multi-syllabic key: {with_multi}")
    deva = sum(1 for w in entries if is_hindi_word(w))
    print(f"devanagari entries:  {deva}  latin entries: {len(entries) - deva}")
    print(f"-> {out_path}")
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
