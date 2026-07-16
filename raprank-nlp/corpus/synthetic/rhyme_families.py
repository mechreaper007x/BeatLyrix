"""
Rhyme scaffolding for the reference-anchored synthetic generators.

LLMs generate left-to-right and cannot "see" a multisyllabic rhyme landing a
line ahead, which is exactly why synthetic ELITE/MID output under-hits the
internal/multisyllabic-rhyme axes (the one thing models are measurably worst
at -- theme/vocabulary they handle fine). This module removes that burden: it
pre-mines real multisyllabic rhyme FAMILIES (groups of 2+ words that rhyme on
their last two syllables) from the local real corpus, then hands the model a
few ready-made families as building blocks it only has to weave into bars --
turning "invent a rhyme scheme" into "use these rhymes."

Only WORD-LEVEL rhyme groups are surfaced -- never a line, phrase, or any span
of consecutive words from a real song -- so nothing copyrightable is copied.
The mined vocabulary is common rhyming words (nation/station, kar-diya/bhar-
diya); the families and their on-disk cache live under the same gitignored
tree as the corpus they're derived from (see raprank-nlp/.gitignore).

Reuses services.rhyme_service's exact multisyllabic-rhyme keying
(_multi_rhyme_key_en / _multi_rhyme_key_hinglish), so a family that groups here
is scored as a real multisyllabic rhyme by the same detector the accept/reject
loop uses -- the scaffold and the scorer agree by construction.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

from services import rhyme_service as rh
from services.language_utils import clean_word, is_hindi_word

_NLP_ROOT = Path(__file__).resolve().parents[2]
_HI_CORPUS = _NLP_ROOT / "corpus" / "data"
_EN_CORPUS = _NLP_ROOT / "corpus" / "real_corpus" / "data"
# Cache lives beside the gitignored corpora it is derived from. corpus/data/ is
# already gitignored, so a file written under it inherits that exclusion.
_CACHE = _NLP_ROOT / "corpus" / "data" / "_rhyme_families_cache.json"

# A family is only useful as scaffolding if several distinct words share the
# rhyme -- a 2-word family gives the model no room, and singletons are noise.
_MIN_FAMILY_SIZE = 3
# Cap words shown per family so the injected block stays short.
_MAX_WORDS_PER_FAMILY = 6
_MIN_WORD_LEN = 3


def _iter_corpus_lyrics(lang: str):
    """Yield lyric strings from the local real corpus for the given lang."""
    root = _EN_CORPUS if lang == "en" else _HI_CORPUS
    if not root.exists():
        return
    for path in glob.glob(str(root / "*" / "*.json")):
        if Path(path).name.startswith("_"):
            continue
        try:
            rec = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        lyrics = rec.get("lyrics")
        if lyrics:
            yield lyrics


def _multi_key(word: str, lang: str):
    """Multisyllabic rhyme key for a single word, using the live scorer's keying.
    English words go through the CMU-based English keyer; Hinglish/Hindi words
    (Devanagari or romanized) go through the Hinglish keyer, matching how
    rhyme_service itself routes words during scoring."""
    if lang == "en":
        return rh._multi_rhyme_key_en(word, n=2)
    # mixed / hi corpus: Devanagari words use the Hindi keyer, romanized use
    # the Hinglish keyer -- both return multisyllabic suffixes.
    if is_hindi_word(word):
        return rh._multi_rhyme_key_hi(word)
    return rh._multi_rhyme_key_hinglish(word)


def _build_families(lang: str) -> list[list[str]]:
    """Bucket every corpus content word by its multisyllabic rhyme key, keep
    buckets with enough distinct members."""
    buckets: dict = defaultdict(set)
    for lyrics in _iter_corpus_lyrics(lang):
        for raw in re.findall(r"[^\s]+", lyrics):
            w = clean_word(raw)
            if len(w) < _MIN_WORD_LEN:
                continue
            key = _multi_key(w, lang)
            if key is not None:
                buckets[key].add(w)
    families = [sorted(ws) for ws in buckets.values() if len(ws) >= _MIN_FAMILY_SIZE]
    # Longest (richest) families first -- they give the model the most options.
    families.sort(key=len, reverse=True)
    return families


def _load_cache() -> dict:
    if _CACHE.exists():
        try:
            return json.loads(_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # cache is an optimization; failing to persist is non-fatal


def get_families(lang: str, rebuild: bool = False) -> list[list[str]]:
    """All mined rhyme families for a lang, cached to disk after first build.
    lang is 'en' or 'mixed' (mixed reads the Hindi/Hinglish corpus)."""
    key = "en" if lang == "en" else "mixed"
    cache = _load_cache()
    if not rebuild and key in cache:
        return cache[key]
    families = _build_families(lang)
    cache[key] = families
    _save_cache(cache)
    return families


def sample_families(lang: str, k: int, rng) -> list[list[str]]:
    """Pick k rhyme families at random for injection into one prompt, each
    trimmed to at most _MAX_WORDS_PER_FAMILY words. `rng` is a random.Random
    instance the caller owns (kept deterministic-per-index in the generators,
    which avoid module-level Math.random-style global state)."""
    families = get_families(lang)
    if not families:
        return []
    chosen = rng.sample(families, min(k, len(families)))
    return [fam[:_MAX_WORDS_PER_FAMILY] for fam in chosen]


def build_scaffold_block(lang: str, k: int, rng) -> str:
    """A ready-to-inject prompt fragment offering the model rhyme building
    blocks, or '' if no families are available (corpus absent). Returns only
    loose word groups -- never a line from a song."""
    fams = sample_families(lang, k, rng)
    if not fams:
        return ""
    lines = [
        "RHYME BUILDING BLOCKS -- these are groups of words that rhyme "
        "multisyllabically. Land some of these as END or INTERNAL rhymes to "
        "carry the technical density (do NOT force all of them, and do NOT "
        "list them -- weave them naturally into original bars):",
    ]
    for fam in fams:
        lines.append("  - " + " / ".join(fam))
    return "\n".join(lines)


def main() -> int:
    """CLI: (re)build and report family counts. Run once to warm the cache."""
    import sys

    for lang in ("mixed", "en"):
        fams = get_families(lang, rebuild=True)
        total_words = sum(len(f) for f in fams)
        print(f"{lang:6}: {len(fams)} families, {total_words} words")
        for fam in fams[:5]:
            print("   e.g. " + " / ".join(fam[:_MAX_WORDS_PER_FAMILY]))
    print(f"\nCache written to {_CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
