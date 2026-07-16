# DHH Dictionary — From-Scratch Hindi/Hinglish Pronunciation & Rhyme Engine

## 1. Context & Motivation

BeatLyrix (`raprank-nlp/`) scores rap lyrics on 19 axes. The rhyme axes work in two
very different ways depending on language:

- **English:** uses **CMUdict** (real phonemes with lexical stress) via the
  `pronouncing` library. `services/rhyme_service.py::_multi_rhyme_key_en` extracts
  the rhyme key from actual phoneme sequences. Reliable.
- **Hindi / Hinglish:** uses **spelling heuristics**, not phonemes.
  `normalize_hinglish` does string surgery (`aa→a`, `bh→b`, consonant dedup) and
  `_multi_rhyme_key_hinglish` / `_multi_rhyme_key_hi` walk vowels in the *written
  form*. There is **no phoneme layer, no stress, and — critically — no schwa-deletion
  handling.**

### Why this matters

Devanagari assigns an inherent schwa (`a`) to every bare consonant, but in real
speech word-final (and some medial) schwas are **deleted**: `घर` is pronounced
`ghar`, not `ghara`. The current keyer, working from spelling, keys the silent
vowel as if it were pronounced — so words that genuinely rhyme in speech fail to
group, and the multisyllabic-rhyme axis under-scores on the Hindi side. This is the
suspected root cause of weak Hindi rhyme scoring (mid-tier multisyllabic barely
moved in the generation-quality A/B pilots).

### Decision

**Build our own** phonemizer and rhyme dictionary from scratch. No external G2P
library (Epitran / Festvox / etc. are explicitly rejected). We implement the
linguistic rules ourselves, tuned to rap vocabulary. "From scratch" means *we write
the phoneme engine* — it does **not** mean ignoring real Hindi phonology (schwa
deletion, nasalization, gemination are facts about the language, implemented in our
own code).

The output is a committable `word → phonemes → rhyme_keys` dictionary of **derived
phonetic facts** — copyright-clean. Raw lyrics used for mining stay gitignored per
the existing project rule.

---

## 2. Hard Constraints

- **Test-first.** Nothing ships without passing a hand-authored gold set (Phase 0).
  Schwa deletion done wrong fails *silently* — the gold set is the only guard against
  shipping a subtly broken keyer.
- **No new committed lyric text.** Source lyrics for mining stay in already-gitignored
  `corpus/data/`. Only the derived dictionary JSON + code commit. The phoneme
  dictionary itself (`word → phones`) is derived facts and is committable; any cache
  that embeds corpus-specific data lives under a gitignored path.
- **Scorer and scaffold agree by construction.** The same rhyme-key function feeds
  both `rhyme_service.py` (scoring) and `corpus/synthetic/rhyme_families.py`
  (scaffolding), so any family that groups is scored as rhyming by the same detector.
- **Environment:** Windows; run under the repo venv `.venv/Scripts/python` with
  `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`.
- **Consistency with English semantics.** The Hindi multisyllabic key must mirror the
  behavior of `_multi_rhyme_key_en` (key from the n-th-to-last nucleus, minimum span
  to count) so both languages behave consistently in the scorer.

---

## 3. Architecture Overview

```
romanized Hinglish word ──▶ [Phase 2] back-transliterate ──▶ Devanagari
Devanagari word ─────────────────────────────────────────▶ [Phase 1] phonemizer
                                                                  │
                                                                  ▼
                                                             phoneme list
                                                                  │
                                                                  ▼
                                                     [Phase 3] rhyme-key extraction
                                                                  │
                          ┌───────────────────────────────────────┼──────────────────────┐
                          ▼                                        ▼                       ▼
              rhyme_service.py scoring        rhyme_families.py scaffold        [Phase 4] cached dictionary
```

- **Phase 1** — Devanagari → phonemes (our own rule engine).
- **Phase 2** — Romanized Hinglish → Devanagari → Phase 1.
- **Phase 3** — Phonemes → rhyme keys (single + multisyllabic).
- **Phase 4** — Mine corpus, cache `word → {phones, rhyme_key, multi_key}`.

---

## 4. Phases

### Phase 0 — Gold Set (the oracle)

**Deliverable:** `services/tests/data/dhh_rhyme_gold.py` (or `.json`) + a pytest.

~100–150 entries, authored/validated by a native ear:

- **Positive pairs** — words that genuinely rhyme, including:
  - tricky schwa cases (`घर`/`डर`, `pyaar`/`yaar`),
  - multisyllabic (`deewana`/`parwana`),
  - spelling variants of the *same* word (`hoon`/`hun`, `kya`/`kiya`) that MUST key
    identically.
- **Negative pairs** — near-spellings that do NOT rhyme.
- **Devanagari + romanized forms** of the same words, which MUST produce identical
  phoneme keys.

**Test asserts:**
1. positive pairs → equal keys,
2. negative pairs → unequal keys,
3. Devanagari form ≡ romanized form for the same word.

> This is the one phase that needs the native ear. Everything downstream is validated
> against it. Option: draft a starter list mined from the corpus, then hand-correct.

---

### Phase 1 — Devanagari → Phonemes

**Deliverable:** `services/dhh_phonemes.py`, API `deva_to_phones(word: str) -> list[str]`.

Deterministic rule engine with our own compact phoneme symbol set (only fine enough
to distinguish rhymes — **not** full IPA):

- Consonant + independent-vowel + matra → phoneme tables. Include nukta forms
  (`क़ ख़ ग़ ज़ ड़ ढ़ फ़`), conjuncts / half-consonants (virama/halant).
- **Schwa deletion** (the hard part — implement the standard algorithm):
  - word-final inherent schwa drops;
  - medial schwa drops in appropriate `V C _ C V` contexts.
  - Must cover every schwa case in the gold set.
- Anusvara / chandrabindu → nasalization.
- Gemination (doubled consonants), visarga.

**Gate:** passes the Phase-0 gold cases scoped to Devanagari input.

---

### Phase 2 — Romanized Hinglish → Devanagari → Phonemes

**Deliverable:** in `services/dhh_phonemes.py`, API `hinglish_to_phones(word: str) -> list[str]`.

The genuinely hard part (many spellings per word). Our own back-transliteration:

- Romanization → Devanagari mapper that **collapses spelling variants to one canonical
  form** (`hoon`/`hun`/`hoo` → same Devanagari → same phonemes). Build the mapping from
  **actual rap-corpus vocabulary**, not generic Hindi — this is our edge over a generic
  library.
- Handle aspirates (`bh/dh/kh/gh/th/ph/ch/sh`), long/short vowel spellings
  (`aa/ee/oo`), `y`/`w` glides, and ambiguous cases via a small disambiguation table.
- Route `is_hindi_word` (Devanagari) input straight to Phase 1.

**Gate:** Devanagari≡romanized equivalence holds for all gold words.

---

### Phase 3 — Rhyme Keys from Phonemes

**Deliverable:** key functions wired into `services/rhyme_service.py`.

- `rhyme_key(phones)` = nucleus vowel + trailing consonants (single-syllable rhyme).
- `multi_rhyme_key(phones, n=2)` = from the n-th-to-last nucleus onward; must span a
  minimum length to count — **mirror `_multi_rhyme_key_en` semantics.**
- Replace the bodies of `_multi_rhyme_key_hi` and `_multi_rhyme_key_hinglish` with
  calls into the new engine.
- Confirm `corpus/synthetic/rhyme_families.py` picks up the new keys unchanged (it
  already routes through `rhyme_service`), so scaffold and scorer stay in lockstep.

**Gate:** full gold pytest green; `rhyme_families.py` rebuild produces sane families.

---

### Phase 4 — The Cached Dictionary

**Deliverable:** a dictionary-builder script + committable `word → {...}` JSON.

- Mine every content word from `corpus/data/` (reuse
  `rhyme_families.py::_iter_corpus_lyrics`), run Phases 1–3, cache
  `word → {phones, rhyme_key, multi_key}`.
- Report coverage: unique words, % phonemized, list of unmatched forms.
- Idempotent / rebuildable, like the existing rhyme-families cache.
- **Target:** grow toward ~10k entries as more artists are mined (mine-and-discard:
  extract phonetic facts, never store the source verses).

---

## 5. Verification (each phase gated)

- Phase-0 gold pytest passes for the phase's scope before moving on.
- Full suite green: `pytest services/tests/ -q`
  - Baseline: **148 passed, 4 known `TestAllusions` failures (pre-existing), 32
    skipped.** Nothing new may break.
- After Phase 3: re-run the synthetic scorer on a few known Hindi rhyming samples and
  confirm previously-missed multisyllabic rhymes now register.
- **Before/after report:** how many gold pairs the old spelling heuristic got wrong
  that the new phoneme engine gets right.

---

## 6. Deliverables Checklist

- [ ] `services/tests/data/dhh_rhyme_gold.py` — gold set
- [ ] gold-set pytest (equality/inequality/script-equivalence)
- [ ] `services/dhh_phonemes.py` — Phase 1 (`deva_to_phones`) + Phase 2 (`hinglish_to_phones`)
- [ ] Phase 3 keys wired into `services/rhyme_service.py` (`_multi_rhyme_key_hi`, `_multi_rhyme_key_hinglish`)
- [ ] dictionary-builder script + cached `word → {phones, rhyme_key, multi_key}` JSON
- [ ] coverage + before/after accuracy report
- [ ] full test suite green (no new failures vs baseline)

---

## 7. Open Questions / Needs Input

- **Gold set authorship** — seed a starter list mined from the corpus for correction,
  or author from scratch by a native ear?
- **Phoneme symbol set** — confirm the compact set is "rhyme-distinguishing only" vs
  full IPA (leaning compact for simplicity and speed).
- **Sequencing** — this project starts *after* the current generation-quality pilot
  (scaffold vs scaffold+refine) is locked, to avoid forking focus.
