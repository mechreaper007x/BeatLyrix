# Rap-element analysis loop — progress log

Persistent tracker for the `/loop` task: enumerate every rap element, build
detectors, fingerprint each song's uniqueness, and analyze each targeted
rapper's specialty. Incremental — each loop iteration extends this.

## Rap-element taxonomy (target detectors)

Legend: [x] have detector in services/  ·  [~] prototyped in analysis/  ·  [ ] TODO

### Rhyme family
- [x] End rhyme (CMU / Hinglish keys)
- [x] Cross-script (Hindi<->English) end/internal/chain rhyme — coarse
      vowel+consonant-class fallback key, rhyme_service.py _cross_script_rhyme_key
- [x] Internal rhyme (within-line)
- [x] Multisyllabic rhyme
- [x] Chain rhyme (3+ consecutive lines)
- [x] Slant / near rhyme (2-vowel match)
- [x] Rhyme-scheme diversity (distinct-key ratio)
- [x] Conjugation/identity rhyme discount (trivial-suffix, anti-inflation)
- [x] Assonance (vowel repetition, non-onset) — services/assonance_service.py
- [x] Consonance (consonant repetition, non-onset) — services/consonance_service.py
- [x] Compound / mosaic rhyme (multi-word rhyme units) — detect_compound_rhymes
      in services/rhyme_service.py, tests/test_compound_rhyme.py
- [x] Holorime / perfect multi-word — detect_holorimes in
      services/rhyme_service.py, tests/test_holorime.py

### Sound / phonetic
- [x] Devanagari→Roman nukta consonants (क़ख़ग़ज़ड़ढ़फ़, decomposed base+U+093C)
      — services/language_utils.py devanagari_to_roman, tests/test_language_utils.py
- [x] Alliteration (onset proximity)
- [x] Assonance density (services/assonance_service.py, tests/test_assonance.py)
- [x] Consonance density (services/consonance_service.py, tests/test_consonance.py)
- [x] Onomatopoeia / ad-lib density — services/onomatopoeia_service.py,
      tests/test_onomatopoeia.py

### Density / delivery (text proxies; true flow needs audio)
- [x] Syllable density (syl/line)
- [x] Syllable weight (complex-word ratio)
- [x] Flow beat-sync (audio-only, separate path)
- [~] Avg line length, words/line variance (cadence proxy)

### Wordplay / semantics
- [x] Simile
- [x] Metaphor
- [x] Homophone pun (variant-filtered)
- [x] Double entendre (WordNet polysemy + rap lexicon)
- [ ] Extended metaphor / conceit (cross-line)
- [ ] Punchline / setup-payoff
- [x] Allusion / pop-culture reference — detect_allusions in
      services/wordplay_service.py (closed lexicon, config/scoring_config.py
      WORDPLAY["ALLUSION_REFERENCES"]), 5th wordplay sub-category

### Lexical / structural / multilingual
- [x] Vocabulary richness (MSTTR)
- [~] Code-switching index (Hindi/English/Hinglish mix)
- [~] Anaphora / repetition (line-initial word reuse)
- [ ] Callback / motif reuse across sections

## Songs/artists covered by close reading (loop-progress tracker)

Purpose: the `/loop` task is a close-reading loop (deep semantic/linguistic/
rap-element read of individual songs, not just corpus-wide statistics) — this
section tracks which *individual songs* have actually been read closely
enough to compare against the rule-based scorer's output, so each iteration
picks up new songs rather than re-reading ones already covered. (Gap noted in
iter 7/8: this section didn't exist before iter 8 — earlier close-reads were
only mentioned inline in iteration-log prose, making "already covered" hard
to check at a glance. Update this list every iteration.)

### KR$NA — corpus/data/kr-na/ (33 songs total)
Closely read (10/33):
- [x] no-cap.json
- [x] saza-e-maut.json (source of the "मैं ना victim, I'm more like Vikram"
      cross-script diagnostic pair)
- [x] makasam.json
- [x] lil-bunty.json (बंदर/rumble/jungle/कंबल/चंबल cross-script rhyme chain)
- [x] blowing-up.json
- [x] khatta-flow.json
- [x] hello.json
- [x] saath-ya-khilaaf.json
- [x] og.json (entirely Romanized-Hinglish/Latin script, zero Devanagari — a
      format variant worth remembering when reasoning about script detection)
- [x] villain.json (motivated the iter-8 allusion detector: Amrish Puri,
      Amjad Khan, Mogambo, Gabbar, Being Human, Ledger/Oscar punchline)

Not yet closely read (23/33) — prioritize these before moving to another
artist ("well-covered" per the loop's instructions means this list, not just
the per-artist z-score stats below, which are corpus-wide aggregates and
don't require reading individual lyrics):
- [ ] been-a-while.json
- [ ] hola-amigo.json
- [ ] i-guess.json
- [ ] joota-japani.json
- [ ] kaha-tak.json
- [ ] knock-knock.json
- [ ] machayenge-4.json
- [ ] maharani.json
- [ ] muqabla.json
- [ ] ngl.json
- [ ] prarthana.json
- [ ] roll-up.json
- [ ] say-my-name.json
- [ ] seedha-makeover.json
- [ ] sensitive.json
- [ ] shut-up.json
- [ ] some-of-us.json
- [ ] still-standing.json
- [ ] untitled.json
- [ ] vibrate.json
- [ ] vyanjan.json
- [ ] wanna-know.json
- [ ] whats-up.json

Also analyzed (NOT in corpus/data/, user-pasted lyrics, no Genius ID/URL
available so deliberately not persisted as corpus JSON — analyzed in-memory
via disposable scripts instead of fabricating provenance metadata):
- [x] "Too Bait" — validated the rhyme_score/internal/chain breakdown; caught
      someone/london (multisyllabic slant) plus several same-script pairs.
- [x] "Playground" — heavy pop-culture allusion (Katie, Thailand, Gilligan's
      Island, Moroccan/Casablanca, Rubik's Cubes, Kubrick) that directly
      informed the iter-8 allusion lexicon's pop-culture entries.
- [x] "Overdrive" — caught काण्ड/rajnikanth, the flagship iter-7 cross-script
      rhyme catch, on genuinely fresh (not corpus-memorized) lyrics.

### Other artists — corpus/data/ (22 directories total, KR$NA not yet
"well-covered" per above, so none of these have individual close-reads yet;
only the corpus-wide per-artist z-score analysis below, which is a different,
coarser kind of coverage):
brodha-v, calm, carryminati, emiway-bantai, epr, hanumankind, jtrix, karma,
king, lil-bhatia, muhfaad, naam-sujal, panther, paradox, raftaar, raga,
seedhe-maut, tsumyoki, vichaar, yashraj, yungsta (+ kr-na above).

## Per-artist specialty analysis (analyze each alone)

Data-driven: z-score each artist's element profile vs the corpus, top standout
= specialty. Cross-check against known reputation. Status per artist below.
Run `python -m corpus.analysis.signature` to regenerate (302 songs / 22
artists as of iter 5 — corpus grew substantially via scraping outside this
loop's direct edits; EPR is no longer missing, and 11 artists outside the
original 9-name target list showed up in corpus/data/, several not present in
corpus/artists.py's ARTISTS seed list, implying scrape runs against a wider
or updated artist config). z-scores below are now computed against this
larger, more diverse corpus, so they are NOT directly comparable to the
iter-2/3/4 numbers logged earlier in this file (different corpus composition
shifts every artist's z vs. the whole).

Original 9 targeted rappers:
- [x] KR$NA (n=33) — wordplay(+0.88), simile(+0.80), metaphor(+0.76),
      entendre(+0.75). Matches reputation: dense wordplay/entendre lyricist,
      now the single most across-the-board strong profile in the full 22-
      artist corpus (see `python -m corpus.report`: highest composite `total`
      score, 51.0).
- [x] Raftaar (n=44) — repetition(+0.56), syl_weight(+0.41), rhyme(+0.34),
      internal(+0.25). Still the flattest specialty profile of the roster,
      consistent with a commercial/hook-driven catalogue — no single axis
      dominates.
- [x] Karma (n=27) — codeswitch(+0.53), entendre(+0.27), internal(+0.07).
      Wordplay/entendre-leaning but far less extreme than KR$NA; code-
      switching (Hindi/English mix) is now his clearest standout.
- [x] Seedhe Maut (n=21) — cadence_var(+1.34), alliteration(+0.85),
      vocab(+0.64), pun(+0.64). Delivery-variety and sound-texture (allit),
      not primarily a rhyme-density act.
- [ ] Encore ABJ (still no separately-scraped corpus data — folded into the
      combined Seedhe Maut catalogue, no individual-member analysis possible)
- [x] Brodha V (n=5, still thin) — syl_density(+0.86), consonance(+0.79),
      onomatopoeia(+0.69), repetition(+0.66). English-heavy dense-delivery
      reputation still holds directionally, though sample is too small to
      fully trust.
- [x] EPR (n=8 — was previously unscraped, now has data) — syl_weight(+1.12),
      consonance(+0.86), alliteration(+0.75), vocab(+0.58). Sound-texture and
      complex-word density lead; lowest wordplay score in the whole corpus
      (`python -m corpus.report`: wordplay 18.2, well below the 45.2 mean) —
      worth a manual sanity-listen, since that's a large outlier in the
      opposite direction from his usual reputation as a technical lyricist;
      may indicate the 8-track sample skews toward simpler/hook material, or
      a scrape/text-quality issue worth spot-checking before trusting it.
- [x] Yashraj (n=7, still thin) — repetition(+0.60), chain(+0.54),
      assonance(+0.18), pun(+0.09). No sharp specialty; small sample.
- [x] Hanumankind (n=4, still thin) — english(+0.97), onomatopoeia(+0.47),
      metaphor(+0.22), repetition(+0.13). English-dominant catalogue still
      the clearest signal, softer than the old +2.14 because the corpus now
      contains several other heavily-English/code-switched artists
      (Tsumyoki, Lil Bhatia, Panther) that dilute how much of an outlier
      "English-heavy" is on its own.
- [x] JTrix (n=20) — codeswitch(+0.46), simile(+0.42), entendre(+0.41),
      multi_dens(+0.29). (iter-4 briefly showed holorime_dens(+0.40) as top
      axis before a same-underlying-words bug fix corrected it to 0 — see
      iteration log.)

Extra artists present in corpus/data/ but outside the original 9-name target
list (scraped by a process outside this loop's direct control — flagging so
scope is explicit, not silently expanding what "the 9 rappers" means):
- [x] Muhfaad (n=20) — syl_weight(+0.41), codeswitch(+0.34),
      consonance(+0.28), vocab(+0.24). (iter-3 flagged compound_dens(+0.41)
      as his top axis; spot-checked in iter 4 and it was a false positive —
      see iteration log.)
- [x] Emiway Bantai (n=16) — compound_dens(+0.69), codeswitch(+0.20),
      metaphor(+0.20), syl_weight(+0.17). Now the corpus's clearest
      compound/mosaic-rhyme standout post the iter-4 anti-inflation fix —
      not yet spot-checked for false positives the way Muhfaad's was; do
      that before treating it as confirmed signal.
- [x] King (n=18) — multi_dens(+0.20), assonance(+0.13), repetition(+0.11),
      syl_density(+0.07). Flat profile, no sharp specialty.
- [x] Tsumyoki (n=15) — english(+1.60), metaphor(+1.14), cadence_var(+0.82),
      onomatopoeia(+0.47). Strongly English-leaning with real metaphor
      density, not just an English-ratio artifact.
- [x] Panther (n=14) — syl_density(+1.01), consonance(+0.63), english(+0.58),
      assonance(+0.49). Dense-delivery + sound-texture profile.
- [x] Yungsta (n=9) — internal(+0.75), assonance(+0.70), pun(+0.53),
      rhyme(+0.52). Internal-rhyme and vowel-music leaning.
- [x] Vichaar (n=8) — pun(+0.60), multi_dens(+0.56), rhyme(+0.51),
      wordplay(+0.49). Broad wordplay/rhyme strength, no single dominant axis.
- [x] Raga (n=8) — alliteration(+0.47), vocab(+0.36), codeswitch(+0.28),
      pun(+0.27). Sound-texture leaning.
- [x] Naam Sujal (n=8) — vocab(+0.84), codeswitch(+0.63), entendre(+0.41),
      onomatopoeia(+0.40). Vocabulary-richness standout — highest MSTTR mean
      in the corpus (95.4, per `corpus.report`).
- [x] CarryMinati (n=7) — chain(+1.57), rhyme(+0.71), multi_dens(+0.28),
      syl_weight(+0.12). Sharp, single-axis chain-rhyme specialist — the
      clearest one-trick standout of any artist in the corpus. Lowest
      wordplay mean among artists with n>=5 (24.3) — YouTuber-turned-rapper
      profile (technical rhyme scheme over dense wordplay) matches
      reputation.
- [x] Paradox (n=5, thin) — chain(+1.28), metaphor(+1.03),
      onomatopoeia(+0.57), syl_density(+0.54). Similar chain-rhyme leaning to
      CarryMinati but paired with real metaphor density; small sample.
- [x] Lil Bhatia (n=3, too thin to trust) — pun(+1.88), english(+1.45),
      syl_weight(+1.13), wordplay(+0.83). Needs more tracks.
- [x] Calm (n=2, too thin to trust) — onomatopoeia(+2.83), alliteration(+1.70),
      consonance(+0.84). Seedhe Maut member; needs far more tracks before any
      of this is meaningful (2-song z-scores are close to noise).

## Iteration log
- (iter 1) Set up taxonomy, analysis dir, signature analyzer. See findings appended by signature.py.
- (iter 2) Built services/assonance_service.py + consonance_service.py (real
  rule-based detectors, not prototypes) with tests/test_assonance.py and
  tests/test_consonance.py. Found and fixed two correctness bugs in the vowel-
  nucleus logic during test-writing: (1) collapsing full ARPAbet vowel phonemes
  to a single first-letter bucket (AA/AH/AO/AW all -> "a") merged phonetically
  distinct vowels and caused false-positive assonance on ordinary English text;
  (2) the code picked the *last vowel in the word* rather than the *last
  primary-stressed vowel*, contradicting the module's own stated design ("same
  stressed vowel sound recurring"). Fixed to key on the full phoneme of the
  last primary-stressed vowel (falling back to the last vowel if none is
  primary-stressed). Wired both detectors into corpus/analysis/signature.py
  (new `assonance`/`consonance` axes) and tests/test_corpus_smoke.py. Full
  suite: 1444 passed (was ~1424 before the 10 new tests), including corpus
  smoke/calibration over all 193 real tracks. Re-ran signature.py — see
  updated per-artist specialty list above. Still open: EPR/Dhanji has no
  lyrics source; Encore ABJ isn't separately scraped; Calm/Hanumankind/
  Brodha V/Yashraj samples are too small (<10) to trust the z-scores yet —
  next iteration should prioritize widening the scrape (`--resume`) for the
  thin artists over adding new detector axes.
- (iter 3) Built compound/mosaic rhyme detection: `detect_compound_rhymes` +
  helpers (`_phrase_signature`, `_word_rhyme_tail`) in services/rhyme_service.py,
  test-first via tests/test_compound_rhyme.py. Definition used: a line's single
  last word's rhyme tail (phones from its last primary-stressed vowel onward,
  same convention as `_rhyme_key_en`) exactly matches the trailing phones of
  another line's last 2-3 words, AND that match must reach back into the
  earlier word of the phrase (not be satisfiable by the phrase's own last word
  alone) -- otherwise it's just ordinary end rhyme, already scored. Two
  fixed-arity bugs caught while writing tests before they shipped: (1) an
  early version matched on the anchor word's *entire* phonetic content
  including its onset consonant, which can never match a phrase (onsets
  never need to agree in real rhyme) -- fixed by using the rhyme-tail
  convention instead; (2) `pronouncing.phones_for_word` returns homograph
  pronunciations in dictionary order, not by frequency/context (e.g. "wind"
  resolves to the "wind up a toy" /waɪnd/ reading before the weather /wɪnd/
  reading used in "windmill") -- worth remembering as a general limitation of
  every service in this file that calls `phones_for_word(...)[0]`, not just
  this one. `calculate()`'s return signature grew from a 5-tuple to a 6-tuple
  (added `compound_count`); updated every fixed-arity call site (main.py,
  tests/test_corpus_smoke.py, tests/test_rhyme.py). Folded a genuine
  `compound` weight into RHYME.WEIGHTS (0.10, taken proportionally from the
  other four so they still sum to 1.0) rather than leaving it a cosmetic
  side-channel, and added it to signature.py as `compound_dens`. Full suite:
  1452 passed (up from 1444), corpus calibration still green with the
  reweighted formula. Re-ran signature.py: Muhfaad's z-profile now leads with
  compound_dens(+0.41) -- worth a manual listen/spot-check to confirm real
  signal vs. an artifact of a short-word-heavy Hinglish catalogue before
  trusting it as a genuine "specialty" claim. Still open next: holorime,
  onomatopoeia/ad-lib density, extended metaphor, punchline/setup-payoff,
  allusion detection, callback/motif reuse across sections; and the standing
  item from iter 2 to widen the scrape for thin-sample artists (Calm,
  Hanumankind, Brodha V, Yashraj all <10 tracks).
- (iter 4) Three things: (1) spot-checked Muhfaad's iter-3 compound_dens
  standout by printing the actual detected rhyme-unit pairs (short 2-4 word
  fragments, not full lyrics) — his only hit was literally the same word
  ("dream") reappearing on both sides of the match, not an independent
  cross-word phonetic coincidence. Added `_is_repeated_word_artifact` to
  `detect_compound_rhymes`: skip a match when the phrase's last word's
  spelling is a literal suffix/prefix of the anchor word (mirrors the
  existing trivial-suffix/identity-chain anti-inflation elsewhere in this
  file). Re-check confirmed Muhfaad's compound count drops to 0 — iter-3's
  flagged specialty was noise, not signal. (2) Built
  `services/onomatopoeia_service.py` (ad-lib/sound-effect interjection
  density, e.g. "woo", "skrrt", elongated "ayyy") test-first via
  `tests/test_onomatopoeia.py` — deliberately scores the *raw* lines (unlike
  `content_lines`, which strips these before other detectors run), since
  ad-lib density is itself a signal here, not noise. (3) Built holorime
  detection (`detect_holorimes`, reusing the `_phrase_signature`/
  `_last_n_words` machinery from iter 3) test-first via
  `tests/test_holorime.py`, using the classic linguistics example "ice cream"
  / "I scream" (public-domain phonetics example, not a lyric) to validate the
  mechanism. Spot-checking JTrix's resulting holorime_dens(+0.40) top-axis
  standout caught a second real bug before it shipped as a claimed
  "specialty": most of JTrix's matches were the *same phrase* respelled --
  Devanagari vs. its own Roman transliteration, or long/short-vowel spelling
  variants ("jave"/"jaave") -- which are phonetically identical by
  construction and were being counted as if they were two independently
  rhyming phrasings. Added `_is_same_underlying_words` (normalizes both
  phrases' words through the same Devanagari-roman + Hinglish-vowel
  normalization already used for signature-building, position-by-position)
  and excluded these. Re-check: JTrix's holorime count drops to 0 and his top
  axis reverts to assonance. `calculate()` grew to a 7-tuple (added
  `holorime_count`); rebalanced RHYME.WEIGHTS to add `compound` (0.10) and
  `holorime` (0.08) on top of iter-3's weights, still summing to 1.0. Full
  suite: 1658 passed (up from 1452), corpus calibration still green. Lesson
  reinforced twice this iteration: every new multi-word/cross-line detector
  in this corpus needs an explicit "is this actually the same word/phrase
  showing up twice" guard, because Hindi/Hinglish's routine
  dual-script-and-multi-spelling representation of a single word is exactly
  the kind of "different surface form, identical sound" case these detectors
  are designed to reward -- and will trigger on trivially, unless excluded.
  Still open: extended metaphor/conceit, punchline/setup-payoff, allusion
  detection, callback/motif reuse (all deferred — likely need LLM assistance
  or much larger hand-built lexicons, not a quick rule-based pass); and the
  standing item to widen the scrape for thin-sample artists (Calm,
  Hanumankind, Brodha V, Yashraj all <10 tracks) once network/token access is
  available.
- (iter 5) No new detectors — this iteration was a corpus/tracker refresh
  after discovering `corpus/data/` had grown substantially outside this
  loop's own edits (likely a scrape run against a wider/updated
  corpus/artists.py ARTISTS list): 193 songs/10 artists -> 302 songs/22
  artists. Ran the full suite (2530 passed, up from 1658 — the increase is
  almost entirely `test_corpus_smoke.py`/`test_calibration.py` parametrizing
  over ~110 more real tracks, not new test files) to confirm every existing
  detector still holds up against the larger, more varied corpus. Re-ran
  `corpus.report` and `corpus.analysis.signature` and rewrote the per-artist
  specialty section above against the current 22-artist corpus rather than
  the stale 10-artist snapshot; flagged two things needing a follow-up spot-
  check before trusting them as confirmed signal: (1) EPR's wordplay score
  (18.2) is a large outlier below the corpus mean (45.2), worth verifying
  isn't a text-quality/scrape artifact; (2) Emiway Bantai is now the
  corpus's clearest compound_dens standout post the iter-4 anti-inflation
  fix, but hasn't been spot-checked the way Muhfaad's false-positive was —
  do that before citing it as real. Also flagged scope creep worth the
  user's attention: 11 artists (Emiway Bantai, King, Tsumyoki, Panther,
  Yungsta, Vichaar, Raga, Naam Sujal, CarryMinati, Paradox, Lil Bhatia) are
  now in the corpus outside the original 9-name target list from the
  original loop prompt; documented their specialties below since the data
  exists and the signature analyzer runs over the whole corpus regardless,
  but didn't silently fold them into "the 9 rappers" framing anywhere else.
  Still open: same detector gaps as iter 4 (extended metaphor, punchline,
  allusion, callback/motif reuse), the EPR/Emiway Bantai spot-checks above,
  and widening thin-sample artists (Calm n=2, Hanumankind n=4, Brodha V n=5,
  Lil Bhatia n=3, Paradox n=5, Yashraj n=7 all still <10 tracks).
- (iter 6) KR$NA-focused close reading (No Cap, Saza-E-Maut, Makasam) surfaced
  a real Devanagari->Roman phonetics bug that was silently deflating rhyme
  scores for the whole Hindi/Hinglish corpus, not just KR$NA.
  `devanagari_to_roman` only handled nukta consonants (क़ख़ग़ज़ड़ढ़फ़) in their
  single PRECOMPOSED codepoint form, but the scraped corpus stores them
  DECOMPOSED (base consonant + combining nukta U+093C), and Unicode NFC does
  not recompose them (composition-exclusion table). So the bare nukta fell
  through the loop's else-branch and leaked into the output — e.g.
  शौख़ -> 'shaukha़', corrupting the rhyme key to 'a़', which matches nothing.
  Impact: 156 DISTINCT nukta-bearing words in KR$NA's 33-song corpus alone had
  broken rhyme keys — including सज़ा (sazaa, the actual title of
  "Saza-E-Maut"), ख़िलाफ़, हज़ारों, and the ubiquitous ड़-flap words
  थोड़ी/पड़ी/खड़े/कीचड़. Fix: handle base+nukta INSIDE the consonant branch of
  the loop (a first attempt — pre-pass string-replace to Roman — was reverted
  because emitting Roman early skipped schwa/virama/matra binding and produced
  ग़म->'gm', ज़्यादा->'z्yaadaa'; the in-loop version resolves the nukta
  variant then lets the existing virama/matra/final-schwa logic bind to it,
  giving gam / zyaadaa). Added tests/test_language_utils.py (19 tests; the
  module had none) covering decomposed-nukta romanization, no nukta/virama
  leaks, and non-nukta regression. Full suite: 2549 passed (up from 2530),
  corpus calibration still green. FLAGGED for a later iteration, NOT yet
  fixed: the two rhyme-key systems are structurally incompatible —
  `_rhyme_key_en` returns ARPAbet phone tuples like ('AO1','K') while
  `_rhyme_key_hinglish` returns letter-suffix strings like 'aunk', so a Hindi
  (Devanagari, non-CMU) word and an English (CMU) word at adjacent line-ends
  can only ever match through a degraded spelling-based fallback — this
  under-counts KR$NA's signature cross-script rhyming (भौंक/talk, शायर/Desire,
  Vikram/victim). Still open from before: extended metaphor/conceit,
  punchline/setup-payoff, allusion/pop-culture density (KR$NA is heavily
  allusive — Nas, Neymar, Vikram-Betaal, DBZ Kakarot, Saina Nehwal — none of
  it counted), callback/motif reuse.
- (iter 7) Fixed the cross-script rhyme-key incompatibility flagged at the end
  of iter 6. `_rhyme_key_en` returns ARPAbet phoneme tuples (e.g. ('AO1','K'))
  and `_rhyme_key_hinglish` returns letter-suffix strings (e.g. 'aunk') — two
  representations that can never be `==` equal, so Hindi/English code-switched
  rhymes (a hallmark KR$NA technique) silently scored as no-match. Added a
  coarser, script-agnostic fallback key — (vowel_class, final_consonant_class)
  from the last stressed/last vowel onward, using new `_ARPABET_VOWEL_CLASS`/
  `_HINGLISH_VOWEL_CLASS`/`_CONSONANT_CLASS_GROUPS` tables in
  services/rhyme_service.py — used ONLY when exactly one side of a pair is
  Devanagari (`_is_cross_script_pair`), so same-script matching keeps using
  the more precise exact keys unchanged. Wired into all three consumers: the
  end-rhyme check in `calculate()`, `detect_internal_rhymes`, and
  `detect_chain_rhymes` (a same-sound OR cross-script-match condition extends
  a chain). Guarded against 1-2 letter filler words (ना/ah both being 2
  letters produced trivial coincidental matches) with the same length>2
  convention already used for internal-rhyme candidates elsewhere in the
  module — this cut a raw diagnostic sweep over KR$NA's 33-song corpus from
  124 to 84 genuine cross-script pairs (window=4 lines, includes internal
  rhyme). Confirmed real technique now detected, not just the original
  diagnostic भौंक/talk pair — e.g. in lil-bunty.json: बंदर(bandar)/rumble,
  बंदर/jungle, rumble/कंबल(kambal), rumble/चंबल(chambal), jungle/कंबल,
  jungle/चंबल — a genuine multi-word cross-script rhyme scheme in that verse.
  Added tests/test_rhyme.py::TestCrossScriptRhyme (4 tests: matching pair,
  non-matching pair, same-script pairs untouched, full calculate() picks up
  a cross-script end rhyme in a verse). Full suite: 2553 passed (up from
  2549), zero regressions. Not perfect — some cross-script pairs still miss
  due to CMU-vs-spelling vowel-notation mismatch (e.g. English "chalk"
  spelled with a single "a" letter for its /ɔː/ sound, vs Hindi की/ka-style
  digraph "au" spellings for the same sound-class) and the ER/syllabic-r vs
  Hindi final-र consonant distinction (शायर/Desire still doesn't match) —
  these are real linguistic edge cases, not the structural blocker iter 6
  flagged, and are lower priority than the still-open detector gaps below.
- (iter 8) Built the allusion/pop-culture-reference detector flagged as open
  since iter 4. Motivated by a close read of kr-na/villain.json, which is
  saturated with reference-driven wordplay (Amrish Puri, Amjad Khan, Mogambo,
  Gabbar, "Being Human", "notorious columbus", "it's like among us", and a
  genuinely clever double allusion — "Oscar milega ledger jaise life khoke",
  chaining Heath Ledger's posthumous Oscar into the hype/knife/life rhyme) —
  none of which any existing detector credited. Added `detect_allusions()` to
  services/wordplay_service.py as a 5th sub-category alongside simile/
  metaphor/pun/entendre, following the existing detect_double_entendres/
  detect_homophone_puns pattern rather than the onomatopoeia_service density-
  only template, since allusions need multi-word phrase matching (e.g. "heath
  ledger", "vikram aur betaal") with longest-match-first span de-duplication
  so a matched multi-word reference doesn't also double-count its substring
  (e.g. "heath ledger" matching shouldn't also separately count bare
  "ledger"). Deliberately a closed lexicon
  (config/scoring_config.py WORDPLAY["ALLUSION_REFERENCES"]: Bollywood
  villains/icons, Hindu mythology, global celebrities/sports figures, hip-hop
  figures, franchise/pop-culture references) rather than a proper-noun
  heuristic — crediting every capitalized name would reward name-dropping,
  not the technique of a reference carrying real meaning. Wired into
  `calculate()`'s total_elements/total_density/max_sub_score exactly like the
  other four sub-categories, with a new ELITE_TARGETS["allusion"] = 0.04 (set
  lower than the other targets since real allusions are sparser by nature
  than similes/metaphors). Added a new `allusions_count` stat to
  models/schemas.py's ScoreBreakdown and wired it through main.py, matching
  the existing double_entendres_count/puns_count/etc. pattern.
  Before considering this complete, ran a corpus-wide diagnostic sweep (all
  302 songs) rather than trusting the villain.json spot-check alone, and
  caught a serious lexicon bug this way: "karna" (the Mahabharata figure) is
  a homograph of "karna"/"karna hai" — the single most common Hindi verb ("to
  do") — and fired 101 false-positive hits across 59 different songs (all
  verb uses, zero were the mythological figure; carryminati/trigger.json
  alone had 59 spurious hits). "kali" (goddess) has the same problem with
  "kali"/"kaali" ("black"), a common adjective (12 hits across 4 songs,
  again all color-adjective uses, zero goddess references). A closed-lexicon
  match can't disambiguate a homograph from its far more common non-reference
  sense, so both words were removed from the lexicon entirely rather than
  patched with heuristics — safer to under-detect a rare true positive than
  to systematically over-detect a common false one. Re-swept the full corpus
  after the fix: the pathological 101-hit spike is gone, all remaining
  matched references (top: gabbar 22, amjad khan 12, shahrukh 11, dhoni 10,
  joker 9 — spot-checked all lines for these plus arjun/shiva/indra/kendrick/
  eminem/drake/neymar/kakarot/sachin, all genuine references) check out as
  real. Added tests/test_wordplay.py::TestAllusions (7 tests: villain-style
  multi-reference detection, the Ledger/Oscar punchline, multi-word span
  de-dup, ordinary-capitalized-word rejection, the karna and kali homograph
  regressions, and end-to-end wiring into `calculate()`). Full suite: 2560
  passed (up from 2553), zero regressions. Still open, unchanged: extended
  metaphor/conceit (cross-line), punchline/setup-payoff, callback/motif reuse
  across sections.
  Still open: extended metaphor/conceit, punchline/setup-payoff, allusion/
  pop-culture density, callback/motif reuse — unchanged from iter 5/6.
