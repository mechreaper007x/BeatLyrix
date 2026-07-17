# Prior Art: Rap Lyrics Technology — Verified Survey

*Compiled 2026-07-16 via deep-research workflow: 5 search angles, 22 sources fetched,
109 claims extracted, top 25 adversarially verified (3 independent votes each) —
23 confirmed, 0 refuted, 2 unverified (infrastructure errors, not refutations).*

**Bottom line for this project:** across 2009–2025 academia, open source, and
industry, no verified system does rhyme/skill scoring for Hindi/Hinglish/
code-switched rap. The entire verified lineage is English (or Finnish) and
follows one pipeline: phonetic transcription (CMU dict / eSpeak / ARPAbet) →
vowel/assonance matching → aggregate rhyme-density-style metrics. raprank-nlp's
DHH phoneme engine + cross-script rhyme keys + tier scoring occupies unclaimed
territory. (Caveat: absence of verification ≠ nonexistence.)

---

## 1. Academic lineage — rap analysis & skill scoring (all 3-0 verified)

### Hirjee & Brown — the foundational rhyme detector (ISMIR 2009 / EMR 2010)
- **What:** Probabilistic rhyme-scoring model based on phoneme frequencies;
  detects internal, line-final, and **imperfect** rhymes (previously ignored in MIR).
- **Training data:** 27,956 manually corrected lines (13,978 rhymed pairs) from
  40 "Golden Age" (1984–1994) rap albums.
- **Phonetics:** CMU Pronouncing Dictionary augmented with hip-hop slang
  ("brotha", "runnin'") + NRL text-to-phoneme rules for OOV words.
- **Skill/style stats:** rhymes per line, rhyme density, internal/bridge/link/
  chain rhymes, multisyllabic proportions. E.g. Rakim 0.63 internal rhymes/line
  vs Run-DMC 0.48; Eminem 2.3, Jay-Z 2.2 rhymes/line. The earliest automated
  rapper style characterization; origin of **rhyme density** as the de facto
  skill metric.
- **Sources:** https://ismir2009.ismir.net/proceedings/OS8-1.pdf ·
  https://kb.osu.edu/handle/1811/48548
- *Unverified (verifier infra errors, plausible):* log-odds scoring adapted from
  BLOSUM protein-homology matrices; ROC cross-validation beating
  minimal-articulatory-feature and Kondrak-alignment baselines.

### Raplyzer / Raplysaattori — per-artist rhyme factor (Eric Malmi, 2015)
- **What:** Detects rhymes and computes rhyme lengths in Finnish and English rap.
- **Method:** eSpeak phonetic transcription → strip consonants → longest
  matching vowel (assonance) sequences → average = per-artist **"rhyme factor"**
  ranking rappers by rhyming ability.
- **Status:** dormant since ~2015-16. MIT-licensed Python.
- **Sources:** https://github.com/ekQ/raplysaattori ·
  https://mining4meaning.com/2015/02/13/raplyzer/

### DopeLearning / DeepBeat (Malmi, Takala, Toivonen, Raiko, Gionis — KDD 2016)
- **What:** Rap lyric generation by predicting the best next line from existing
  songs; RankSVM + novel deep neural network.
- **Evaluation:** rhyme density metric (Raplyzer lineage); generated lyrics
  outperformed the best human rappers by 21% on rhyme density.
- **Deployment:** deepbeat.org (dead); usage logs showed ML rankings correlate
  with user preferences.
- **Sources:** https://arxiv.org/abs/1505.04771 · https://github.com/ekQ/dopelearning

### HAVAE — rhyme2vec (Liang et al., Nankai U.; PRICAI 2018 / WWW Journal 2019)
- **What:** Hierarchical attention variational autoencoder jointly learning
  semantic + prosodic representations of rap lyrics.
- **Method:** **rhyme2vec** — phoneme-based rhyme embeddings from phonetic
  transcriptions, handling monorhyme and alternate-rhyme schemes.
- **Results:** beat DopeLearning on rhyme density 2.278 vs 1.436 (Hirjee &
  Brown's metric); evaluated on next-line prediction, generation, genre
  classification.
- **Source:** https://link.springer.com/article/10.1007/s11280-019-00672-2

### RapViz (Müller, Panzer & Beck — IVAPP/VISIGRAPP 2025)
- **What:** Most recent academic system; auto-detects assonance rhymes and
  renders interactive, audio-synchronized visualizations of rhyme groups/schemes.
- **Method:** syllable-pair rhyme score (0–1) = vowel-similarity-gated weighted
  mean of vowel similarity, stress, and consonant-suffix similarity over
  ARPAbet/CMU transcriptions, using **perceptual confusion matrices** (Phatak &
  Allen 2007 vowels; Woods et al. 2010 consonants). Rhyme groups via
  DBSCAN/HDBSCAN over the similarity matrix (D = 1 − S), parameters
  auto-selected by silhouette score.
- **Source:** https://www.scitepress.org/Papers/2025/131907/131907.pdf
  (DOI 10.5220/0013190700003912)

### RapAnalysis (Marozick, Elfandi, Mayer — 2020)
- **What:** Python/Flask rhyme-detection web service; color-codes rhyme groups
  (HSL hue per group) via NLTK + CMU dict phoneme matching.
- **Status:** dead — created Oct 2020, last push Nov 2020. MIT.
- **Source:** https://github.com/alexmarozick/RapAnalysis

### Cross-cutting pattern (verified synthesis)
Every verified rap-analysis system 2009–2025 follows the same pipeline:
**phonetic transcription → vowel/assonance-focused similarity → aggregate
style/skill metrics.** Rhyme density (Hirjee & Brown, popularized by Malmi) is
the standard rap-skill/lyrical-complexity metric across the whole literature.

---

## 2. Indian / Hinglish adjacent work (fetched; below top-25 verification cut)

No direct competitor found — only adjacent resources:

| Project | Year | What it is |
|---|---|---|
| **Bollyrics** (arXiv 2007.12916) | 2020 | Automatic lyric *generator* for romanized Hindi songs; no rhyme scoring |
| **Gupta, Choudhury & Bali** (MSR India, LREC) | 2012 | Mining Hindi–English transliteration pairs from online lyrics sites (word-by-word aligned transliteration) |
| **L3Cube-HingCorpus** (arXiv 2204.08398) | 2022 | First large-scale Roman-script Hindi-English code-mixed corpus: 52.93M sentences / 1.04B tokens (Twitter, not lyrics) |
| **Hinglish-Hindi parallel corpus** (Kaggle, stutig29) | 2020 | Hindi film-song lyrics in Devanagari + romanized parallel form |
| **CS-LLM survey** (arXiv 2505.00035) | 2025 | 327 studies, 30+ datasets, 80+ languages of code-switching NLP — context, no rap |

**Zero published work on Hindi/Hinglish rap rhyme detection, cross-script
(Devanagari↔Roman) rhyme matching, or Desi hip-hop skill scoring.**

---

## 3. Commercial products (fetched; unverified tier)

| Product | Year | What it does | Status |
|---|---|---|---|
| **AutoRap by Smule** | 2012 | Speech→rap via pitch correction + beat-matching flow; 5000+ beats; battles judged by *community voting, not algorithms* | active |
| **Uberduck** | ~2020 | Text-to-rap synthetic vocals (speech/singing/rapping) + developer API | active |
| **Rap Fame / Battle Me** | — | Sought explicitly; nothing verifiable surfaced on algorithmic scoring | open question |

No Indian rap-tech startup surfaced in any search angle. No commercial product
implements genuine rhyme/skill scoring (vs. generation or community voting).

---

## 4. Datasets / corpora (fetched; unverified tier)

- **Genius Song Lyrics w/ language info** (Kaggle, carlosgdcj) — multi-million
  song corpus, per-song language tags, data through 2022
- **brunokreiner/genius-lyrics** (HuggingFace) — 480,855 English-classified
  songs via lyricsgenius + Spotify Million Playlist metadata
- **rikdifos/rap-lyrics** (Kaggle) — "Hip-Hop Encounters Data Science", 36 rappers
- **Cropinky/rap_lyrics_english** (HuggingFace) — Genius-API-scraped rap lyrics
- **arXiv 2510.07037** (2025) — 3,814 songs / 146 hip-hop artists / 1980–2020
  (~2.3M words), stratified by US region and era

**No public Hindi/Hinglish rap lyric dataset exists** on Kaggle, HuggingFace, or
in academic releases (as of this sweep).

---

## Open questions (not settled by this survey)

1. Any Indian-language rap NLP beyond the adjacent work above (Punjabi, Tamil)?
2. Do Rap Fame/Battle Me implement algorithmic scoring or only community voting?
3. Legal usability/annotation quality of the public English rap datasets.
4. The two unverified Hirjee & Brown methodology claims (BLOSUM-inspired scoring;
   ROC benchmarking) — recheck ISMIR 2009 PDF §3–5 directly.

## Methodology note

Findings above marked "verified" passed 3-0 adversarial verification against
primary sources (paper PDFs, GitHub repos/APIs). "Fetched; unverified tier"
items came from primary sources (App Store listings, dataset cards, arXiv) but
did not go through the verification vote — treat as reliable-but-unchecked.
Status assessments reflect repo timestamps / listings as of July 2026.
