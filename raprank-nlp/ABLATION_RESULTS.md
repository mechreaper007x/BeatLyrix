# DPST / BarsNet — Experiment Scoreboard

Eval protocol: full real DHH corpus (`corpus/data/`, 521 tracks after dedup),
tiers from `indian_tiers.artist_tier_map()`. Script: `test_dpst_dhh.py` or the
`scratch_eval_v2x.py` variants. Raw outputs in `dpst_dhh_eval_*.out`.

| Version | Training data | Extras | Overall | Elite rec. | Mid rec. | Comm rec. | Verdict |
|---|---|---|---|---|---|---|---|
| **v19** | mixed, pre-cleanup, random split | — | **73.7%** | 30.0% | 42.6% | 94.7% | **PRODUCTION** (commercial-biased but best overall) |
| v22 | mixed, cleaned, dual-script, held-out val | multi-task aux | 68.1% | 20.0% | 9.8% | 95.6% | honest but worse |
| v23 | = v22 | aux OFF | 65.1% | 8.3% | 14.8% | 94.1% | multi-task helps → keep aux |
| v24 | real-only | aux, CPU | 36.7% | **58.3%** | **67.2%** | 23.5% | elite signal EXISTS in real text; class-prior inverted |
| v25 | synthetic-only | aux, CPU | 62.2% | 10.8% | 26.2% | 86.8% | synth elite ≠ real elite (fully blind test) |
| BarsNet | pretrain all + tier from real-only | own arch | *training* | | | | |

## Established facts (each measured, not assumed)

1. **v19's classifier head DOES take the 10 element features** (input dim
   522 = 512 + 10, verified from checkpoint weights). All versions consume
   elements; v22+ additionally predict them (aux head).
2. **Script bias**: same lyrics, Devanagari → elite 0.46, romanized →
   commercial 0.73 (Emiway M4 A/B). Char tower reads orthography.
3. **Multi-task aux helps** (v22 vs v23: +3.0 overall, +11.7 elite recall).
4. **Real text contains the elite/mid signal** (v24 elite 58%, mid 67%) —
   the "text ceiling" theory was wrong; it's a class-prior/distribution issue.
5. **Synthetic elite is a different dialect than real elite** (v25 blind:
   EPR 75% but KR$NA 3%, Karma 0%, Seedhe Maut 8%, conf 0.84 = confidently
   wrong). Generator writes vocab-dense/consonant-heavy "EPR-style" elite,
   not punchline/flow-switch "KR$NA-style" elite.
6. **DPST architecture flaws** (fixed in BarsNet): phonetic tower had NO
   positional encoding (order-blind bag-of-phonemes → cannot see rhyme);
   inputs truncated to ~1/4 song (120 words / 1024 chars); no line
   boundaries in phoneme stream; unnormalized tower reprs vs 0-1 features.

## BarsNet design (services/barsnet.py, kernel mishmay/train-barsnet)

Own attention from primitives. LineEncoder with BACKWARD positional encoding
(rhyme-relevant slots align across lines) → RhymeGeometry (L×L suffix cosine
matrix → conv; rhyme schemes become visible 2-D patterns) → SongEncoder over
line vecs. <SYL>/<LB> tokens from DHH G2P. CharCNN texture branch (3072
chars). Heads: tier (trained on real+consented ONLY), elements (all data),
source-adversarial (gradient reversal — erases generator fingerprints).
Stage 1: masked-span pretraining (spans + line-endings) on all 1,820 texts.
Stage 2: balanced-sampler fine-tune. ~2.1M params.

Dataset: `kaggle_dataset/barsnet_dataset.json` (2,060 records) + meta.
Eval: `test_barsnet_dhh.py`.

## Next decisions

- BarsNet run 1 = shape check: want elite recall > 20% without commercial
  collapse, and val curve NOT peaking at epoch ~4. Tune λs/LR after.
- Bar to replace production v19: beat 73.7% overall with balanced recall.
- Generator work (from fact 5): seed refine-scaffold with KR$NA/Seedhe
  Maut-register attributes, not density maximization.
- Also available: threshold calibration on v19 (cheap, could improve its
  elite recall without retraining).
