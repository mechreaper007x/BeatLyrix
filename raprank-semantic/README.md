---
title: RapRank Hindi Semantic Scoring
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
---

# RapRank Hindi Semantic Scoring Microservice

Meaning-based lyric scoring for Hindi/Hinglish rap. Two multilingual models (both cover Devanagari **and** Romanized Hinglish):

- **Embeddings** (coherence / theme / lexical spread): [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — a sentence-transformer, so cosine similarity is actually meaningful.
- **Masked-LM surprisal**: [google/muril-base-cased](https://huggingface.co/google/muril-base-cased).

> Why two models: raw masked-LM embeddings (MuRIL) are anisotropic — every sentence sits at cosine ≈0.99, so they can't tell a coherent verse from word-salad. Sentence-transformers are trained to fix exactly that.

Four label-free semantic axes that surface-level phonetic/lexicon scoring can't capture:

| Score | What it measures |
|---|---|
| `coherence_score` | Do adjacent bars connect, or is it word salad? (adjacent-line embedding cosine) |
| `semantic_surprisal_score` | How unexpected are word choices — a cleverness proxy (masked-LM pseudo-surprisal) |
| `lexical_sophistication_score` | Semantic spread of vocabulary, beyond surface TTR (pairwise embedding distance) |
| `theme_consistency_score` | How tightly lines hold the central theme (line-vs-verse-centroid cosine) |

## Endpoint

```
POST /semantic
Content-Type: application/json

{ "lyrics": "your lyrics here" }
```

Returns the four 0-100 scores plus a `metrics` object with the raw cosine /
surprisal values (kept for debugging and future percentile calibration).

> The 0-100 rescale anchors in `main.py` are heuristic. They are the first thing
> to recalibrate once a labelled/synthetic dataset exists (see the project's
> GMM percentile-calibration roadmap).
