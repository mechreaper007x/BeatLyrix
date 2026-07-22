---
title: RapRank NLP Service
emoji: 🎤
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# RapRank NLP Service & BarsNet V2 🎤⚡

> **Multi-Lingual Hip-Hop NLP Engine, G2P Syllabifier & Deep Dual-Tower Neural Transformer**

Part of the **BeatLyrix / RapRank** microservices suite.

---

## 🌟 Capabilities

* **BarsNet V2 Transformer Model (2.39M Parameters)**: Neural dual-tower architecture with **Backward Rotary Position Embeddings (RoPE)**, **Rhyme Geometry 2D Matrix Convolutions**, and **Symmetric Co-Attention**.
* **Phonetic G2P Engine**: Custom neural Grapheme-to-Phoneme translator with multi-syllabic boundary insertion (`<SYL>`).
* **Lyrical Quality Index (LQI)**: Computes 16 technical, phonetic, and structural feature axes:
  - Syllable Density & Syllable Weight Ratio
  - Multisyllabic End-Rhyme & Internal Rhyme Schemes (LLPC)
  - Wordplay & Literary Devices (Similes, Puns, Double Entendres, Metaphors, Allusions)
  - Vocabulary Uniqueness & MSTTR
  - Sound Devices (Assonance, Consonance, Onomatopoeia)
  - Code-Switching (Hindi / English ratio), Anaphora Repetition, and Cadence Variance
* **Multi-Head Consensus Classifier**: Combines **BarsNet V2**, **Random Forest**, **SVM**, and **Bayesian Networks** into a 4-model consensus tier prediction (`elite`, `mid`, `commercial`).

---

## 🚀 Fast API Endpoints

- **`GET /health`**: Liveness probe
- **`POST /analyze`**: Text-only lyrics scoring & BarsNet V2 inference
- **`POST /transcribe-and-analyze`**: End-to-end audio transcription (Whisper), beat-sync flow scoring, and NLP analysis

---

## 📁 Key File Structure

```
raprank-nlp/
├── main.py                          # FastAPI application & endpoint handlers
├── prepare_barsnet_dataset.py       # Dataset preprocessor for BarsNet V2
├── config/
│   └── scoring_config.py            # Calibrated weights & threshold curves
├── models/
│   └── schemas.py                   # Pydantic request/response schemas
└── services/
    ├── barsnet.py                   # BarsNet V2 PyTorch model implementation
    ├── dpst_quality_service.py      # Inference wrapper for BarsNet V2 & G2P
    ├── lyrical_compiler.py          # LQI Lexer & Parser Compiler
    ├── bayesian_scoring_service.py  # Bayesian Belief Network classifier
    ├── rf_quality_service.py        # Random Forest classifier
    ├── svm_quality_service.py       # Support Vector Machine classifier
    ├── wordplay_service.py          # Puns, entendres, and simile detectors
    ├── vocabulary_service.py        # MSTTR lexical richness analyzer
    ├── assonance_service.py         # Vowel harmony analyzer
    ├── consonance_service.py        # Consonant cluster analyzer
    ├── onomatopoeia_service.py      # Ad-lib sound effect detector
    └── prosody_service.py           # Code-switching & anaphora analyzer
```

---

## 📜 License

Distributed under the MIT License. Part of the BeatLyrix open-source suite.
