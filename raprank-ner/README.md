---
title: RapRank Hindi NER
emoji: 🎤
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# RapRank Hindi NER Microservice

Named Entity Recognition for Hindi/Hinglish rap lyrics using [ai4bharat/IndicNER](https://huggingface.co/ai4bharat/IndicNER).

## Endpoint

```
POST /ner
Content-Type: application/json

{ "text": "your lyrics here" }
```

Returns detected PERSON / ORG / LOC entities — Devanagari and Romanized Hinglish both supported.
