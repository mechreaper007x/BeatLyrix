# RapRank NLP — test suite

Regression + calibration tests for the rule-based scorers in `services/`
(syllable, rhyme, alliteration, vocabulary, wordplay), covering English,
Romanized Hinglish, and Devanagari (Hindi) text.

## Layout

```
tests/
  conftest.py            # path setup + optional corpus loader
  test_syllable.py       # English / Hinglish / Devanagari syllable counting
  test_rhyme.py          # end / internal / multisyllabic rhyme keys + ordering
  test_alliteration.py   # onset-proximity alliteration
  test_vocabulary.py     # MSTTR vocabulary richness
  test_wordplay.py       # similes / metaphors / puns / entendres + small-sample calibration
  test_corpus_smoke.py   # every corpus track scores without error, bounded output
  test_calibration.py    # corpus-level distribution health + cross-source discrimination
```

## Running

```bash
# from raprank-nlp/
pip install -r requirements.txt -r requirements-dev.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('stopwords','wordnet','omw-1.4')]"

# On Windows, force UTF-8 so Devanagari test ids print:
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m pytest tests/ -q
```

## Corpus-driven tests

`test_corpus_smoke.py` and `test_calibration.py` run every scorer over a local
lyric corpus and check for crashes, bounded output, and healthy score
distributions. The corpus is produced by local, git-ignored tooling and lives
under a git-ignored data directory, so these two suites **skip cleanly** when
the corpus is absent (e.g. a fresh checkout or CI). The unit tests always run.

Corpus records are plain JSON of the shape:

```json
{ "artist": "<source label>", "title": "<title>", "lyrics": "<text>",
  "line_count": <int>, "primary_language": "hi|en|mixed" }
```

Drop any files of this shape into the git-ignored data directory to exercise the
corpus tests against your own material.
