"""
Corpus-driven robustness: run every real scraped track through every local
scorer and assert it completes without error and returns finite, bounded
scores. This is the regression net — it exercises the scorers on thousands of
lines of real Hindi/Hinglish/English rap the unit tests can't anticipate.

Skips cleanly if the corpus hasn't been scraped (corpus/data/ empty).
"""
import math

import pytest

from services import (
    alliteration_service, assonance_service, consonance_service,
    onomatopoeia_service, rhyme_service, syllable_service, vocabulary_service,
    wordplay_service,
)


def _finite_pct(x):
    return isinstance(x, (int, float)) and math.isfinite(x) and 0.0 <= x <= 100.0


def test_corpus_present(corpus_tracks):
    if not corpus_tracks:
        pytest.skip("corpus not scraped (corpus/data empty)")
    assert len(corpus_tracks) >= 50


class TestEveryTrackScores:
    def test_syllable(self, track):
        if not track.get("lyrics"):
            pytest.skip("no corpus")
        score, avg, weight, ratio = syllable_service.calculate(track["lyrics"])
        assert _finite_pct(score) and _finite_pct(weight)
        assert avg >= 0 and 0.0 <= ratio <= 1.0

    def test_rhyme(self, track):
        if not track.get("lyrics"):
            pytest.skip("no corpus")
        score, pairs, multi, internal, chain, compound, holorime = rhyme_service.calculate(track["lyrics"])
        assert _finite_pct(score) and _finite_pct(internal) and _finite_pct(chain)
        assert multi >= 0 and isinstance(pairs, list) and compound >= 0 and holorime >= 0

    def test_alliteration(self, track):
        if not track.get("lyrics"):
            pytest.skip("no corpus")
        score, details = alliteration_service.calculate(track["lyrics"])
        assert _finite_pct(score) and isinstance(details, list)

    def test_assonance(self, track):
        if not track.get("lyrics"):
            pytest.skip("no corpus")
        score, details = assonance_service.calculate(track["lyrics"])
        assert _finite_pct(score) and isinstance(details, list)

    def test_consonance(self, track):
        if not track.get("lyrics"):
            pytest.skip("no corpus")
        score, details = consonance_service.calculate(track["lyrics"])
        assert _finite_pct(score) and isinstance(details, list)

    def test_onomatopoeia(self, track):
        if not track.get("lyrics"):
            pytest.skip("no corpus")
        score, details = onomatopoeia_service.calculate(track["lyrics"])
        assert _finite_pct(score) and isinstance(details, list)

    def test_vocabulary(self, track):
        if not track.get("lyrics"):
            pytest.skip("no corpus")
        score, ttr = vocabulary_service.calculate(track["lyrics"])
        assert _finite_pct(score) and 0.0 <= ttr <= 1.0

    def test_wordplay(self, track):
        if not track.get("lyrics"):
            pytest.skip("no corpus")
        score, meta = wordplay_service.calculate(track["lyrics"])
        assert _finite_pct(score)
        for k in ("simile_count", "metaphor_count", "puns_count",
                  "double_entendres_count"):
            assert meta[k] >= 0
