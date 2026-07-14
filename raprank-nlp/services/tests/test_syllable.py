"""
Syllable counting + density scoring.

Covers three scripts the scorer must handle: English (CMU/pyphen),
Romanized Hinglish (vowel-group fallback), and Devanagari (matra/nucleus
counting with schwa deletion). The Hinglish and Devanagari cases below encode
the *correct* linguistic counts and were added to drive fixes for two bugs:

  * Devanagari nukta (़) / anusvara (ं) / chandrabindu (ँ) / visarga (ः) were
    counted as full consonant syllables, inflating Hindi counts.
  * Romanized Hinglish words were undercounted because pyphen returns >1 for
    them, so the accurate vowel-group fallback never ran.
"""
import pytest

from services import syllable_service as syl


class TestEnglish:
    @pytest.mark.parametrize("word,expected", [
        ("cat", 1), ("dog", 1), ("table", 2), ("water", 2),
        ("beautiful", 3), ("innovation", 4),
        ("business", 2), ("computer", 3),
    ])
    def test_common_english_counts(self, word, expected):
        assert syl.count_syllables(word) == expected

    @pytest.mark.parametrize("word,low,high", [
        ("every", 2, 3),     # "ev-ry" (2) or "ev-er-y" (3) — CMU says 3
        ("fire", 1, 2),      # "fire" (1) or "fi-er" (2)
        ("different", 2, 3),
    ])
    def test_ambiguous_english_counts_in_range(self, word, low, high):
        assert low <= syl.count_syllables(word) <= high

    def test_no_word_is_zero(self):
        for w in ["a", "the", "strength", "queue", "rhythm"]:
            assert syl.count_syllables(w) >= 1


class TestHinglish:
    """Romanized Hindi words must not be undercounted by English hyphenation."""

    @pytest.mark.parametrize("word,expected", [
        ("zindagi", 3),     # zin-da-gi
        ("paisa", 2),       # pai-sa
        ("aukaat", 2),      # au-kaat
        ("caucasian", 3),   # cau-ca-sian
        ("badnaam", 2),     # bad-naam
        ("mohabbat", 3),    # mo-hab-bat
    ])
    def test_hinglish_not_undercounted(self, word, expected):
        assert syl.count_syllables(word) >= expected


class TestDevanagari:
    @pytest.mark.parametrize("word,expected", [
        ("दिल", 1),          # dil
        ("प्यार", 1),         # pyaar
        ("ज़िंदगी", 3),        # zin-da-gi  (nukta + anusvara must not add syllables)
        ("ज़मीन", 2),         # za-meen    (nukta must not add a syllable)
        ("आसमान", 3),        # aa-sa-maan
        ("मोहब्बत", 3),       # mo-hab-bat
    ])
    def test_devanagari_counts(self, word, expected):
        assert syl.count_syllables(word) == expected

    def test_signs_do_not_add_syllables(self):
        # A bare consonant+matra with anusvara should equal the same without it
        assert syl.count_syllables("हूँ") == syl.count_syllables("हू")


class TestDensityScoring:
    def test_empty_lyrics(self):
        assert syl.calculate("") == (0.0, 0.0, 0.0, 0.0)

    def test_score_monotonic_in_density(self):
        sparse = "\n".join(["main hoon yahan"] * 6)          # ~3 syl/line
        dense = "\n".join(
            ["intellectual conversation manifestation revolution"] * 6)
        assert syl.calculate(dense)[0] > syl.calculate(sparse)[0]

    def test_score_bounds(self):
        verse = "Main asli mein asli hoon no cap\nBeat pe jungli hoon no cap\n" \
                "Aaj kal yahan bohot se log\nBhaunke milte occasion saath"
        score, avg, weight, ratio = syl.calculate(verse)
        assert 0.0 <= score <= 100.0
        assert 0.0 <= weight <= 100.0
        assert avg > 0
