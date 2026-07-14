"""
Assonance detection — requires 3+ distinct content words in a line sharing a
vowel nucleus (non-onset vowel music, independent of alliteration/rhyme).
"""
from services import assonance_service as aso


class TestAssonance:
    def test_clear_assonance_detected(self):
        line = "make late shape amazing gate"
        score, details = aso.calculate(line + "\n" + line)
        assert score > 0.0
        assert any("EY" in d for d in details)

    def test_no_assonance_scores_zero(self):
        # Separated by more than WINDOW_SIZE_LINES stopword-only lines so the
        # two sentences fall outside each other's cross-line cluster window
        # (they otherwise coincidentally share nuclei, e.g. fox/dogs/coffee).
        filler = "\n".join(["the a an is of to"] * 5)
        text = "the quick brown fox jumped over lazy dogs\n" \
               f"{filler}\n" \
               "morning coffee tastes wonderful every single day\n"
        score, _ = aso.calculate(text)
        assert score == 0.0

    def test_two_words_insufficient(self):
        score, _ = aso.calculate("make late elephant walked away\n")
        assert score == 0.0

    def test_stop_words_excluded(self):
        # "the" is a stopword and must not count toward the group; the
        # remaining content words also carry distinct vowel nuclei.
        score, _ = aso.calculate("the the the sun sky red big work\n")
        assert score == 0.0

    def test_score_bounded(self):
        text = "\n".join(["make late shape amazing gate replace"] * 5)
        score, _ = aso.calculate(text)
        assert 0.0 <= score <= 100.0

    def test_vowel_nucleus_shared_across_separate_lines(self):
        # No single line has 3+ words sharing a vowel nucleus, but the same
        # "EY" nucleus recurs across 4 nearby lines.
        text = "\n".join([
            "make your own way",
            "late night escape",
            "shape the day",
            "amazing grace",
        ])
        score, details = aso.calculate(text)
        assert score > 0.0
        assert any("EY" in d for d in details)
