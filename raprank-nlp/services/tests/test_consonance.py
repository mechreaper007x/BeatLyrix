"""
Consonance detection — requires 3+ distinct content words in a line sharing a
non-onset consonant sound. Onset repetition (alliteration) must NOT count here.
"""
from services import consonance_service as cons


class TestConsonance:
    def test_clear_consonance_detected(self):
        line = "blank think junk plank drink"
        score, details = cons.calculate(line + "\n" + line)
        assert score > 0.0
        assert any("NGK" in d or "K" in d for d in details)

    def test_no_consonance_scores_zero(self):
        # Separated by more than WINDOW_SIZE_LINES stopword-only lines so the
        # two sentences fall outside each other's cross-line cluster window
        # (they otherwise coincidentally share a consonant, e.g. brown/morning).
        filler = "\n".join(["the a an is of to"] * 5)
        text = "the quick brown fox jumped over lazy dogs\n" \
               f"{filler}\n" \
               "morning coffee tastes wonderful every single day\n"
        score, _ = cons.calculate(text)
        assert score == 0.0

    def test_no_shared_noninitial_consonant(self):
        # Each word's only consonant sound is its onset (dropped by design), so
        # no non-onset consonant is shared across words -> no consonance.
        score, _ = cons.calculate("otter echo away ivy oboe\n")
        assert score == 0.0

    def test_two_words_insufficient(self):
        score, _ = cons.calculate("blank think elephant walked away\n")
        assert score == 0.0

    def test_score_bounded(self):
        text = "\n".join(["blank think junk plank drink shrink"] * 5)
        score, _ = cons.calculate(text)
        assert 0.0 <= score <= 100.0

    def test_noninitial_consonant_shared_across_separate_lines(self):
        # No single line has 3+ words sharing a non-onset consonant, but the
        # "-NGK" ending recurs across 4 nearby lines.
        text = "\n".join([
            "walking down this blank street",
            "always trying not to think",
            "counting every single drink",
            "hoping that my mind wont sink",
        ])
        score, details = cons.calculate(text)
        assert score > 0.0
        assert any("NG" in d for d in details)
