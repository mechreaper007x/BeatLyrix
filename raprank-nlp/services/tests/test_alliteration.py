"""
Alliteration detection — requires 3+ adjacent content words sharing an onset.
"""
from services import alliteration_service as al


class TestAlliteration:
    def test_clear_alliteration_detected(self):
        line = "silent snakes slither slowly southward"
        score, details = al.calculate(line + "\n" + line.replace("s", "s"))
        assert score > 0.0
        assert any("S" in d for d in details)

    def test_no_alliteration_scores_zero(self):
        text = "the quick brown animal jumped over lazy dogs\n" \
               "morning coffee tastes wonderful every single day\n"
        score, details = al.calculate(text)
        assert score == 0.0

    def test_two_words_insufficient(self):
        # Only two 'b' words -- below the 3-occurrence minimum, should not score
        score, _ = al.calculate("big brown elephant walked away\n")
        assert score == 0.0

    def test_three_words_sufficient(self):
        # Three distinct 'b' words meets the minimum (MIN_OCCURRENCES_PER_GROUP=3)
        score, _ = al.calculate("big brown bear walked away\n")
        assert score > 0.0

    def test_midline_word_repetition_counts(self):
        # Same word repeated mid-line (not line-initial) is still
        # alliteration -- "laal ... laal ... laal" should score > 0, not 0,
        # even though none of the occurrences is the first word of its line.
        score, details = al.calculate("raste pe chalke kare talve laal, ab talve laal jab ye joote laal\n")
        assert score > 0.0
        assert any("repeated" in d for d in details)

    def test_stop_words_excluded(self):
        # "the the the" are stop words -> should not count as alliteration
        score, _ = al.calculate("the the the cat ran fast\n")
        assert score == 0.0

    def test_score_bounded(self):
        text = "\n".join(["dark daring dangerous demons dance"] * 5)
        score, _ = al.calculate(text)
        assert 0.0 <= score <= 100.0

    def test_distinct_words_alliterate_across_separate_lines(self):
        # No single line has 3+ shared-onset words, but 4 different lines each
        # lead with a distinct "D" word within the cross-line window.
        text = "\n".join([
            "Dilli dekhi shehron mein",
            "Dilli ghoomi galiyon mein",
            "Dilli jeeti maidano mein",
            "Dilli haari kabhi nahi",
        ])
        score, details = al.calculate(text)
        assert score > 0.0
        assert any("D" in d for d in details)

    def test_same_word_repeated_hook_across_lines(self):
        # The KRSNA "Dekh kaun aaya wapas" pattern: the hook's first word
        # repeats across many consecutive lines. Per-line scoping alone
        # cannot see this (each line has < 3 distinct alliterating words) --
        # this is the case that motivated the cross-line rewrite.
        hook = "Dekh kaun aaya wapas"
        filler = "yeh sab bekar baatein hain jo mujhe pasand nahi aati"
        text = "\n".join([hook] * 8 + [filler] * 4)
        score, details = al.calculate(text)
        assert score > 0.0
        assert any("repeated" in d for d in details)
