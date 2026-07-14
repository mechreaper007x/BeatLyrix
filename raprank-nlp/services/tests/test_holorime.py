"""
Holorime ("perfect" multi-word rhyme): two different phrases whose FULL
phonetic content matches end-to-end, not just a shared tail -- the classic
linguistics example is "ice cream" / "I scream". Distinct from compound rhyme
(a single word vs. a longer phrase) and from ordinary end rhyme (anchored on
one word each side).
"""
from services import rhyme_service as rh


class TestHolorimeDetection:
    def test_full_phrase_match_detected(self):
        lines = [
            "on a hot day we all want ice cream",
            "when something scares me I scream",
        ]
        count, pairs = rh.detect_holorimes(lines)
        assert count >= 1
        assert any("ice" in p and "cream" in p and "scream" in p for p in pairs)

    def test_identical_repeated_line_not_flagged(self):
        # Same phrase repeated verbatim is not "two phrasings", just repetition.
        lines = ["we want ice cream", "we want ice cream"]
        count, _ = rh.detect_holorimes(lines)
        assert count == 0

    def test_non_matching_phrases_score_zero(self):
        lines = ["we want ice cream", "we walked to the market"]
        count, pairs = rh.detect_holorimes(lines)
        assert count == 0 and pairs == []

    def test_short_input_safe(self):
        count, pairs = rh.detect_holorimes(["one line only"])
        assert count == 0 and pairs == []

    def test_calculate_returns_holorime_count(self):
        lines = "on a hot day we all want ice cream\nwhen something scares me I scream\n"
        result = rh.calculate(lines)
        assert len(result) == 7
        score, pairs, multi, internal, chain, compound, holorime = result
        assert holorime >= 0
