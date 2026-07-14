"""
Vocabulary richness via Mean Segmental Type-Token Ratio (MSTTR).
"""
from services import vocabulary_service as vo


class TestVocabulary:
    def test_empty(self):
        assert vo.calculate("") == (0.0, 0.0)

    def test_repetitive_scores_low(self):
        repetitive = "\n".join(["money money money money money"] * 10)
        diverse = (
            "eloquent verbose articulate profound\n"
            "intricate elaborate sophisticated nuanced\n"
            "meticulous deliberate calculated precise\n"
            "eclectic diverse manifold multifaceted\n"
        )
        assert vo.calculate(diverse)[0] > vo.calculate(repetitive)[0]

    def test_ttr_bounds(self):
        score, ttr = vo.calculate("alpha beta gamma delta epsilon zeta")
        assert 0.0 <= ttr <= 1.0
        assert 0.0 <= score <= 100.0

    def test_all_unique_high_ttr(self):
        _, ttr = vo.calculate("apple banana cherry mango orange papaya guava")
        assert ttr > 0.9
