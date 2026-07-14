"""
Onomatopoeia / ad-lib density -- vocalized sound-effect interjections
("woo", "skrrt", "brrr", elongated "ayyy"), distinct from ordinary filler
words (already handled as stopwords elsewhere) and from the section-header /
parenthetical stripping `language_utils.content_lines` performs before
scoring other axes. This axis intentionally scores the *raw* lines, since
ad-lib density is itself a stylistic signal, not noise to discard.
"""
from services import onomatopoeia_service as ono


class TestOnomatopoeia:
    def test_parenthetical_adlib_detected(self):
        text = "walking down the street tonight\n(woo)\nfeeling like the city's mine\n(woo)\n"
        score, details = ono.calculate(text)
        assert score > 0.0
        assert any("woo" in d.lower() for d in details)

    def test_elongated_interjection_detected(self):
        text = "ayyy we made it to the top\nlooking down at everyone\n"
        score, details = ono.calculate(text)
        assert score > 0.0
        assert any("ayyy" in d.lower() for d in details)

    def test_known_word_detected(self):
        text = "brrr the money keeps coming in\nnever stopping never slowing\n"
        score, details = ono.calculate(text)
        assert score > 0.0

    def test_plain_lyrics_score_zero(self):
        text = "walking down the street tonight\nfeeling like the city is mine\n" \
               "counting all the stars above\nthinking of a better life\n"
        score, _ = ono.calculate(text)
        assert score == 0.0

    def test_ordinary_filler_words_not_counted(self):
        # "yeah" / "yo" are ordinary filler, not a sound-effect ad-lib.
        text = "yeah we made it yo\nyeah we did it yo\n"
        score, _ = ono.calculate(text)
        assert score == 0.0

    def test_empty_input_safe(self):
        score, details = ono.calculate("")
        assert score == 0.0 and details == []

    def test_score_bounded(self):
        text = "\n".join(["(woo) brrr ayyy skrrt"] * 10)
        score, _ = ono.calculate(text)
        assert 0.0 <= score <= 100.0
