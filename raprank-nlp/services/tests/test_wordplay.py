"""
Wordplay detection — similes, metaphors, homophone puns, double entendres.

The small-sample test encodes a fix for a density-explosion bug: on a very
short verse a single detected device produced density ~0.25 which maxed out a
sub-score, awarding ~97/100 to lyrics with no real wordplay. Density must be
smoothed by a minimum line denominator so short inputs can't spike.
"""
from services import wordplay_service as wp


class TestDetectors:
    def test_simile_detected(self):
        n, m, sims, _ = wp.detect_similes_and_metaphors(
            "I shine like a diamond in the rough tonight\n")
        assert n >= 1

    def test_metaphor_detected(self):
        _, m, _, metas = wp.detect_similes_and_metaphors(
            "I am a beast when I grab the microphone\n")
        assert m >= 1

    def test_hindi_simile_jaise(self):
        n, _, _, _ = wp.detect_similes_and_metaphors(
            "sher jaise dahaadta hoon main stage pe aake\n")
        assert n >= 1





class TestCalibration:
    def test_trivial_short_verse_not_elite(self):
        """A short, device-free verse must not score near-elite (small-sample bug)."""
        verse = (
            "Main asli mein asli hoon no cap\n"
            "Beat pe jungli hoon no cap\n"
            "Aaj kal yahan bohot se log\n"
            "Bhaunke milte occasion saath\n"
        )
        score, _ = wp.calculate(verse)
        assert score < 70.0

    def test_word_variants_not_puns(self):
        """Inflections / spelling variants of the same word are not puns."""
        assert wp._is_word_variant("lagta", "lagte")     # verb conjugation
        assert wp._is_word_variant("poora", "pura")      # spelling variant
        assert not wp._is_word_variant("sun", "son")     # genuine homophone pair

    def test_conjugation_heavy_verse_low_puns(self):
        verse = (
            "main tujhe dekhta tu mujhe dekhti\n"
            "wo humko rokta hum unko rokte\n"
            "sab kuch hota tha sab kuch hoti\n"
        )
        _, meta = wp.calculate(verse)
        assert meta["puns_count"] <= 1

    def test_score_bounded(self):
        score, meta = wp.calculate("random words here\nnothing clever going on\n")
        assert 0.0 <= score <= 100.0
        assert set(["simile_count", "metaphor_count", "puns_count",
                    "double_entendres_count"]).issubset(meta)

    def test_dense_wordplay_beats_plain(self):
        dense = (
            "I am a lion, sharp like a blade in the night\n"
            "cold as ice but I burn like a flame so bright\n"
            "I am a king, fierce like a storm in the fight\n"
            "hard as stone, I strike like a bolt of light\n"
        )
        plain = (
            "I woke up and I went to the store today\n"
            "bought some milk and then I drove back home\n"
            "sat on the couch and watched a little tv\n"
            "then I made some food and went to sleep\n"
        )
        assert wp.calculate(dense)[0] > wp.calculate(plain)[0]
