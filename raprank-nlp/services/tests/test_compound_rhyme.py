"""
Compound / mosaic rhyme: a single word's full sound rhyming with the combined
sound of two (or three) words on the other side -- e.g. a one-word line-ending
rhyming with a multi-word phrase where neither the phrase's last word alone,
nor the single word, would match on its own without the extra word's sound.

This is distinct from ordinary end/multisyllabic rhyme (anchored on a single
last word each side) and must not double-count trivial single-word matches.
"""
from services import rhyme_service as rh


class TestCompoundSignature:
    def test_single_word_signature_is_en_mode(self):
        sig = rh._phrase_signature(["toaster"])
        assert sig is not None
        tokens, mode = sig
        assert mode == "en" and len(tokens) > 0

    def test_two_word_phrase_signature_concatenates_in_order(self):
        sig1 = rh._phrase_signature(["roast"])
        sig2 = rh._phrase_signature(["her"])
        combo = rh._phrase_signature(["roast", "her"])
        assert combo is not None
        tokens, mode = combo
        assert tokens == sig1[0] + sig2[0]

    def test_hinglish_fallback_when_no_cmu_entry(self):
        sig = rh._phrase_signature(["dilwaala"])
        assert sig is not None
        _, mode = sig
        assert mode == "translit"


class TestCompoundDetection:
    def test_word_vs_two_word_phrase_detected(self):
        # "argyle" (single word) vs "aargh aisle" (two-word phrase): the whole
        # word's sound is only reproduced by combining both words on the other
        # side, and neither word's spelling literally reappears inside
        # "argyle" -- a clean, non-inflated compound/mosaic rhyme.
        lines = [
            "his socks had a argyle",
            "she let out a aargh aisle",
        ]
        count, pairs = rh.detect_compound_rhymes(lines)
        assert count >= 1
        assert any("argyle" in p and "aargh" in p and "aisle" in p for p in pairs)

    def test_repeated_word_not_flagged_as_compound(self):
        # "cupcake" (single word) vs "cup cake" (two-word phrase): this passes
        # the phonetic match, but "cake" literally reappears inside
        # "cupcake"'s own spelling -- credit here would just be rewarding the
        # same word twice, not an independent phonetic coincidence, so it
        # must NOT count (mirrors the identity/trivial-suffix anti-inflation
        # already applied to end/chain rhyme).
        lines = ["she baked a cupcake", "he wants a cup cake"]
        count, _ = rh.detect_compound_rhymes(lines)
        assert count == 0

    def test_pure_single_word_end_rhyme_not_flagged_as_compound(self):
        # "cat" / "hat" is ordinary end rhyme on single words -- must not also
        # fire as compound (no word-boundary crossing on either side).
        lines = ["the black cat", "a red hat"]
        count, _ = rh.detect_compound_rhymes(lines)
        assert count == 0

    def test_non_rhyming_lines_score_zero(self):
        lines = ["I fixed the toaster", "we walked to the market"]
        count, pairs = rh.detect_compound_rhymes(lines)
        assert count == 0 and pairs == []

    def test_short_input_safe(self):
        count, pairs = rh.detect_compound_rhymes(["one line only"])
        assert count == 0 and pairs == []

    def test_calculate_returns_compound_count(self):
        lines = "I fixed the toaster\ngo on and roast her\n"
        result = rh.calculate(lines)
        assert len(result) == 7
        score, pairs, multi, internal, chain, compound, holorime = result
        assert compound >= 0
