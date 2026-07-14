"""
Rhyme detection: end rhyme, internal rhyme, multisyllabic, and chains.

English rhyme keys come from the CMU dictionary (last stressed vowel onward);
Hinglish uses a romanized vowel-nucleus heuristic. These tests pin the phonetic
key behavior and check that a clearly-rhyming verse out-scores a
non-rhyming one.
"""
import pytest

from services import rhyme_service as rh


class TestEnglishRhymeKeys:
    @pytest.mark.parametrize("a,b", [
        ("nation", "station"), ("cat", "hat"), ("time", "crime"),
        ("light", "night"), ("fire", "desire"), ("play", "day"),
    ])
    def test_true_rhymes_share_key(self, a, b):
        assert rh._rhyme_key_en(a) == rh._rhyme_key_en(b)

    @pytest.mark.parametrize("a,b", [
        ("cat", "dog"), ("orange", "table"), ("time", "money"),
    ])
    def test_non_rhymes_differ(self, a, b):
        assert rh._rhyme_key_en(a) != rh._rhyme_key_en(b)


class TestHinglishNormalization:
    @pytest.mark.parametrize("word,expected", [
        ("aaya", "aya"), ("bhaari", "bari"), ("khaali", "kali"),
    ])
    def test_normalize_collapses_digraphs_and_long_vowels(self, word, expected):
        assert rh.normalize_hinglish(word) == expected

    def test_hinglish_rhyme_keys_match_for_rhyming_pair(self):
        # "aaya" / "gaaya" rhyme on -aya
        assert rh._rhyme_key_hinglish("aaya") == rh._rhyme_key_hinglish("gaaya")


class TestEndToEnd:
    def test_rhyming_verse_beats_non_rhyming(self):
        rhyming = (
            "I write the flow tonight so bright\n"
            "burning up the mic with all my might\n"
            "reaching for the sky in endless flight\n"
            "everything I grip is holding tight\n"
        )
        non_rhyming = (
            "I walk into the room and sit\n"
            "the weather outside is very cold\n"
            "she bought some bread from the store\n"
            "we talked about the news today\n"
        )
        assert rh.calculate(rhyming)[0] > rh.calculate(non_rhyming)[0]

    def test_short_input_safe(self):
        score, pairs, multi, internal, chain, compound, holorime = rh.calculate("one line only")
        assert score == 0.0 and pairs == []

    def test_scores_bounded(self):
        score, pairs, multi, internal, chain, compound, holorime = rh.calculate(
            "raat ki baat hai\nsaath tere jaat hai\ndil ki yeh baat hai\n")
        assert 0.0 <= score <= 100.0
        assert 0.0 <= internal <= 100.0
        assert 0.0 <= chain <= 100.0
        assert compound >= 0
        assert holorime >= 0

    def test_duplicate_hook_lines_not_double_counted(self):
        """Repeated identical hook lines should be de-duplicated before scoring."""
        hooky = "\n".join(["paisa paisa chahiye mujhe"] * 8)
        score, pairs, *_ = rh.calculate(hooky)
        # All lines identical -> after de-dup only one line -> no rhyme pairs
        assert pairs == []

    def test_repeated_two_word_phrase_not_a_rhyme(self):
        """Two DIFFERENT lines that both end in the same literal 2-word phrase
        (e.g. "...kya hai" / "...kya hai") are phrase repetition, not a rhyme.
        Before the fix, the identity-rhyme fallback compared the second-to-
        last word of each line -- but never checked whether THAT word was
        also identical, so it credited the phrase repeating itself as a
        (sometimes multisyllabic) rhyme. Found via close-reading KR$NA's
        been-a-while.json ("mera taaj kya hai" / "mera takht kya hai" /
        "...luck kya hai" all correctly rhyme on "hai", but the fallback also
        spuriously matched "kya" against "kya")."""
        verse = (
            "mera taaj kya hai\n"
            "mera takht kya hai\n"
            "yeh sab kuch bakwaas\n"
        )
        score, pairs, *_ = rh.calculate(verse)
        matched_words = {(p.word_a.lower(), p.word_b.lower()) for p in pairs}
        assert ("kya", "kya") not in matched_words
        assert pairs == []


class TestCrossScriptRhyme:
    """Hindi/English code-switched rhyme (e.g. KR$NA's "bhaunk"/"talk") can't
    match on the exact ARPAbet-tuple vs. Hinglish-string keys -- those are
    different representations that are never `==`. The coarse vowel+consonant
    class key below is a script-agnostic fallback used only for such pairs."""

    def test_matching_cross_script_pair(self):
        assert rh._cross_script_keys_match("भौंक", "talk")

    def test_non_rhyming_cross_script_pair_does_not_match(self):
        assert not rh._cross_script_keys_match("भौंक", "soft")

    def test_same_script_pairs_are_untouched(self):
        # Cross-script fallback must not fire when both words share a script.
        assert not rh._is_cross_script_pair("bhaunk", "talk")
        assert not rh._is_cross_script_pair("talk", "chalk")
        assert not rh._is_cross_script_pair("भौंक", "शौख़")

    def test_cross_script_end_rhyme_detected_with_devanagari(self):
        verse = (
            "gali mein kutta jab bhi भौंक\n"
            "sunne ko koi taiyar nahi bas talk\n"
        )
        score, pairs, *_ = rh.calculate(verse)
        matched_words = {(p.word_a, p.word_b) for p in pairs}
        assert ("भौंक", "talk") in matched_words or ("talk", "भौंक") in matched_words


class TestLyricismDiscrimination:
    """Unskilled rhyme patterns (conjugation rhyme, monotone schemes) must not be
    credited like skilled lyricism."""

    def test_conjugation_rhyme_not_multisyllabic(self):
        # -aya/-ani grammatical endings rhyme automatically; not a skill.
        assert rh._is_trivial_suffix_rhyme("rulaya", "kamaya")
        assert rh._is_trivial_suffix_rhyme("sunani", "jalani")
        # a real multisyllabic rhyme shares stem material, not just the ending
        assert not rh._is_trivial_suffix_rhyme("insaan", "imaan")

    def test_monotone_scheme_scores_below_diverse(self):
        # Same two rhyme sounds hammered every line vs varied rich rhymes.
        monotone = (
            "aaj maine socha tujhko rulaya\n"
            "kal tune humko yaha bulaya\n"
            "phir usne saara kuch chhupaya\n"
            "ab maine dekho sab samjhaya\n"
            "wo baatein karke tujhe sataya\n"
            "raatein guzari maine gaya\n"
        )
        diverse = (
            "intricate lyrical miracle spiritual\n"
            "burning the concrete streets of the capital\n"
            "flowing meticulous vicious and criminal\n"
            "shifting the atmosphere breaking the physical\n"
            "razor sharp similes cutting political\n"
            "elevate every bar into a pinnacle\n"
        )
        assert rh.calculate(monotone)[0] < rh.calculate(diverse)[0]
