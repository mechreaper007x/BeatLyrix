"""
Tests for services/language_utils.devanagari_to_roman, focused on the
decomposed-nukta handling (base consonant + combining U+093C).

Nukta consonants (क़ख़ग़ज़ड़ढ़फ़) are pervasive in Hindi/Urdu-influenced rap
vocabulary (ख़ौफ़, ज़िंदगी, राज़, फ़र्ज़ ...). They are commonly stored
DECOMPOSED (base + U+093C) rather than as the single precomposed codepoint,
and Unicode NFC does not recompose them (composition-exclusion list). Before
the fix, the decomposed nukta fell through untranslated and corrupted the
romanization -- and therefore the rhyme key -- of every such word.
"""
from __future__ import annotations

import unicodedata

import pytest

from services.language_utils import devanagari_to_roman

NUKTA = "़"


def _decomposed(s: str) -> str:
    """Force NFD so nukta letters are stored as base + U+093C, mirroring how
    the scraped corpus stores them."""
    return unicodedata.normalize("NFD", s)


@pytest.mark.parametrize(
    "word, expected",
    [
        ("ज़िंदगी", "zindagee"),   # z, not j
        ("राज़", "raaz"),          # z, not j
        ("फ़र्ज़", "farz"),        # f, not ph; z, not j
        ("ग़म", "gam"),            # medial schwa preserved
        ("ख़ौफ़", "khauf"),        # f, not ph
        ("शौख़", "shaukh"),
        ("काफ़ी", "kaafee"),
        ("हाज़िर", "haazir"),
        ("ज़्यादा", "zyaadaa"),    # virama binding survives nukta
        ("ग़ज़ल", "gazal"),
    ],
)
def test_decomposed_nukta_romanizes_correctly(word, expected):
    assert devanagari_to_roman(_decomposed(word)) == expected


@pytest.mark.parametrize("word", ["ज़िंदगी", "फ़र्ज़", "ख़ौफ़", "ग़ज़ल", "क़ीमत"])
def test_no_nukta_or_virama_leaks(word):
    out = devanagari_to_roman(_decomposed(word))
    assert NUKTA not in out
    assert "्" not in out  # virama must never leak either
    assert out.isascii()


@pytest.mark.parametrize(
    "word, expected",
    [
        ("भौंक", "bhaunk"),
        ("प्यार", "pyaar"),
        ("दिल", "dil"),
        ("शहर", "shahar"),
    ],
)
def test_non_nukta_words_unregressed(word, expected):
    assert devanagari_to_roman(word) == expected
