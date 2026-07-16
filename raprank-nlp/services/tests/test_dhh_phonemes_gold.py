# -*- coding: utf-8 -*-
"""
Gold-set tests for the from-scratch DHH phoneme engine (Phase 0 of
docs/dhh_dictionary/PLAN.md). These are the gate every phase must pass.

Skips cleanly while services/dhh_phonemes.py does not exist yet, so the
suite stays green until Phase 1 lands; from then on these are hard gates.

API under test (services/dhh_phonemes.py):
    deva_to_phones(word: str) -> list[str]
    hinglish_to_phones(word: str) -> list[str]
    to_phones(word: str) -> list[str]          # routes by script
    rhyme_key(phones: list[str]) -> tuple | None
    multi_rhyme_key(phones: list[str], n: int = 2) -> tuple | None
"""
import pytest

dhh = pytest.importorskip(
    "services.dhh_phonemes",
    reason="Phase 1 (services/dhh_phonemes.py) not built yet",
)

from services.tests.data.dhh_rhyme_gold import (  # noqa: E402
    GOLD_MULTI_NEGATIVE,
    GOLD_MULTI_POSITIVE,
    GOLD_NEGATIVE,
    GOLD_POSITIVE,
    GOLD_SCRIPT_EQUIV,
)


def _key(word: str):
    phones = dhh.to_phones(word)
    assert phones, f"no phonemes for {word!r}"
    return dhh.rhyme_key(phones)


def _multi_key(word: str):
    phones = dhh.to_phones(word)
    assert phones, f"no phonemes for {word!r}"
    return dhh.multi_rhyme_key(phones)


class TestGoldPositive:
    @pytest.mark.parametrize("a,b", GOLD_POSITIVE, ids=[f"{a}~{b}" for a, b in GOLD_POSITIVE])
    def test_rhyme_keys_match(self, a, b):
        ka, kb = _key(a), _key(b)
        assert ka is not None, f"no rhyme key for {a!r}"
        assert ka == kb, f"{a!r} ({ka}) should rhyme with {b!r} ({kb})"


class TestGoldNegative:
    @pytest.mark.parametrize("a,b", GOLD_NEGATIVE, ids=[f"{a}!~{b}" for a, b in GOLD_NEGATIVE])
    def test_rhyme_keys_differ(self, a, b):
        ka, kb = _key(a), _key(b)
        assert ka != kb, f"{a!r} and {b!r} must NOT share a rhyme key (both {ka})"


class TestGoldMultiPositive:
    @pytest.mark.parametrize("a,b", GOLD_MULTI_POSITIVE, ids=[f"{a}~~{b}" for a, b in GOLD_MULTI_POSITIVE])
    def test_multi_keys_match(self, a, b):
        ka, kb = _multi_key(a), _multi_key(b)
        assert ka is not None, f"no multi key for {a!r}"
        assert ka == kb, f"{a!r} ({ka}) should multi-rhyme with {b!r} ({kb})"


class TestGoldMultiNegative:
    @pytest.mark.parametrize("a,b", GOLD_MULTI_NEGATIVE, ids=[f"{a}!~~{b}" for a, b in GOLD_MULTI_NEGATIVE])
    def test_multi_keys_differ(self, a, b):
        ka, kb = _multi_key(a), _multi_key(b)
        assert ka != kb, f"{a!r} and {b!r} must NOT share a multi key (both {ka})"


class TestScriptEquivalence:
    @pytest.mark.parametrize(
        "deva,variants",
        GOLD_SCRIPT_EQUIV,
        ids=[deva for deva, _ in GOLD_SCRIPT_EQUIV],
    )
    def test_devanagari_equals_romanized(self, deva, variants):
        ref = dhh.deva_to_phones(deva)
        assert ref, f"no phonemes for Devanagari {deva!r}"
        for rom in variants:
            got = dhh.hinglish_to_phones(rom)
            assert got == ref, (
                f"{rom!r} -> {got} but {deva!r} -> {ref}; "
                "romanized variant must collapse to the Devanagari pronunciation"
            )
