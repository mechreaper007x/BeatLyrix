"""
prosody_service -- code-switching, anaphora/repetition, text cadence variance.

Validates the three promoted axes score sensibly and never crash, plus a corpus
smoke over real songs via the auto-parametrized `track` fixture.
"""
from __future__ import annotations

from services import prosody_service as ps


def test_returns_all_axes_and_bounds():
    r = ps.calculate("hello world this is a line\nanother line right here now\none more line to close")
    for k in ("codeswitch_score", "repetition_score", "cadence_text_score"):
        assert k in r
        assert 0.0 <= r[k] <= 100.0
    assert "raw" in r


def test_codeswitch_high_on_hinglish():
    hinglish = (
        "Every night main sochta hoon about my future bright\n"
        "Money aur power that's the only cheez I need\n"
        "Chhote se sheher se nikla ab main hoon at the lead\n"
        "Sapne bade the always par kiya maine succeed"
    )
    english = (
        "I grind every night just to make it out the ends\n"
        "Working really hard to finally make amends\n"
        "Money and the power are the only things I need\n"
        "Came up from the bottom now I'm finally in the lead"
    )
    assert ps.codeswitch_ratio(hinglish) > ps.codeswitch_ratio(english)
    assert ps.calculate(hinglish)["codeswitch_score"] > 50.0


def test_repetition_high_on_anaphora():
    anaphora = (
        "Every night I pray for a better day\n"
        "Every night I fight to find my way\n"
        "Every day I rise and I grind again\n"
        "Every day I climb till I finally win"
    )
    no_anaphora = (
        "Money on my mind and the world in my hand\n"
        "Rising to the top from a grain of sand\n"
        "Nobody can stop what the future has planned\n"
        "Kings never kneel, that's the only command"
    )
    assert ps.repetition_ratio(anaphora) > ps.repetition_ratio(no_anaphora)
    assert ps.calculate(anaphora)["repetition_score"] > 50.0


def test_cadence_varies_with_line_length_spread():
    monotone = "\n".join(["one two three four five"] * 6)          # all 5 words
    varied = "yo\nthis line is a much much longer bar right here now\nok\nshort\nanother medium length line here"
    assert ps.cadence_var_raw(varied) > ps.cadence_var_raw(monotone)


def test_empty_and_short_input_safe():
    assert ps.calculate("")["codeswitch_score"] == 0.0
    assert ps.calculate("single line only")["repetition_score"] == 0.0


def test_corpus_smoke(track):
    """Auto-parametrized over the corpus (conftest). Must never crash; bounded."""
    r = ps.calculate(track["lyrics"])
    for k in ("codeswitch_score", "repetition_score", "cadence_text_score"):
        assert 0.0 <= r[k] <= 100.0
