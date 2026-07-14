"""
Synthetic lyrics generator, seeded ONLY from the numeric tier targets in
corpus.synthetic.tier_profiles -- never from real lyric text. Every prompt
sent to Gemma-4/Mistral contains numbers (rhyme/syllable/vocabulary/wordplay
density targets) and a tier label, nothing lifted from the scraped corpus.

Hinglish (Hindi/Devanagari + English code-switched) ONLY -- pure "hi" and
pure "en" generation were dropped entirely. This product only ever needs
genuinely code-switched output, and splitting effort across three language
modes diluted both the prompt and the validation loop for no benefit.

Each candidate is scored locally with the exact same services the live
scoring pipeline uses (services/rhyme_service.py etc., via
corpus/analysis/signature.py) and is only kept if its measured axes land
close enough to the tier's expected scores -- this is what makes the
tier label a trustworthy ground-truth for Bayesian-network training rather
than an assumed one.

Usage (pilot batch, resumable like corpus/scrape_corpus.py):
    MISTRAL_API_KEY=xxxx python -m corpus.synthetic.generate --count 300 --resume
    MISTRAL_API_KEY=xxxx python -m corpus.synthetic.generate --tier elite --count 20

Generation uses Gemini's Gemma-4 model instead of Mistral when GEMINI_API_KEY is
set -- confirmed via direct testing that open-mistral-nemo almost never produces
genuine intra-line code-switching (intraline_mix_fraction 0.00 on most attempts
even with worked examples), while gemma-4-31b-it hits intraline_mix_fraction=1.00
on a first-try, unrepaired sample. The narrow repair-pass edit calls stay on
Mistral, which already handles those fine.
    GEMINI_API_KEY=xxxx MISTRAL_API_KEY=xxxx python -m corpus.synthetic.generate --count 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import httpx

# Log messages below print Devanagari lyric excerpts directly (e.g. rejection
# reasons quoting the offending line) -- on Windows, stdout defaults to the
# cp1252 console codepage, which can't encode those characters and crashes
# the whole batch mid-run with UnicodeEncodeError. Force UTF-8 unconditionally.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpus.synthetic.tier_profiles import (  # noqa: E402
    TIER_NAMES,
    build_extra_targets,
    build_targets,
    expected_axis_score,
    expected_extra_axis_score,
)
from services import (  # noqa: E402
    alliteration_service as al,
    assonance_service as aso,
    consonance_service as cons,
    onomatopoeia_service as ono,
    prosody_service as pro,
    rhyme_service as rh,
    syllable_service as sy,
    vocabulary_service as vo,
    wordplay_service as wp,
)
from services.language_utils import (  # noqa: E402
    clean_word, content_lines, get_multilingual_stopwords, is_hindi_word,
)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

OUT_DIR = Path(__file__).resolve().parents[1] / "synthetic_data"
# Mixed (Hinglish) generation has a much higher rejection rate than a single
# monolingual pass would -- the model has to simultaneously satisfy script
# (Devanagari), register (rap not poetry), AND genuine intra-line
# code-switching, and empirically needs more retries to land all three at
# once. Give it more attempts rather than accepting a much higher drop rate.
MAX_ATTEMPTS_MIXED = 7
ACCEPT_FRACTION = 0.6  # fraction of measured axes that must land in-tolerance

# --- Gemini/Gemma backend (see module docstring) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
_GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


class _RateLimiter:
    """Simple async leaky-bucket: spaces calls >= 60/rpm seconds apart across
    all concurrent callers. The Gemini API doesn't expose per-key RPM in
    ListModels or response headers, so this defaults conservatively (8 RPM) --
    override with GEMINI_RPM once the account's actual tier limit is known."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._next_ok = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next_ok)
            delay = start - now
            self._next_ok = start + self._interval
        if delay > 0:
            await asyncio.sleep(delay)


_gemini_rate_limiter = _RateLimiter(int(os.getenv("GEMINI_RPM", "8")))


async def _gemini_generate(prompt: str, temperature: float = 0.9, max_output_tokens: int = 16384) -> str:
    """Calls the Gemini API's Gemma-4 endpoint directly over REST (no SDK
    dependency needed -- httpx is already a project dependency). Gemma-4 is a
    "thinking" model: generateContent responses include a scratchpad part
    tagged thought=true ahead of the real answer part, so thought parts must
    be filtered out or the reasoning trace leaks into the lyrics.

    Confirmed via direct timing against this project's real (~12.6K char)
    production prompt: the thinking scratchpad alone can run 70-170+s before
    the model even starts the actual answer, and a low max_output_tokens
    cuts it off mid-thought with zero final text -- 16384 gives enough
    headroom for both the scratchpad and a full verse. A 90s timeout was
    just too short for a reasoning model against a prompt this dense; 250s
    per HTTP call (with a couple of internal retries on transient 5xx,
    which showed up intermittently and unrelated to prompt content) is the
    realistic budget."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
    }
    last_exc: Exception | None = None
    for retry in range(3):
        await _gemini_rate_limiter.wait()
        try:
            async with httpx.AsyncClient(timeout=250.0) as http_client:
                resp = await http_client.post(_GEMINI_URL, params={"key": GEMINI_API_KEY}, json=payload)
        except httpx.TransportError as exc:
            # Covers both timeouts (httpx.TimeoutException) AND network-level
            # failures like httpx.ReadError/ConnectError -- confirmed via a
            # real pipeline run that ReadError shows up as often as outright
            # timeouts against this API and was previously NOT retried here
            # (only TimeoutException was caught), silently burning a whole
            # generate_one attempt on what was actually a transient network hiccup.
            last_exc = exc
            await asyncio.sleep(5.0 * (retry + 1))
            continue
        if resp.status_code == 429:
            raise RuntimeError(f"429 rate_limited: {resp.text[:200]}")
        if resp.status_code >= 500:
            # Transient server-side error, not a content/quality problem --
            # confirmed intermittent across otherwise-identical calls.
            last_exc = RuntimeError(f"{resp.status_code} server error: {resp.text[:200]}")
            await asyncio.sleep(5.0 * (retry + 1))
            continue
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason")
            raise RuntimeError(f"no candidates in Gemini response (blockReason={block_reason}): {data}")
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "\n".join(p["text"] for p in parts if p.get("text") and not p.get("thought"))
        if not text.strip():
            if candidate.get("finishReason") == "MAX_TOKENS":
                last_exc = RuntimeError(
                    "hit MAX_TOKENS before any final (non-thought) text was produced -- "
                    "thinking scratchpad consumed the whole output budget"
                )
                continue
            raise RuntimeError(f"empty Gemini response text (finishReason={candidate.get('finishReason')})")
        return text.strip()
    raise last_exc or RuntimeError("Gemini call failed after retries")


def _devanagari_ratio(lyrics: str) -> float:
    """Fraction of non-whitespace characters that are Devanagari script.
    This is a HARD gate, separate from the 19-axis tolerance system in
    score_against_tier() -- english_ratio there is computed via phonetic
    English-word recognition (pronouncing.phones_for_word), which doesn't
    check script at all, so Romanized Hindi ("Beta, tu apna kal dekh") can
    score as low english_ratio (those words aren't in CMU) and still pass
    the 60%-of-axes tolerance despite containing zero real Devanagari.
    Confirmed in practice: a "hi" sample that was ~98% Romanized Hindi/English
    translation pairs passed score_against_tier() outright."""
    chars = [ch for ch in lyrics if not ch.isspace()]
    if not chars:
        return 0.0
    deva = sum(1 for ch in chars if "ऀ" <= ch <= "ॿ")
    return deva / len(chars)


_MIN_DEVANAGARI_RATIO = 0.15

# Templated openers the model keeps defaulting to despite prompt instructions
# telling it not to -- negative prompting alone proved unreliable: after
# banning "In the heart of the city..."/"In the city of dreams..." by name,
# the model just converged on a NEW stock opener, "In the echo of...", used
# verbatim to open 4 independently generated songs in one 27-song batch.
# This is a hard post-hoc gate instead of relying on the model to self-police.
_BANNED_OPENERS = re.compile(
    r"in the [\w'\s]{0,20}(echo|heart|shadow|mirror|city of dreams|realm|attic|ring)"
    r"|echo (of|chamber)"
    r"|yo,? i've been (grinding|hustling)",
    re.IGNORECASE,
)

# Generators like to prefix the first line with a label ("Bar 1:", "Verse 1:",
# "Hook:") which defeats a ^-anchored match against the banned phrases below --
# strip it before checking so "Bar 1: In the echo of..." is still caught.
_LINE_LABEL_PREFIX = re.compile(r"^(bar|verse|hook|chorus|intro|outro)\s*\d*\s*[:\-]\s*", re.IGNORECASE)


# "I was the [storm/fire/architect] that X, now I'm [left with/burning in] Y" --
# named and banned in the prompt itself, but negative prompting alone proved
# unreliable there too (same lesson as _BANNED_OPENERS): the model keeps using
# this exact contrast-metaphor shape mid-verse where no opener-only gate would
# ever see it. Scanned across the whole song, not just line 1.
_CONTRAST_METAPHOR_CLICHE = re.compile(
    r"i was the \w+.{0,60}now i'?m (left with|the one|burning in|drowning in)",
    re.IGNORECASE,
)


def _has_contrast_metaphor_cliche(lyrics: str) -> bool:
    return bool(_CONTRAST_METAPHOR_CLICHE.search(lyrics))


# ── Anti-poetry filters ──────────────────────────────────────────────────
# The LLM keeps generating shayari/nazm/free-verse poetry instead of rap bars.
# These are hard post-hoc gates that catch the most common poetry patterns.

# 1. Shayari/Nazm anaphora: repetitive "मैं हूँ वो..." / "main hoon wo..." / "I am the..."
#    repeated 4+ times -- this is classic Urdu poetry structure, not rap.
# NOTE: this used to end in a `\b` word-boundary anchor after "वो", which is a
# real, silent bug -- "वो" ends in the dependent vowel sign ो (U+094B), which
# is Unicode category Mc (a combining mark), NOT a \w character to Python's re
# engine. \b requires a transition between \w/non-\w, so with a plain space
# following "वो" (also non-\w), there is NO transition at all and \b fails to
# match -- meaning EVERY line ending "...मैं हूँ वो <next word>" silently never
# matched, and this whole anaphora gate never fired on real samples. Confirmed
# directly: a sample with "मैं हूँ वो" repeated 6 times passed this gate
# undetected. Fixed by replacing \b with a Devanagari-aware negative lookahead
# (blocks only continuation into another Devanagari character, e.g. "वोट",
# without relying on \w's incomplete Devanagari coverage).
_SHAYARI_ANAPHORA_HI = re.compile(
    r"मैं\s+हूँ\s+वो(?![ऀ-ॿ])",
    re.IGNORECASE,
)
_SHAYARI_ANAPHORA_EN = re.compile(
    r"(?:^|\n)\s*I\s+am\s+the\s+\w+.{0,30}(?:\n\s*I\s+am\s+the\s+\w+){2,}",
    re.IGNORECASE,
)


def _has_shayari_anaphora(lyrics: str) -> bool:
    """Catches repetitive anaphora patterns: "मैं हूँ वो X, मैं हूँ वो Y..." used 2+
    times, or "I am the X... I am the Y... I am the Z..." repeated 3+ times.
    This is shayari/nazm structure, not rap -- the prompt itself now instructs
    the model to use "मैं हूँ वो" AT MOST once in a verse, so 2+ occurrences
    means the model ignored that instruction, not that it's using a legitimate
    hook repetition (a real repeated hook is caught/allowed separately via
    repetition_density, not via this specific poetic-anaphora phrase)."""
    # Hindi anaphora: count occurrences across lines
    lines = content_lines(lyrics)
    anaphora_count = 0
    for line in lines:
        if _SHAYARI_ANAPHORA_HI.search(line):
            anaphora_count += 1
    if anaphora_count >= 2:
        return True
    # English anaphora: check for multi-line "I am the..." repetition
    if _SHAYARI_ANAPHORA_EN.search(lyrics):
        return True
    return False


# 2. Literary/poetic English diction -- multi-word phrases no rapper would say out loud.
#    Uses multi-word phrases to avoid false positives on single words used naturally in rap
#    (e.g. "storm" in "I'm bringing the storm" is fine; "I am the storm" is poetry).
_LITERARY_DICTION = re.compile(
    r"a canvas of (?:night|darkness|sorrow|despair)|"
    r"the alley'?s? belly|"
    r"a ritual of (?:pain|sorrow|loss)|"
    r"the soul'?s? (?:journey|cry|song|calling|flame)|"
    r"the heart'?s? (?:desire|cry|ache|longing)|"
    r"the spirit(?:ual)? (?:journey|calling|quest|awakening)|"
    r"a tapestry of (?:dreams?|sorrow|pain)|"
    r"a mosaic of (?:memories?|shadows?|tears?)|"
    r"the realm of (?:shadows?|darkness?|dreams?)|"
    r"in the depths? of (?:my|the|a) (?:soul|heart|being)|"
    r"from the ashes? (?:i|we|he|she|they) (?:rose|emerged|arose|was reborn)|"
    r"the phoenix (?:that|which|who) (?:rose|emerged|unfolded)|"
    r"the rainbow (?:that|which|who) (?:followed|came|appeared)|"
    r"a blank canvas (?:of|for|waiting)|"
    r"pure (?:love|light|hope)|eternal (?:love|light|hope)|"
    r"boundless (?:love|light|hope|energy)|endless (?:love|light|hope)",
    re.IGNORECASE,
)


def _has_literary_diction(lyrics: str) -> bool:
    """Catches ornate, multi-word literary phrases that sound like poetry readings,
    not rap performances. Uses multi-word patterns to avoid false positives on
    single words ("storm", "fire", "echo") used naturally in rap context."""
    return bool(_LITERARY_DICTION.search(lyrics))


# 3. "So here's to..." toast structure -- a common LLM poetry pattern
_SO_HERES_TO = re.compile(
    r"so\s+(?:here'?s|let'?s)\s+(?:to|raise|celebrate)\s+the",
    re.IGNORECASE,
)


def _has_so_heres_to(lyrics: str) -> bool:
    """Catches the 'So here's to the grind, to the hustle, to the fight'
toast structure -- a common LLM poetry pattern, not real rap."""
    return bool(_SO_HERES_TO.search(lyrics))


# 4. Stacked nature metaphors: "I am the storm... I am the rainbow... I am the phoenix..."
#    Require 3+ nature metaphors within close proximity (within 6 lines) to avoid
#    false positives where a rapper naturally mentions "fire" and "storm" in different contexts.
_NATURE_METAPHOR_INLINE = re.compile(
    r"(?:i am|i'?m|mein hoon|main hoon)\s+(?:the\s+)?(?:storm|fire|rain|wind|"
    r"thunder|lightning|ocean|river|flame|sun|moon|star|phoenix|rainbow|"
    r"dawn|sunrise|sea|wave|earth|sky|cloud|mountain|"
    r"बादल|तूफ़ान|आग|बिजली|चाँद|तारा|सूरज|समंदर|नदी)",
    re.IGNORECASE,
)


def _has_stacked_nature_metaphors(lyrics: str) -> bool:
    """Catches the 'I am the storm/fire/phoenix/rainbow' stacked-nature-metaphor
    pattern when 3+ such metaphors appear within 6 lines of each other.
    Single nature metaphors used naturally in different parts of the verse
    are acceptable -- only the stacked/clustered pattern is rejected."""
    lines = content_lines(lyrics)
    # Collect line indices where nature metaphors appear
    metaphor_indices = []
    for i, line in enumerate(lines):
        if _NATURE_METAPHOR_INLINE.search(line):
            metaphor_indices.append(i)
    # Check if 3+ metaphors appear within a 6-line window
    for i in range(len(metaphor_indices)):
        window_end = metaphor_indices[i] + 6
        count = sum(1 for idx in metaphor_indices[i:] if idx < window_end)
        if count >= 3:
            return True
    return False


# 5. Narrative prose: long average sentence length suggests paragraph-style
#    storytelling, not punchy rap bars.
# 6. Repetitive "Remember when..." / "मुझे याद है..." nostalgia opening
_NOSTALGIA_OPENING = re.compile(
    r"^(?:मुझे\s+याद\s+है|याद\s+है\s+(?:जब|वो|वह)|remember\s+(?:when|the\s+time)|"
    r"there was a (?:time|moment|day)|back in (?:the days?|those days?))",
    re.IGNORECASE,
)


def _has_nostalgia_opening(lyrics: str) -> bool:
    """Catches narrative nostalgia openers like 'मुझे याद है, बचपन में...' or
    'Remember when I was young...' -- these start prose narratives, not rap bars."""
    raw_lines = [ln.strip() for ln in lyrics.strip().split("\n") if ln.strip()]
    if not raw_lines:
        return False
    first_line = raw_lines[0]
    return bool(_NOSTALGIA_OPENING.match(first_line))


# 7. Prose sentence detector: if average non-empty line has >25 words,
#    the model is writing paragraphs with line breaks, not bars.
_PROSE_SENTENCE_AVG_WORDS = 25


def _has_prose_sentences(lyrics: str) -> bool:
    """Catches paragraphs-with-line-breaks: if the average content line has
    more than 25 words, the model is writing narrative prose, not rap bars.
    Real rap bars average 8-15 words per line."""
    lines = content_lines(lyrics)
    if not lines:
        return False
    avg_words = sum(len(ln.split()) for ln in lines) / len(lines)
    return avg_words > _PROSE_SENTENCE_AVG_WORDS


# 8. Generic LLM rap clichés -- phrases ChatGPT/Mistral always default to
_GENERIC_RAP_CLICHES = re.compile(
    r"(?:paint(?:ing|s)?|weav(?:e|ing)|spinn?(?:ing|s)?|spitt?(?:ing|s)?|craft(?:ing|s)?)\s+"
    r"(?:pictures?|stories?|tales?|words?|my\s+story)\s+with",
    re.IGNORECASE,
)


def _has_generic_rap_cliches(lyrics: str) -> bool:
    """Catches overused LLM rap phrases like 'paint pictures with my words'
    or 'spinning stories with my bars' -- generic, not authentic rap."""
    return bool(_GENERIC_RAP_CLICHES.search(lyrics))


# 9. Hindi repetitive "मैं ने देखा / मैंने जाना / मैंने कहा" narrative prose
_HINDI_NARRATIVE_PROSE = re.compile(
    r"मैं(?:ने)?\s+(?:देखा|जाना|कहा|सुना|पूछा|बताया|समझा|सीखा)",
    re.IGNORECASE,
)


def _has_hindi_narrative_prose(lyrics: str) -> bool:
    """Catches Hindi prose narrative patterns: 'मैंने देखा...', 'मैंने जाना...',
    'मैंने कहा...' repeated within a 6-line window -- this is storytelling, not rapping.
    Single occurrences spread across the verse are acceptable (natural narration)."""
    lines = content_lines(lyrics)
    match_indices = []
    for i, line in enumerate(lines):
        if _HINDI_NARRATIVE_PROSE.search(line):
            match_indices.append(i)
    # Check if 3+ matches appear within a 6-line window
    for i in range(len(match_indices)):
        window_end = match_indices[i] + 6
        count = sum(1 for idx in match_indices[i:] if idx < window_end)
        if count >= 3:
            return True
    return False


def _has_banned_opener(lyrics: str) -> bool:
    # content_lines() strips whole-line parentheticals as ad-libs (e.g. "(Woo)"),
    # which also erases a cliché opener wrapped in parens (e.g. "(In the echo of
    # a shattered mirror)") entirely -- content_lines()[0] would then silently be
    # the NEXT line instead, letting the banned phrase through. Check the raw
    # first non-empty line too, before ad-lib stripping, so wrapped clichés are
    # still caught.
    raw_lines = [ln.strip() for ln in lyrics.strip().split("\n") if ln.strip()]
    lines = content_lines(lyrics)
    candidates = raw_lines[:1] + lines[:1]
    return any(
        _BANNED_OPENERS.search(_LINE_LABEL_PREFIX.sub("", c))
        for c in candidates
    )


_TRANSLATION_GLOSS = re.compile(r"[ऀ-ॿ][^()]{3,80}\(([a-zA-Z][a-zA-Z\s,'.-]{3,80})\)")


def _has_translation_gloss(lyrics: str) -> bool:
    """Catches a formulaic ChatGPT-ism: a Devanagari clause immediately
    followed by its own Romanized/English gloss in parentheses on the same
    line (e.g. "मैंने गलत किया, अब मैं पछताता हूँ (Maine galat kiya, ab main
    pachataata hoon)") -- real code-switching doesn't translate itself twice
    in the same breath, this is a machine-translation artifact."""
    return bool(_TRANSLATION_GLOSS.search(lyrics))


def _has_alternating_translation_lines(lyrics: str) -> bool:
    """Catches the multi-line version of the same self-translation artifact
    _has_translation_gloss() catches inline: a whole-Devanagari line
    immediately followed by a whole-English line that's just its translation,
    repeated across the verse (e.g. "मुझे याद है..." / "That last message..."
    alternating for every couplet). Real code-switching mixes scripts within
    a line or across unrelated lines -- it doesn't restate every line in the
    other language right below it."""
    lines = content_lines(lyrics)
    scripts = []
    for line in lines:
        letters = [ch for ch in line if ch.isalpha()]
        if not letters:
            scripts.append(None)
            continue
        deva = sum(1 for ch in letters if "ऀ" <= ch <= "ॿ")
        ratio = deva / len(letters)
        if ratio > 0.7:
            scripts.append("deva")
        elif ratio < 0.1:
            scripts.append("latin")
        else:
            scripts.append("mixed")

    alternations = 0
    for a, b in zip(scripts, scripts[1:]):
        if {a, b} == {"deva", "latin"}:
            alternations += 1
        else:
            alternations = 0
        if alternations >= 3:
            return True
    return False


def _intraline_mix_fraction(lyrics: str) -> float:
    """Fraction of content lines that are genuinely script-mixed (Devanagari
    AND Latin words in the SAME line), as opposed to a whole line being
    purely one script. This is the direct measure of "real Hinglish" that
    english_ratio/codeswitch_density (whole-verse aggregates) can't catch --
    a verse can hit those aggregate targets perfectly while still being
    built entirely from segregated mono-lingual lines (e.g. 2 English lines
    then 1 Devanagari line, repeated) rather than lines that mix words."""
    lines = content_lines(lyrics)
    if not lines:
        return 0.0
    mixed_count = 0
    for line in lines:
        letters = [ch for ch in line if ch.isalpha()]
        if not letters:
            continue
        deva = sum(1 for ch in letters if "ऀ" <= ch <= "ॿ")
        ratio = deva / len(letters)
        if 0.1 <= ratio <= 0.9:
            mixed_count += 1
    return mixed_count / len(lines)


# The prompt's own worked examples (WRONG/RIGHT code-switching blocks, plus the
# RAP/POETRY example lines) are meant to illustrate a TECHNIQUE only -- the
# prompt explicitly says "do NOT copy the words" -- but the model sometimes
# lifts them almost verbatim into the actual output anyway (confirmed in
# practice: an "accepted" mixed sample had 5 of its 11 lines be near-verbatim
# copies of the RIGHT worked-example block below, e.g. "मुझे पता है situation
# is mad, पर मैं हूँ that fighter jo/जो कभी नहीं रुकता" appeared essentially
# unchanged). This is plagiarism of the PROMPT itself, not of any real song,
# but it's still fake/non-original content that must be rejected. Check every
# generated line against every example line here via normalized word overlap.
_PROMPT_EXAMPLE_LINES = (
    "मुझे पता है, रात को शहर जागा रहता है, पर अंधेरे में भी रोशनी मैं ही लाता हूँ।",
    "I know the city stays awake at night, but I'm the one who brings light in the dark.",
    "मुझे मालूम है, रास्ता आसान नहीं है, हर रोड़े से लड़ता हूँ, रुकता नहीं कभी,",
    "I know the road ain't easy, but I'm the one who fights through every obstacle.",
    "मुझे पता है situation is mad, पर मैं हूँ that fighter jo कभी नहीं रुकता",
    "रात को grind करता हूँ, दिन को polish, that's the recipe for this whole hustle",
    "तू सोचता है मैं गिर गया, but nah, मैं तो सिर्फ warming up हूँ",
    "हर गली से आया हूँ मैं, every corner taught me kuch naya",
    "they said I'd fail, पर मैंने prove किया, अपनी मेहनत से हर सवाल का जवाब दिया",
    "ये सफर आसान नहीं था, but मैं रुका नहीं, मैंने अपना रास्ता खुद बनाया",
    "Tu kehta hai main gir gaya, main kehta hu main ne sambhaali",
    "Stack the paper, dodge the static, they don't want to see me make it",
    "Gully ka ladka, sapno ka rajah, beat pe baitha king",
    "They said I couldn't, I did it, now they watching from the stands",
    "Late nights, cold coffee, lyrics on a napkin, dreamin' big",
)
_EXAMPLE_LINE_WORDSETS = [
    frozenset(w for w in re.findall(r"[\w']+", ex.lower()) if len(w) > 2)
    for ex in _PROMPT_EXAMPLE_LINES
]


def _copies_prompt_example(lyrics: str) -> bool:
    """True if 2+ generated lines each share >=60% of their words (by a
    normalized word-overlap ratio) with one of the prompt's own worked-example
    lines -- i.e. the model copied the illustration instead of writing
    original content about the actual topic."""
    lines = content_lines(lyrics)
    copied = 0
    for line in lines:
        words = frozenset(w for w in re.findall(r"[\w']+", line.lower()) if len(w) > 2)
        if not words:
            continue
        for ex_words in _EXAMPLE_LINE_WORDSETS:
            if not ex_words:
                continue
            overlap = len(words & ex_words) / min(len(words), len(ex_words))
            if overlap >= 0.6:
                copied += 1
                break
    return copied >= 2


_REPAIR_MIXED_PROMPT = """Below is a draft rap verse that's supposed to be Hinglish (Hindi + English mixed), but
most of its lines are written in ONLY Hindi or ONLY English, with no mixing within a line. Your job is a narrow
EDITING task, not free rewriting: go line by line, and for every line that is currently written in just one
language, rewrite that SAME line so it blends Hindi (in actual Devanagari script, e.g. देवनागरी -- never
Romanized/transliterated Hindi) and English words TOGETHER within that one line, the way real Hinglish speakers
switch languages mid-sentence. Keep the same meaning and the same topic for each line. Leave any line that
ALREADY mixes Devanagari and English within itself unchanged. Do NOT turn one line into two lines (e.g. do NOT
add a translation line right after it) -- the output must have the exact same number of lines as the input, one
bar per line. Do NOT add commentary, headers, explanations, or a summary. Output ONLY the rewritten verse.

Example of the fix, applied to two lines (topic-neutral, do NOT copy the words, copy the technique):
  BEFORE (pure Hindi line): मुझे पता है ये रास्ता आसान नहीं है, पर मैं हार नहीं मानूंगा।
  AFTER (mixed within the line): मुझे पता है ये road आसान नहीं है, but मैं हार नहीं मानूंगा।
  BEFORE (pure English line): I've been grinding every night just to make it to the top.
  AFTER (mixed within the line): रात भर grind करता हूँ, बस इस top तक पहुंचने के लिए।

DRAFT VERSE TO FIX:
{lyrics}
"""


async def _repair_intraline_mix(client, lyrics: str) -> str | None:
    """Targeted follow-up call, tried when a "mixed"-language generation comes
    back with genuinely segregated mono-lingual lines (intraline_mix_fraction
    too low). Empirically, asking the model to freely GENERATE genuine
    intra-line Hinglish from an abstract instruction fails almost every time
    (confirmed via debug batches: most first-pass "mixed" generations score
    intraline_mix_fraction=0.00 even with a worked example and several
    attempts) -- but models are much more reliable at a narrow, bounded EDIT
    task on their own already-written text than at free generation against a
    stylistic instruction. So instead of only re-rolling the whole verse from
    scratch, also try asking the model to mechanically rewrite just the
    pure-script lines of its own draft into mixed-script lines, preserving
    meaning and line count."""
    return await _run_repair_call(client, _REPAIR_MIXED_PROMPT.format(lyrics=lyrics))


_REPAIR_ANAPHORA_PROMPT = """Below is a draft rap verse that overuses the Hindi phrase "मैं हूँ वो" ("I am the
one who...") as a repeated line-opener -- this reads as shayari/nazm poetry, not rap, and is not allowed to
appear more than once in the verse. Your job is a narrow EDITING task, not free rewriting: find every line that
starts with or contains "मैं हूँ वो" AFTER THE FIRST such occurrence, and rewrite ONLY that line using a direct,
active-verb boast instead of the passive "मैं हूँ वो जो..." shape -- keep the same meaning/content of the line,
just change the grammatical construction. Leave the first occurrence of "मैं हूँ वो" (if any) and every other line
unchanged. Do NOT change the number of lines. Do NOT add commentary or headers. Output ONLY the rewritten verse.

Example fix:
  BEFORE: मैं हूँ वो जो कभी नहीं रुकता, हर मुश्किल को मैं पार करता हूँ
  AFTER:  कभी नहीं रुकता मैं, हर मुश्किल को पार करता हूँ

DRAFT VERSE TO FIX:
{lyrics}
"""


async def _repair_anaphora(client, lyrics: str) -> str | None:
    """Targeted follow-up call, tried when a generation comes back overusing
    "मैं हूँ वो" as a repeated line-opener (shayari anaphora). Same rationale
    as _repair_intraline_mix: negative prompting alone doesn't stop the model
    from defaulting to this exact phrase for Hindi boast lines even after an
    explicit "use at most once" instruction was added to the prompt (confirmed
    via debug batches -- the phrase kept reappearing 2-6+ times per verse
    regardless) -- but a narrow, bounded edit on the model's own draft is far
    more reliable than another free-generation attempt."""
    return await _run_repair_call(client, _REPAIR_ANAPHORA_PROMPT.format(lyrics=lyrics))


async def _run_repair_call(client, prompt: str) -> str | None:
    try:
        response = await asyncio.wait_for(
            client.chat.complete_async(
                model="open-mistral-nemo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            ),
            timeout=90.0,
        )
        repaired = response.choices[0].message.content.strip()
    except Exception:
        return None
    repaired = re.sub(r"^```[a-z]*\n?|```$", "", repaired, flags=re.MULTILINE).strip()
    repaired = re.sub(r'^"|"$', "", repaired.strip(), flags=re.MULTILINE).strip()
    repaired = _strip_meta_annotation_lines(repaired)
    repaired = _strip_leaked_metrics_footer(repaired)
    return repaired


_LEAKED_META_TERMS = re.compile(
    r"\b(multisyllabic|internal rhyme|rhyme[- ]?chain|compound rhyme|mosaic rhyme|holorime|"
    r"assonance|consonance|onomatopoeia|rhyme scheme|wordplay density|alliteration density|"
    r"cadence variance|repetition density|codeswitch density|syllable density|"
    r"vocabulary uniqueness|msttr)\b"
    r"|\bin (devanagari|hinglish)[,:]? (it read|it says|translation)"
    r"|\b(it read|it says|translated as|literally means)[,:] ?[\"‘’“”]",
    re.IGNORECASE,
)


def _has_leaked_meta_terms(lyrics: str) -> bool:
    """Catches the model narrating its own writing instructions or
    translating itself mid-verse (e.g. "In Devanagari, it read, ..." or
    "Multisyllabic, like a rap bar's claim") -- these are technical/meta
    terms from the prompt itself leaking into the lyrics, which the prompt
    explicitly forbids and no genuine rap bar would ever contain."""
    return bool(_LEAKED_META_TERMS.search(lyrics))


# The model sometimes appends a trailing self-analysis footer summarizing the
# axes it thinks it hit, e.g. "(Word count: 240, Average syllables per line:
# 14.0, ... Alliteration density: 90, English-word ratio: 0.45, ...)" -- a
# leaked restatement of the prompt's own scoring rubric, never part of the
# actual lyrics. Strip it (whole trailing block) before any gate/measurement
# runs, since left in place it inflates line counts and pollutes every axis
# that scans raw text (e.g. it can bump english_ratio/codeswitch_density
# purely from the English words in the footer's own labels).
_TRAILING_METRICS_FOOTER = re.compile(
    r"\n+\(\s*(word count|average syllable|syllable[s]? per line|complex-word)\b.*\)\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Some responses also annotate every line with a trailing "(14)"-style
# syllable count -- another leaked scoring artifact, not part of the bar
# itself. Strip a trailing "(<digits>)" at end-of-line, and the same
# annotation when it leaks as a line PREFIX instead (e.g. "(14) एक तख्ती...").
_TRAILING_SYLLABLE_COUNT = re.compile(r"\s*\(\d{1,2}\)\s*$", re.MULTILINE)
_LEADING_SYLLABLE_COUNT = re.compile(r"^\s*\(\d{1,2}\)\s*", re.MULTILINE)


def _strip_leaked_metrics_footer(lyrics: str) -> str:
    lyrics = _TRAILING_METRICS_FOOTER.sub("", lyrics)
    lyrics = _TRAILING_SYLLABLE_COUNT.sub("", lyrics)
    lyrics = _LEADING_SYLLABLE_COUNT.sub("", lyrics)
    return lyrics.strip()


_META_ANNOTATION_LINE = re.compile(
    r"^\(?\s*(in\s+)?(devanagari|hindi|english|hinglish|romanized|transliterat\w*)"
    r"(\s+(script|version|translation))?\s*\)?$",
    re.IGNORECASE,
)


def _strip_meta_annotation_lines(lyrics: str) -> str:
    """Strips leaked self-referential lines like "(In Devanagari)" that the
    model sometimes emits as a stage direction about its own output rather
    than actual lyric content."""
    kept = [
        line for line in lyrics.split("\n")
        if not _META_ANNOTATION_LINE.match(line.strip())
    ]
    return "\n".join(kept).strip()


def _line_uniqueness_ratio(lyrics: str) -> float:
    """Fraction of content lines that are unique -- catches degenerate
    verses that loop the same 3-4 lines over and over (observed in practice:
    a 24-line "verse" that was the same 4 lines repeated 6 times, which the
    axis-tolerance system didn't reject since high repetition_density is a
    legitimate target for some tiers)."""
    lines = [ln.strip().lower() for ln in content_lines(lyrics)]
    if not lines:
        return 1.0
    return len(set(lines)) / len(lines)


def _english_ratio(lyrics: str) -> float:
    """Fraction of content tokens that are plausibly English (ASCII, in CMU).
    Duplicated from corpus/analysis/signature.py rather than imported, since
    that module pulls in tests/conftest.py (a pytest runtime dependency) --
    generation shouldn't need pytest installed to run."""
    import pronouncing
    stops = get_multilingual_stopwords()
    en = tot = 0
    for line in content_lines(lyrics):
        for raw in line.split():
            w = clean_word(raw)
            if not w or len(w) < 2 or w in stops or is_hindi_word(w):
                continue
            tot += 1
            if pronouncing.phones_for_word(w.lower()):
                en += 1
    return en / tot if tot else 0.0


def measure(lyrics: str) -> dict[str, float]:
    """Measure the same axes tier targets are defined over, using the live scorers."""
    nlines = max(len(content_lines(lyrics)), 1)
    _, _, multisyl_count, internal_score, chain_score, compound_count, holorime_count = rh.calculate(lyrics)
    aso_score, _ = aso.calculate(lyrics)
    cons_score, _ = cons.calculate(lyrics)
    ono_score, _ = ono.calculate(lyrics)
    _, msttr = vo.calculate(lyrics)
    _, syl_density, _, syl_weight_ratio = sy.calculate(lyrics)
    _, wp_meta = wp.calculate(lyrics)

    return {
        "rhyme_internal_density": internal_score,       # already a 0-100 sub-score
        "rhyme_multisyllabic_density": min(multisyl_count / nlines / build_targets("elite")["rhyme_multisyllabic_density"] * 100.0, 100.0),
        "rhyme_chain_density": chain_score,
        "rhyme_compound_density": min(compound_count / nlines / build_targets("elite")["rhyme_compound_density"] * 100.0, 100.0),
        "rhyme_holorime_density": min(holorime_count / nlines / build_targets("elite")["rhyme_holorime_density"] * 100.0, 100.0),
        "assonance_density": aso_score,
        "consonance_density": cons_score,
        "onomatopoeia_density": ono_score,
        "syllable_density_per_line": syl_density,       # raw avg syllables/line -- compared via curve below
        "syllable_complex_word_ratio": syl_weight_ratio,
        "vocabulary_msttr": msttr,
        "wordplay_simile_density": 100.0 * wp_meta["simile_count"] / nlines,
        "wordplay_metaphor_density": 100.0 * wp_meta["metaphor_count"] / nlines,
        "wordplay_pun_density": 100.0 * wp_meta["puns_count"] / nlines,
        "wordplay_entendre_density": 100.0 * wp_meta["double_entendres_count"] / nlines,
    }


def measure_extra(lyrics: str) -> dict[str, float]:
    """Measure the axes not covered by build_targets()/measure() above (see
    tier_profiles.build_extra_targets for why these are modeled separately) --
    values here are RAW (0-1 fractions or an already-0-100 score), matching
    what tier_profiles.expected_extra_axis_score()'s curve lookups expect."""
    allit_score, _ = al.calculate(lyrics)
    prosody = pro.calculate(lyrics)
    return {
        "alliteration_density": allit_score,             # already 0-100
        "english_ratio": _english_ratio(lyrics),          # raw 0-1 fraction
        "codeswitch_density": prosody["raw"]["codeswitch"],
        "repetition_density": prosody["raw"]["repetition"],
        "cadence_var_density": prosody["raw"]["cadence_var"],
    }


def _axis_actual_score(axis: str, tier: str, raw: dict[str, float]) -> float:
    """Normalize `measure()`'s mixed raw/score values onto the same 0-100 scale
    expected_axis_score() uses, so the two are directly comparable."""
    from corpus.synthetic.tier_profiles import AXIS_KIND, _CURVES

    kind = AXIS_KIND[axis]
    value = raw[axis]
    if kind == "curve":
        thresholds, scores = _CURVES[axis]
        from config import scoring_config as cfg
        return cfg.evaluate_piecewise_curve(value, thresholds, scores)
    return value  # linear_ratio / density_direct axes are already 0-100


# These devices are rare, deliberate, all-or-nothing stylistic choices (a
# writer either lands a holorime somewhere or doesn't -- there's no dial to
# turn to hit "50% of a holorime"). At non-elite tiers, empirically an LLM
# (and, presumably, most real mid/commercial writers) simply omits them
# entirely rather than sprinkling in a moderate amount -- so below elite,
# "less than expected" (including zero) is a legitimate mid/commercial
# outcome, not a miss. Only flag these axes if the sample OVER-shoots into
# elite-level density, since that's the actual tier-confusing failure mode.
_ONE_SIDED_BELOW_ELITE = frozenset({
    "rhyme_multisyllabic_density", "rhyme_chain_density",
    "rhyme_compound_density", "rhyme_holorime_density", "onomatopoeia_density",
})


def score_against_tier(tier: str, lyrics: str, lang: str = "mixed") -> tuple[bool, dict[str, float], dict[str, float]]:
    """Returns (accepted, actual_scores, expected_scores) for every tracked axis
    (both the tier-scaled devices in measure() and the style/language axes in
    measure_extra() -- alliteration, english/codeswitch, repetition, cadence_var)."""
    from corpus.synthetic.tier_profiles import AXIS_KIND, EXTRA_AXIS_KIND

    raw = measure(lyrics)
    actual = {axis: _axis_actual_score(axis, tier, raw) for axis in raw}
    expected = {axis: expected_axis_score(tier, axis) for axis in raw}

    extra_raw = measure_extra(lyrics)
    # extra axes are already 0-100 (alliteration) or a plain ratio*100 (english) --
    # only codeswitch/repetition/cadence_var need the curve re-applied to compare
    # against expected_extra_axis_score(), which itself curves the raw target.
    for axis, val in extra_raw.items():
        kind = EXTRA_AXIS_KIND[axis]
        if kind == "score_direct":
            actual[axis] = val
        elif kind == "curve_identity":
            actual[axis] = val * 100.0
        else:
            from config import scoring_config as _cfg
            from corpus.synthetic.tier_profiles import _EXTRA_CURVES
            thresholds, scores = _EXTRA_CURVES[axis]
            actual[axis] = _cfg.evaluate_piecewise_curve(val, thresholds, scores)
        expected[axis] = expected_extra_axis_score(tier, axis, lang)

    passed = 0
    for axis in actual:
        kind = AXIS_KIND.get(axis) or EXTRA_AXIS_KIND.get(axis)
        # rhyme sub-devices (internal/multisyllabic/chain/compound/holorime) are a
        # stylistic yes/no choice for an LLM, not a dial it can titrate to a precise
        # percentage -- widen tolerance for those vs. the smoother curve-based axes.
        base_tol = 35.0 if kind == "linear_ratio" else 30.0
        tol = max(base_tol, expected[axis] * 0.6)
        diff = actual[axis] - expected[axis]
        if tier != "elite" and axis in _ONE_SIDED_BELOW_ELITE and diff < 0:
            passed += 1
        elif abs(diff) <= tol:
            passed += 1
    accepted = passed / len(actual) >= ACCEPT_FRACTION
    return accepted, actual, expected


PROMPT_TEMPLATE = """You are an expert Hindi/English/Hinglish rap lyricist writing an ORIGINAL 16-24 line verse about: {topic}.

Do NOT reference or reuse any existing song's lyrics. Write entirely new lines that hit these numeric targets:

- Internal rhyme density score (0-100, target): {rhyme_internal_density:.0f}
- Multisyllabic rhyme density score (0-100, target): {rhyme_multisyllabic_density:.0f}
- Rhyme-chain density score (0-100, target): {rhyme_chain_density:.0f}
- Compound/mosaic rhyme density score (0-100, target): {rhyme_compound_density:.0f}
- Holorime density score (0-100, target): {rhyme_holorime_density:.0f}
- Assonance (vowel-music) density score (0-100, target): {assonance_density:.0f}
- Consonance density score (0-100, target): {consonance_density:.0f}
- Ad-lib/onomatopoeia density score (0-100, target): {onomatopoeia_density:.0f}
- Average syllables per line (target): {syllable_density_per_line:.1f}
- Complex-word (3+ syllable) ratio (target): {syllable_complex_word_ratio:.2f}
- Vocabulary uniqueness / MSTTR (target, 0-1): {vocabulary_msttr:.2f}
- Simile density, % of lines (target): {wordplay_simile_density:.0f}%
- Metaphor density, % of lines (target): {wordplay_metaphor_density:.0f}%
- Pun density, % of lines (target): {wordplay_pun_density:.0f}%
- Double-entendre density, % of lines (target): {wordplay_entendre_density:.0f}%
- Alliteration density score (0-100, target): {alliteration_density:.0f}
- English-word ratio among content words (target, 0-1): {english_ratio:.2f}
- Code-switch density -- how often the language flips within/between lines (target, 0-1): {codeswitch_density:.2f}
- Word/phrase repetition density, e.g. a repeated hook or refrain (target, 0-1): {repetition_density:.2f}
- Cadence/rhythm variance across lines (target, 0-1): {cadence_var_density:.2f}

Concrete guide for a ~20-line verse (use this, the density scores above are just the underlying math):
- internal (mid-line) rhymes in about {internal_lines:.0f} of the 20 lines -- example technique (topic-neutral,
  do NOT copy): "I'm sipping that potion, devotion in motion, causing commotion" -- 3+ words INSIDE one line
  rhyme with each other, not just the line's end word.
- multisyllabic (2+ syllable) rhyme pairs in about {multisyl_lines:.0f} of the 20 lines -- example: a line ending
  "...breaking the cycle" rhymed with a later line ending "...faking a title" (2-syllable rhyme "-icle"/"-itle",
  not just the last syllable).
- a stacked rhyme chain (3+ consecutive lines sharing a rhyme sound) about {chain_lines:.0f} time(s)
- a compound/mosaic rhyme (multi-word phrase rhyming with another) about {compound_lines:.0f} time(s)
- a holorime (whole phrase rhymes with another whole phrase) about {holorime_lines:.0f} time(s)
If the counts above are 0 or near-0 for a device, skip it entirely -- do not force it in.

Language: write in {language_label}. Any Hindi content MUST be written in actual Devanagari script (the native
script used for Hindi, e.g. देवनागरी) -- NEVER Romanized/transliterated Hindi spelled out in Latin letters
(e.g. never "mujhe maloom hai" -- spell it using Devanagari characters instead). The english_ratio/codeswitch_density
targets above tell you how much of the verse should be plain English words (in Latin script) vs. how often the
language should flip between Devanagari Hindi and English within/across lines -- hit those proportions instead of
defaulting to a fixed language mix regardless of target.

If codeswitch_density above is moderate/high, genuine code-switching means Hindi and English words interleaved
WITHIN THE SAME LINE, not whole lines or whole stanzas written in one language followed by whole lines/stanzas in
the other. Concrete example (topic-neutral, do NOT copy): "मुझे पता है situation is mad, पर मैं हूँ fighter" --
English words dropped into a Devanagari sentence and vice versa, in the SAME breath. Do NOT write four lines fully
in Hindi and then four lines fully in English (or vice versa) -- that reads as two mono-lingual sections stitched
together, which is NOT Hinglish and will be rejected outright.

WRONG (the #1 mistake models make -- do NOT do this): writing a full Devanagari line, then immediately writing
a full English line that just TRANSLATES it, and repeating that pattern down the whole verse -- and this WRONG
pattern is JUST AS WRONG at bigger scale: writing several full-Hindi lines in a row, then several full-English
lines in a row (a "Hindi stanza" followed by an "English stanza"), is the exact same mistake, just spread across
more lines instead of one couplet. Example of the WRONG pattern (do not write anything shaped like this):
  मुझे पता है, रात को शहर जागा रहता है, पर अंधेरे में भी रोशनी मैं ही लाता हूँ।
  I know the city stays awake at night, but I'm the one who brings light in the dark.
  मुझे मालूम है, रास्ता आसान नहीं है, हर रोड़े से लड़ता हूँ, रुकता नहीं कभी,
  I know the road ain't easy, but I'm the one who fights through every obstacle.
That is NOT code-switching, no matter how the lines are arranged -- it's one language followed by its own
translation, and it will be rejected.

RIGHT -- here is the mandatory technique: for AT LEAST every other line, START the line in one language and
SWITCH to the other language partway through the SAME line, mid-clause, the way real Hinglish speakers actually
talk -- never finish a full clause in one language and then restate that same clause in the other. Worked example
of 6 consecutive lines written the RIGHT way (topic-neutral, do NOT copy the words, copy the TECHNIQUE of
switching languages mid-line):
  मुझे पता है situation is mad, पर मैं हूँ that fighter jo कभी नहीं रुकता
  रात को grind करता हूँ, दिन को polish, that's the recipe for this whole hustle
  तू सोचता है मैं गिर गया, but nah, मैं तो सिर्फ warming up हूँ
  हर गली से आया हूँ मैं, every corner taught me kuch naya
  they said I'd fail, पर मैंने prove किया, अपनी मेहनत से हर सवाल का जवाब दिया
  ये सफर आसान नहीं था, but मैं रुका नहीं, मैंने अपना रास्ता खुद बनाया
Notice: none of those lines is a translation of another line -- each line says ONE new thing, with Hindi and
English words interleaved inside it. Write your verse the same way.

HARD REQUIREMENT, not a suggestion: at least 1 out of every 4 lines (so, ~5+ lines in a 20-line verse) MUST
individually contain BOTH Devanagari and Latin-script words in that SAME line -- not spread across two separate
lines. A verse where every single line is purely one script (even if the verse as a whole alternates between
all-Hindi stanzas and all-English stanzas) will be REJECTED regardless of the overall english_ratio -- writing
4 full lines in Hindi, then 4 full lines in English, then repeating, is exactly the WRONG pattern above at
stanza scale instead of line scale, and it is rejected just the same.

Vary your opening line every time -- do NOT default to a generic scene-setter like "In the heart of the city...",
"In the city of dreams...", or "मुझे मालूम है/था मेरी मंज़िल..." ("I know/knew my destination..."). Start
mid-thought, with a specific image, an action, a direct address, or a punchline -- anything but a stock intro line.
Also avoid the "I was the [storm/fire/match/architect] that [caused it], now I'm [left with/burning in] [the
aftermath]" contrast-metaphor structure -- that exact rhetorical shape (declare what you WERE, then contrast with
what you are NOW, via a single repeated element/nature metaphor) has been overused already; build the verse's
structure and imagery around this specific detail instead so it can't fall back on that template: {anchor}.
Weave in alliteration (repeated starting sounds across nearby words) to roughly match the alliteration density
target -- skip it if the target is low. If the repetition target is moderate/high, repeat a hook phrase or refrain
at least once or twice; if it's low, avoid repeating any line or phrase. Vary line length/rhythm to roughly match
the cadence variance target -- low means keep a steady, consistent rhythm; high means mix short punchy lines with
longer flowing ones.

═══════════════════════════════════════════════════════════════════════════════
THIS MUST READ AS RAP BARS, NOT AS POETRY. THIS IS THE MOST IMPORTANT RULE.
═══════════════════════════════════════════════════════════════════════════════

Every line must pass the "out loud test": if it sounds like a poetry reading, not like
someone spitting bars over 808s, rewrite it.

RAP REGISTER (write like this):
- Conversational, direct, street-level -- the way you'd actually talk to someone.
- Contractions everywhere: I'm, ain't, gotta, tryna, gonna, wanna, can't, won't.
- Boasts, call-outs, direct address ("you", "they", "y'all") -- talk TO someone.
- Short punchy clauses, punchline at the end of the bar.
- Slang, colloquialisms, code-switching between Hindi/English naturally.
- Confidence, swagger, attitude -- even when the topic is heavy.
- Concrete, specific imagery: real places, real objects, real people.

POETRY REGISTER (DO NOT write like this -- this gets rejected):
- Ornate imagery: "a canvas of night", "the alley's belly", "a ritual of pain"
- Archaic/formal language: "whilst", "thou", "shall", "upon"
- Free-verse line breaks that only make sense visually on a page
- Narrative prose with line breaks (telling a story paragraph-style, not spitting bars)
- Repetitive anaphora: "मैं हूँ वो..." x8, "I am the..." x5 (this is shayari, not rap)

HARD BAN on a specific phrase: do NOT use the construction "मैं हूँ वो" (literally "I am the one who...") as a
line-opener MORE THAN ONCE in the whole verse -- this is the single most common shayari-poetry tell and models
default to it constantly for Hindi boast lines. Use direct, active-verb boasts instead of this passive
self-declaration shape. Instead of "मैं हूँ वो जो कभी नहीं रुकता" (I am the one who never stops), write something
active like "मैं रुकता नहीं कभी", "कभी नहीं रुकता मैं", "रुकना मुझे आता ही नहीं", or switch the boast into English/mixed
entirely ("I never stop, कभी नहीं"). Same restriction in English: do not open more than one line with "I am the...".
- Philosophical abstractions: "the soul's journey", "the heart's desire", "the spirit calls"
- Romantic/pastoral imagery: sunsets, moonlight, stars, roses, rivers as metaphors
- Toast structures: "So here's to the grind, to the hustle, to the fight"
- Nostalgia openers: "मुझे याद है, बचपन में...", "Remember when I was young..."
- Stacked nature metaphors: "I am the storm, I am the rainbow, I am the phoenix"
- Generic LLM rap clichés: "paint pictures with my words", "spinning stories with my bars"

RAP EXAMPLES (what real bars sound like):
  "Tu kehta hai main gir gaya, main kehta hu main ne sambhaali"
  "Stack the paper, dodge the static, they don't want to see me make it"
  "Gully ka ladka, sapno ka rajah, beat pe baitha king"
  "They said I couldn't, I did it, now they watching from the stands"
  "Late nights, cold coffee, lyrics on a napkin, dreamin' big"

POETRY EXAMPLES (what you must NOT write -- this gets rejected):
  "मैं हूँ वो आग, जो तेरे अंदर भड़कती है" (shayari anaphora)
  "I am the storm that came, I am the rainbow that followed" (nature metaphor stack)
  "In the echo of a shattered mirror, I found myself" (ornate imagery)
  "The moon weeps for the fallen star" (romantic abstraction)
  "So here's to the grind, to the hustle, to the fight" (toast structure)
  "मुझे याद है, बचपन में हमारा घर था..." (narrative prose opener)
  "I remember when I was young, my family was poor..." (narrative prose)
═══════════════════════════════════════════════════════════════════════════════

If a line sounds like it belongs in a poetry chapbook instead of blasting out of
a speaker, rewrite it plainer and punchier. Every bar should hit like a punchline,
not drift like a meditation.

A score of 100 means "as dense/complex as it gets for elite technical rap"; 0 means plain, simple, commercial writing.
For LOW targets: keep vocabulary common, sentences short and plain, rhymes simple single-syllable end rhymes only.
For HIGH/MODERATE targets: actually weave in the internal/multisyllabic rhyme techniques shown above at the
stated line-count -- don't just write plain end-rhymed couplets, that scores as 0 density regardless of topic.

CRITICAL: the numbers above are internal instructions for YOU only. The verse itself must be about "{topic}" and
must NEVER mention, name, or describe rhyme schemes, syllables, metrics, scores, or any of the technical terms
used above (e.g. never write words like "internal rhyme", "multisyllabic", "assonance", "consonance",
"onomatopoeia", "metric", "score" in the lyrics). Do NOT append a summary/analysis footer after the verse (e.g.
"(Word count: ..., Average syllables per line: ..., Alliteration density: ...)") and do NOT annotate individual
lines with a trailing syllable count like "(14)" -- the response must end right after the last bar of the verse,
nothing after it. Output ONLY the verse lines, one bar per line, no title, no explanation, no markdown, no
meta-commentary about the writing process itself, no trailing self-scoring summary."""

_LANG_LABEL = "Hinglish -- Hindi in Devanagari script mixed with English, code-switching within/across lines"

_TOPICS = (
    "hustling to make it out of a small town",
    "a love that fell apart",
    "late-night city life and ambition",
    "loyalty and betrayal among friends",
    "celebrating success after a hard grind",
    "family struggles and growing up poor",
    "confidence and swagger on the mic",
    "missing home while chasing a dream",
    "a party night with friends",
    "rivalry and competition with other rappers",
    "a mentor or elder's advice shaping who you became",
    "the grind of a 9-to-5 job you're trying to escape",
    "getting cheated by a business partner",
    "a childhood memory that still haunts you",
    "proving doubters and critics wrong",
    "addiction and the fight to get clean",
    "the loneliness of fame or success",
    "a road trip with no destination",
    "standing up to corruption or injustice",
    "a breakup you caused and regret",
    "raising your kid the way you wish you'd been raised",
    "the first time you ever performed on stage",
    "losing a friend to violence or an overdose",
    "flexing wealth you worked hard for",
    "the pressure of being the first in your family to succeed",
    "an old rivalry that turned into respect",
    "surviving a betrayal by someone you trusted with everything",
    "the grind of studio nights before a big break",
    "watching your neighborhood change and gentrify",
    "a phone call that changed everything",
    "faith and doubt during the hardest year of your life",
    "the thrill and danger of street life",
    "reconnecting with an estranged parent",
    "the weight of being the one everyone depends on",
    "chasing a dream your family thinks is a waste of time",
    "a rival crew disrespecting your block",
    "the comedown after the party ends",
    "falling in love unexpectedly",
    "surviving heartbreak by throwing yourself into work",
    "the moment you realized you'd made it",
)

# Randomized concrete detail injected alongside the topic, decoupled from it
# (different modulus base) so topic x anchor gives ~1600 distinct combinations
# for 300 songs instead of just 40 -- without this, an LLM sampling the same
# topic repeatedly converges on the same rhetorical template (observed: elite/en
# heartbreak/comedown topics kept reusing an "I was the storm/match/architect
# that X, now I'm left with Y" contrast-metaphor shape almost verbatim across
# independently generated songs, TF-IDF cosine similarity up to 0.55).
_ANCHORS = (
    "a cracked phone screen you never got fixed",
    "the smell of your grandmother's kitchen",
    "a specific scar and how you got it",
    "the sound of a train passing at 3am",
    "a pair of worn-out sneakers",
    "a voice note you never listened to",
    "the last text message someone sent you",
    "a photograph stuck in a drawer",
    "the exact color of a sunset over a specific block",
    "a broken watch that still says the wrong time",
    "the taste of the first meal you bought with your own money",
    "a specific street corner where everything changed",
    "a childhood nickname nobody uses anymore",
    "the weight of a house key that no longer opens anything",
    "a song that plays on the radio at the wrong moment",
    "a specific animal (a stray dog, a caged bird, a crow on a wire)",
    "the sound of rain on a tin roof",
    "a receipt from a place that's now closed",
    "a specific piece of jewelry passed down in the family",
    "the way a specific person laughed",
    "a bus route you used to take every day",
    "a candle burning in an empty room",
    "the static of an old radio or TV",
    "a specific meal shared in silence",
    "a torn poster on a wall",
    "the sound of keys jingling before someone leaves",
    "a specific handwriting you'd recognize anywhere",
    "an umbrella that didn't survive the storm",
    "a specific game played on the street as a kid",
    "the smell of rain on hot pavement",
    "a locked door and no key",
    "a specific brand of cheap cigarettes or chai",
    "a mirror with a crack running through it",
    "a specific text you typed but never sent",
    "an old cassette tape or CD that still works",
    "a specific bridge or overpass",
    "the sound of a specific instrument from a nearby window",
    "a stray thread on a jacket that means something",
    "a specific number (an age, a date, an amount owed)",
    "the last thing someone said before walking away",
)


_ASSUMED_VERSE_LINES = 20


def build_prompt(tier: str, topic_idx: int = 0) -> str:
    targets = build_targets(tier)
    expected = {axis: expected_axis_score(tier, axis) for axis in targets}
    extra_targets = build_extra_targets(tier)
    topic = _TOPICS[topic_idx % len(_TOPICS)]
    anchor = _ANCHORS[(topic_idx * 7 + 3) % len(_ANCHORS)]
    # wordplay_*_density targets are raw 0-1 fractions ("15% of lines") but the
    # template prints them as "{...:.0f}%" -- pre-multiply by 100 or every
    # tier's wordplay instruction silently renders as "0%".
    pct_targets = {k: v * 100 for k, v in targets.items() if k.startswith("wordplay_")}
    return PROMPT_TEMPLATE.format(
        language_label=_LANG_LABEL,
        topic=topic,
        anchor=anchor,
        internal_lines=targets["rhyme_internal_density"] * _ASSUMED_VERSE_LINES,
        multisyl_lines=targets["rhyme_multisyllabic_density"] * _ASSUMED_VERSE_LINES,
        chain_lines=targets["rhyme_chain_density"] * _ASSUMED_VERSE_LINES,
        compound_lines=targets["rhyme_compound_density"] * _ASSUMED_VERSE_LINES,
        holorime_lines=targets["rhyme_holorime_density"] * _ASSUMED_VERSE_LINES,
        **{
            **targets,
            **pct_targets,
            **extra_targets,
            **{k: v for k, v in expected.items() if k in (
                "rhyme_internal_density", "rhyme_multisyllabic_density", "rhyme_chain_density",
                "rhyme_compound_density", "rhyme_holorime_density", "assonance_density",
                "consonance_density", "onomatopoeia_density",
            )},
        },
    )


async def run_content_gates(client, lyrics: str, attempt: int, max_attempts: int) -> tuple[str, str | None]:
    """Runs the anti-poetry/anti-artifact content gates in sequence. Returns
    (lyrics, None) if every gate passes -- `lyrics` may have been repaired by
    a narrow follow-up edit call (anaphora repair) along the way -- or
    (lyrics, rejection_reason) on the first gate that fails.

    Shared by every generator variant (numeric-target-only, reference-anchored,
    persona-anchored) so poetry-register/artifact detection isn't duplicated
    or, as `generate_with_reference.py`/`generate_from_consented.py` did
    before this was extracted, skipped entirely."""
    if _has_banned_opener(lyrics):
        return lyrics, "opening line matches a banned templated opener"

    if _has_contrast_metaphor_cliche(lyrics):
        return lyrics, "uses the banned 'I was the X, now I'm Y' contrast-metaphor cliche"

    if _has_translation_gloss(lyrics):
        return lyrics, "contains a Devanagari-then-parenthetical-gloss artifact"

    if _has_alternating_translation_lines(lyrics):
        return lyrics, "alternates Devanagari lines with their own English translation"

    if _has_leaked_meta_terms(lyrics):
        return lyrics, "leaked technical/meta terms from the prompt into the lyrics"

    if _has_shayari_anaphora(lyrics):
        # Negative prompting alone doesn't stop this specific phrase
        # (confirmed: it kept reappearing 2-6+ times even after an
        # explicit "at most once" instruction was added) -- try a
        # narrow follow-up edit before burning a full retry attempt,
        # same rationale/pattern as the intraline-mix repair.
        repaired = await _repair_anaphora(client, lyrics)
        if repaired and not _has_shayari_anaphora(repaired) and len(content_lines(repaired)) >= 6:
            print(f"    ~ repaired anaphora via follow-up edit (attempt {attempt}/{max_attempts})")
            lyrics = repaired
        else:
            return lyrics, (
                "shayari/nazm anaphora pattern (repetitive 'मैं हूँ वो...' / 'I am the...'); "
                "repair attempt also failed"
            )

    if _has_literary_diction(lyrics):
        return lyrics, "literary/poetic diction detected (ornate imagery, not rap register)"

    if _has_so_heres_to(lyrics):
        return lyrics, "toast structure detected ('So here's to...')"

    if _has_stacked_nature_metaphors(lyrics):
        return lyrics, "stacked nature metaphors (I am the storm/fire/phoenix x3+)"

    if _has_nostalgia_opening(lyrics):
        return lyrics, "narrative nostalgia opener ('मुझे याद है...' / 'Remember when...')"

    if _has_prose_sentences(lyrics):
        return lyrics, f"prose sentences detected (avg line > {_PROSE_SENTENCE_AVG_WORDS} words, not bars)"

    if _has_generic_rap_cliches(lyrics):
        return lyrics, "generic LLM rap clichés detected ('paint pictures with my words' etc.)"

    if _has_hindi_narrative_prose(lyrics):
        return lyrics, "Hindi prose narrative pattern ('मैंने देखा/जाना/कहा...' x3+)"

    if _copies_prompt_example(lyrics):
        return lyrics, "copied the prompt's own worked-example lines instead of writing original content"

    return lyrics, None


async def generate_one(client, tier: str, topic_idx: int = 0) -> dict | None:
    max_attempts = MAX_ATTEMPTS_MIXED
    for attempt in range(1, max_attempts + 1):
        # vary the topic on retry too -- a rejected sample retried with the
        # exact same prompt tends to regenerate the same drifted shape.
        prompt = build_prompt(tier, topic_idx + attempt)
        try:
            if GEMINI_API_KEY:
                # Mistral (open-mistral-nemo) almost never produces genuine
                # intra-line Hinglish code-switching from free generation
                # (confirmed: intraline_mix_fraction=0.00 on most attempts
                # even with a worked example). Gemma-4 handles it reliably --
                # it's a "thinking" model, so it needs real headroom (up to
                # ~170s+ against this ~12.6K char prompt, sometimes an
                # intermittent 5xx) rather than a short timeout; that's
                # handled inside _gemini_generate itself (250s/call, 3
                # retries on timeout/5xx, 16K output budget).
                lyrics = await _gemini_generate(prompt, temperature=0.9)
            else:
                response = await asyncio.wait_for(
                    client.chat.complete_async(
                        model="open-mistral-nemo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.9,
                    ),
                    # Outer guard independent of the SDK's own timeout handling --
                    # observed in practice: some calls under concurrency=6 simply
                    # hung forever (no exception, no response), stalling 2 of 27
                    # workers indefinitely with no backoff/retry ever triggering.
                    timeout=90.0,
                )
                lyrics = response.choices[0].message.content.strip()
        except Exception as exc:
            detail = str(exc) or repr(exc)
            print(f"    ! generation call failed (attempt {attempt}): {type(exc).__name__}: {detail}")
            # Back off before retrying -- concurrent batches have hit sustained
            # Mistral 429 rate-limit bursts; retrying instantly makes it worse.
            is_rate_limited = "429" in detail or "rate_limited" in detail
            delay = min(15.0 * attempt, 60.0) if is_rate_limited else min(5.0 * attempt, 20.0)
            await asyncio.sleep(delay)
            continue

        lyrics = re.sub(r"^```[a-z]*\n?|```$", "", lyrics, flags=re.MULTILINE).strip()
        lyrics = re.sub(r"\*\*(.+?)\*\*", r"\1", lyrics)  # strip stray markdown bold
        lyrics = re.sub(r'^"|"$', "", lyrics.strip(), flags=re.MULTILINE).strip()
        lyrics = _strip_meta_annotation_lines(lyrics)
        lyrics = _strip_leaked_metrics_footer(lyrics)
        if len(content_lines(lyrics)) < 6:
            continue

        deva_ratio = _devanagari_ratio(lyrics)
        if deva_ratio < _MIN_DEVANAGARI_RATIO:
            print(f"    ~ rejected (attempt {attempt}/{max_attempts}), "
                  f"devanagari_ratio={deva_ratio:.2f} < {_MIN_DEVANAGARI_RATIO} (Romanized, not Devanagari)")
            continue

        # english_ratio/codeswitch_density are whole-verse aggregates and
        # can be hit perfectly by a verse built from segregated
        # mono-lingual lines (e.g. 2 English lines, then 1 full-Devanagari
        # line, repeated) -- that passes the aggregate numbers but is NOT
        # genuine Hinglish. Directly require a minimum fraction of lines
        # to mix scripts WITHIN the line itself. Confirmed via debug dump:
        # a segregated-line sample scored intraline_mix_fraction=0.0 and
        # would otherwise have passed every other gate.
        mix_frac = _intraline_mix_fraction(lyrics)
        if mix_frac < 0.2:
            # Free generation almost never lands genuine intra-line mixing
            # on its own (confirmed: most attempts score 0.00 even with a
            # worked example in the prompt) -- before burning a whole retry
            # attempt on a fresh re-roll, try a narrow follow-up EDIT call
            # that asks the model to mechanically rewrite just the
            # pure-script lines of THIS draft into mixed-script lines.
            # Models are much more reliable at bounded edits than at
            # following an abstract stylistic instruction from scratch.
            repaired = await _repair_intraline_mix(client, lyrics)
            repaired_mix_frac = _intraline_mix_fraction(repaired) if repaired else 0.0
            if repaired and repaired_mix_frac >= 0.2 and len(content_lines(repaired)) >= 6:
                print(f"    ~ repaired via follow-up edit (attempt {attempt}/{max_attempts}), "
                      f"intraline_mix_fraction {mix_frac:.2f} -> {repaired_mix_frac:.2f}")
                lyrics = repaired
            else:
                print(f"    ~ rejected (attempt {attempt}/{max_attempts}), "
                      f"intraline_mix_fraction={mix_frac:.2f} < 0.2 "
                      f"(lines are segregated mono-lingual, not genuinely code-switched within a line); "
                      f"repair attempt also failed (repaired={repaired_mix_frac:.2f})")
                if os.environ.get("DEBUG_DUMP_LYRICS"):
                    print("    ---- rejected lyrics dump ----")
                    print(lyrics)
                    print("    -------------------------------")
                continue

        uniq_ratio = _line_uniqueness_ratio(lyrics)
        if uniq_ratio < 0.5:
            print(f"    ~ rejected (attempt {attempt}/{max_attempts}), "
                  f"line_uniqueness={uniq_ratio:.2f} < 0.5 (degenerate repetition)")
            continue

        lyrics, gate_reason = await run_content_gates(client, lyrics, attempt, max_attempts)
        if gate_reason:
            print(f"    ~ rejected (attempt {attempt}/{max_attempts}), {gate_reason}")
            if os.environ.get("DEBUG_DUMP_LYRICS"):
                print("    ---- rejected lyrics dump ----")
                print(lyrics)
                print("    -------------------------------")
            continue

        accepted, actual, expected = score_against_tier(tier, lyrics)

        # NOTE: an english_ratio/codeswitch_density post-hoc gate used to live
        # here, but english_ratio (measure_extra's _english_ratio()) filters out
        # every Devanagari word via is_hindi_word() *before* computing the
        # ratio -- so for genuinely bilingual Hinglish it always reads close to
        # 100% (it measures "of the non-Hindi words, how many are English",
        # which is tautologically ~100%, not "what fraction of the verse is
        # English"). That gate was rejecting good, genuinely code-switched
        # output across the board. devanagari_ratio (above) and
        # intraline_mix_fraction (above) already correctly verify genuine
        # code-switching without this broken metric.

        if accepted:
            return {
                "id": str(uuid.uuid4()),
                "tier": tier,
                "language": "mixed",
                "target_profile": build_targets(tier),
                "actual_scores": actual,
                "expected_scores": expected,
                "attempts": attempt,
                "lyrics": lyrics,
            }
        print(f"    ~ rejected (attempt {attempt}/{max_attempts}), drifted from {tier} targets")
    return None


def _plan_batch(count: int) -> list[str]:
    """Spread `count` samples evenly across tiers.

    Used to also plan a `lang` per sample (weighted by corpus/artists.py's
    primary_language mix via language_mix()) -- now that generation is
    Hinglish-only, there's nothing left to weight; every slot is "mixed", so
    the plan is just a tier per sample."""
    return [TIER_NAMES[i % len(TIER_NAMES)] for i in range(count)]


async def run(count: int, tier_filter: str | None, resume: bool) -> int:
    from mistralai import Mistral

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: set MISTRAL_API_KEY", file=sys.stderr)
        return 2
    client = Mistral(api_key=api_key)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = _plan_batch(count)
    if tier_filter:
        plan = [t for t in plan if t == tier_filter]

    accepted_total = rejected_total = 0
    for i, tier in enumerate(plan, 1):
        tier_dir = OUT_DIR / tier / "mixed"
        tier_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(tier_dir.glob("*.json")))
        if resume and existing > i // len(TIER_NAMES):
            continue

        record = await generate_one(client, tier, topic_idx=i)
        if record is None:
            rejected_total += 1
            print(f"  [{i}/{len(plan)}] {tier}/mixed -> DROPPED after {MAX_ATTEMPTS_MIXED} attempts")
            continue

        path = tier_dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        accepted_total += 1
        print(f"  [{i}/{len(plan)}] {tier}/mixed -> accepted in {record['attempts']} attempt(s) ({path.name})")
        time.sleep(0.3)  # gentle pacing against Mistral rate limits

    print(f"\nDONE: {accepted_total} accepted, {rejected_total} dropped, out of {len(plan)} planned")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic Hinglish lyrics from numeric tier targets")
    ap.add_argument("--count", type=int, default=300, help="pilot batch size (default 300)")
    ap.add_argument("--tier", choices=TIER_NAMES, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    return asyncio.run(run(args.count, args.tier, args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
