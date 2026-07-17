"""
Rap-craft rules, expressed the way a human would teach a rapper -- NOT as
numbers. Fed to the reasoning model (minimax-m3) in place of the 15 numeric
axis targets that a non-reasoning model needed.

`RAP_CRAFT_GLOSSARY` defines every technique the local scorers measure, in
plain craft language. `TIER_CRAFT` says, per quality tier, HOW HEAVILY to
lean on each technique -- again qualitatively (elite = stack them densely;
commercial = keep it simple and catchy).

The numeric tier targets in corpus.synthetic.tier_profiles are NOT gone --
they still drive the accept/reject validation loop in generate_with_reference
(so every kept sample is verified to actually land in-tier and gets a
trustworthy ground-truth label). They are just no longer shown to the model;
the model reasons from craft rules and its output is checked against the
numbers after the fact.
"""
from __future__ import annotations

RAP_CRAFT_GLOSSARY = """THE CRAFT OF RAP -- the techniques a skilled lyricist controls:

⚠️  RAP IS NOT POETRY. Rap is spoken, performed, delivered INTO a mic over a beat.
Every line must pass the "out loud test": if it sounds like a poetry reading,
not like someone spitting bars over 808s, rewrite it.

RAP REGISTER (how real rappers talk):
- Conversational, direct, street-level language -- the way you'd actually talk to someone.
- Contractions everywhere: I'm, ain't, gotta, tryna, gonna, wanna, can't, won't, don't.
- Boasts, call-outs, direct address ("you", "they", "y'all").
- Short punchy clauses, punchline at the end of the bar.
- Slang, colloquialisms, code-switching between Hindi/English naturally.
- Confidence, swagger, attitude -- even when the topic is heavy.

AVOID (this is poetry/shayari/nazm, NOT rap):
- Ornate imagery ("a canvas of night", "the alley's belly", "a ritual of pain")
- Repetitive anaphora ("मैं हूँ वो..." x10, "I am the..." x8)
- Philosophical abstractions ("the soul's journey", "the heart's desire")
- Toast structures ("So here's to the grind, to the hustle")
- Stacked nature metaphors ("I am the storm, I am the rainbow, I am the phoenix")
- Nostalgia openers ("मुझे याद है, बचपन में...", "Remember when I was young...")
- Narrative prose with line breaks (telling a story paragraph-style, not spitting bars)

RAP EXAMPLES:
  "Tu kehta hai main gir gaya, main kehta hu main ne sambhaali"
  "Stack the paper, dodge the static, they don't want to see me make it"
  "They said I couldn't, I did it, now they watching from the stands"

POETRY EXAMPLES (rejected):
  "मैं हूँ वो आग, जो तेरे अंदर भड़कती है" (shayari anaphora)
  "I am the storm that came, I am the rainbow that followed" (nature metaphor stack)
  "In the echo of a shattered mirror, I found myself" (ornate imagery)
  "So here's to the grind, to the hustle, to the fight" (toast structure)

RHYME
- End rhyme: the last word of two lines share a sound (cat / hat). The basic building block.
- Internal rhyme: rhymes land WITHIN a single line, not only at the end ("I'm SIPPING on a POTION with DEVOTION").
- Multisyllabic rhyme: two or more syllables rhyme as a unit, not just the final one ("calibrated / celebrated",
  "kar diya / bhar diya") -- the mark of a technical rapper vs. a simple one.
- Rhyme chain: the same rhyme sound is sustained across 3, 4, 5+ lines in a row instead of switching every couplet.
- Compound / mosaic rhyme: a multi-word phrase rhymes with another word or phrase ("hold me" / "goalie",
  "raat bhar" / "haath par").
- Holorime: two whole phrases sound almost identical end to end -- the rarest, hardest device.

SOUND TEXTURE
- Assonance: the same VOWEL sound repeats across several words in a line ("the RAIN in SPAIN", "dhaEk sEk mEl").
- Consonance: a CONSONANT sound (not the first letter of the word) repeats across words ("blank / think / junk").
- Onomatopoeia / ad-libs: vocalized sound effects woven in ("brr", "skrrt", "boom", "ayy", "dhak dhak").

FLOW & DELIVERY
- Syllable density: how many syllables are packed per line. Long, dense bars read as more technical; short, sparse
  lines read as simpler / more commercial.
- Complex words: use of longer, multi-syllable words vs. only short everyday ones.
- Vocabulary richness: how wide and varied the word choice is -- an elite writer rarely repeats the same word;
  a commercial hook repeats on purpose.

WORDPLAY
- Simile: an explicit comparison with "like / as" ("jaise", "jaisa", "sa") -- but use SPARINGLY and concretely.
  GOOD: "flow paani jaisa, beh raha non-stop"
  BAD: "like a river of dreams flowing to the sea" (too poetic, too abstract)
- Metaphor: an implicit comparison -- calling one thing another without "like".
  GOOD: "ye game mera jaal hai, sab fase mere"
  BAD: "I am the architect of my own destiny" (literary, abstract)
- Pun: playing on a word that sounds like, or means, two things at once.
- Double entendre: a line that carries two meanings simultaneously, often one plain and one clever/cultural.

MORE TEXTURE & STRUCTURE
- Alliteration: nearby words start with the same sound ("BIG BAD business", "chalti CHAAL mein CHAMKE").
- Repetition: deliberately repeating a word, phrase, or hook/refrain for emphasis or catchiness.
- Cadence variance: mixing up line length and rhythm across the verse instead of every line landing
  the same length and beat -- short punchy lines against longer flowing ones."""


TIER_CRAFT: dict[str, str] = {
    "elite": """QUALITY TARGET: ELITE / technical lyricist (think KR$NA, Raftaar, Karma, Eminem, Kendrick Lamar, J.Cole).
Write dense, re-listenable BARS -- not a poem, not a shayari, not a nazm. RAP.

REGISTER: Conversational, direct, street-level. Talk TO someone, not AT them.
- Start mid-thought or mid-action, never with a philosophical opener.
- Use slang, code-switching, boasts, call-outs -- the way a real MC delivers.
- Punchline at the end of bars. Confidence and swagger throughout.

TECHNIQUE:
- Pack the lines -- long, syllable-heavy bars, sophisticated and varied vocabulary, rarely repeat a word.
- Stack INTERNAL rhymes inside most lines, not just at the ends.
- Lean hard on MULTISYLLABIC rhymes -- rhyme 2-3 syllables together, constantly.
- Sustain rhyme CHAINS across 3-5 consecutive lines before switching the sound.
- Occasionally land a compound / mosaic rhyme (a multi-word phrase rhyming with another).
- Layer wordplay throughout: metaphors, the odd pun or double meaning.
- Stack alliteration densely within lines -- a hallmark of a technical writer showing off control.
- Vary cadence heavily -- mix long dense bars with sudden short punches, never settle into one rhythm.
- Avoid repeating words/phrases -- a rich vocabulary and rare repetition is part of the elite sound.

AVOID:
- Narrative prose with line breaks (telling a story paragraph-style, not spitting bars)
- Repetitive anaphora ("मैं हूँ वो..., मैं हूँ वो..., मैं हूँ वो..." x10 -- this is shayari, not rap)
- Literary/poetic diction ("a canvas of night", "the soul's journey", "a ritual of pain")
- Philosophical abstractions ("the heart's desire", "the spirit calls")
- Toast structures ("So here's to the grind, to the hustle")
- Nature metaphors stacked ("I am the storm, I am the rainbow, I am the phoenix")

This should sound like the most technically impressive rap you can write.""",

    "mid": """QUALITY TARGET: MID / competent (think Divine, Brodha V, Hanumankind).
Write solid, structured verses that are clearly skilled but NOT densely technical.
This tier sits exactly HALFWAY between elite and commercial on every axis below --
do not default to elite-level density just because you can write well.

REGISTER: Same as elite -- conversational, direct, street-level rap. NOT poetry.
- Talk about real things: money, grind, rivals, goals, daily life.
- Use contractions, slang, code-switching naturally.
- Short to moderate lines, punchy delivery.

TECHNIQUE:
- Clear, satisfying END rhymes on every couplet.
- SOME internal rhyme, but only in roughly half the lines -- the other half should have none.
- Include AT LEAST 2-3 clear multisyllabic rhymes somewhere in the verse (2 syllables rhyming as
  a unit, e.g. "kar diya / bhar diya") -- do not skip this entirely, but do not chain it either.
- KEEP LINES MODERATE LENGTH -- roughly 6-9 syllables per line, NOT long elite-style dense bars.
  Short, punchy lines that don't overload each line with content.
- Use everyday, moderately varied vocabulary and let some words/phrases repeat naturally.
- A few similes or a metaphor across the whole verse -- not layered wordplay.
- Light, occasional alliteration.
- Moderate cadence variance.

AVOID:
- Narrative prose with line breaks ("I remember when I was young, my family was poor...")
- Repetitive anaphora patterns ("मैं हूँ वो..." x8 is shayari, not rap)
- Literary/poetic diction and philosophical abstractions
- Toast structures and nostalgia arcs

Competent and listenable, but consciously restrained -- leave the dense stacking to elite.""",

    "commercial": """QUALITY TARGET: COMMERCIAL / hook-driven (think King, CarryMinati, Emiway Bantai, Yo Yo Honey Singh, Badshah).
Write simple, catchy, radio/club-friendly RAP -- not a lullaby, not a bedtime story.

REGISTER: Conversational and direct, but simpler than elite/mid.
- Simple slang, everyday language, confident delivery.
- Chantable hooks, memorable phrases.
- Direct address, boasts, party vibes.
- Heavy repetition: same phrase/word repeated across lines IS the hook.

TECHNIQUE:
- Simple SINGLE-syllable end rhymes, easy and predictable.
- Repetitive and hook-like -- repeating words, phrases, and a refrain is EXPECTED.
- Short, easy lines (4-6 syllables max) built from common everyday vocabulary.
- Almost NO internal rhyme.
- ZERO multisyllabic rhyme -- if you hear "calibrated / celebrated", that's elite.
- Almost NO wordplay -- direct brags, statements, or a chantable hook.
- Almost NO alliteration -- the words don't need to connect phonetically.
- NO assonance or consonance devices.
- Steady, consistent cadence -- every line lands at roughly the same length.
- Repeat a hook phrase or refrain 3+ times per verse.
- Vocabulary should be extremely simple and repetitive.

The score ceiling for this tier is STRICTLY LOW. Even a perfect commercial
track should not exceed ~45/100 on the LQI. If your output has internal
rhymes, multisyllabic rhymes, varied vocabulary, or any wordplay devices,
it has drifted into mid or elite territory -- simplify it.

AVOID:
- Narrative prose with line breaks
- Literary/poetic diction
- Philosophical or emotional depth -- keep it surface-level and catchy
- Longer lines -- keep everything short and punchy
- Internal rhyme, multisyllabic rhyme, or compound rhyme
- Assonance, consonance, or alliteration devices
- Varied vocabulary -- repetition IS the point

Prioritize catchiness and simplicity over ANY technical density.""",
}


def craft_block(tier: str) -> str:
    """Glossary + this tier's craft guidance -- the qualitative replacement for
    the numeric target list."""
    return f"{RAP_CRAFT_GLOSSARY}\n\n{TIER_CRAFT[tier]}"
