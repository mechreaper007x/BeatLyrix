"""
Scoring calibration, weights, tolerances, and thresholds for RapRank NLP.
No thresholds or scoring parameters are hardcoded in the service logic directly.
"""
from __future__ import annotations
import json
import pathlib

# ── Allusion Reference Loader ────────────────────────────────────────────────
_ALLUSION_JSON = pathlib.Path(__file__).parent.parent / "corpus" / "allusion_references.json"

def _load_allusion_references() -> set:
    """
    Loads the allusion reference set from corpus/allusion_references.json.
    This is the ONLY place allusions are defined -- edit the JSON file,
    never hardcode entries here. Falls back to an empty set on error.
    """
    if not _ALLUSION_JSON.exists():
        import logging
        logging.getLogger(__name__).warning(
            "allusion_references.json not found at %s -- allusion detection disabled",
            _ALLUSION_JSON
        )
        return set()
    try:
        raw = json.loads(_ALLUSION_JSON.read_text(encoding="utf-8"))
        entries: set = set()
        for key, val in raw.items():
            if key.startswith("_"):          # skip metadata keys
                continue
            if isinstance(val, list):
                entries.update(v.lower() for v in val)
            elif isinstance(val, dict):      # nested category dict
                for sublist in val.values():
                    if isinstance(sublist, list):
                        entries.update(v.lower() for v in sublist)
        return entries
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Failed to load allusion_references.json: %s", exc)
        return set()


# ── Flow / Rhythm Calibration ───────────────────────────────────────────────
FLOW = {
    "POCKET_TOLERANCE_MS": 50.0,         # Pocket timing tolerance (ms)
    "GRID_SUBDIVISIONS": 2,              # 8th-note grid (2 subdivisions per beat)
    "SPEED_BONUS_MIN_SYLLABLES": 3.5,    # Minimum syllables/sec to start receiving speed bonus
    "SPEED_BONUS_MAX_SYLLABLES": 6.0,    # Max syllables/sec (up to 10 points bonus)
    "SPEED_BONUS_WEIGHT": 10.0,
    "CADENCE_VARIANCE_WINDOW": 4.0,      # Cadence variance measuring window (seconds)
    "CADENCE_VARIANCE_WEIGHT": 5.0,
    
    # Pocket Ratio Piecewise Scoring Curve. NOT ECDF-recalibrated in the DHH
    # scoring-engine pass (see corpus/calibrate.py): flow_service.calculate_beat_sync
    # requires separated vocal/accompaniment audio bytes, which the 302-track
    # corpus.data/ JSON files don't carry (lyrics/metadata only, no audio) --
    # there is no corpus signal to fit against without a separate audio corpus.
    "CURVE_THRESHOLDS": [0.40, 0.60, 0.75],
    "CURVE_SCORES": [40.0, 65.0, 85.0],
}

# ── Rhyme Calibration ────────────────────────────────────────────────────────
RHYME = {
    "WINDOW_SIZE_LINES": 5,              # Number of lines ahead to scan for end rhymes (min(a+5))
    "DEVA_SUFFIX_LENGTH": 3,             # Hindi Devanagari normal suffix length
    "DEVA_MULTI_SUFFIX_LENGTH": 5,       # Hindi Devanagari multisyllabic suffix length
    
    # Combined Rhyme Score weights
    "WEIGHTS": {
        "end_rhyme": 0.15,
        "internal": 0.20,
        "multisyllabic": 0.25,
        "chains": 0.22,
        "compound": 0.10,
        "holorime": 0.08,
    },

    # Density targets for max sub-scores. internal/multisyllabic/chain refit
    # via corpus.calibrate ECDF against the 302-track corpus (~90th
    # percentile of each raw ratio -- top-decile density maxes the sub-score).
    # compound/holorime could NOT be ECDF-fit: both raw ratios are exactly
    # 0.0 across all 302 real tracks, including known technical showcases
    # (KR$NA "Vyanjan") -- these devices are genuinely near-absent from real
    # DHH lyrics, so there is no positive corpus signal to calibrate against.
    # Left at their original hand-set values (a single occurrence in an
    # average-length song already maxes the sub-score, matching their
    # extreme rarity as literary devices).
    "ELITE_TARGETS": {
        "internal_density": 0.21,        # ECDF (DHH phoneme keyer, 505 tracks): 90th pct = 0.208.
                                          # Was 0.30 under the old spelling-heuristic keyer -- the
                                          # phoneme engine is stricter (no false matches from spelled
                                          # schwas), so the same corpus measures lower.
        "multisyllabic_density": 0.33,   # ECDF re-run with phoneme keyer: 90th pct = 0.333 (unchanged)
        "chain_density": 0.14,           # ECDF re-run with phoneme keyer: 95th pct = 0.130 (~unchanged)
        "compound_density": 0.05,        # unchanged -- no positive corpus signal (see note above)
        "holorime_density": 0.03,        # unchanged -- no positive corpus signal (see note above)
    },

    # Compound/mosaic rhyme: minimum matched-unit length (CMU phones for
    # English, normalized letters for Hinglish) required before a single-word
    # vs. multi-word phrase match counts -- filters out short/trivial words
    # (e.g. "her") coincidentally matching a phrase tail.
    "COMPOUND_MIN_PHONES": 4,
    "COMPOUND_MIN_LETTERS": 5,

    # Holorime ("perfect" multi-word rhyme, e.g. "ice cream" / "I scream"):
    # minimum matched-unit length for a FULL phrase-to-phrase match (stricter
    # than compound since the entire phrase, not just a tail, must agree).
    "HOLORIME_MIN_PHONES": 5,
    "HOLORIME_MIN_LETTERS": 6,

    # End Rhyme Ratio Piecewise Scoring Curve. Thresholds = corpus 25th/50th/
    # 75th/90th percentiles of the raw end-rhyme ratio (ECDF-fit via
    # corpus.calibrate --metric rhyme.end_rhyme_ratio); score anchors
    # unchanged from the prior hand-set curve.
    "CURVE_THRESHOLDS": [0.25, 0.38, 0.52, 0.67],
    "CURVE_SCORES": [30.0, 55.0, 75.0, 90.0],

    # Hindi/Hinglish grammatical (inflectional) endings that rhyme automatically:
    # every past-tense verb ends in -aya, every feminine verb in -ani/-ati, etc.
    # Matching on these alone is zero-skill "conjugation rhyme" and must NOT be
    # credited as multisyllabic/chain rhyme (which distinguishes real lyricists
    # from repetitive commercial tracks). Longest matches are checked first.
    "TRIVIAL_RHYME_SUFFIXES": (
        "aayaa", "aaya", "aya", "aye", "ayi", "aye", "aaye",
        "ana", "ani", "ane", "ana", "aana", "aani", "aane",
        "ata", "ati", "ate", "aata", "aati", "aate",
        "unga", "ungi", "oonga", "oongi", "enge", "enge", "ega", "egi",
        "iya", "iye", "iyaan", "wala", "wali", "wale", "waala", "waale",
        "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
        "diya", "liya", "kiya", "tani", "nani", "hani", "vani",
        "taa", "naa", "hai", "hain", " hoon", "hoon",
        "ta", "ti", "te", "na", "ne", "ya", "ye", "da", "de", "la", "le", "ra", "ri", "re",
    ),

    # A trivial-suffix rhyme still counts toward the basic end-rhyme ratio, but
    # its contribution is scaled down by this factor so filler-heavy verses can't
    # inflate the end-rhyme score with conjugation rhymes.
    "TRIVIAL_END_RHYME_WEIGHT": 0.4,

    # Rhyme-scheme diversity multiplier. Skilled lyricists cycle through many
    # distinct rhyme sounds; monotonous tracks hammer the same 1-2 sounds every
    # line. diversity = distinct end-rhyme keys / keyed lines. Below FLOOR the
    # rhyme score is scaled to MIN_MULT; at/above CEIL it is untouched.
    "DIVERSITY_MIN_LINES": 6,
    "DIVERSITY_FLOOR": 0.25,
    "DIVERSITY_CEIL": 0.40,
    "DIVERSITY_MIN_MULT": 0.65,
}

# ── LLPC live rhyme formula (services/lyrical_compiler.py) ───────────────────
# This is the formula actually behind the LIVE "Rhyme Complexity" score users
# see (main.py wires ScoreBreakdown.rhyme_score straight from
# compile_lyrics()["rhyme_complexity"]) -- a completely separate, simpler
# implementation from the RHYME block above (which only backs rhyme_service.py,
# dead code for live scoring, still used by tests/corpus tooling).
#
# rhyme_density and normalized_entropy were previously blended as raw
# fractions (density*0.40 + entropy*0.60)*100 with no curve -- but ECDF
# against the 302-track corpus (corpus.calibrate --metric llpc.rhyme_density /
# llpc.normalized_entropy) showed neither raw component's real-world range
# gets anywhere near 1.0 (99th percentile ~0.57 and ~0.32 respectively), so
# the resulting rhyme_complexity was capped at ~42.75 even for the best real
# track in the corpus (mean 18.6) -- a severe, previously-undetected
# compression of the live score. Each raw component is now mapped through its
# own percentile-fit piecewise curve (25th/50th/75th/90th percentiles ->
# 30/55/75/90, matching the convention used elsewhere in this file) before
# the unchanged 0.40/0.60 blend -- the blend ratio itself wasn't shown to be
# the problem (neither raw component dominates or is near-constant), only
# the raw-fraction-to-percentage mapping was.
#
# Refit via corpus.calibrate (ECDF over the 302-track corpus) after fixing
# two bugs in the underlying rhyme_key/window: the key required BOTH of the
# last two vowels and BOTH of the last two trailing consonants to match
# (fragmenting true end-rhymes into distinct non-matching keys whenever the
# second-to-last syllable differed), and the SymbolTable's 3-line scope
# window counted against the raw lyric line index rather than the rhyme
# stream, so blank/section-header lines silently ate into the window. Both
# bugs structurally penalized wide-rhyme-alphabet schemes (many distinct end
# sounds, each reused a few lines apart) relative to tight AABB couplets --
# thresholds below are the 25th/50th/75th/90th percentiles of the corrected
# raw metric.
LLPC_RHYME = {
    "DENSITY_THRESHOLDS": [0.39, 0.48, 0.55, 0.63],
    "DENSITY_SCORES": [30.0, 55.0, 75.0, 90.0],
    "ENTROPY_THRESHOLDS": [0.39, 0.44, 0.49, 0.54],
    "ENTROPY_SCORES": [30.0, 55.0, 75.0, 90.0],
}

# ── Sound / Phonetic Calibration (assonance & consonance) ────────────────────
# Assonance = repeated VOWEL nucleus across content words in a line (non-onset
# vowel music, e.g. "the rain in Spain"). Consonance = repeated NON-ONSET
# consonant sound across content words (e.g. "blank/think/junk" on -nk). Onset
# repetition is alliteration and is scored separately, so consonance explicitly
# ignores the first sound of each word to avoid double-counting.
SOUND = {
    # A line "fires" for a sound when at least this many DISTINCT content words
    # share the vowel nucleus (assonance) / interior-or-final consonant
    # (consonance). 3 keeps it from rewarding an incidental pair.
    "MIN_WORDS_PER_GROUP": 3,
    "MIN_WORD_LEN": 2,

    # Per-line weight is capped so one very long line can't dominate the density.
    "MAX_GROUP_WEIGHT": 3.0,

    # Occurrences of the same sound cluster across lines if no gap between
    # consecutive hits exceeds this many lines -- lets a repeated hook spread
    # over several lines register instead of vanishing at each line boundary.
    # Kept to adjacent-line-only (1): vowel/non-onset-consonant sound classes
    # are few enough (a dozen or so) that a wider window matches almost every
    # song's ordinary word choice by chance, not a deliberate device -- this
    # was measured empirically against the 302-track corpus (see Verification
    # in the alliteration/assonance/consonance cross-line plan): windows > 1
    # pushed both metrics' corpus mean above 88-98, i.e. saturated.
    "WINDOW_SIZE_LINES": 1,

    # Density (weighted firing lines / valid lines) piecewise curve → 0-100.
    # ECDF-fit against the real 302-track corpus (corpus.calibrate --metric
    # assonance.density / consonance.density) at the 25th/50th/75th
    # percentiles of the raw pre-curve density, matching the convention used
    # for ALLITERATION -- the prior thresholds were hand-picked and did not
    # line up with the corpus's actual percentile breakpoints (e.g. the old
    # assonance first threshold of 0.3 sat at only the ~7th percentile).
    "ASSONANCE_THRESHOLDS": [0.46, 0.61, 0.74],
    "ASSONANCE_SCORES": [20.0, 55.0, 85.0],
    "CONSONANCE_THRESHOLDS": [0.70, 0.97, 1.20],
    "CONSONANCE_SCORES": [20.0, 55.0, 85.0],

    # Known non-lexical ad-libs / onomatopoeia (English + Hinglish). Deliberately
    # excludes ordinary filler words ("yeah", "yo") already treated as stopwords
    # elsewhere -- this axis is about vocalized *sound effects*, not filler.
    "ONOMATOPOEIA_WORDS": {
        "woo", "skrrt", "brr", "brrr", "ayy", "ayyy", "uh", "huh", "ha", "haha",
        "boom", "bam", "pow", "bang", "vroom", "zoom", "ding", "clang", "crash",
        "buzz", "hiss", "sizzle", "whoosh", "swoosh", "thud", "clap", "snap",
        "aah", "ooh", "hola", "oye", "arre", "wah", "vah", "aha", "oho",
    },

    # A word counts as an elongated interjection (e.g. "ayyy", "yooo", "brrrr")
    # when any single letter repeats at least this many times in a row --
    # distinct from the known-word list, catches ad-libbed stylization of any
    # base word.
    "ONOMATOPOEIA_ELONGATION_MIN_REPEAT": 3,

    # Density (ad-lib hits / raw lines, including lines the rest of the
    # pipeline strips as pure ad-libs) piecewise curve -> 0-100. Onomatopoeia
    # is a rare, deliberate device -- median across the corpus is exactly
    # 0.0 (most tracks use none), so a 25th/50th/75th percentile fit would be
    # degenerate (two zero thresholds). ECDF-fit at the 75th/90th/95th
    # percentiles instead (corpus.calibrate --metric onomatopoeia.density),
    # matching the shape of the prior hand-picked thresholds but grounded in
    # the corpus's actual sparse-tail distribution rather than guessed.
    "ONOMATOPOEIA_THRESHOLDS": [0.0163, 0.0535, 0.1066],
    "ONOMATOPOEIA_SCORES": [20.0, 55.0, 85.0],
}

# ── Alliteration Calibration ────────────────────────────────────────────────
# Alliteration = repeated ONSET (first) sound across content words. Previously
# scoped to a single line only, which made repeated hooks/refrains spread
# across consecutive lines (e.g. a chorus line whose first word repeats 8x)
# structurally invisible to any threshold, since only the sound-device axis
# was missing a cross-line window that rhyme_service.py already has via
# RHYME["WINDOW_SIZE_LINES"]. These constants were previously hardcoded
# directly in alliteration_service.py, unvalidated against the real corpus --
# moved here to match every other sound-device service's calibration
# discipline (see module docstring).
ALLITERATION = {
    # Occurrences of the same onset sound cluster across lines if no gap
    # between consecutive hits exceeds this many lines. There are more
    # distinct onset sounds than vowel nuclei or non-onset consonant classes,
    # so alliteration tolerates a slightly wider window than
    # SOUND["WINDOW_SIZE_LINES"] without saturating the corpus -- verified
    # empirically (see THRESHOLDS note below).
    "WINDOW_SIZE_LINES": 2,

    # A cluster fires once it has at least this many TOTAL occurrences of the
    # onset sound close together (window above) -- occurrences may be
    # distinct words ("Big blue bouncy balls") or the same word repeated
    # ("kaun kaun kaun"), matching alliteration's actual definition: words
    # that start with the same sound and sit close together, full stop.
    # There is no separate word-position requirement -- unlike the previous
    # design, a repeated phrase mid-line ("talve laal ... talve laal") counts
    # exactly like a repeated line-initial hook word.
    "MIN_OCCURRENCES_PER_GROUP": 3,
    "MIN_WORD_LEN": 2,
    "MAX_GROUP_WEIGHT": 3.0,

    # Within a firing cluster, credit is split into a variety component
    # (distinct words beyond the first, full weight) and a repetition
    # component (occurrences of an already-seen word, weighted down by
    # REPEATED_WORD_WEIGHT_SCALE) -- pure repetition of one word is still
    # alliteration, but genuine word variety ("Big blue bouncy balls") is a
    # more skilled device than saying the same word four times, so it should
    # score higher for the same occurrence count.
    "REPEATED_WORD_WEIGHT_SCALE": 0.5,

    # Raw density (total cluster weight / valid lines) is divided by this
    # before the curve below. Re-fit after the recurring-hook-line fix in
    # alliteration_service.py (a verbatim chorus block reappearing later in
    # the song no longer re-earns MAX_GROUP_WEIGHT credit a second time --
    # previously a single repeated hook, printed as two separate physical
    # chorus blocks, could independently peg the per-cluster cap in each
    # block, letting one 4-word hook contribute a third or more of a song's
    # total density).
    #
    # A 3-point curve (25th/50th/75th pctile) normalized at the 80th
    # percentile was tried first but left evaluate_piecewise_curve's
    # always-ramps-to-100-at-value=1.0 tail badly compressed: everything from
    # the 80th to 99th percentile got jammed into the last 0.06-wide
    # normalized band (0.94->1.0), so a solidly-good-but-not-exceptional song
    # (e.g. 80th percentile) scored 96, barely distinguishable from a 99th
    # percentile song pinned at literal 100. Normalizing at the 95th
    # percentile instead (1.22) spreads that same tail across a much wider
    # 0.90->1.0 band, and only the genuine top ~5% saturate.
    "DENSITY_NORM": 1.22,

    # Density (normalized, see DENSITY_NORM above) piecewise curve -> 0-100.
    # 4-point fit (25th/50th/75th/90th percentile of normalized density),
    # matching the resolution VOCABULARY/SYLLABLE already use for their top
    # quartile instead of leaving it to a single steep tail segment.
    "THRESHOLDS": [0.42, 0.58, 0.74, 0.90],
    "SCORES": [20.0, 45.0, 65.0, 88.0],
}

# ── Audio Pipeline (Demucs separation, CPU-only) ────────────────────────────
AUDIO_PIPELINE = {
    # Demucs vocal/accompaniment separation is CPU-bound and has no hard
    # timeout of its own, so very long audio risks the same kind of
    # multi-minute event-loop block the Whisper service had. This threshold
    # was previously hardcoded at 90s in services/separation_service.py --
    # low enough to skip separation on almost every real song (most rap
    # tracks run 2.5-4+ minutes), silently degrading flow-score accuracy
    # (accompaniment isn't actually isolated) with no visibility into why.
    #
    # Raised to a duration that at least covers most real tracks rather than
    # skipping by default. NOTE: this number has not been empirically timed
    # against Demucs in the actual deployment environment (the `demucs`
    # package isn't installed in this dev sandbox, only in the built
    # container per requirements.txt) -- tune it against real wall-clock
    # timing on whatever hardware raprank-nlp actually runs on before
    # trusting it in production.
    "DEMUCS_MAX_DURATION_S": 300.0,
    "DEMUCS_MAX_SIZE_MB": 20.0,  # fallback when duration can't be determined
}

# ── Wordplay Calibration ─────────────────────────────────────────────────────
WORDPLAY = {
    # Known rap-specific double-meaning candidate words (English & Romanized Hindi)
    "RAP_DOUBLE_ENTENDRE_WORDS": {
        "bar", "bars", "key", "keys", "draft", "bank", "charge", "beat", "case", "court",
        "spin", "rock", "roll", "arms", "joint", "high", "line", "lines", "hit", "rap",
        "flow", "crack", "note", "notes", "banda", "paisa", "bhaari", "maal", "trap",
        "ice", "cold", "fire", "smoke", "spit", "pocket", "drive", "ride", "run", "cuff",
        "lock", "deal", "plate", "scale", "pound", "gram", "dope", "green", "burn",
        "kashmiri", "cashmere", "pupil", "pupils", "motiyabind", "aadhar", "cheeta", "cheetah",
        "kutta", "kutto", "lagaan", "brahman", "don", "doon", "shakkar", "dalle", "dalal",
        "saabu", "shroud", "shahrule", "khatoon", "bhed", "bhedchaal", "manjha", "dheel",
        "columbia", "yggdrasil", "tiananmen", "borphukan", "bismil", "tsushima", "tasleema",
        "amadeus", "memento", "fibonacci", "shakuni", "chaturanga", "mitochondria", "notochord"
    },

    # Allusion / pop-culture reference lexicon.
    # !! DO NOT ADD ENTRIES HERE !!
    # Edit corpus/allusion_references.json instead -- it loads at import time
    # and can be updated without touching any Python code.
    # The JSON supports categorised groups (bollywood_cinema, anime_manga, etc.)
    # so new domains can be added without hunting for the right place in code.
    "ALLUSION_REFERENCES": _load_allusion_references(),


    # Minimum line denominator for density calcs. Without this, a single device
    # in a 4-line verse yields density 0.25 and pegs a sub-score at 100, awarding
    # ~97/100 to trivial short inputs. Smoothing over >= this many lines fixes it.
    "MIN_LINES_FOR_DENSITY": 8,

    # Evidence gate: eliteness in a single category (max_sub_score) is only fully
    # credited once at least this many literary devices are detected overall. A
    # lone (often false-positive) detection can no longer crown a verse elite.
    "MIN_ELEMENTS_FOR_ELITE": 4,

    # Weights for final wordplay combination
    "WEIGHT_OVERALL_DENSITY": 0.50,
    "WEIGHT_MAX_SUB_SCORE": 0.50,
    
    # Density targets for max sub-scores (divisor for maxing out)
    # ECDF-fit against the real 302-track corpus (corpus.calibrate --metric
    # wordplay.<x>_density) at the 95th percentile of each raw density,
    # matching the convention used for RHYME's chain_density -- simile/
    # metaphor/entendre were previously near-unreachable outliers (e.g. old
    # simile target 0.15 sat above the corpus's 99th percentile of 0.1509)
    # while pun's old target of 0.08 sat below its own 90th percentile,
    # pegging 15% of the corpus at a saturated 100. allusion is left
    # unchanged: it requires a live external NER call (services/
    # wordplay_service.py detect_allusions) unavailable in this dev sandbox,
    # so no corpus-wide raw density could be computed for it here.
    "ELITE_TARGETS": {
        "simile": 0.0956,
        "metaphor": 0.0512,
        "pun": 0.0999,
        "entendre": 0.0714,
        "allusion": 0.04,                 # unchanged -- see note above, no corpus signal available
    },
    
    # WordNet polysemy thresholds
    "ENTENDRE_MIN_SENSES_RAP": 2,
    "ENTENDRE_MIN_SENSES_GENERAL": 999,

    
    # Overall Wordplay Density Curve. ECDF-fit at the 75th/95th percentiles
    # (corpus.calibrate --metric wordplay.total_density) -- close to the
    # prior hand-picked values, now grounded and with a slightly improved
    # discrimination spread (pstdev 24.5 vs 22.8).
    "CURVE_THRESHOLDS": [0.1474, 0.2351],
    "CURVE_SCORES": [60.0, 85.0],
}

# ── Vocabulary Calibration (MSTTR) ──────────────────────────────────────────
VOCABULARY = {
    "MSTTR_SEGMENT_SIZE": 50,            # Word segment count for TTR calculation

    # Uniqueness (MSTTR) Piecewise Scoring Curve. ECDF-fit against the real
    # 302-track corpus (corpus.calibrate --metric vocabulary.msttr) at the
    # 10th/50th/75th/95th percentiles -- the prior thresholds sat below even
    # the corpus's 5th percentile (0.61), so nearly every real track already
    # scored 80+ regardless of actual lexical variety (corpus mean was 82.95,
    # barely discriminating). Refit spreads the scale across where real
    # tracks actually sit (corpus mean now 61.44, spread 21.5 vs prior 14.7).
    "CURVE_THRESHOLDS": [0.67, 0.83, 0.88, 0.93],
    "CURVE_SCORES": [30.0, 60.0, 80.0, 95.0],
}

# ── LQI Vocabulary Richness Bonus (lyrical_compiler.py) ─────────────────────
# Artists with near-zero word repetition are penalised by the rhyme density
# component because flow-pocketing deliberately avoids repeated end-words.
# This bonus partially compensates without breaking the scale for repetitive tracks.
VOCAB_BONUS = {
    # Proxy MSTTR threshold above which the bonus kicks in (0.0 – 1.0 range)
    "MSTTR_THRESHOLD": 0.85,
    # Maximum bonus points added to LQI at proxy MSTTR == 1.0
    "MAX_BONUS_POINTS": 4.0,
    # Minimum number of content words in the track to qualify for bonus
    "MIN_WORDS_REQUIRED": 20,
    # Sliding window step for MSTTR calculation (fraction of segment size)
    "SEGMENT_STEP_FRACTION": 0.5,
}


# ── Syllables Calibration ────────────────────────────────────────────────────
SYLLABLE = {
    "MIN_WORDS_FOR_DENSITY": 3,          # Ignore lines with fewer than 3 words (ad-libs)
    "COMPLEX_WORD_SYLLABLES": 3,         # Words with >= 3 syllables are complex
    
    # Syllable Density (Average syllables per line) Curve. ECDF-fit against
    # the real 302-track corpus (corpus.calibrate --metric
    # syllable.avg_per_line) at the 10th/25th/50th/75th/90th percentiles --
    # improves discrimination spread (pstdev 21.8 vs 14.2 under the prior
    # hand-picked thresholds) without changing the mean.
    "DENSITY_THRESHOLDS": [7.91, 8.76, 9.74, 10.96, 12.24],
    "DENSITY_SCORES": [30.0, 50.0, 70.0, 85.0, 95.0],

    # Syllable Weight (Complex word ratio) Curve. ECDF-fit at the
    # 10th/75th percentiles (corpus.calibrate --metric syllable.weight_ratio)
    # -- nearly identical to the prior hand-picked values, now grounded.
    "WEIGHT_THRESHOLDS": [0.0484, 0.1535],
    "WEIGHT_SCORES": [30.0, 80.0],
}

# ── Prosody / structural axes (services/prosody_service.py) ─────────────────
# Code-switching, anaphora/repetition, and text cadence variance -- promoted
# from the offline signature.py prototypes to live scored axes. Thresholds are
# on the raw 0-1 fractions each heuristic returns; curves map them to 0-100 via
# evaluate_piecewise_curve. Descriptive axes -- NOT folded into total_score.
PROSODY = {
    # Fraction of lines mixing English + non-English words. ECDF-fit against
    # the real 302-track corpus (corpus.calibrate --metric prosody.codeswitch)
    # at the 25th/50th/75th percentiles -- the prior thresholds sat well
    # below the corpus's actual distribution (old first threshold 0.10 was
    # below the 5th percentile), skewing the corpus mean to ~80/100 and
    # under-discriminating.
    "CODESWITCH_THRESHOLDS": [0.39, 0.55, 0.69],
    "CODESWITCH_SCORES": [30.0, 65.0, 90.0],

    # Anaphora: fraction of consecutive line pairs sharing a first word. Real
    # anaphora is rare/deliberate (corpus median is only 0.037), so ECDF-fit
    # at the 75th/90th/95th percentiles (corpus.calibrate --metric
    # prosody.repetition), matching the sparse-tail convention used for
    # ONOMATOPOEIA.
    "REPETITION_THRESHOLDS": [0.098, 0.1667, 0.2491],
    "REPETITION_SCORES": [35.0, 70.0, 90.0],

    # Words-per-line stdev (already /4.0-normalised, ~0-1). ECDF-fit at the
    # 25th/50th/75th percentiles (corpus.calibrate --metric
    # prosody.cadence_var) -- prior thresholds skewed the corpus mean to
    # ~81/100; refit spreads the scale across where real tracks sit
    # (pstdev 27.4 vs 11.9).
    "CADENCE_THRESHOLDS": [0.4329, 0.5104, 0.6488],
    "CADENCE_SCORES": [30.0, 65.0, 90.0],
}


# ── Main Endpoint Aggregation Weights ────────────────────────────────────────
MAIN_WEIGHTS = {
    "WITH_FLOW": {
        "rhyme": 0.20,
        "syllable": 0.15,
        "alliteration": 0.08,
        "vocabulary": 0.08,
        "wordplay": 0.15,
        "syllable_weight": 0.08,
        "flow": 0.20,
        "assonance": 0.02,
        "consonance": 0.02,
        "onomatopoeia": 0.02,
    },
    "TEXT_ONLY": {
        "rhyme": 0.40,
        "syllable": 0.00,
        "alliteration": 0.01,
        "vocabulary": 0.01,
        "wordplay": 0.49,
        "syllable_weight": 0.04,
        "assonance": 0.02,
        "consonance": 0.02,
        "onomatopoeia": 0.01,
    }
}

# NOTE: Scoring is now 100% local/rule-based. Mistral was removed from the
# scoring path entirely (it only describes song story/structure now), so the
# former HYBRID_COMBINATION_WEIGHTS (local/llm blend per axis) is gone --
# every axis and the total come straight from the services in services/.



def evaluate_piecewise_curve(
    value: float,
    thresholds: list[float],
    scores: list[float],
    max_score: float = 100.0
) -> float:
    """
    Evaluates a value dynamically along a piecewise linear scoring curve.
    Uses thresholds and target score markers from config to interpolate the score.
    """
    if value <= 0.0:
        return 0.0

    # Below the first threshold
    if value < thresholds[0]:
        return (value / thresholds[0]) * scores[0]

    # Interpolate within matching interval
    for i in range(1, len(thresholds)):
        t_prev, t_curr = thresholds[i - 1], thresholds[i]
        s_prev, s_curr = scores[i - 1], scores[i]
        if value < t_curr:
            ratio = (value - t_prev) / (t_curr - t_prev)
            return s_prev + ratio * (s_curr - s_prev)

    # Above the last threshold
    t_last = thresholds[-1]
    s_last = scores[-1]
    if t_last < 1.0:
        ratio = (value - t_last) / (1.0 - t_last)
        return min(s_last + ratio * (max_score - s_last), max_score)
    
    return min(s_last, max_score)
