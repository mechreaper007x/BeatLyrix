"""
Scoring calibration, weights, tolerances, and thresholds for RapRank NLP.
No thresholds or scoring parameters are hardcoded in the service logic directly.
"""
from __future__ import annotations

# ── Flow / Rhythm Calibration ───────────────────────────────────────────────
FLOW = {
    "POCKET_TOLERANCE_MS": 50.0,         # Pocket timing tolerance (ms)
    "GRID_SUBDIVISIONS": 2,              # 8th-note grid (2 subdivisions per beat)
    "SPEED_BONUS_MIN_SYLLABLES": 3.5,    # Minimum syllables/sec to start receiving speed bonus
    "SPEED_BONUS_MAX_SYLLABLES": 6.0,    # Max syllables/sec (up to 10 points bonus)
    "SPEED_BONUS_WEIGHT": 10.0,
    "CADENCE_VARIANCE_WINDOW": 4.0,      # Cadence variance measuring window (seconds)
    "CADENCE_VARIANCE_WEIGHT": 5.0,
    
    # Pocket Ratio Piecewise Scoring Curve
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
        "end_rhyme": 0.30,
        "internal": 0.25,
        "multisyllabic": 0.25,
        "chains": 0.20,
    },
    
    # Density targets for max sub-scores
    "ELITE_TARGETS": {
        "internal_density": 0.25,        # 25% internal rhyme density per line is elite
        "multisyllabic_density": 0.15,   # 15% multisyllabic rhyme density is elite
        "chain_density": 0.25,           # 25% of lines stacked in continuous chains is elite
    },
    
    # End Rhyme Ratio Piecewise Scoring Curve
    "CURVE_THRESHOLDS": [0.20, 0.40, 0.60, 0.80],
    "CURVE_SCORES": [40.0, 65.0, 85.0, 95.0],
}

# ── Wordplay Calibration ─────────────────────────────────────────────────────
WORDPLAY = {
    # Known rap-specific double-meaning candidate words (English & Romanized Hindi)
    "RAP_DOUBLE_ENTENDRE_WORDS": {
        "bar", "bars", "key", "keys", "draft", "bank", "charge", "beat", "case", "court",
        "spin", "rock", "roll", "arms", "joint", "high", "line", "lines", "hit", "rap",
        "flow", "crack", "note", "notes", "banda", "paisa", "bhaari", "maal", "trap",
        "ice", "cold", "fire", "smoke", "spit", "pocket", "drive", "ride", "run", "cuff",
        "lock", "deal", "plate", "scale", "pound", "gram", "dope", "green", "burn"
    },
    
    # Weights for final wordplay combination
    "WEIGHT_OVERALL_DENSITY": 0.50,
    "WEIGHT_MAX_SUB_SCORE": 0.50,
    
    # Density targets for max sub-scores (divisor for maxing out)
    "ELITE_TARGETS": {
        "simile": 0.15,                  # 15% similes is elite
        "metaphor": 0.10,               # 10% metaphors is elite
        "pun": 0.08,                     # 8% puns is elite
        "entendre": 0.12,                # 12% double-meaning words is elite
    },
    
    # WordNet polysemy thresholds
    "ENTENDRE_MIN_SENSES_RAP": 2,
    "ENTENDRE_MIN_SENSES_GENERAL": 7,
    
    # Overall Wordplay Density Curve
    "CURVE_THRESHOLDS": [0.15, 0.30],
    "CURVE_SCORES": [60.0, 85.0],
}

# ── Vocabulary Calibration (MSTTR) ──────────────────────────────────────────
VOCABULARY = {
    "MSTTR_SEGMENT_SIZE": 50,            # Word segment count for TTR calculation
    
    # Uniqueness (MSTTR) Piecewise Scoring Curve
    "CURVE_THRESHOLDS": [0.50, 0.65, 0.78, 0.88],
    "CURVE_SCORES": [30.0, 60.0, 80.0, 95.0],
}

# ── Syllables Calibration ────────────────────────────────────────────────────
SYLLABLE = {
    "MIN_WORDS_FOR_DENSITY": 3,          # Ignore lines with fewer than 3 words (ad-libs)
    "COMPLEX_WORD_SYLLABLES": 3,         # Words with >= 3 syllables are complex
    
    # Syllable Density (Average syllables per line) Curve
    "DENSITY_THRESHOLDS": [6.0, 8.0, 10.0, 12.0, 14.0],
    "DENSITY_SCORES": [30.0, 50.0, 70.0, 85.0, 95.0],
    
    # Syllable Weight (Complex word ratio) Curve
    "WEIGHT_THRESHOLDS": [0.05, 0.15],
    "WEIGHT_SCORES": [30.0, 80.0],
}

# ── Main Endpoint Aggregation Weights ────────────────────────────────────────
MAIN_WEIGHTS = {
    "WITH_FLOW": {
        "rhyme": 0.20,
        "syllable": 0.15,
        "alliteration": 0.10,
        "vocabulary": 0.10,
        "wordplay": 0.15,
        "syllable_weight": 0.10,
        "flow": 0.20,
    },
    "TEXT_ONLY": {
        "rhyme": 0.25,
        "syllable": 0.20,
        "alliteration": 0.15,
        "vocabulary": 0.15,
        "wordplay": 0.15,
        "syllable_weight": 0.10,
    }
}


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
