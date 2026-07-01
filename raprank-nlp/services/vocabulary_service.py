"""
Vocabulary uniqueness scoring via Type-Token Ratio (TTR).

TTR = unique_content_words / total_content_words

Stop words are excluded so filler words don't dilute the signal.
Works for English, Hindi (Devanagari), and mixed lyrics.
"""
from __future__ import annotations

from services.language_utils import clean_word, content_lines, get_multilingual_stopwords
from config import scoring_config


def calculate(lyrics: str) -> tuple[float, float]:
    """
    Returns (vocabulary_score 0-100, msttr 0.0-1.0).
    Uses Mean Segmental Type-Token Ratio (MSTTR) with segment size from config.
    """
    words: list[str] = []
    stop_words = get_multilingual_stopwords()
    for line in content_lines(lyrics):
        for raw in line.split():
            w = clean_word(raw)
            if w and len(w) > 1 and w not in stop_words:
                words.append(w)

    if not words:
        return 0.0, 0.0

    segment_size = scoring_config.VOCABULARY["MSTTR_SEGMENT_SIZE"]
    ttr_values = []
    
    # Divide the words into segments of size segment_size
    for i in range(0, len(words), segment_size):
        segment = words[i:i + segment_size]
        if len(segment) == segment_size:
            ttr_values.append(len(set(segment)) / segment_size)
            
    # Fallback to standard TTR if text is shorter than the segment size
    if not ttr_values:
        msttr = len(set(words)) / len(words)
    else:
        msttr = sum(ttr_values) / len(ttr_values)

    # Dynamic piecewise scoring curve evaluation
    score = scoring_config.evaluate_piecewise_curve(
        msttr,
        scoring_config.VOCABULARY["CURVE_THRESHOLDS"],
        scoring_config.VOCABULARY["CURVE_SCORES"]
    )

    return round(score, 2), round(msttr, 4)
