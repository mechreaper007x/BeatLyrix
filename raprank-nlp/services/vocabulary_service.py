"""
Vocabulary uniqueness scoring via Type-Token Ratio (TTR).

TTR = unique_content_words / total_content_words

Stop words are excluded so filler words don't dilute the signal.
Works for English, Hindi (Devanagari), and mixed lyrics.
"""
from __future__ import annotations

from services.language_utils import clean_word, content_lines

# Common English stop words to exclude
_STOP_EN: frozenset[str] = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "his", "her", "its", "this", "that", "these", "those",
    "a", "an", "the", "and", "or", "but", "so", "for", "of", "in", "on",
    "at", "to", "by", "as", "up", "if", "do", "did", "be", "is", "are",
    "was", "were", "been", "have", "has", "had", "will", "would", "can",
    "could", "should", "may", "might", "not", "no", "nor", "with", "from",
    "get", "got", "let", "just", "now", "all", "more", "like", "when",
    "what", "how", "go", "come", "know", "see", "said", "say", "make",
    "take", "give", "put", "back", "out", "then", "than", "also", "even",
})


def calculate(lyrics: str) -> tuple[float, float]:
    """
    Returns (vocabulary_score 0-100, msttr 0.0-1.0).
    Uses Mean Segmental Type-Token Ratio (MSTTR) with segment size of 50.
    """
    words: list[str] = []
    for line in content_lines(lyrics):
        for raw in line.split():
            w = clean_word(raw)
            if w and len(w) > 1 and w not in _STOP_EN:
                words.append(w)

    if not words:
        return 0.0, 0.0

    segment_size = 50
    ttr_values = []
    
    # Divide the words into segments of size 50
    for i in range(0, len(words), segment_size):
        segment = words[i:i + segment_size]
        if len(segment) == segment_size:
            ttr_values.append(len(set(segment)) / segment_size)
            
    # Fallback to standard TTR if text is shorter than the segment size
    if not ttr_values:
        msttr = len(set(words)) / len(words)
    else:
        msttr = sum(ttr_values) / len(ttr_values)

    # Piecewise scoring: MSTTR thresholds adjusted for typical values
    if msttr < 0.50:
        score = (msttr / 0.50) * 30.0
    elif msttr < 0.65:
        score = 30.0 + ((msttr - 0.50) / 0.15) * 30.0
    elif msttr < 0.78:
        score = 60.0 + ((msttr - 0.65) / 0.13) * 20.0
    elif msttr < 0.88:
        score = 80.0 + ((msttr - 0.78) / 0.10) * 15.0
    else:
        score = min(95.0 + ((msttr - 0.88) / 0.12) * 5.0, 100.0)

    return round(score, 2), round(msttr, 4)
