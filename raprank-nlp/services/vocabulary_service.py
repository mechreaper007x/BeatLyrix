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
    Returns (vocabulary_score 0-100, ttr 0.0-1.0).
    """
    words: list[str] = []
    for line in content_lines(lyrics):
        for raw in line.split():
            w = clean_word(raw)
            if w and len(w) > 1 and w not in _STOP_EN:
                words.append(w)

    if not words:
        return 0.0, 0.0

    ttr = len(set(words)) / len(words)

    # Piecewise scoring: TTR ≥ 0.7 is excellent in rap
    if ttr < 0.3:
        score = (ttr / 0.3) * 30.0
    elif ttr < 0.5:
        score = 30.0 + ((ttr - 0.3) / 0.2) * 30.0
    elif ttr < 0.7:
        score = 60.0 + ((ttr - 0.5) / 0.2) * 25.0
    elif ttr < 0.85:
        score = 85.0 + ((ttr - 0.7) / 0.15) * 12.0
    else:
        score = min(97.0 + (ttr - 0.85) * 20.0, 100.0)

    return round(score, 2), round(ttr, 4)
