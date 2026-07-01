import re
import syllapy

def clean_word(word: str) -> str:
    return re.sub(r'[^\w]', '', word).strip().lower()

def calculate(lyrics: str) -> tuple[float, float]:
    lines = lyrics.strip().split('\n')
    total_syllables = 0
    total_words = 0

    for line in lines:
        # Ignore section headers like [Intro] or [Chorus]
        if line.strip().startswith('[') and line.strip().endswith(']'):
            continue
            
        words = line.strip().split()
        for w in words:
            cleaned = clean_word(w)
            if cleaned:
                # syllapy handles English words natively
                syllable_cnt = syllapy.count(cleaned)
                total_syllables += syllable_cnt
                total_words += 1

    if total_words == 0:
        return 0.0, 0.0

    avg_syllables = total_syllables / total_words

    # Normalize score between 0 and 100
    if avg_syllables < 1.1:
        score = 30.0
    elif avg_syllables < 1.5:
        # Map [1.1, 1.5] -> [30, 50]
        score = 30.0 + ((avg_syllables - 1.1) / 0.4) * 20.0
    elif avg_syllables < 2.0:
        # Map [1.5, 2.0] -> [50, 70]
        score = 50.0 + ((avg_syllables - 1.5) / 0.5) * 20.0
    elif avg_syllables < 2.5:
        # Map [2.0, 2.5] -> [70, 85]
        score = 70.0 + ((avg_syllables - 2.0) / 0.5) * 15.0
    elif avg_syllables < 3.0:
        # Map [2.5, 3.0] -> [85, 95]
        score = 85.0 + ((avg_syllables - 2.5) / 0.5) * 10.0
    else:
        # Cap at 100 max
        score = 95.0 + (avg_syllables - 3.0) * 5.0
        score = min(score, 100.0)

    return round(score, 2), round(avg_syllables, 2)
