import re

def clean_word(word: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', word).strip().lower()

def calculate(lyrics: str) -> tuple[float, list[str]]:
    lines = lyrics.strip().split('\n')
    pairs = []
    valid_lines_count = 0

    for line in lines:
        if line.strip().startswith('[') and line.strip().endswith(']'):
            continue
        
        valid_lines_count += 1
        words = line.strip().split()
        if len(words) < 2:
            continue

        for i in range(len(words) - 1):
            w1 = clean_word(words[i])
            w2 = clean_word(words[i+1])
            
            if w1 and w2 and w1[0] == w2[0]:
                pairs.append(f"{words[i]} - {words[i+1]}")

    if valid_lines_count == 0:
        return 0.0, []

    density = len(pairs) / valid_lines_count

    # Normalize density to 0-100 score
    if density == 0.0:
        score = 0.0
    elif density < 0.1:
        score = (density / 0.1) * 40.0
    elif density < 0.2:
        score = 40.0 + ((density - 0.1) / 0.1) * 25.0
    elif density < 0.3:
        score = 65.0 + ((density - 0.2) / 0.1) * 15.0
    elif density < 0.5:
        score = 80.0 + ((density - 0.3) / 0.2) * 12.0
    else:
        score = 92.0 + (density - 0.5) * 10.0
        score = min(score, 100.0)

    return round(score, 2), pairs
