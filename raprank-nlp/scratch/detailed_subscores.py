import sys
from pathlib import Path
import statistics as st
from collections import defaultdict

ROOT = Path("C:/Users/Savyasachi Mishra/Desktop/Beatlyrix/raprank-nlp")
sys.path.insert(0, str(ROOT))

from services.tests.conftest import corpus
from services.lyrical_compiler import compile_lyrics
from services import (
    wordplay_service,
    alliteration_service,
    assonance_service,
    consonance_service,
    onomatopoeia_service,
)
from config import scoring_config

tracks = corpus()
artist_scores = defaultdict(list)

# Pre-load config weights
w = scoring_config.MAIN_WEIGHTS["TEXT_ONLY"]

for t in tracks:
    artist = t["artist"]
    lyrics = t["lyrics"]
    
    # Run full compilation to get standard fields
    res = compile_lyrics(lyrics)
    
    # Get extra sub-scores
    wp_score, _ = wordplay_service.calculate(lyrics)
    allit_score, _ = alliteration_service.calculate(lyrics)
    asso_score, _ = assonance_service.calculate(lyrics)
    cons_score, _ = consonance_service.calculate(lyrics)
    ono_score, _ = onomatopoeia_service.calculate(lyrics)
    
    # Calculate cadence variance
    lines = [line.strip() for line in lyrics.split("\n") if line.strip() and not (line.strip().startswith("[") and line.strip().endswith("]"))]
    syllable_counts = []
    # Simple syllable count per line
    from services.lyrical_compiler import LyricalLexer
    for line in lines:
        lexer = LyricalLexer(line)
        tokens = lexer.scan()
        vowel_tokens = [t for line_tokens in tokens for t in line_tokens if t.type == "T_VOWEL"]
        syllable_counts.append(len(vowel_tokens))
    
    if len(syllable_counts) >= 3:
        syl_stdev = st.pstdev(syllable_counts)
    else:
        syl_stdev = 0.0
    cadence_score = scoring_config.evaluate_piecewise_curve(
        syl_stdev,
        [5.0, 7.0, 9.0, 11.0],
        [10.0, 30.0, 60.0, 85.0],
    )
    
    row = {
        "lqi": res["lyrical_score"],
        "rhyme": res["rhyme_complexity"],
        "syllable": res["syllable_density"],
        "lexical": res["lexical_information_density"],
        "wordplay": wp_score,
        "allit": allit_score,
        "assonance": asso_score,
        "consonance": cons_score,
        "onomatopoeia": ono_score,
        "cadence": cadence_score
    }
    artist_scores[artist].append(row)

# Print average subscores
print("="*140)
print(f"{'Artist':<22} | {'LQI':<6} | {'Rhyme':<6} | {'Syllable':<8} | {'Lexical':<7} | {'Wordplay':<8} | {'Allit':<6} | {'Asso':<6} | {'Cons':<6} | {'Ono':<6} | {'Cad':<6}")
print("="*140)

# Sort by LQI desc
sorted_artists = sorted(artist_scores.keys(), key=lambda a: -st.mean([x["lqi"] for x in artist_scores[a]]))

for artist in sorted_artists:
    rows = artist_scores[artist]
    lqi = st.mean([x["lqi"] for x in rows])
    rhyme = st.mean([x["rhyme"] for x in rows])
    syl = st.mean([x["syllable"] for x in rows])
    lex = st.mean([x["lexical"] for x in rows])
    wp = st.mean([x["wordplay"] for x in rows])
    allit = st.mean([x["allit"] for x in rows])
    asso = st.mean([x["assonance"] for x in rows])
    cons = st.mean([x["consonance"] for x in rows])
    ono = st.mean([x["onomatopoeia"] for x in rows])
    cad = st.mean([x["cadence"] for x in rows])
    
    print(f"{artist:<22} | {lqi:<6.2f} | {rhyme:<6.2f} | {syl:<8.2f} | {lex:<7.2f} | {wp:<8.2f} | {allit:<6.2f} | {asso:<6.2f} | {cons:<6.2f} | {ono:<6.2f} | {cad:<6.2f}")
