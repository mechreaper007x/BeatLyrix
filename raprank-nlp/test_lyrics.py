import sys
import os
import json
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from services.lyrical_compiler import compile_lyrics
from services import (
    wordplay_service,
    alliteration_service,
    assonance_service,
    consonance_service,
    onomatopoeia_service,
    bayesian_scoring_service as b,
    svm_quality_service as s,
    rf_quality_service as r
)

def format_color(val, threshold_low=45, threshold_high=70):
    if val < threshold_low:
        return f"\033[91m{val:.2f} (Low/Commercial)\033[0m"
    elif val < threshold_high:
        return f"\033[93m{val:.2f} (Mid-Tier)\033[0m"
    else:
        return f"\033[92m{val:.2f} (Elite/Lyrical)\033[0m"

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_lyrics.py \"your raw lyrics here\"")
        print("  python test_lyrics.py path/to/lyrics_file.txt")
        sys.exit(1)

    input_data = sys.argv[1]
    
    # Check if input is a file path
    if os.path.exists(input_data):
        with open(input_data, "r", encoding="utf-8") as f:
            lyrics = f.read()
        print(f"[*] Reading lyrics from file: {input_data}")
    else:
        lyrics = input_data
        print("[*] Analyzing raw lyrics input...")

    lyrics = lyrics.strip()
    if not lyrics:
        print("[-] Error: Lyrics input is empty.")
        sys.exit(1)

    # 1. Compile raw LQI metrics
    res = compile_lyrics(lyrics)
    
    lqi = res["lyrical_score"]
    rhyme_complexity = res["rhyme_complexity"]
    syllable_score = res["syllable_density"]
    lexical_score = res["lexical_information_density"]
    
    # Extract subscores
    wp_score, wp_meta = wordplay_service.calculate(lyrics)
    allit_score, _ = alliteration_service.calculate(lyrics)
    asso_score, _ = assonance_service.calculate(lyrics)
    cons_score, _ = consonance_service.calculate(lyrics)
    ono_score, _ = onomatopoeia_service.calculate(lyrics)
    
    # 2. Predict classifiers
    try:
        b_mod = b.load()
        s_mod = s.load()
        r_mod = r.load()
        
        sc = {
            "rhyme": rhyme_complexity,
            "syllable": syllable_score,
            "vocabulary": lexical_score,
            "wordplay": wp_score,
            "alliteration": allit_score,
            "assonance": asso_score,
            "consonance": cons_score,
            "onomatopoeia": ono_score,
            "cadence": res.get("cadence_variance", 50.0),
        }
        
        bp = b.predict_posterior(b_mod, sc)
        b_tier = max(bp, key=bp.get)
        s_tier = s.predict_tier_from_scores(s_mod, sc)["tier"]
        r_tier = r.predict_tier_from_scores(r_mod, sc)["tier"]
    except Exception as e:
        b_tier = s_tier = r_tier = "Not Available (models untrained)"
        bp = {}

    # Print results
    print("\n" + "="*60)
    print("                 RAPRANK NLP ANALYSIS REPORT                ")
    print("="*60)
    print(f"Total LQI Score:      {format_color(lqi)}")
    print(f"Rhyme Complexity:    {format_color(rhyme_complexity)}")
    print(f"Wordplay Score:      {format_color(wp_score)}")
    print(f"Vocabulary Richness:  {format_color(lexical_score)}")
    print(f"Syllable Density:     {syllable_score:.2f}%")
    print(f"Assonance Match:      {asso_score:.2f}%")
    print(f"Consonance Match:     {cons_score:.2f}%")
    print(f"Onomatopoeia Hits:    {ono_score:.2f}%")
    
    print("\n" + "-"*40)
    print("AI QUALITY CLASSIFIERS")
    print("-"*40)
    print(f"Flow Critic (RF) Predicts:    {r_tier.upper()}")
    print(f"The Gatekeeper (SVM) Predicts: {s_tier.upper()}")
    print(f"The Oracle (Bayes) Predicts:   {b_tier.upper()}")
    if bp:
        print(f"  Oracle Posterior Probabilities: Elite: {bp.get('elite',0)*100:.1f}%, Mid: {bp.get('mid',0)*100:.1f}%, Commercial: {bp.get('commercial',0)*100:.1f}%")

    print("\n" + "-"*40)
    print("LITERARY METADATA BREAKDOWN")
    print("-"*40)
    print(f"Similes:           {wp_meta.get('simile_count', 0)}")
    print(f"Metaphors:         {wp_meta.get('metaphor_count', 0)}")
    print(f"Puns/Homophones:   {wp_meta.get('pun_count', 0)}")
    print(f"Double Entendres:  {wp_meta.get('double_entendre_count', 0)}")
    print(f"Punchlines:        {wp_meta.get('punchline_count', 0)}")
    print(f"Allusions:         {wp_meta.get('allusion_count', 0)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
