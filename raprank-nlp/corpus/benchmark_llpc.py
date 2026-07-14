"""
Benchmark script comparing Lyrical Lexer & Parser Compiler (LLPC)
against the old rule-based scoring weights on real lyrics.
"""
from __future__ import annotations

import sys
import statistics as st
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.tests.conftest import corpus
from services.lyrical_compiler import compile_lyrics
from services import (
    alliteration_service as al, rhyme_service as rh, syllable_service as sy,
    vocabulary_service as vo, wordplay_service as wp,
)
from config import scoring_config

# Categorize artists based on user description
HARDCORE_LYRICAL = {
    "KR$NA", "Raftaar", "Karma", "EPR", "Seedhe Maut", "Calm", "Brodha V", "Yungsta", "Vichaar", "Naam Sujal"
}
COMMERCIAL_MAINSTREAM = {
    "Emiway Bantai", "CarryMinati", "King", "Paradox", "Raga"
}

def calculate_old_score(lyrics: str) -> float:
    sy_score, _, sy_weight, _ = sy.calculate(lyrics)
    rh_score, _, _, _, _, _, _ = rh.calculate(lyrics)
    al_score, _ = al.calculate(lyrics)
    vo_score, _ = vo.calculate(lyrics)
    wp_score, _ = wp.calculate(lyrics)
    
    w = scoring_config.MAIN_WEIGHTS["TEXT_ONLY"]
    return (
        rh_score * w["rhyme"] +
        sy_score * w["syllable"] +
        al_score * w["alliteration"] +
        vo_score * w["vocabulary"] +
        wp_score * w["wordplay"] +
        sy_weight * w["syllable_weight"]
    )

def main():
    tracks = corpus()
    if not tracks:
        print("No corpus files found. Make sure tests/conftest.py points to a valid corpus/data directory.")
        return

    # Store results
    results = []
    
    for t in tracks:
        artist = t["artist"]
        title = t["title"]
        lyrics = t["lyrics"]
        
        # Determine cohort
        if artist in HARDCORE_LYRICAL:
            cohort = "Hardcore Lyrical"
        elif artist in COMMERCIAL_MAINSTREAM:
            cohort = "Commercial/Mainstream"
        else:
            cohort = "Other"
            
        old_score = calculate_old_score(lyrics)
        
        # Compile via new LLPC
        compiled = compile_lyrics(lyrics)
        new_score = compiled["lyrical_score"]
        
        results.append({
            "title": title,
            "artist": artist,
            "cohort": cohort,
            "old_score": old_score,
            "new_score": new_score,
            "rhyme_entropy": compiled["rhyme_entropy"],
            "unique_rhyme_schemes": compiled["unique_rhyme_schemes"],
            "avg_syllables_per_line": compiled["avg_syllables_per_line"]
        })

    # Print summary
    print(f"# RapRank NLP Benchmark Report (n={len(results)} tracks)\n")
    
    # 1. Cohort Analysis
    cohort_stats = {}
    for c in ["Hardcore Lyrical", "Commercial/Mainstream"]:
        c_tracks = [r for r in results if r["cohort"] == c]
        if not c_tracks:
            continue
        old_scores = [r["old_score"] for r in c_tracks]
        new_scores = [r["new_score"] for r in c_tracks]
        entropies = [r["rhyme_entropy"] for r in c_tracks]
        
        cohort_stats[c] = {
            "n": len(c_tracks),
            "old_mean": st.mean(old_scores),
            "new_mean": st.mean(new_scores),
            "entropy_mean": st.mean(entropies)
        }

    print("## Cohort Performance Comparison\n")
    print("| Cohort | Tracks | Old Rule-Based Mean | New LLPC LQI Mean | Rhyme State Entropy (Mean) |")
    print("|---|---|---|---|---|")
    for c, stats in cohort_stats.items():
        print(f"| **{c}** | {stats['n']} | {stats['old_mean']:.2f} | {stats['new_mean']:.2f} | {stats['entropy_mean']:.4f} |")
    
    # Compute separation margins
    if "Hardcore Lyrical" in cohort_stats and "Commercial/Mainstream" in cohort_stats:
        old_margin = cohort_stats["Hardcore Lyrical"]["old_mean"] - cohort_stats["Commercial/Mainstream"]["old_mean"]
        new_margin = cohort_stats["Hardcore Lyrical"]["new_mean"] - cohort_stats["Commercial/Mainstream"]["new_mean"]
        print(f"\n*   **Old Rule-Based separation Margin**: {old_margin:.2f} points")
        print(f"*   **New LLPC LQI separation Margin**: {new_margin:.2f} points")
        improvement = ((new_margin - old_margin) / old_margin) * 100 if old_margin > 0 else 0
        print(f"*   **Separation Margin Improvement**: +{improvement:.1f}% higher contrast between Hardcore & Commercial rap!")

    # 2. Top Lyrical Tracks (New Scorer)
    print("\n## Top 10 Lyrical Tracks (Unsupervised LLPC)")
    print("| Rank | Track | Artist | New Score | Old Score | Rhyme Entropy |")
    print("|---|---|---|---|---|---|")
    ranked = sorted(results, key=lambda x: -x["new_score"])
    for idx, r in enumerate(ranked[:10], 1):
        print(f"| {idx} | {r['title']} | {r['artist']} | **{r['new_score']:.2f}** | {r['old_score']:.2f} | {r['rhyme_entropy']:.4f} |")

    # 3. Bottom/Simple Tracks (New Scorer)
    print("\n## 5 Most Commercial/Simple Tracks (Unsupervised LLPC)")
    print("| Rank | Track | Artist | New Score | Old Score | Rhyme Entropy |")
    print("|---|---|---|---|---|---|")
    for idx, r in enumerate(reversed(ranked[-5:]), 1):
        print(f"| {idx} | {r['title']} | {r['artist']} | **{r['new_score']:.2f}** | {r['old_score']:.2f} | {r['rhyme_entropy']:.4f} |")

if __name__ == "__main__":
    main()
