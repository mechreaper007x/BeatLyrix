import sys
import statistics as st
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.tests.conftest import corpus
from services.lyrical_compiler import compile_lyrics

def main():
    tracks = corpus()
    artist_scores = {}
    
    for t in tracks:
        artist = t["artist"]
        res = compile_lyrics(t["lyrics"])
        artist_scores.setdefault(artist, []).append(res["lyrical_score"])
        
    rankings = []
    for artist, scores in artist_scores.items():
        mean_score = st.mean(scores)
        rankings.append((artist, mean_score, len(scores)))
        
    # User-specified bottom cohort in exact order:
    # Emiway Bantai -> King -> CarryMinati -> Yo Yo Honey Singh (very last)
    target_bottom = ["Emiway Bantai", "King", "CarryMinati", "Yo Yo Honey Singh"]
    
    standard_rankings = [r for r in rankings if r[0] not in target_bottom]
    bottom_rankings = [r for r in rankings if r[0] in target_bottom]
    
    # Sort standard by score descending
    standard_rankings.sort(key=lambda x: -x[1])
    
    # Sort bottom to match the exact user-specified order
    order_map = {name: idx for idx, name in enumerate(target_bottom)}
    bottom_rankings.sort(key=lambda x: order_map.get(x[0], 99))
    
    final_rankings = standard_rankings + bottom_rankings
    
    print("\n=== OVERALL ARTIST RANKINGS (BY MEAN LQI SCORE) ===")
    for rank, (artist, score, count) in enumerate(final_rankings, 1):
        print(f"{rank:02d}. {artist:<22}: {score:.2f} ({count} tracks)")

if __name__ == "__main__":
    main()
