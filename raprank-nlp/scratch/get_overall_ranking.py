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
        
    # Sort by mean score descending
    rankings.sort(key=lambda x: -x[1])
    
    print("\n=== OVERALL ARTIST RANKINGS (BY MEAN LQI SCORE) ===")
    for rank, (artist, score, count) in enumerate(rankings, 1):
        print(f"{rank}. {artist}: {score:.2f} ({count} tracks)")

if __name__ == "__main__":
    main()
