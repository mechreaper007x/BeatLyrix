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
    target_artists = ["Emiway Bantai", "CarryMinati", "King"]
    
    for artist in target_artists:
        art_tracks = [t for t in tracks if t["artist"] == artist]
        if not art_tracks:
            print(f"No tracks found for {artist}")
            continue
            
        scores = []
        entropies = []
        track_details = []
        
        for t in art_tracks:
            res = compile_lyrics(t["lyrics"])
            scores.append(res["lyrical_score"])
            entropies.append(res["rhyme_entropy"])
            track_details.append((t["title"], res["lyrical_score"], res["rhyme_entropy"]))
            
        print(f"\n=== {artist} (n={len(art_tracks)}) ===")
        print(f"Mean LQI Score: {st.mean(scores):.2f}")
        print(f"Mean Rhyme Entropy: {st.mean(entropies):.4f}")
        print("Tracks:")
        # Sort tracks by score descending
        for title, score, entropy in sorted(track_details, key=lambda x: -x[1]):
            print(f"  - {title}: Score={score:.2f}, Entropy={entropy:.4f}")

if __name__ == "__main__":
    main()
