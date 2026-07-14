import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.tests.conftest import load_corpus
from services.lyrical_compiler import compile_lyrics
from services import rhyme_service

corpus = load_corpus()
by_id = {}
for t in corpus:
    path = t.get("_path", "").replace("\\", "/")
    parts = path.split("/")
    artist = parts[-2] if len(parts) >= 2 else "unknown"
    title = parts[-1].replace(".json", "")
    by_id[f"{artist}/{title}"] = t["lyrics"]

targets = [
    "tsumyoki/dont-even-text",
    "paradox/glitch",
    "tsumyoki/idk",
    "king/iiconic",
    "karma/airplane-mode",
    "brodha-v/aigiri-nandini",
    "kr-na/still-standing",
    "seedhe-maut/brahamachari",
]

for sid in targets:
    lyrics = by_id.get(sid)
    if not lyrics:
        print(f"MISSING {sid}")
        continue
    compiled = compile_lyrics(lyrics)
    rb = rhyme_service.calculate(lyrics)
    print(sid)
    print(f"  LIVE (compile_lyrics)   rhyme_complexity={compiled['rhyme_complexity']}  "
          f"detected_rhyme_count={compiled['detected_rhyme_count']}  "
          f"unique_schemes={compiled['unique_rhyme_schemes']}  entropy={compiled['rhyme_entropy']}")
    print(f"  OFFLINE (rhyme_service) score={rb[0]}")
