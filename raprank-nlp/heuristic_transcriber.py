import os
import tempfile
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RapRank Heuristic Transcriber",
    description="Rule-based audio/lyrics synchronization using silence detection (No AI models).",
    version="1.0.0",
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "heuristic_transcriber"}

@app.post("/sync")
async def sync_lyrics(
    file: UploadFile = File(...),
    lyrics: str = Form(...)
) -> JSONResponse:
    """
    Takes an audio file and raw lyrics.
    Uses Librosa to find non-silent intervals.
    Maps lines of lyrics to the found intervals mathematically.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided.")
    if not lyrics:
        raise HTTPException(status_code=400, detail="No lyrics provided.")

    audio_bytes = await file.read()
    
    # 1. Clean and split lyrics into lines
    lines = [line.strip() for line in lyrics.split("\n") if line.strip() and not line.strip().startswith("[")]
    if not lines:
        return JSONResponse(content={"words": []})

    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".mp3"
        
    tmp_path = None
    try:
        # Save to temp file for librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        logger.info(f"Loading audio {file.filename} for heuristic chunking...")
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        
        # 2. Heuristic Silence Detection (Rule-based Voice Activity Detection)
        # top_db=30 means anything 30dB below the peak is considered silence.
        intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
        
        # Convert frame intervals to seconds
        time_intervals = [(interval[0] / sr, interval[1] / sr) for interval in intervals]
        logger.info(f"Detected {len(time_intervals)} audio segments.")

        # 3. Mathematical mapping:
        # We need to map `len(lines)` lines onto `len(time_intervals)` intervals.
        # If there are more lines than intervals, we group lines into the same interval.
        # If there are fewer lines, we group intervals.
        
        words_data = []
        
        # Fallback if no intervals found
        if not time_intervals:
            duration = librosa.get_duration(y=y, sr=sr)
            time_intervals = [(0, duration)]

        # Map evenly based on percentage
        total_intervals = len(time_intervals)
        total_lines = len(lines)
        
        current_word_idx = 0
        
        for i, line in enumerate(lines):
            # Find the corresponding interval proportionally
            interval_idx = int((i / total_lines) * total_intervals)
            interval_idx = min(interval_idx, total_intervals - 1)
            
            start_time, end_time = time_intervals[interval_idx]
            
            line_words = line.split()
            word_count = len(line_words)
            if word_count == 0:
                continue
                
            # Sub-divide the interval for each word in the line
            duration = end_time - start_time
            time_per_word = duration / word_count
            
            for j, word in enumerate(line_words):
                w_start = start_time + (j * time_per_word)
                w_end = w_start + time_per_word
                
                words_data.append({
                    "id": current_word_idx,
                    "word": word,
                    "start": round(w_start, 3),
                    "end": round(w_end, 3),
                    "probability": 1.0 # Heuristic approach assumes 100% confidence in text
                })
                current_word_idx += 1
                
        logger.info(f"Successfully mapped {len(words_data)} words using heuristic alignment.")
        
        return JSONResponse(content={
            "text": lyrics,
            "detected_language": "heuristic",
            "language_probability": 1.0,
            "words": words_data
        })

    except Exception as e:
        logger.error(f"Heuristic alignment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    # Run on a distinct port from the main NLP service
    logger.info("Starting Heuristic Transcriber on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
