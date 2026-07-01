"""
Flow / Beat-sync scoring using librosa.

Algorithm:
  1. Load audio with librosa (ffmpeg backend handles mp3/wav/ogg).
  2. Detect tempo and beat frame positions.
  3. Convert beat frames → beat times (seconds).
  4. For each word onset from faster-whisper timestamps,
     find the nearest beat and measure deviation.
  5. Classify word as "on-beat" if deviation ≤ beat_period / 8.
  6. Score on_beat_ratio with a piecewise curve.

Flow score is only available when audio bytes AND word timestamps
are both provided (i.e. /transcribe-and-analyze or /analyze with words=[...]).
"""
from __future__ import annotations

import logging
import os
import tempfile

logger = logging.getLogger(__name__)

try:
    import librosa
    import numpy as np
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    logger.warning("librosa not installed — flow/beat-sync scoring disabled.")


def calculate_beat_sync(
    audio_bytes: bytes,
    filename: str,
    words: list[dict],
) -> tuple[float, dict]:
    """
    Compute a flow/beat-sync score for a rap track.

    Args:
        audio_bytes: Raw audio file content.
        filename:    Original filename — used to infer audio format.
        words:       List of {word, start, end} dicts from faster-whisper.

    Returns:
        (flow_score 0-100, metadata dict)
        On failure returns (0.0, {"error": "reason"}).
    """
    if not _LIBROSA_OK:
        return 0.0, {"error": "librosa not available"}

    if not words:
        return 0.0, {"error": "no word timestamps provided"}

    suffix = (os.path.splitext(filename)[1].lower() or ".mp3")
    tmp_path: str | None = None

    try:
        # Write audio to temp file (librosa needs a path for mp3 decoding)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Load audio — preserve native sample rate, force mono
        y, sr = librosa.load(tmp_path, sr=None, mono=True)

        # Beat tracking
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo_arr)[0])

        if beat_frames.size == 0:
            return 0.0, {"error": "no beats detected in audio"}

        beat_times: list[float] = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        beat_period = 60.0 / tempo  # seconds per beat
        grid_size = beat_period / 4.0 # 16th notes
        tolerance = grid_size / 3.0 # strict pocket: ±8.3% of a quarter note

        # Evaluate each word onset
        word_starts = [w["start"] for w in words if "start" in w]
        if not word_starts:
            return 0.0, {"error": "word timestamp data missing 'start' field"}

        on_beat = 0
        deviations: list[float] = []

        for onset in word_starts:
            closest_beat = min(beat_times, key=lambda bt: abs(bt - onset))
            offset = onset - closest_beat
            nearest_grid_offset = round(offset / grid_size) * grid_size
            dev = abs(offset - nearest_grid_offset)
            
            deviations.append(dev)
            if dev <= tolerance:
                on_beat += 1

        on_beat_ratio = on_beat / len(word_starts)
        avg_dev_ms = (sum(deviations) / len(deviations)) * 1000.0

        # Piecewise scoring curve
        r = on_beat_ratio
        if r < 0.10:
            score = (r / 0.10) * 20.0
        elif r < 0.25:
            score = 20.0 + ((r - 0.10) / 0.15) * 20.0
        elif r < 0.40:
            score = 40.0 + ((r - 0.25) / 0.15) * 20.0
        elif r < 0.60:
            score = 60.0 + ((r - 0.40) / 0.20) * 20.0
        elif r < 0.80:
            score = 80.0 + ((r - 0.60) / 0.20) * 12.0
        else:
            score = min(92.0 + (r - 0.80) * 40.0, 100.0)

        metadata = {
            "tempo_bpm": round(tempo, 1),
            "on_beat_ratio": round(on_beat_ratio, 4),
            "avg_deviation_ms": round(avg_dev_ms, 1),
            "words_analyzed": len(word_starts),
        }
        logger.info(
            "Beat sync: tempo=%.1f BPM, on_beat=%.1f%%, score=%.1f",
            tempo, on_beat_ratio * 100, score,
        )
        return round(score, 2), metadata

    except Exception as exc:
        logger.exception("Beat sync analysis failed: %s", exc)
        return 0.0, {"error": str(exc)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning("Could not remove temp file %s: %s", tmp_path, e)
