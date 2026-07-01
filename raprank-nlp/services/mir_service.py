"""
MIR Service — Hybrid beat-tracking and tempo extractor.
Uses Essentia if available; falls back to Librosa and NumPy.
"""
from __future__ import annotations

import logging
import os
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Try importing Essentia
_ESSENTIA_AVAILABLE = False
try:
    import essentia
    import essentia.standard as es
    _ESSENTIA_AVAILABLE = True
    logger.info("Essentia library is successfully loaded.")
except ImportError:
    logger.info("Essentia is not available. Using Librosa/NumPy hybrid backend for MIR.")

# Try importing Librosa
_LIBROSA_AVAILABLE = False
try:
    import librosa
    import numpy as np
    _LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("Librosa is not available. Rhythmic tracking is limited.")


def detect_beats_and_tempo(accompaniment_bytes: bytes, filename: str) -> tuple[float, list[float]]:
    """
    Given accompaniment audio bytes and filename, returns (tempo_bpm, beat_times_seconds).
    """
    if not accompaniment_bytes:
        logger.warning("Empty accompaniment bytes passed to MIR.")
        return 120.0, []

    suffix = os.path.splitext(filename)[1].lower() or ".mp3"
    tmp_dir = tempfile.mkdtemp(prefix="mir_temp_")
    audio_path = os.path.join(tmp_dir, f"accompaniment{suffix}")

    try:
        # Write bytes to temp file
        with open(audio_path, "wb") as f:
            f.write(accompaniment_bytes)

        if _ESSENTIA_AVAILABLE:
            try:
                logger.info("Using Essentia standard RhythmExtractor2013 for beat tracking...")
                # Load audio
                loader = es.MonoLoader(filename=audio_path)
                audio = loader()

                # Rhythm extractor
                rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
                bpm, ticks, estimates, bpm_intervals = rhythm_extractor(audio)
                
                # Convert ticks (beat times in seconds) to list
                beat_times = [float(tick) for tick in ticks]
                logger.info("Essentia detected tempo: %.1f BPM, %d beats", bpm, len(beat_times))
                return float(bpm), beat_times
            except Exception as e:
                logger.exception("Essentia beat tracking failed, falling back to Librosa: %s", e)

        if _LIBROSA_AVAILABLE:
            logger.info("Using Librosa beat_track for beat detection...")
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Run beat track
            tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(np.atleast_1d(tempo_arr)[0])
            
            if beat_frames.size == 0:
                logger.warning("No beats detected in audio via Librosa.")
                return tempo, []

            beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
            logger.info("Librosa detected tempo: %.1f BPM, %d beats", tempo, len(beat_times))
            return tempo, beat_times

        # Ultimate fallback
        logger.warning("No MIR audio backends (Essentia or Librosa) are available!")
        return 120.0, []

    except Exception as e:
        logger.exception("Error during MIR beat and tempo tracking: %s", e)
        return 120.0, []

    finally:
        # Clean up temp files
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning("Could not delete MIR temp directory: %s", e)
