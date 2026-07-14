"""
Shared audio-decode helper.

Before this existed, every audio-analysis step (MIR beat tracking, Wav2Vec2/
spectral onset refinement) independently wrote the same raw bytes to its own
temp file and called librosa.load() separately -- duplicated disk I/O and
codec decoding, most wastefully when vocals_bytes and accompaniment_bytes are
literally the same underlying bytes (Demucs separation skipped/unavailable).
The audio equivalent of decoding a video into frames once instead of
re-parsing the container per consumer.

Decode once at the file's native sample rate; each consumer resamples the
in-memory array to whatever rate it needs (e.g. Wav2Vec2's 16000 Hz) via a
cheap in-memory resample rather than re-reading/re-decoding from bytes.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

import librosa
import numpy as np

logger = logging.getLogger(__name__)

DecodedAudio = tuple[np.ndarray, int]


def decode_audio_bytes(audio_bytes: bytes, filename: str) -> DecodedAudio | None:
    """
    Decode raw audio bytes into a mono waveform at the file's native sample
    rate. Returns (waveform, sample_rate), or None on empty input/failure.
    """
    if not audio_bytes:
        return None

    suffix = os.path.splitext(filename)[1].lower() or ".mp3"
    tmp_dir = tempfile.mkdtemp(prefix="audio_decode_")
    tmp_path = os.path.join(tmp_dir, f"audio{suffix}")

    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        return y, sr
    except Exception as e:
        logger.exception("Failed to decode audio bytes: %s", e)
        return None
    finally:
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning("Could not delete audio decode temp dir: %s", e)


def resample(decoded: DecodedAudio, target_sr: int) -> DecodedAudio:
    """Cheap in-memory resample of an already-decoded waveform -- no disk I/O,
    unlike decoding from bytes again at a different rate."""
    y, sr = decoded
    if sr == target_sr:
        return y, sr
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr), target_sr
