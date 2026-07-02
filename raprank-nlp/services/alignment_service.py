"""
Alignment Service — Syllable and word forced alignment using Wav2Vec2.
Refines Whisper's rough word timestamps to millisecond-accurate physical onsets.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import numpy as np

logger = logging.getLogger(__name__)

# Try loading torch and torchaudio
_TORCH_OK = False
try:
    import config.ffmpeg_patch
    import torch
    import torchaudio
    _TORCH_OK = True
except ImportError:

    logger.warning("Torch/TorchAudio not available. Forced alignment will fall back to spectral onset detection.")

# Load pre-trained Wav2Vec2 model at module level if available
_MODEL = None
_BUNDLE = None
_LABELS_MAP = None

if _TORCH_OK:
    try:
        # Using ASR Base model (lightweight, highly accurate for character classification)
        _BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        _MODEL = _BUNDLE.get_model()
        _MODEL.eval()
        _LABELS_MAP = {c: i for i, c in enumerate(_BUNDLE.get_labels())}
        logger.info("Wav2Vec2 forced alignment model loaded successfully.")
    except Exception as exc:
        logger.warning("Failed to load Wav2Vec2 model: %s. Using fallback onset detectors.", exc)


def refine_word_onsets(
    vocals_bytes: bytes,
    filename: str,
    rough_words: list[dict]
) -> list[tuple[float, float]]:
    """
    Takes isolated vocal bytes and Whisper's rough word timestamps.
    Returns a list of tuples (refined_onset, probability).
    """
    if not rough_words:
        return []

    # Fallback to spectral onset detection if models are not available
    if not _TORCH_OK or _MODEL is None or _BUNDLE is None:
        spectral_onsets = get_fallback_spectral_onsets(vocals_bytes, filename)
        return [(t, 1.0) for t in spectral_onsets]

    # Fallback if track is too long / has too many words to prevent event-loop freezing on CPU
    if len(rough_words) > 150:
        logger.warning(
            "Track has %d words (threshold: 150). Skipping CPU-heavy Wav2Vec2 forced alignment. Falling back to spectral onset detection.",
            len(rough_words),
        )
        spectral_onsets = get_fallback_spectral_onsets(vocals_bytes, filename)
        return [(t, 1.0) for t in spectral_onsets]

    suffix = os.path.splitext(filename)[1].lower() or ".mp3"
    tmp_dir = tempfile.mkdtemp(prefix="align_temp_")
    audio_path = os.path.join(tmp_dir, f"vocals{suffix}")

    try:
        # Save vocals to temp file
        with open(audio_path, "wb") as f:
            f.write(vocals_bytes)

        # Load vocal waveform
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample to Wav2Vec2 target rate (16000 Hz)
        target_sr = _BUNDLE.sample_rate
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        # Force mono channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        refined_onsets = []

        # We process each word using local slicing to prevent trellis memory overhead
        for w in rough_words:
            start = w.get("start")
            end = w.get("end")
            word_text = w.get("word", "").strip()
            prob = w.get("probability", 1.0)

            if start is None or end is None or not word_text:
                continue

            # Standardize Hinglish/Hindi text to Roman uppercase matching the Wav2Vec2 vocabulary
            cleaned_word = "".join(c.upper() for c in word_text if c.isalnum() or c == "'")
            if not cleaned_word:
                refined_onsets.append((start, prob))
                continue

            # Slice window: pad slightly around rough timestamps
            pad = 0.25
            slice_start_s = max(start - pad, 0.0)
            slice_end_s = end + pad
            
            frame_start = int(slice_start_s * sr)
            frame_end = int(slice_end_s * sr)
            
            waveform_slice = waveform[:, frame_start:frame_end]
            if waveform_slice.shape[1] < 1000:  # too short
                refined_onsets.append((start, prob))
                continue

            try:
                # Generate Wav2Vec2 emission probabilities for this audio slice
                with torch.inference_mode():
                    emissions, _ = _MODEL(waveform_slice)
                    # emissions shape: [batch=1, frames, classes]
                    emissions = torch.log_softmax(emissions, dim=-1)
                
                emission_slice = emissions[0].cpu().numpy()
                
                # Get target index of the word's starting phoneme/letter
                first_char = cleaned_word[0]
                char_idx = _LABELS_MAP.get(first_char)

                if char_idx is None:
                    refined_onsets.append((start, prob))
                    continue

                # Locate frame in slice with maximum probability for the starting letter
                # Wav2Vec2 downsamples audio by 320x (16000 Hz / 320 = 50 frames per second, i.e., 20ms per frame)
                slice_frames = emission_slice.shape[0]
                if slice_frames > 0:
                    max_frame = int(np.argmax(emission_slice[:, char_idx]))
                    # Convert slice frame back to global timeline seconds
                    refined_onset = slice_start_s + (max_frame * 320.0 / sr)
                    
                    # Sanity check: keep refinement within reasonable bounds
                    if abs(refined_onset - start) < 0.4:
                        refined_onsets.append((refined_onset, prob))
                    else:
                        refined_onsets.append((start, prob))
                else:
                    refined_onsets.append((start, prob))

            except Exception as e:
                logger.debug("Wav2Vec2 word slice alignment failed, using fallback: %s", e)
                refined_onsets.append((start, prob))

        logger.info("Wav2Vec2 refined %d word onsets.", len(refined_onsets))
        return refined_onsets

    except Exception as e:
        logger.exception("Forced alignment pipeline failed: %s. Falling back to spectral onsets.", e)
        spectral_onsets = get_fallback_spectral_onsets(vocals_bytes, filename)
        return [(t, 1.0) for t in spectral_onsets]

    finally:
        # Cleanup
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning("Could not delete alignment temp directory: %s", e)


def get_fallback_spectral_onsets(vocals_bytes: bytes, filename: str) -> list[float]:
    """
    Fallback onset detector using Librosa spectral flux peaks.
    Used when torch/wav2vec2 is not present or throws exceptions.
    """
    try:
        import librosa
        suffix = os.path.splitext(filename)[1].lower() or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(vocals_bytes)
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            # Find spectral onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # Peak pick onset times
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
            logger.info("Librosa fallback detected %d vocal onsets.", len(onset_times))
            return onset_times
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        logger.warning("Librosa fallback onset detector failed: %s", e)
        return []


def align_structured_lyrics_to_whisper(
    lyrics: str,
    whisper_words: list[dict]
) -> list[dict]:
    """
    Performs dynamic programming global sequence alignment to map Whisper's
    word timestamps onto the user's ground-truth structured lyrics.
    Optimised using band DP (Sakoe-Chiba band), precomputed phonetic keys, and similarity caching.
    """
    from services.language_utils import clean_word, devanagari_to_roman, is_hindi_word
    import difflib

    # 1. Parse structured lyrics into words and precompute normalized comparison keys
    structured_words = []
    lines = lyrics.strip().split("\n")
    word_id = 0
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line or (line.startswith("[") and line.endswith("]")):
            continue
        for word in line.split():
            cleaned = clean_word(word)
            if cleaned:
                # Precompute romanized and normalized form for comparison
                rom = devanagari_to_roman(cleaned) if is_hindi_word(cleaned) else cleaned
                rom_norm = rom.lower().replace("aa", "a").replace("ee", "i").replace("oo", "u")
                structured_words.append({
                    "id": word_id,
                    "word": word,
                    "cleaned": cleaned.lower(),
                    "rom_norm": rom_norm,
                    "line_idx": line_idx,
                    "start": None,
                    "end": None,
                    "probability": None
                })
                word_id += 1

    if not structured_words or not whisper_words:
        return []

    # 2. Parse whisper words and precompute comparison keys
    cleaned_whisper = []
    for idx, w in enumerate(whisper_words):
        word_text = w.get("word", "").strip()
        cleaned = clean_word(word_text)
        rom = devanagari_to_roman(cleaned) if is_hindi_word(cleaned) else cleaned
        rom_norm = rom.lower().replace("aa", "a").replace("ee", "i").replace("oo", "u")
        cleaned_whisper.append({
            "id": idx,
            "word": word_text,
            "cleaned": cleaned.lower(),
            "rom_norm": rom_norm,
            "start": w.get("start"),
            "end": w.get("end"),
            "probability": w.get("probability", 1.0)
        })

    # Similarity helper with caching
    similarity_cache = {}
    def word_similarity(idx1: int, idx2: int) -> float:
        r1 = structured_words[idx1]["rom_norm"]
        r2 = cleaned_whisper[idx2]["rom_norm"]
        if r1 == r2:
            return 1.0
        pair = (r1, r2)
        if pair not in similarity_cache:
            similarity_cache[pair] = difflib.SequenceMatcher(None, r1, r2).ratio()
        return similarity_cache[pair]

    # 3. Band Dynamic Programming Alignment (Sakoe-Chiba band)
    n = len(structured_words)
    m = len(cleaned_whisper)
    
    # Initialize DP table with very low value for out-of-band cells
    dp = np.full((n + 1, m + 1), -1e9)
    tb = np.zeros((n + 1, m + 1), dtype=int)  # 0: Match, 1: Skip S, 2: Skip W
    
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = -i * 0.5
        tb[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = -j * 0.5
        tb[0][j] = 2

    # Band size (how far search can wander from expected diagonal)
    band_size = 100
    ratio = m / n if n > 0 else 1.0

    for i in range(1, n + 1):
        center = int(i * ratio)
        start_j = max(1, center - band_size)
        end_j = min(m, center + band_size)
        
        for j in range(start_j, end_j + 1):
            sim = word_similarity(i - 1, j - 1)
            match_score = dp[i-1][j-1] + (1.5 * sim - 0.2)
            skip_s = dp[i-1][j] - 0.5
            skip_w = dp[i][j-1] - 0.5
            
            best = max(match_score, skip_s, skip_w)
            dp[i][j] = best
            
            if best == match_score:
                tb[i][j] = 0
            elif best == skip_s:
                tb[i][j] = 1
            else:
                tb[i][j] = 2

    # Traceback
    i, j = n, m
    matches = {}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and tb[i][j] == 0:
            matches[i-1] = j-1
            i -= 1
            j -= 1
        elif i > 0 and tb[i][j] == 1:
            i -= 1
        else:
            j -= 1

    # Map timestamps and probabilities
    for s_idx, w_idx in matches.items():
        structured_words[s_idx]["start"] = cleaned_whisper[w_idx]["start"]
        structured_words[s_idx]["end"] = cleaned_whisper[w_idx]["end"]
        structured_words[s_idx]["probability"] = cleaned_whisper[w_idx]["probability"]

    # 4. Fill missing timestamps via interpolation
    last_known_start = 0.0
    last_known_end = 0.0
    
    # Forward pass: fill leading missing values with first match, or interpolate
    for k in range(n):
        if structured_words[k]["start"] is not None:
            last_known_start = structured_words[k]["start"]
            last_known_end = structured_words[k]["end"]
        else:
            # Find next match for interpolation
            next_idx = -1
            for l in range(k + 1, n):
                if structured_words[l]["start"] is not None:
                    next_idx = l
                    break
            
            if next_idx != -1:
                next_start = structured_words[next_idx]["start"]
                next_end = structured_words[next_idx]["end"]
                # Interpolate
                steps = next_idx - (k - 1)
                ratio = 1.0 / steps
                structured_words[k]["start"] = last_known_start + ratio * (next_start - last_known_start)
                structured_words[k]["end"] = last_known_end + ratio * (next_end - last_known_end)
                structured_words[k]["probability"] = 0.5
            else:
                structured_words[k]["start"] = last_known_start
                structured_words[k]["end"] = last_known_end
                structured_words[k]["probability"] = 0.5

            last_known_start = structured_words[k]["start"]
            last_known_end = structured_words[k]["end"]

    return structured_words
