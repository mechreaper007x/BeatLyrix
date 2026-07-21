"""
Alignment Service — Syllable and word forced alignment using Wav2Vec2-CTC.
Refines Whisper's rough word timestamps to millisecond-accurate physical onsets.

Model choice: WAV2VEC2_ASR_BASE_960H (torchaudio's bundled English/LibriSpeech
model) was previously wired up here but never actually installed in
requirements.txt, so this whole path was dead code -- every request silently
used the cruder spectral-onset fallback below. Also, that specific model has
no Devanagari support in its vocabulary at all, so even once installed it
would only ever refine English words, not the Hindi/Hinglish majority of this
corpus. Replaced with `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`, a
Hindi CTC model (16kHz, char-level Devanagari vocabulary) loaded via
`transformers` instead of torchaudio's bundle system (a custom HF checkpoint
isn't packaged as a torchaudio pipeline bundle).

Caveat this model choice creates: its vocabulary is 62/67 Devanagari tokens
and ZERO Latin tokens -- it cannot refine Romanized/Hinglish words at all
(and this app's "hinglish" Whisper mode deliberately produces Romanized
output). So refinement here is script-aware: Devanagari words go through the
Hindi CTC model; Latin-script words fall back to a local spectral-onset snap
(the same signal-processing idea, just without a matching neural model for
that script) rather than being silently passed through unchanged.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import numpy as np

from services.audio_utils import DecodedAudio, decode_audio_bytes, resample
from services.language_utils import is_devanagari_char

logger = logging.getLogger(__name__)

_MODEL_ID = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"

# Try loading torch/torchaudio/transformers
_TORCH_OK = False
try:
    import config.ffmpeg_patch
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    _TORCH_OK = True
except (ImportError, Exception) as exc:
    logger.warning("Torch/torchaudio/transformers not available or DLL failed (%s). Forced alignment will fall back to spectral onset detection.", exc)

# Load pre-trained Hindi Wav2Vec2-CTC model at module level if available
_MODEL = None
_PROCESSOR = None
_LABELS_MAP: dict[str, int] | None = None
_TARGET_SR = 16000

if _TORCH_OK:
    try:
        _PROCESSOR = Wav2Vec2Processor.from_pretrained(_MODEL_ID)
        _MODEL = Wav2Vec2ForCTC.from_pretrained(_MODEL_ID)
        _MODEL.eval()
        _TARGET_SR = _PROCESSOR.feature_extractor.sampling_rate
        _LABELS_MAP = {c: i for c, i in _PROCESSOR.tokenizer.get_vocab().items()}
        logger.info("Hindi Wav2Vec2-CTC forced alignment model loaded successfully (%s).", _MODEL_ID)
    except Exception as exc:
        logger.warning("Failed to load Hindi Wav2Vec2-CTC model: %s. Using fallback onset detectors.", exc)
        _MODEL = None


def refine_word_onsets(
    vocals_bytes: bytes,
    filename: str,
    rough_words: list[dict],
    decoded: DecodedAudio | None = None,
) -> list[tuple[float, float]]:
    """
    Takes isolated vocal bytes (or an already-decoded waveform) and Whisper's
    rough word timestamps. Returns a list of tuples (refined_onset, probability).

    `decoded`, if provided, must be the native-sample-rate decode of
    `vocals_bytes` (see services/audio_utils.py) -- avoids re-decoding bytes
    that a caller (e.g. flow_service) already decoded once.
    """
    if not rough_words:
        return []

    # Fallback to spectral onset detection if the Hindi model isn't available
    if not _TORCH_OK or _MODEL is None:
        spectral_onsets = get_fallback_spectral_onsets(vocals_bytes, filename, decoded=decoded)
        return [(t, 1.0) for t in spectral_onsets]

    # Fallback if track is too long / has too many words to prevent event-loop freezing on CPU
    if len(rough_words) > 150:
        logger.warning(
            "Track has %d words (threshold: 150). Skipping CPU-heavy Wav2Vec2 forced alignment. Falling back to spectral onset detection.",
            len(rough_words),
        )
        spectral_onsets = get_fallback_spectral_onsets(vocals_bytes, filename, decoded=decoded)
        return [(t, 1.0) for t in spectral_onsets]

    resolved = decoded if decoded is not None else decode_audio_bytes(vocals_bytes, filename)
    if resolved is None:
        return []
    y, sr = resample(resolved, _TARGET_SR)
    waveform = torch.from_numpy(y).unsqueeze(0)  # [1, samples], mono

    refined_onsets: list[tuple[float, float]] = []

    # We process each word using local slicing to prevent trellis memory overhead
    for w in rough_words:
        start = w.get("start")
        end = w.get("end")
        word_text = w.get("word", "").strip()
        prob = w.get("probability", 1.0)

        if start is None or end is None or not word_text:
            continue

        pad = 0.25
        slice_start_s = max(start - pad, 0.0)
        slice_end_s = end + pad
        frame_start = int(slice_start_s * sr)
        frame_end = int(slice_end_s * sr)
        waveform_slice = waveform[:, frame_start:frame_end]

        if waveform_slice.shape[1] < 1000:  # too short
            refined_onsets.append((start, prob))
            continue

        # Script-aware routing: the Hindi CTC model's vocabulary is
        # Devanagari-only (see module docstring) -- it cannot meaningfully
        # refine Latin-script/Romanized words, so those go through the
        # spectral-onset snap instead of a doomed vocab lookup.
        if not any(is_devanagari_char(c) for c in word_text):
            snapped = _snap_to_local_spectral_onset(y, sr, slice_start_s, slice_end_s, start)
            refined_onsets.append((snapped, prob))
            continue

        try:
            refined_onset = _refine_devanagari_word_onset(
                waveform_slice, slice_start_s, sr, word_text, start
            )
            refined_onsets.append((refined_onset, prob))
        except Exception as e:
            logger.debug("Wav2Vec2 word slice alignment failed, using fallback: %s", e)
            refined_onsets.append((start, prob))

    logger.info("Wav2Vec2 refined %d word onsets.", len(refined_onsets))
    return refined_onsets


def _refine_devanagari_word_onset(
    waveform_slice, slice_start_s: float, sr: int, word_text: str, rough_start: float
) -> float:
    """Locate the acoustic onset of *word_text*'s first Devanagari character
    within its padded slice via Wav2Vec2-CTC emissions. Returns rough_start
    unchanged if the character isn't in the model's vocabulary or the
    refinement lands implausibly far from Whisper's rough guess."""
    first_char = next((c for c in word_text if is_devanagari_char(c)), None)
    char_idx = _LABELS_MAP.get(first_char) if first_char else None
    if char_idx is None:
        return rough_start

    with torch.inference_mode():
        logits = _MODEL(waveform_slice).logits
        emissions = torch.log_softmax(logits, dim=-1)

    emission_slice = emissions[0].cpu().numpy()
    slice_frames = emission_slice.shape[0]
    if slice_frames == 0:
        return rough_start

    # Wav2Vec2-CTC's frame stride: model total downsampling factor.
    stride_samples = waveform_slice.shape[1] / slice_frames
    max_frame = int(np.argmax(emission_slice[:, char_idx]))
    refined_onset = slice_start_s + (max_frame * stride_samples / sr)

    # Sanity check: keep refinement within reasonable bounds
    if abs(refined_onset - rough_start) < 0.4:
        return refined_onset
    return rough_start


def _snap_to_local_spectral_onset(
    y: np.ndarray, sr: int, slice_start_s: float, slice_end_s: float, rough_start: float
) -> float:
    """For words the Hindi CTC model's vocabulary can't cover (Latin/Romanized
    script), snap Whisper's rough onset to the nearest real spectral-flux
    onset peak within this word's own local time window -- still a genuine
    acoustic refinement, just without a matching neural model for the script."""
    try:
        import librosa
        frame_start = max(int(slice_start_s * sr), 0)
        frame_end = min(int(slice_end_s * sr), len(y))
        y_slice = y[frame_start:frame_end]
        if len(y_slice) < 512:
            return rough_start

        onset_env = librosa.onset.onset_strength(y=y_slice, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        if len(onset_frames) == 0:
            return rough_start

        onset_times_local = librosa.frames_to_time(onset_frames, sr=sr)
        onset_times_global = onset_times_local + slice_start_s
        closest = min(onset_times_global, key=lambda t: abs(t - rough_start))

        if abs(closest - rough_start) < 0.4:
            return float(closest)
        return rough_start
    except Exception as e:
        logger.debug("Local spectral onset snap failed, using rough timestamp: %s", e)
        return rough_start


def get_fallback_spectral_onsets(
    vocals_bytes: bytes, filename: str, decoded: DecodedAudio | None = None
) -> list[float]:
    """
    Fallback onset detector using Librosa spectral flux peaks over the whole
    track. Used when the Hindi Wav2Vec2-CTC model is not present/fails
    entirely (not for the script-aware per-word case above, which has its
    own local variant: `_snap_to_local_spectral_onset`).
    """
    try:
        import librosa
        resolved = decoded if decoded is not None else decode_audio_bytes(vocals_bytes, filename)
        if resolved is None:
            return []
        y, sr = resolved

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
        logger.info("Librosa fallback detected %d vocal onsets.", len(onset_times))
        return onset_times
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
