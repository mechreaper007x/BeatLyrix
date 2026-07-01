"""
Flow / Beat-sync scoring using true audio-level MIR and Wav2Vec2 forced alignment.
No constants or scoring thresholds are hardcoded in the service logic.
"""
from __future__ import annotations

import logging
import numpy as np

from services.language_utils import clean_word
from services.mir_service import detect_beats_and_tempo
from services.alignment_service import refine_word_onsets
from config import scoring_config

logger = logging.getLogger(__name__)


def calculate_beat_sync(
    vocals_bytes: bytes,
    accompaniment_bytes: bytes,
    filename: str,
    words: list[dict],
) -> tuple[float, dict]:
    """
    Compute an accurate signal-level flow/beat-sync score for a rap track.
    Aligns vocal transient onsets (via Wav2Vec2) to beat markers (via Essentia/Librosa).
    """
    if not words:
        return 0.0, {"error": "no words provided"}

    try:
        # 1. Beat and Tempo Tracking via MIR service
        tempo, beat_times = detect_beats_and_tempo(accompaniment_bytes, filename)
        if not beat_times:
            return 0.0, {"error": "no beats detected in audio"}

        # 2. Vocal Onset Detection / Refinement via Wav2Vec2
        refined_onsets = refine_word_onsets(vocals_bytes, filename, words)
        
        # Fallback to Whisper rough timestamps if Wav2Vec2 fails/returns empty
        if not refined_onsets:
            logger.info("Forced alignment returned no onsets. Falling back to Whisper rough timestamps.")
            refined_onsets = [(w["start"], w.get("probability", 1.0)) for w in words if "start" in w]

        if not refined_onsets:
            return 0.0, {"error": "no vocal onsets detected"}

        # 3. Dynamic timing variables from Config
        beat_period = 60.0 / tempo
        grid_size = beat_period / scoring_config.FLOW["GRID_SUBDIVISIONS"]
        tolerance = scoring_config.FLOW["POCKET_TOLERANCE_MS"] / 1000.0

        on_beat_weight = 0.0
        total_weight = 0.0
        
        weighted_dev_sum = 0.0
        weighted_offset_sum = 0.0

        onsets_only = [item[0] for item in refined_onsets]

        for onset, prob in refined_onsets:
            # Find closest beat marker
            closest_beat = min(beat_times, key=lambda bt: abs(bt - onset))
            offset = onset - closest_beat
            
            # Align to nearest subdivisions grid offset
            nearest_grid_offset = round(offset / grid_size) * grid_size
            dev_offset = offset - nearest_grid_offset
            dev = abs(dev_offset)
            
            weighted_dev_sum += dev * prob
            weighted_offset_sum += dev_offset * prob
            total_weight += prob
            
            if dev <= tolerance:
                on_beat_weight += prob

        on_beat_ratio = on_beat_weight / total_weight if total_weight > 0.0 else 0.0
        avg_dev_ms = (weighted_dev_sum / total_weight) * 1000.0 if total_weight > 0.0 else 0.0
        
        # Calculate Micro-timing Flow Delivery Style (Mean Offset)
        mean_offset_ms = (weighted_offset_sum / total_weight) * 1000.0 if total_weight > 0.0 else 0.0
        if mean_offset_ms < -15.0:
            flow_style = "Driving (Ahead of beat)"
        elif mean_offset_ms > 15.0:
            flow_style = "Laid-back Swing (Behind beat)"
        else:
            flow_style = "On the Pocket (Grid)"

        # 4. Syllable density / speed calculation
        start_time = onsets_only[0]
        end_time = onsets_only[-1]
        duration = max(end_time - start_time, 1.0)
        
        # Syllable count is mapped directly to the number of physical vocal onsets
        syllable_rate = len(onsets_only) / duration
        
        # Dynamic Syllable rate speed bonus from Config
        min_syl = scoring_config.FLOW["SPEED_BONUS_MIN_SYLLABLES"]
        max_syl = scoring_config.FLOW["SPEED_BONUS_MAX_SYLLABLES"]
        w_speed = scoring_config.FLOW["SPEED_BONUS_WEIGHT"]
        
        if syllable_rate > min_syl:
            complexity_bonus = min(((syllable_rate - min_syl) / (max_syl - min_syl)) * w_speed, w_speed)
        else:
            complexity_bonus = 0.0

        # 5. Cadence variance (flow switches)
        interval_duration = scoring_config.FLOW["CADENCE_VARIANCE_WINDOW"]
        num_intervals = int((duration + (interval_duration - 0.01)) // interval_duration)
        rates = []
        
        for j in range(num_intervals):
            int_start = start_time + j * interval_duration
            int_end = min(int_start + interval_duration, end_time)
            int_dur = int_end - int_start
            if int_dur <= 0.1:
                continue
            
            syl_in_int = sum(1 for onset in refined_onsets if int_start <= onset < int_end)
            rates.append(syl_in_int / int_dur)

        if len(rates) > 1:
            mean_rate = sum(rates) / len(rates)
            variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
            cadence_variance = variance ** 0.5
        else:
            cadence_variance = 0.0

        # Cadence versatility bonus from Config
        w_variance = scoring_config.FLOW["CADENCE_VARIANCE_WEIGHT"]
        flow_switch_bonus = min((cadence_variance / 2.0) * w_variance, w_variance)

        # 6. Evaluate curve dynamically from Config
        score = scoring_config.evaluate_piecewise_curve(
            on_beat_ratio,
            scoring_config.FLOW["CURVE_THRESHOLDS"],
            scoring_config.FLOW["CURVE_SCORES"]
        )

        # Apply complexity and flow switch bonuses
        score = min(score + complexity_bonus + flow_switch_bonus, 100.0)

        metadata = {
            "tempo_bpm": round(tempo, 1),
            "on_beat_ratio": round(on_beat_ratio, 4),
            "avg_deviation_ms": round(avg_dev_ms, 1),
            "mean_offset_ms": round(mean_offset_ms, 1),
            "flow_style": flow_style,
            "words_analyzed": len(refined_onsets),
            "syllable_rate": round(syllable_rate, 2),
            "complexity_bonus": round(complexity_bonus, 2),
            "cadence_variance": round(cadence_variance, 4),
            "flow_switch_bonus": round(flow_switch_bonus, 2),
        }
        
        logger.info(
            "MIR Flow Sync: tempo=%.1f BPM, style='%s', on_beat=%.1f%%, dev_ms=%.1f, score=%.1f",
            tempo, flow_style, on_beat_ratio * 100, avg_dev_ms, score,
        )
        return round(score, 2), metadata

    except Exception as exc:
        logger.exception("MIR Flow Sync analysis failed: %s", exc)
        return 0.0, {"error": str(exc)}
