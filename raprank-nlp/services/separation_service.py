import logging
import os
import shutil
import tempfile

from config import scoring_config

logger = logging.getLogger(__name__)

_DEMUCS_AVAILABLE = False
try:
    import config.ffmpeg_patch
    import demucs.separate
    _DEMUCS_AVAILABLE = True
except (ImportError, Exception) as exc:
    logger.warning("demucs not available or DLL failed (%s). Audio source separation will be skipped (falling back to mixed audio).", exc)


def separate_vocals(audio_bytes: bytes, filename: str) -> tuple[tuple[bytes, bytes] | None, str | None]:
    """
    Separate vocals and accompaniment using Demucs.

    Returns (result, skip_reason):
      - On success: ((vocals_bytes, accompaniment_bytes), None)
      - On skip/failure: (None, "<human-readable reason>") -- callers should
        surface this reason (e.g. via FlowMetadata) rather than silently
        falling back to mixed audio with no visibility into why.
    """
    if not _DEMUCS_AVAILABLE:
        reason = "demucs not installed in this environment"
        logger.info(reason)
        return None, reason

    cfg = scoring_config.AUDIO_PIPELINE
    suffix = os.path.splitext(filename)[1].lower() or ".mp3"
    tmp_dir = tempfile.mkdtemp(prefix="demucs_sep_")
    input_path = os.path.join(tmp_dir, f"input{suffix}")

    try:
        # Write original audio to temp file
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        # Check audio duration to avoid OOM/timeout on CPU
        try:
            import librosa
            duration = librosa.get_duration(path=input_path)
            logger.info("Audio duration: %.2f seconds", duration)
            max_duration = cfg["DEMUCS_MAX_DURATION_S"]
            if duration > max_duration:
                reason = (
                    f"audio duration ({duration:.0f}s) exceeds the "
                    f"{max_duration:.0f}s CPU-separation threshold"
                )
                logger.warning("%s. Skipping Demucs separation, falling back to mixed audio.", reason)
                return None, reason
        except Exception as dur_err:
            logger.warning("Could not determine audio duration: %s. Using size-based fallback.", dur_err)
            max_mb = cfg["DEMUCS_MAX_SIZE_MB"]
            size_mb = len(audio_bytes) / 1024 / 1024
            if size_mb > max_mb:
                reason = f"audio file size ({size_mb:.1f}MB) exceeds the {max_mb:.0f}MB fallback threshold"
                logger.warning("%s. Skipping Demucs separation.", reason)
                return None, reason

        logger.info("Starting audio source separation using Demucs...")
        
        # We run demucs programmatically.
        # Options:
        #   --two-stems vocals: split into vocals and no_vocals
        #   -d cpu: force cpu device
        #   -o: output directory
        args = [
            "--two-stems", "vocals",
            "-d", "cpu",
            "-o", tmp_dir,
            input_path
        ]
        
        try:
            demucs.separate.main(args)
        except SystemExit as e:
            if e.code != 0:
                raise RuntimeError(f"Demucs process exited with code {e.code}")

        # The output files are saved under:
        # tmp_dir/htdemucs/input/vocals.wav
        # tmp_dir/htdemucs/input/no_vocals.wav (or whichever model name)
        
        model_dirs = [d for d in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, d)) and d != "input"]
        if not model_dirs:
            raise FileNotFoundError("Demucs output directory not found")
        
        model_name = model_dirs[0]
        output_folder = os.path.join(tmp_dir, model_name, "input")
        
        vocals_path = os.path.join(output_folder, "vocals.wav")
        no_vocals_path = os.path.join(output_folder, "no_vocals.wav")

        if not os.path.exists(vocals_path) or not os.path.exists(no_vocals_path):
            raise FileNotFoundError("Demucs vocals or no_vocals output files not found")

        with open(vocals_path, "rb") as f:
            vocals_bytes = f.read()

        with open(no_vocals_path, "rb") as f:
            accompaniment_bytes = f.read()

        logger.info("Audio source separation completed successfully.")
        return (vocals_bytes, accompaniment_bytes), None

    except Exception as e:
        logger.exception("Audio source separation failed: %s", e)
        return None, f"separation failed: {e}"

    finally:
        # Cleanup temp directory
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning("Could not delete temp directory %s: %s", tmp_dir, e)
