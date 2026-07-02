import logging
import os
import shutil
import tempfile

logger = logging.getLogger(__name__)

_DEMUCS_AVAILABLE = False
try:
    import config.ffmpeg_patch
    import demucs.separate
    _DEMUCS_AVAILABLE = True
except ImportError:
    logger.warning("demucs not installed. Audio source separation will be skipped (falling back to mixed audio).")


def separate_vocals(audio_bytes: bytes, filename: str) -> tuple[bytes, bytes] | None:
    """
    Separate vocals and accompaniment using Demucs.
    Returns (vocals_bytes, accompaniment_bytes) or None on failure/unavailability.
    """
    if not _DEMUCS_AVAILABLE:
        logger.info("Demucs is not available in the current environment.")
        return None

    suffix = os.path.splitext(filename)[1].lower() or ".mp3"
    tmp_dir = tempfile.mkdtemp(prefix="demucs_sep_")
    input_path = os.path.join(tmp_dir, f"input{suffix}")

    try:
        # Write original audio to temp file
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

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
        return vocals_bytes, accompaniment_bytes

    except Exception as e:
        logger.exception("Audio source separation failed: %s", e)
        return None

    finally:
        # Cleanup temp directory
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning("Could not delete temp directory %s: %s", tmp_dir, e)
