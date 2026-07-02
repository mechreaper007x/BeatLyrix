import os
import shutil
import glob
import logging

logger = logging.getLogger(__name__)

def patch_ffmpeg_path():
    # If ffmpeg is already in the system path, do nothing
    if shutil.which("ffmpeg"):
        return

    # Check the WinGet user packages directory for Gyan.FFmpeg
    local_appdata = os.getenv("LOCALAPPDATA")
    if not local_appdata:
        return

    # Try searching for Gyan.FFmpeg or BtbN.FFmpeg packages in user WinGet folder
    search_pattern = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages", "*FFmpeg*", "**", "ffmpeg.exe")
    matches = glob.glob(search_pattern, recursive=True)
    if matches:
        # Find the bin directory
        bin_dir = os.path.dirname(matches[0])
        # Prepend to PATH so it takes priority
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
        logger.info(f"Dynamically added FFmpeg to PATH: {bin_dir}")

# Apply the patch immediately upon import
patch_ffmpeg_path()
