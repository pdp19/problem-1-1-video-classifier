from __future__ import annotations

import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def ensure() -> str:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return "system"
    try:
        import static_ffmpeg
    except ImportError as exc:
        raise RuntimeError("ffmpeg/ffprobe not found; install ffmpeg or `pip install static-ffmpeg`") from exc
    static_ffmpeg.add_paths(weak=True)
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError("static-ffmpeg could not provide ffmpeg/ffprobe")
    return "static-ffmpeg"
