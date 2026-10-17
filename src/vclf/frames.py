from __future__ import annotations

import io
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .ffmpeg_bin import ensure as ensure_ffmpeg


def sample_times(duration_s: float, n: int) -> list[float]:
    if n <= 0 or duration_s <= 0:
        return []
    return [duration_s * (i + 0.5) / n for i in range(n)]


def grab_frame(path: str | Path, t: float, width: int, height: int) -> Image.Image | None:
    ensure_ffmpeg()
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
        "-frames:v", "1", "-vf", vf, "-f", "image2pipe", "-vcodec", "png", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


def grab_frames(
    path: str | Path, times: list[float], width: int, height: int, workers: int = 4
) -> list[Image.Image | None]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda t: grab_frame(path, t, width, height), times))
