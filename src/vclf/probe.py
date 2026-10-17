from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from .ffmpeg_bin import ensure as ensure_ffmpeg


@dataclass
class VideoMeta:
    path: str
    duration_s: float
    width: int
    height: int
    fps: float
    video_codec: str
    bitrate_bps: int
    size_bytes: int
    has_audio: bool
    audio_channels: int

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "N/A"):
        return 0.0
    return float(Fraction(rate))


def probe(path: str | Path) -> VideoMeta:
    ensure_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,channels",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    fmt = info.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    return VideoMeta(
        path=str(path),
        duration_s=float(fmt.get("duration") or 0.0),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=round(fps, 3),
        video_codec=str(video.get("codec_name") or ""),
        bitrate_bps=int(fmt.get("bit_rate") or 0),
        size_bytes=int(fmt.get("size") or 0),
        has_audio=audio is not None,
        audio_channels=int(audio.get("channels") or 0) if audio else 0,
    )
