from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ..frames import grab_frames, sample_times

TILE_W, TILE_H = 320, 180
GRID = 4
HEADER_H = 28


def _fmt_ts(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_contact_sheet(video_path: Path, duration_s: float, title: str) -> Image.Image:
    n = GRID * GRID
    times = sample_times(duration_s, n)
    frames = grab_frames(video_path, times, TILE_W, TILE_H)
    sheet = Image.new("RGB", (TILE_W * GRID, TILE_H * GRID + HEADER_H), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), f"{title}   duration {_fmt_ts(duration_s)}", fill=(235, 235, 235))
    for i, (t, frame) in enumerate(zip(times, frames)):
        x = (i % GRID) * TILE_W
        y = HEADER_H + (i // GRID) * TILE_H
        if frame is not None:
            sheet.paste(frame, (x, y))
        else:
            draw.rectangle([x, y, x + TILE_W - 1, y + TILE_H - 1], fill=(60, 0, 0))
        label = _fmt_ts(t)
        draw.rectangle([x, y, x + 8 + 7 * len(label), y + 16], fill=(0, 0, 0))
        draw.text((x + 4, y + 2), label, fill=(255, 255, 0))
    return sheet


def write_contact_sheet(video_path: Path, duration_s: float, title: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_contact_sheet(video_path, duration_s, title).save(out_path, quality=85)
    return out_path
