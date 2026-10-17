from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config, subtitles
from ..dataset import VideoRecord
from ..probe import probe
from .contact_sheet import write_contact_sheet
from .draft import LabelContext


def sheet_path(video_id: str, sheets_dir: Path | None = None) -> Path:
    return (sheets_dir or config.CONTACT_SHEETS_DIR) / f"{video_id}.jpg"


def make_context(rec: VideoRecord, sheets_dir: Path | None = None, force: bool = False) -> LabelContext:
    meta = probe(rec.path)
    out = sheet_path(rec.video_id, sheets_dir)
    if force or not out.exists():
        write_contact_sheet(rec.path, meta.duration_s, rec.video_id, out)
    cues = subtitles.load_cues(rec.subtitle_path) if rec.subtitle_path else []
    return LabelContext(
        video_id=rec.video_id,
        sheet_path=out.resolve(),
        duration_s=meta.duration_s,
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        has_audio=meta.has_audio,
        has_subtitles=bool(cues),
        speech_coverage=subtitles.speech_coverage(cues, meta.duration_s),
        words_per_minute=subtitles.words_per_minute(cues, meta.duration_s),
        excerpts=subtitles.excerpts(cues, meta.duration_s),
        domain=rec.domain,
        sub_category=rec.sub_category,
    )


def make_contexts(
    records: list[VideoRecord], sheets_dir: Path | None = None, force: bool = False, workers: int = 4
) -> list[LabelContext]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda r: make_context(r, sheets_dir, force), records))
