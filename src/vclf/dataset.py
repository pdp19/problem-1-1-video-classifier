from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    path: Path
    subtitle_path: Path | None
    duration_bucket: str
    domain: str
    sub_category: str


def download(data_dir: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    target = data_dir or config.DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=config.HF_DATASET_ID, repo_type="dataset", local_dir=str(target))
    return target


def load_records(data_dir: Path | None = None) -> list[VideoRecord]:
    root = data_dir or config.DATA_DIR
    qa_path = root / "qa" / "test.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(
            f"{qa_path} not found. Run `python src/main.py download` or set VCLF_DATA_DIR."
        )
    meta: dict[str, dict] = {}
    with qa_path.open() as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                meta.setdefault(row["video_id"], row)
    records = []
    for vid in sorted(meta):
        row = meta[vid]
        srt = config.subtitles_dir(root) / f"{vid}.srt"
        records.append(
            VideoRecord(
                video_id=vid,
                path=config.videos_dir(root) / f"{vid}.mp4",
                subtitle_path=srt if srt.exists() else None,
                duration_bucket=row.get("duration", ""),
                domain=row.get("domain", ""),
                sub_category=row.get("sub_category", ""),
            )
        )
    return records
