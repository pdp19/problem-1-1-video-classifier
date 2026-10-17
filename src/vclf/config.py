from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HF_DATASET_ID = "aintropy-ai/video-mme-sample10"

DATA_DIR = Path(os.environ.get("VCLF_DATA_DIR", REPO_ROOT / "data" / "video-mme-sample10"))

ARTIFACTS_DIR = Path(os.environ.get("VCLF_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
CONTACT_SHEETS_DIR = ARTIFACTS_DIR / "contact_sheets"
FEATURES_PATH = ARTIFACTS_DIR / "features.parquet"
TIMINGS_PATH = ARTIFACTS_DIR / "feature_timings.csv"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODEL_PATH = ARTIFACTS_DIR / "models" / "classifier.joblib"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

LABELS_DIR = REPO_ROOT / "labels"
DRAFT_LABELS_PATH = LABELS_DIR / "labels_draft.csv"
GOLD_LABELS_PATH = LABELS_DIR / "labels.csv"
SPLIT_PATH = LABELS_DIR / "split.csv"


def videos_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or DATA_DIR) / "videos"


def subtitles_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or DATA_DIR) / "subtitles"
