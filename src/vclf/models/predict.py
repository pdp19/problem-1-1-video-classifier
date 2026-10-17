from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd

from .. import config
from ..taxonomy import CLASS_BY_NAME, CLASS_NAMES

FALLBACK_CONFIDENCE = 0.5
UNSEEN_ZS_MIN = 0.6
UNSEEN_MAX_EMB_DIV = 0.08


def load_model(path: Path | None = None) -> dict:
    p = path or config.MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `python src/main.py train` first")
    return joblib.load(p)


def classify_windows(rows: pd.DataFrame, bundle: dict) -> dict:
    est, classes = bundle["estimator"], bundle["classes"]
    X = rows[bundle["columns"]] if bundle["columns"] else rows[["window"]]
    p = est.predict_proba(X)
    probs = pd.DataFrame(0.0, index=rows.index, columns=classes)
    for j, c in enumerate(est.classes_):
        probs[c] = p[:, j]
    video_p = probs.mean(axis=0)
    label = str(video_p.idxmax())
    conf = float(video_p.max())
    source = "classifier"

    zs_cols = [f"zs_{c}" for c in CLASS_NAMES]
    if set(zs_cols) <= set(rows.columns):
        zs = rows[zs_cols].mean(axis=0)
        unseen = [c for c in CLASS_NAMES if c not in classes]
        for c in unseen:
            if zs[f"zs_{c}"] >= UNSEEN_ZS_MIN and rows["vid_emb_div"].iloc[0] <= UNSEEN_MAX_EMB_DIV:
                label, conf, source = c, float(zs[f"zs_{c}"]), "zero-shot (class has no training data)"
    route = label if conf >= FALLBACK_CONFIDENCE else "fallback"
    return {
        "label": label,
        "confidence": round(conf, 3),
        "route": route,
        "chunking_strategy": CLASS_BY_NAME[label].chunking if route != "fallback"
        else "fixed ~30 s windows snapped to shot cuts + transcript boundaries",
        "source": source,
        "probabilities": {c: round(float(v), 3) for c, v in video_p.sort_values(ascending=False).items()},
    }


def find_subtitles(video: Path) -> Path | None:
    for cand in (video.with_suffix(".srt"), video.parent.parent / "subtitles" / f"{video.stem}.srt"):
        if cand.exists():
            return cand
    return None


def predict_file(path: Path, bundle: dict | None = None) -> dict:
    from ..features.extract import extract_video

    bundle = bundle or load_model()
    t0 = time.perf_counter()
    rows, timings = extract_video(path, path.stem, find_subtitles(path))
    t1 = time.perf_counter()
    result = classify_windows(pd.DataFrame(rows), bundle)
    t2 = time.perf_counter()
    result.update({
        "video": str(path),
        "duration_s": round(timings["duration_s"], 1),
        "model": bundle["variant"],
        "latency_ms": {"features": round((t1 - t0) * 1000, 1), "classifier": round((t2 - t1) * 1000, 2),
                       **{k: round(v, 1) for k, v in timings.items() if k not in ("total", "duration_s")}},
    })
    return result

