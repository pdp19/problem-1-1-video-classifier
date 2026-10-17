from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import subtitles
from ..frames import grab_frames, sample_times
from ..probe import probe
from ..taxonomy import CLASS_NAMES
from . import EMB_DIM, META_FEATURES
from .audio import audio_features, decode_clip, get_vad
from .encoder import INPUT_SIZE, get_encoder
from .packets import packet_features, read_packets
from .visual import embedding_diversity, lowvis_features

N_WINDOWS = 4
FRAMES = 16
AUDIO_CLIP_S = 8.0


def _window_cues(cues: list[subtitles.Cue], start: float, end: float) -> list[subtitles.Cue]:
    return [subtitles.Cue(c.start - start, c.end - start, c.text) for c in cues if c.end > start and c.start < end]


def extract_video(path: Path, video_id: str, subtitle_path: Path | None = None,
                  n_windows: int = N_WINDOWS, frames_n: int = FRAMES) -> tuple[list[dict], dict]:
    timings: dict[str, float] = {}

    def timed(stage, fn, *a, **kw):
        t0 = time.perf_counter()
        result = fn(*a, **kw)
        timings[stage] = timings.get(stage, 0.0) + (time.perf_counter() - t0) * 1000
        return result

    meta = timed("probe", probe, path)
    dur = meta.duration_s
    pixels = meta.width * meta.height
    packets = timed("packets", read_packets, path)
    cues = timed("subtitles", lambda: subtitles.load_cues(subtitle_path) if subtitle_path else [])

    times = sample_times(dur, frames_n)
    pil_frames = timed("frames", grab_frames, path, times, INPUT_SIZE, INPUT_SIZE)
    frames = [np.asarray(f) if f is not None else None for f in pil_frames]
    valid = [i for i, f in enumerate(frames) if f is not None]
    encoder = get_encoder()
    emb_all = np.zeros((len(frames), EMB_DIM), dtype=np.float32)
    if valid:
        emb_all[valid] = timed("encode", encoder.encode, [frames[i] for i in valid])

    vad = get_vad() if meta.has_audio else None
    video_level = {
        "meta_log_duration": math.log10(max(dur, 1.0)),
        "meta_fps": meta.fps,
        "meta_aspect": meta.width / max(meta.height, 1),
        "meta_height": float(meta.height),
        "meta_log_bpp": math.log10(meta.bitrate_bps / max(pixels * max(meta.fps, 1.0), 1) + 1e-6),
        "meta_has_audio": float(meta.has_audio),
        "meta_has_subs": float(bool(cues)),
        "vid_emb_div": embedding_diversity(emb_all[valid]) if valid else 0.0,
    }
    assert set(video_level) == set(META_FEATURES)

    rows = []
    per_win = frames_n // n_windows
    for w in range(n_windows):
        start, end = dur * w / n_windows, dur * (w + 1) / n_windows
        span = end - start
        row: dict = {"video_id": video_id, "window": w, "start_s": start, "end_s": end, **video_level}
        row.update(timed("packets", packet_features, packets.window(start, end), pixels, span))
        wc = _window_cues(cues, start, end)
        row["sub_speech_cov"] = subtitles.speech_coverage(wc, span)
        row["sub_wpm"] = subtitles.words_per_minute(wc, span)

        if meta.has_audio:
            mid = (start + end) / 2
            clip_len = min(AUDIO_CLIP_S, span)
            audio = timed("audio", decode_clip, path, mid - clip_len / 2, clip_len)
            row.update(timed("audio", audio_features, audio, vad))
        else:
            row.update(audio_features(np.zeros(0, dtype=np.float32)))

        idx = [i for i in range(w * per_win, (w + 1) * per_win) if frames[i] is not None]
        row.update(timed("lowvis", lowvis_features, [frames[i] for i in idx]))
        if idx:
            emb = emb_all[idx]
            mean = emb.mean(axis=0)
            mean /= max(np.linalg.norm(mean), 1e-8)
            row["emb_div"] = embedding_diversity(emb)
            zs = encoder.zero_shot(mean[None, :])[0]
        else:
            mean = np.zeros(EMB_DIM, dtype=np.float32)
            row["emb_div"] = 0.0
            zs = np.full(len(encoder.text), 1.0 / len(encoder.text))
        row.update({f"emb_{i:03d}": float(v) for i, v in enumerate(mean)})
        row.update({f"zs_{c}": float(p) for c, p in zip(CLASS_NAMES, zs)})
        rows.append(row)
    timings["total"] = sum(v for k, v in timings.items())
    timings["duration_s"] = dur
    return rows, timings


def extract_many(records, workers: int = 4, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    get_encoder()
    results = {}

    def work(rec):
        rows, t = extract_video(rec.path, rec.video_id, rec.subtitle_path)
        return rec.video_id, rows, t

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (vid, rows, t) in enumerate(pool.map(work, records), 1):
            results[vid] = (rows, t)
            if i % 10 == 0 or i == len(records):
                log(f"  features {i}/{len(records)}")
    feats = pd.DataFrame([r for vid in sorted(results) for r in results[vid][0]])
    timings = pd.DataFrame([{"video_id": vid, **results[vid][1]} for vid in sorted(results)])
    return feats, timings
