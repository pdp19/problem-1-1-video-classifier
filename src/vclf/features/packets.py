from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ffmpeg_bin import ensure as ensure_ffmpeg


@dataclass
class PacketTable:
    pts: np.ndarray
    size: np.ndarray
    key: np.ndarray

    def window(self, start: float, end: float) -> "PacketTable":
        m = (self.pts >= start) & (self.pts < end)
        return PacketTable(self.pts[m], self.size[m], self.key[m])


def read_packets(path: str | Path) -> PacketTable:
    ensure_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,size,flags", "-of", "csv=p=0", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe packets failed for {path}: {proc.stderr.strip()}")
    pts, size, key = [], [], []
    for line in proc.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 3 or parts[0] in ("", "N/A"):
            continue
        pts.append(float(parts[0]))
        size.append(int(parts[1]))
        key.append("K" in parts[2])
    order = np.argsort(pts, kind="stable")
    return PacketTable(np.asarray(pts)[order], np.asarray(size, dtype=np.float64)[order], np.asarray(key, dtype=bool)[order])


PACKET_FEATURES = (
    "pk_kf_per_min", "pk_kf_interval_cv", "pk_p_bpp_log", "pk_p_size_cv",
    "pk_p_burst", "pk_tiny_p_frac", "pk_sec_rate_cv", "pk_spike_per_min", "pk_key_p_ratio_log",
)


def packet_features(tab: PacketTable, pixels: int, duration_s: float) -> dict[str, float]:
    out = dict.fromkeys(PACKET_FEATURES, 0.0)
    if len(tab.pts) < 10 or duration_s <= 0:
        return out
    minutes = duration_s / 60.0
    key_t = tab.pts[tab.key]
    out["pk_kf_per_min"] = len(key_t) / minutes
    if len(key_t) >= 3:
        iv = np.diff(key_t)
        out["pk_kf_interval_cv"] = float(iv.std() / max(iv.mean(), 1e-6))
    p = tab.size[~tab.key]
    if len(p) >= 5:
        bpp = p / max(pixels, 1)
        out["pk_p_bpp_log"] = float(np.log10(bpp.mean() + 1e-6))
        out["pk_p_size_cv"] = float(p.std() / max(p.mean(), 1e-6))
        out["pk_p_burst"] = float(np.percentile(p, 95) / max(np.median(p), 1.0))
        out["pk_tiny_p_frac"] = float((bpp < 0.002).mean())
        k = 25
        if len(p) > k:
            pad = np.pad(p, (k // 2, k // 2), mode="edge")
            local = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)[: len(p)]
            out["pk_spike_per_min"] = float((p > 4 * np.maximum(local, 1.0)).sum() / minutes)
        if tab.key.any():
            out["pk_key_p_ratio_log"] = float(np.log10(tab.size[tab.key].mean() / max(p.mean(), 1.0)))
    secs = np.floor(tab.pts - tab.pts[0]).astype(int)
    per_sec = np.bincount(secs, weights=tab.size)
    if len(per_sec) >= 3:
        out["pk_sec_rate_cv"] = float(per_sec.std() / max(per_sec.mean(), 1e-6))
    return out
