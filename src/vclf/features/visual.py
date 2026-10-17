from __future__ import annotations

import numpy as np

LOWVIS_FEATURES = (
    "lv_brightness", "lv_saturation", "lv_colorfulness", "lv_edge_density",
    "lv_flat_frac", "lv_dark_frac", "lv_frame_diff", "lv_dup_frac",
)


def _gray(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _small(gray: np.ndarray, factor: int = 4) -> np.ndarray:
    h, w = (gray.shape[0] // factor) * factor, (gray.shape[1] // factor) * factor
    return gray[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def trim_letterbox(img: np.ndarray) -> np.ndarray:
    content = img[..., :3].max(axis=2) > 0
    rows, cols = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
    if len(rows) < 8 or len(cols) < 8:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def lowvis_features(frames: list[np.ndarray]) -> dict[str, float]:
    out = dict.fromkeys(LOWVIS_FEATURES, 0.0)
    if not frames:
        return out
    bright, sat, colorful, edges, flat, dark, smalls = [], [], [], [], [], [], []
    for raw in frames:
        img = trim_letterbox(raw)
        rgb = img[..., :3].astype(np.float32)
        g = _gray(img)
        bright.append(g.mean() / 255)
        mx, mn = rgb.max(axis=2), rgb.min(axis=2)
        sat.append(((mx - mn) / (mx + 1e-6)).mean())
        rg, yb = rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
        colorful.append((np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 255)
        gx, gy = np.abs(np.diff(g, axis=1))[:-1], np.abs(np.diff(g, axis=0))[:, :-1]
        mag = gx + gy
        edges.append((mag > 40).mean())
        flat.append((mag < 2).mean())
        dark.append(float(g.mean() < 20))
        smalls.append(_small(_gray(raw)))
    out.update(
        lv_brightness=float(np.mean(bright)), lv_saturation=float(np.mean(sat)),
        lv_colorfulness=float(np.mean(colorful)), lv_edge_density=float(np.mean(edges)),
        lv_flat_frac=float(np.mean(flat)), lv_dark_frac=float(np.mean(dark)),
    )
    if len(smalls) >= 2:
        diffs = [np.abs(a - b).mean() / 255 for a, b in zip(smalls, smalls[1:])]
        out["lv_frame_diff"] = float(np.mean(diffs))
        out["lv_dup_frac"] = float(np.mean([d < 0.02 for d in diffs]))
    return out


def embedding_diversity(emb: np.ndarray) -> float:
    if len(emb) < 2:
        return 0.0
    sim = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)
    return float(1.0 - sim[iu].mean())
