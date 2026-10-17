from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from .regret import expected_regret, routing_regret


def ece(y_true, proba: np.ndarray, classes, bins: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    conf = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def top2_accuracy(y_true, proba: np.ndarray, classes) -> float:
    top2 = np.argsort(-proba, axis=1)[:, :2]
    cls = np.asarray(classes)
    return float(np.mean([t in cls[row] for t, row in zip(y_true, top2)]))


def point_metrics(y_true, proba: np.ndarray, classes) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    present = sorted(set(y_true))
    return {
        "macro_f1": float(f1_score(y_true, pred, labels=present, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "top2_acc": top2_accuracy(y_true, proba, classes),
        "ece": ece(y_true, proba, classes),
        "regret": routing_regret(y_true, pred),
        "regret_cost_aware": expected_regret(y_true, proba, list(classes)),
    }


def bootstrap(y_true, proba: np.ndarray, classes, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    point = point_metrics(y_true, proba, classes)
    samples = {k: [] for k in point}
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        m = point_metrics(y_true[idx], proba[idx], classes)
        for k, v in m.items():
            samples[k].append(v)
    return {
        k: {"value": point[k],
            "lo": float(np.percentile(samples[k], 100 * alpha / 2)),
            "hi": float(np.percentile(samples[k], 100 * (1 - alpha / 2)))}
        for k in point
    }


def per_class_f1(y_true, proba: np.ndarray, classes) -> dict[str, dict]:
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    out = {}
    for c in classes:
        support = int((y_true == c).sum())
        if support == 0:
            continue
        out[c] = {"f1": float(f1_score(y_true == c, pred == c, zero_division=0)), "support": support}
    return out


def confusion(y_true, proba: np.ndarray, classes) -> np.ndarray:
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    return confusion_matrix(y_true, pred, labels=list(classes))
