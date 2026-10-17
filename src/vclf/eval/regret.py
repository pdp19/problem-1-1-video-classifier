from __future__ import annotations

import numpy as np

from ..taxonomy import CLASS_NAMES

EFFECTIVENESS = np.array([
    # TH   SC   EN   SB   EG   ST   SP   GP      chunker
    [1.0, 0.7, 0.5, 0.3, 0.4, 0.0, 0.2, 0.3],  # transcript topic segmentation
    [0.3, 1.0, 0.6, 0.4, 0.3, 0.5, 0.4, 0.5],  # screen/slide change detection
    [0.4, 0.6, 1.0, 0.6, 0.5, 0.2, 0.7, 0.5],  # shot -> scene grouping
    [0.1, 0.1, 0.4, 1.0, 0.3, 0.4, 0.4, 0.7],  # play/event segmentation
    [0.4, 0.4, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5],  # activity-drift sliding windows
    [0.1, 0.2, 0.2, 0.4, 0.3, 1.0, 0.3, 0.3],  # motion-triggered, idle dropped
    [0.2, 0.2, 0.6, 0.4, 0.3, 0.1, 1.0, 0.3],  # audio structure + shots
    [0.1, 0.3, 0.3, 0.5, 0.2, 0.2, 0.2, 1.0],  # game-state / HUD changes
])
assert EFFECTIVENESS.shape == (len(CLASS_NAMES), len(CLASS_NAMES))
assert np.allclose(np.diag(EFFECTIVENESS), 1.0)

COST = (1.0 - EFFECTIVENESS).T
_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}


def routing_regret(y_true, y_pred, cost: np.ndarray = COST) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean([cost[_INDEX[t], _INDEX[p]] for t, p in zip(y_true, y_pred)]))


def expected_regret(y_true, proba: np.ndarray, classes, cost: np.ndarray = COST) -> float:
    idx = [_INDEX[c] for c in classes]
    sub = cost[np.ix_(idx, idx)]
    choice = np.argmin(proba @ sub, axis=1)
    preds = [classes[i] for i in choice]
    return routing_regret(y_true, preds, cost)
