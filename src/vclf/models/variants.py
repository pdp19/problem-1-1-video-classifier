from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import features
from ..taxonomy import CLASS_NAMES

VISUAL_GROUPS = {"lowvis", "zs", "emb"}
CHEAP_GROUPS = ("meta", "packet", "subs", "audio")


class ZeroShot(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        return self

    def predict_proba(self, X):
        cols = [f"zs_{c}" for c in self.classes_]
        p = X[cols].to_numpy(dtype=float) + 1e-9
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class Majority(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.prior_ = counts / counts.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _logreg(scalar_cols: list[str], emb_cols: list[str], C: float, pca: int, balanced: bool) -> Pipeline:
    parts = []
    if scalar_cols:
        parts.append(("scalar", StandardScaler(), scalar_cols))
    if emb_cols:
        parts.append(("emb", Pipeline([("pca", PCA(n_components=pca, random_state=0)),
                                       ("scale", StandardScaler())]), emb_cols))
    return Pipeline([
        ("cols", ColumnTransformer(parts)),
        ("clf", LogisticRegression(C=C, max_iter=5000, class_weight="balanced" if balanced else None)),
    ])


def _hgb(cols: list[str]) -> Pipeline:
    return Pipeline([
        ("cols", ColumnTransformer([("x", "passthrough", cols)])),
        ("clf", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.08, max_iter=150, min_samples_leaf=8,
            l2_regularization=1.0, class_weight="balanced", random_state=0)),
    ])


@dataclass
class Variant:
    name: str
    groups: tuple[str, ...]
    build: Callable[[], BaseEstimator]
    description: str
    params: dict = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return features.columns(*self.groups) if self.groups else []

    @property
    def needs_frames(self) -> bool:
        return bool(VISUAL_GROUPS & set(self.groups))


def _split(groups: tuple[str, ...]) -> tuple[list[str], list[str]]:
    cols = features.columns(*groups)
    emb_set = set(features.GROUPS["emb"])
    emb = [c for c in cols if c in emb_set]
    return [c for c in cols if c not in emb], emb


def make_variants(C: float = 0.3, pca: int = 24) -> list[Variant]:
    all_groups = (*CHEAP_GROUPS, "lowvis", "zs", "emb")
    no_emb = (*CHEAP_GROUPS, "lowvis", "zs")
    vis = ("lowvis", "zs", "emb")
    return [
        Variant("majority", (), Majority, "always the most common class (floor)"),
        Variant("zero_shot", ("zs",), ZeroShot, "MobileCLIP zero-shot prompts, no training"),
        Variant("lr_cheap", CHEAP_GROUPS, lambda: _logreg(*_split(CHEAP_GROUPS), C, pca, True),
                "logistic regression on metadata + packets + subtitles + audio (no frames decoded)", {"C": C}),
        Variant("hgb_cheap", CHEAP_GROUPS, lambda: _hgb(features.columns(*CHEAP_GROUPS)),
                "gradient-boosted trees on the cheap tier"),
        Variant("lr_visual", vis, lambda: _logreg(*_split(vis), C, pca, True),
                "logistic regression on frames only (pixel stats + zero-shot + embedding)", {"C": C, "pca": pca}),
        Variant("hgb_all_noemb", no_emb, lambda: _hgb(features.columns(*no_emb)),
                "gradient-boosted trees on every scalar feature incl. zero-shot scores"),
        Variant("lr_all", all_groups, lambda: _logreg(*_split(all_groups), C, pca, True),
                "logistic regression on every feature group (embedding PCA-compressed)", {"C": C, "pca": pca}),
        Variant("lr_all_unweighted", all_groups, lambda: _logreg(*_split(all_groups), C, pca, False),
                "lr_all without class re-weighting (class-balance ablation)", {"C": C, "pca": pca}),
        Variant("lr_all_noemb", no_emb, lambda: _logreg(*_split(no_emb), C, pca, True),
                "logistic regression on every scalar feature incl. zero-shot scores (no raw embedding)", {"C": C}),
        Variant("ens_lr_hgb", all_groups, lambda: VotingClassifier(
                    [("lr", _logreg(*_split(all_groups), C, pca, True)), ("hgb", _hgb(features.columns(*no_emb)))],
                    voting="soft"),
                "soft-voting average of lr_all and hgb_all_noemb", {"C": C, "pca": pca}),
    ]


def known_classes(y) -> list[str]:
    present = set(y)
    return [c for c in CLASS_NAMES if c in present]
