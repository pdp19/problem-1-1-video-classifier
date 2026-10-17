import numpy as np
import pandas as pd

from vclf.eval import metrics
from vclf.eval.regret import COST, EFFECTIVENESS, expected_regret, routing_regret
from vclf.models.train import cascade_curve, make_split, select_one_se, CVResult
from vclf.models.variants import ZeroShot, make_variants
from vclf.taxonomy import CLASS_NAMES

CLASSES = ["a", "b", "c"]


def test_regret_matrix_properties():
    assert np.allclose(np.diag(COST), 0) and (COST >= 0).all() and (COST <= 1).all()
    assert COST.shape == EFFECTIVENESS.shape == (8, 8)
    th, st = CLASS_NAMES.index("talking_head"), CLASS_NAMES.index("static_camera")
    assert COST[st, th] == 1.0


def test_routing_regret_zero_when_correct():
    assert routing_regret(["gameplay", "talking_head"], ["gameplay", "talking_head"]) == 0.0
    assert routing_regret([], []) == 0.0
    assert routing_regret(["static_camera"], ["talking_head"]) == 1.0


def test_cost_aware_router_never_worse_on_confident_correct_probs():
    classes = ["talking_head", "edited_narrative"]
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert expected_regret(["talking_head", "edited_narrative"], p, classes) == 0.0


def test_point_metrics_perfect_predictions():
    classes = ["talking_head", "gameplay", "sports_broadcast"]
    y = ["talking_head", "gameplay", "sports_broadcast", "talking_head"]
    p = np.eye(3)[[0, 1, 2, 0]]
    m = metrics.point_metrics(y, p, classes)
    assert m["macro_f1"] == m["accuracy"] == m["top2_acc"] == 1.0
    assert m["ece"] == 0.0 and m["regret"] == 0.0 and m["regret_cost_aware"] == 0.0
    assert all(v["f1"] == 1.0 for v in metrics.per_class_f1(y, p, classes).values())


def test_ece_detects_overconfidence():
    y = ["a", "b"] * 50
    p = np.tile([0.99, 0.01, 0.0], (100, 1))
    assert metrics.ece(y, p, CLASSES) > 0.45


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(0)
    y = np.array(["talking_head", "gameplay"] * 20)
    p = rng.dirichlet([1, 1], size=40)
    out = metrics.bootstrap(y, p, ["talking_head", "gameplay"], n=200)
    for v in out.values():
        assert v["lo"] <= v["value"] + 1e-9 and v["value"] <= v["hi"] + 1e-9


def test_split_is_stratified_and_deterministic():
    labels = pd.DataFrame({"video_id": [f"v{i}" for i in range(40)], "label": ["x"] * 30 + ["y"] * 10})
    s1, s2 = make_split(labels), make_split(labels)
    assert s1.equals(s2)
    test = s1[s1.split == "test"].merge(labels)
    assert len(test) == 10 and (test.label == "y").sum() in (2, 3)


def test_one_se_rule_prefers_cheaper_when_close():
    rows = [
        {"variant": "big", "cv_macro_f1": 0.80, "cv_macro_f1_se": 0.03, "latency_p50_ms": 3000},
        {"variant": "small", "cv_macro_f1": 0.78, "cv_macro_f1_se": 0.03, "latency_p50_ms": 400},
        {"variant": "bad", "cv_macro_f1": 0.60, "cv_macro_f1_se": 0.03, "latency_p50_ms": 10},
    ]
    assert select_one_se(rows)["variant"] == "small"


def test_cascade_curve_endpoints():
    CC = ["talking_head", "gameplay"]
    idx = pd.Index(["v1", "v2"])
    labels = pd.Series(["talking_head", "gameplay"], index=idx)
    cheap = CVResult("cheap", pd.DataFrame([[0.6, 0.4], [0.6, 0.4]], index=idx, columns=CC), [0.5], labels)
    full = CVResult("full", pd.DataFrame([[0.9, 0.1], [0.1, 0.9]], index=idx, columns=CC), [1.0], labels)
    cur = cascade_curve(cheap, full, 100.0, 1000.0, CC)
    low, high = cur.iloc[0], cur.iloc[-1]
    assert low.cheap_coverage == 1.0 and low.mean_latency_ms == 100.0
    assert high.cheap_coverage == 0.0 and high.macro_f1 == 1.0 and high.mean_latency_ms == 1000.0


def test_zero_shot_estimator_renormalises_over_seen_classes():
    X = pd.DataFrame({f"zs_{c}": [1 / 8] * 2 for c in CLASS_NAMES})
    X["zs_gameplay"] = [0.5, 0.0]
    est = ZeroShot().fit(X, ["gameplay", "talking_head"])
    p = est.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1) and est.predict(X)[0] == "gameplay"


def test_variants_have_unique_names_and_valid_columns():
    vs = make_variants()
    assert len({v.name for v in vs}) == len(vs)
    assert {"lr_cheap", "lr_all", "zero_shot"} <= {v.name for v in vs}
    assert not any(v.needs_frames for v in vs if v.name in ("lr_cheap", "hgb_cheap"))
