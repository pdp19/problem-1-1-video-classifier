from __future__ import annotations

import json
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .. import config
from ..eval import metrics
from .variants import Variant, known_classes, make_variants

SEED = 42
STAGES_CHEAP = ("probe", "packets", "subtitles", "audio")
STAGES_FRAMES = ("frames", "encode", "lowvis")


def load_table(features_path=None, labels_path=None) -> pd.DataFrame:
    feats = pd.read_parquet(features_path or config.FEATURES_PATH)
    labels = pd.read_csv(labels_path or config.GOLD_LABELS_PATH, keep_default_na=False)[["video_id", "label"]]
    df = feats.merge(labels, on="video_id", how="inner")
    missing = set(labels.video_id) - set(feats.video_id)
    if missing:
        raise RuntimeError(f"{len(missing)} labelled videos have no features (e.g. {sorted(missing)[0]})")
    return df


def make_split(labels: pd.DataFrame, test_frac: float = 0.25, seed: int = SEED) -> pd.DataFrame:
    train_ids, test_ids = train_test_split(
        labels.video_id, test_size=test_frac, stratify=labels.label, random_state=seed)
    split = pd.DataFrame({"video_id": list(train_ids) + list(test_ids),
                          "split": ["train"] * len(train_ids) + ["test"] * len(test_ids)})
    return split.sort_values("video_id").reset_index(drop=True)


def load_or_create_split(labels: pd.DataFrame) -> pd.DataFrame:
    if config.SPLIT_PATH.exists():
        split = pd.read_csv(config.SPLIT_PATH)
        if set(split.video_id) == set(labels.video_id):
            return split
    split = make_split(labels)
    split.to_csv(config.SPLIT_PATH, index=False)
    return split


def video_proba(est, X: pd.DataFrame, video_ids: pd.Series, classes: list[str]) -> pd.DataFrame:
    p = est.predict_proba(X)
    frame = pd.DataFrame(0.0, index=X.index, columns=classes)
    for j, c in enumerate(est.classes_):
        frame[c] = p[:, j]
    frame["video_id"] = video_ids.values
    return frame.groupby("video_id")[classes].mean()


def fit(variant: Variant, df: pd.DataFrame):
    X = df[variant.columns] if variant.columns else df[["window"]]
    return variant.build().fit(X, df["label"])


@dataclass
class CVResult:
    variant: str
    oof: pd.DataFrame
    repeat_f1: list[float]
    labels: pd.Series


def cross_validate(variant: Variant, df: pd.DataFrame, classes: list[str],
                   n_splits: int = 5, n_repeats: int = 5, seed: int = SEED) -> CVResult:
    vids = df.groupby("video_id")["label"].first()
    sums = pd.DataFrame(0.0, index=vids.index, columns=classes)
    repeat_f1 = []
    X_all = df[variant.columns] if variant.columns else df[["window"]]
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof = pd.DataFrame(0.0, index=vids.index, columns=classes)
        for tr, te in cv.split(X_all, df["label"], groups=df["video_id"]):
            est = variant.build().fit(X_all.iloc[tr], df["label"].iloc[tr])
            vp = video_proba(est, X_all.iloc[te], df["video_id"].iloc[te], classes)
            oof.loc[vp.index] = vp.values
        sums += oof
        repeat_f1.append(metrics.point_metrics(vids.loc[oof.index], oof.to_numpy(), classes)["macro_f1"])
    return CVResult(variant.name, sums / n_repeats, repeat_f1, vids)


def tune_logreg(df: pd.DataFrame, classes: list[str], log=print) -> dict:
    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        for pca in (4, 8, 16, 32):
            v = next(x for x in make_variants(C=C, pca=pca) if x.name == "lr_all")
            res = cross_validate(v, df, classes, n_repeats=3)
            score = float(np.mean(res.repeat_f1))
            if best is None or score > best["cv_macro_f1"]:
                best = {"C": C, "pca": pca, "cv_macro_f1": score}
    log(f"  tuned lr_all: C={best['C']} pca={best['pca']} (CV macro-F1 {best['cv_macro_f1']:.3f})")
    return best


def variant_latency_ms(variant: Variant, timings: pd.DataFrame) -> pd.Series:
    if not variant.groups:
        return pd.Series(0.0, index=timings.index)
    stages = ["probe"]
    if set(variant.groups) & {"meta", "packet", "subs", "audio"}:
        stages += list(STAGES_CHEAP[1:])
    if variant.needs_frames:
        stages += ["frames", "encode"] + (["lowvis"] if "lowvis" in variant.groups else [])
    return timings[[s for s in dict.fromkeys(stages) if s in timings]].sum(axis=1)


def select_one_se(rows: list[dict]) -> dict:
    best = max(rows, key=lambda r: r["cv_macro_f1"])
    threshold = best["cv_macro_f1"] - best["cv_macro_f1_se"]
    ok = [r for r in rows if r["cv_macro_f1"] >= threshold]
    return min(ok, key=lambda r: (r["latency_p50_ms"], -r["cv_macro_f1"]))


def cascade_curve(cheap: CVResult, full: CVResult, cheap_ms: float, full_ms: float, classes) -> pd.DataFrame:
    y = cheap.labels.loc[cheap.oof.index]
    pc, pf = cheap.oof.to_numpy(), full.oof.loc[cheap.oof.index].to_numpy()
    rows = []
    for tau in np.round(np.arange(0.3, 1.0001, 0.05), 2):
        accept = pc.max(axis=1) >= tau
        p = np.where(accept[:, None], pc, pf)
        m = metrics.point_metrics(y, p, classes)
        rows.append({"tau": tau, "cheap_coverage": float(accept.mean()),
                     "macro_f1": m["macro_f1"], "regret": m["regret"],
                     "mean_latency_ms": float(cheap_ms + (1 - accept.mean()) * (full_ms - cheap_ms))})
    return pd.DataFrame(rows)


def run(log=print, n_repeats: int = 5) -> dict:
    t_start = time.perf_counter()
    df = load_table()
    timings = pd.read_csv(config.TIMINGS_PATH).set_index("video_id")
    labels = df.groupby("video_id")["label"].first().reset_index()
    split = load_or_create_split(labels)
    df = df.merge(split, on="video_id")
    classes = known_classes(df["label"])
    train, test = df[df.split == "train"], df[df.split == "test"]
    log(f"{labels.shape[0]} videos ({train.video_id.nunique()} train / {test.video_id.nunique()} test), "
        f"{len(df)} window rows, {len(classes)} classes with data")

    tuned = tune_logreg(train, classes, log)
    variants = make_variants(C=tuned["C"], pca=tuned["pca"])

    rows, cv_results = [], {}
    for v in variants:
        res = cross_validate(v, train, classes, n_repeats=n_repeats)
        cv_results[v.name] = res
        boot = metrics.bootstrap(res.labels.loc[res.oof.index], res.oof.to_numpy(), classes, n=1000)
        lat = variant_latency_ms(v, timings.loc[train.video_id.unique()])
        rows.append({
            "variant": v.name, "description": v.description, "groups": "+".join(v.groups) or "-",
            "cv_macro_f1": float(np.mean(res.repeat_f1)),
            "cv_macro_f1_se": float(np.std(res.repeat_f1, ddof=1) / np.sqrt(len(res.repeat_f1))),
            "cv_macro_f1_lo": boot["macro_f1"]["lo"], "cv_macro_f1_hi": boot["macro_f1"]["hi"],
            "cv_accuracy": boot["accuracy"]["value"], "cv_balanced_acc": boot["balanced_acc"]["value"],
            "cv_regret": boot["regret"]["value"], "cv_ece": boot["ece"]["value"],
            "latency_p50_ms": float(lat.median()), "latency_p95_ms": float(lat.quantile(0.95)),
        })
        log(f"  {v.name:18s} CV macro-F1 {rows[-1]['cv_macro_f1']:.3f} ± {rows[-1]['cv_macro_f1_se']:.3f}  "
            f"latency p50 {rows[-1]['latency_p50_ms']:.0f} ms")
    cv_table = pd.DataFrame(rows)

    chosen = select_one_se([r for r in rows if r["variant"] not in ("majority",)])
    best_variant = next(v for v in variants if v.name == chosen["variant"])
    log(f"selected (one-SE rule): {best_variant.name}")

    cheap_ms = cv_table.set_index("variant").loc["lr_cheap", "latency_p50_ms"]
    full_ms = cv_table.set_index("variant").loc["lr_all", "latency_p50_ms"]
    cascade = cascade_curve(cv_results["lr_cheap"], cv_results["lr_all"], cheap_ms, full_ms, classes)

    est = fit(best_variant, train)
    X_test = test[best_variant.columns] if best_variant.columns else test[["window"]]
    t0 = time.perf_counter()
    test_p = video_proba(est, X_test, test["video_id"], classes)
    infer_ms = (time.perf_counter() - t0) * 1000 / max(len(test_p), 1)
    y_test = test.groupby("video_id")["label"].first().loc[test_p.index]
    test_boot = metrics.bootstrap(y_test, test_p.to_numpy(), classes)
    test_per_class = metrics.per_class_f1(y_test, test_p.to_numpy(), classes)
    test_conf = metrics.confusion(y_test, test_p.to_numpy(), classes)
    test_pred = pd.DataFrame({"video_id": test_p.index, "label": y_test.values,
                              "pred": np.asarray(classes)[test_p.to_numpy().argmax(1)],
                              "confidence": test_p.to_numpy().max(1)})

    all_cv = cross_validate(best_variant, df, classes, n_repeats=n_repeats)
    all_boot = metrics.bootstrap(all_cv.labels.loc[all_cv.oof.index], all_cv.oof.to_numpy(), classes)
    zs_all = cross_validate(next(v for v in variants if v.name == "zero_shot"), df, classes, n_repeats=1)

    final = fit(best_variant, df)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"variant": best_variant.name, "estimator": final, "columns": best_variant.columns,
                 "classes": classes, "groups": best_variant.groups, "tuned": tuned}, config.MODEL_PATH)
    model_kb = config.MODEL_PATH.stat().st_size / 1024

    summary = {
        "n_videos": int(labels.shape[0]), "n_train": int(train.video_id.nunique()),
        "n_test": int(test.video_id.nunique()), "classes": classes,
        "class_counts": labels.label.value_counts().to_dict(),
        "tuned": tuned, "selected": best_variant.name, "selection_rule": "one-standard-error, then lowest latency",
        "test": test_boot, "test_per_class": test_per_class,
        "test_confusion": test_conf.tolist(),
        "all_data_cv": all_boot, "all_data_cv_per_class": metrics.per_class_f1(
            all_cv.labels.loc[all_cv.oof.index], all_cv.oof.to_numpy(), classes),
        "all_data_cv_confusion": metrics.confusion(all_cv.labels.loc[all_cv.oof.index], all_cv.oof.to_numpy(), classes).tolist(),
        "zero_shot_all_data_macro_f1": metrics.point_metrics(
            zs_all.labels.loc[zs_all.oof.index], zs_all.oof.to_numpy(), classes)["macro_f1"],
        "classifier_inference_ms_per_video": infer_ms, "model_file_kb": model_kb,
        "train_seconds": time.perf_counter() - t_start,
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(config.REPORTS_DIR / "cv_variants.csv", index=False)
    cascade.to_csv(config.REPORTS_DIR / "cascade.csv", index=False)
    test_pred.to_csv(config.REPORTS_DIR / "test_predictions.csv", index=False)
    all_cv.oof.assign(label=all_cv.labels).to_csv(config.REPORTS_DIR / "all_data_oof.csv")
    (config.REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"test macro-F1 {test_boot['macro_f1']['value']:.3f} "
        f"[{test_boot['macro_f1']['lo']:.3f}, {test_boot['macro_f1']['hi']:.3f}]  "
        f"accuracy {test_boot['accuracy']['value']:.3f}  regret {test_boot['regret']['value']:.3f}")
    log(f"all-data CV macro-F1 {all_boot['macro_f1']['value']:.3f} "
        f"[{all_boot['macro_f1']['lo']:.3f}, {all_boot['macro_f1']['hi']:.3f}]")
    log(f"model {model_kb:.0f} KB, {infer_ms:.2f} ms/video classifier time, done in {summary['train_seconds']:.0f}s")
    return summary
