from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402

FAMILY = {"majority": "#999999", "zero_shot": "#a463f2", "lr_cheap": "#4269d0", "hgb_cheap": "#97bbf5",
          "lr_visual": "#6cc5b0", "hgb_all_noemb": "#efb118", "lr_all": "#ff725c", "lr_all_unweighted": "#ff8ab7"}


def _pareto(cv: pd.DataFrame, selected: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    cv = cv.assign(x=cv.latency_p50_ms.clip(lower=1.0)).sort_values("cv_macro_f1", ascending=False)
    label_x = cv.x.max() * 2.2
    last_y = None
    for _, r in cv.iterrows():
        c = FAMILY.get(r.variant, "#555")
        sel = r.variant == selected
        ax.errorbar(r.x, r.cv_macro_f1,
                    yerr=[[r.cv_macro_f1 - r.cv_macro_f1_lo], [r.cv_macro_f1_hi - r.cv_macro_f1]],
                    fmt="o", color=c, ecolor=c, elinewidth=1, capsize=3, alpha=0.9,
                    markersize=8 if sel else 5, markeredgecolor="black" if sel else c)
        label = f"{r.variant}  {r.cv_macro_f1:.3f}" + ("  (selected)" if sel else "")
        if r.x >= 500:
            ty = r.cv_macro_f1 if last_y is None else min(r.cv_macro_f1, last_y - 0.042)
            last_y = ty
            ax.annotate(label, (r.x, r.cv_macro_f1), xytext=(label_x, ty), fontsize=8, va="center",
                        color="black" if sel else c, fontweight="bold" if sel else "normal",
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.7))
        else:
            ax.annotate(label, (r.x, r.cv_macro_f1), textcoords="offset points", xytext=(7, 3), fontsize=8, color=c)
    ax.set_xlim(0.7, cv.x.max() * 12)
    ax.set_xscale("log")
    ax.set_xlabel("feature latency per video, p50 (ms, log scale, 1 CPU worker)")
    ax.set_ylabel("CV macro-F1 (95% bootstrap CI)")
    ax.set_title("Accuracy vs cost: every variant on identical folds", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e5e5", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pareto.png", dpi=150)
    plt.close(fig)


def _confusion(mat: np.ndarray, classes: list[str], title: str, name: str) -> None:
    mat = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if mat[i, j]:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "#222", fontsize=9)
    short = [c.replace("_", "\n", 1) for c in classes]
    ax.set_xticks(range(len(classes)), short, fontsize=7.5)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def _cascade(cascade: pd.DataFrame, full_f1: float) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    ax1.plot(cascade.mean_latency_ms, cascade.macro_f1, "-o", color="#4269d0", markersize=4)
    for _, r in cascade.iloc[::2].iterrows():
        ax1.annotate(f"τ={r.tau:.2f}\n{r.cheap_coverage:.0%} cheap", (r.mean_latency_ms, r.macro_f1),
                     textcoords="offset points", xytext=(4, -18), fontsize=7, color="#555")
    ax1.axhline(full_f1, color="#ff725c", linestyle="--", linewidth=1)
    ax1.text(cascade.mean_latency_ms.min(), full_f1, " lr_all on every video", color="#ff725c", fontsize=8, va="bottom")
    ax1.set_xlabel("mean feature latency per video (ms)")
    ax1.set_ylabel("CV macro-F1")
    ax1.set_title("Cascade: cheap model first, frames only when unsure", loc="left", fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(color="#e5e5e5", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cascade.png", dpi=150)
    plt.close(fig)


def _fmt(m: dict) -> str:
    return f"{m['value']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def run(log=print) -> None:
    rep = config.REPORTS_DIR
    summary = json.loads((rep / "summary.json").read_text())
    cv = pd.read_csv(rep / "cv_variants.csv")
    cascade = pd.read_csv(rep / "cascade.csv")
    timings = pd.read_csv(config.TIMINGS_PATH)
    classes = summary["classes"]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _pareto(cv, summary["selected"])
    _confusion(summary["all_data_cv_confusion"], classes,
               f"{summary['selected']}: all-data repeated CV (90 videos)", "confusion_cv.png")
    _confusion(summary["test_confusion"], classes,
               f"{summary['selected']}: locked test split ({summary['n_test']} videos)", "confusion_test.png")
    full_f1 = float(cv.set_index("variant").loc["lr_all", "cv_macro_f1"])
    _cascade(cascade, full_f1)

    stages = ["probe", "packets", "subtitles", "audio", "frames", "encode", "lowvis", "total"]
    lat = timings[stages].describe(percentiles=[0.5, 0.95]).T[["50%", "95%"]].round(0)
    t, a = summary["test"], summary["all_data_cv"]
    ok = cascade[cascade.macro_f1 >= full_f1 - 0.01]
    best_cascade = ok.sort_values("mean_latency_ms").iloc[0] if len(ok) else cascade.iloc[-1]

    lines = [
        "# Results",
        "",
        f"Selected model: **{summary['selected']}** (rule: {summary['selection_rule']}). "
        f"Trained on {summary['n_train']} videos, scored once on {summary['n_test']} locked test videos. "
        f"Model file {summary['model_file_kb']:.0f} KB, classifier time {summary['classifier_inference_ms_per_video']:.2f} ms/video.",
        "",
        "| metric | locked test split | all-data repeated CV (90 videos) |",
        "|---|---|---|",
    ]
    for k, name in [("macro_f1", "macro-F1"), ("balanced_acc", "balanced accuracy"), ("accuracy", "accuracy"),
                    ("top2_acc", "top-2 accuracy"), ("ece", "ECE (calibration)"), ("regret", "routing regret"),
                    ("regret_cost_aware", "routing regret, cost-aware router")]:
        lines.append(f"| {name} | {_fmt(t[k])} | {_fmt(a[k])} |")
    lines += [
        "",
        f"Zero-shot MobileCLIP alone: macro-F1 {summary['zero_shot_all_data_macro_f1']:.3f} on all 90 videos.",
        "",
        "## Every variant (training videos only, 5x5 grouped CV)",
        "",
        "| variant | features | CV macro-F1 [95% CI] | regret | ECE | latency p50 ms |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in cv.sort_values("latency_p50_ms").iterrows():
        lines.append(f"| {r.variant} | {r.groups} | {r.cv_macro_f1:.3f} [{r.cv_macro_f1_lo:.3f}, {r.cv_macro_f1_hi:.3f}] "
                     f"| {r.cv_regret:.3f} | {r.cv_ece:.3f} | {r.latency_p50_ms:.0f} |")
    lines += [
        "",
        "## Per-class F1 (all-data CV)",
        "",
        "| class | F1 | videos |",
        "|---|---|---|",
    ]
    for c, m in summary["all_data_cv_per_class"].items():
        lines.append(f"| {c} | {m['f1']:.2f} | {m['support']} |")
    lines += [
        "",
        "## Cascade",
        "",
        f"Accepting `lr_cheap` when its confidence ≥ {best_cascade.tau:.2f} keeps macro-F1 at "
        f"{best_cascade.macro_f1:.3f} (vs {full_f1:.3f} for `lr_all` everywhere), decodes frames for only "
        f"{1 - best_cascade.cheap_coverage:.0%} of videos, and cuts mean latency to {best_cascade.mean_latency_ms:.0f} ms.",
        "",
        "## Feature latency per video (ms, 1 worker, M3 CPU)",
        "",
        "| stage | p50 | p95 |",
        "|---|---|---|",
    ]
    for s, r in lat.iterrows():
        lines.append(f"| {s} | {r['50%']:.0f} | {r['95%']:.0f} |")
    lines += ["", "![pareto](figures/pareto.png)", "![confusion](figures/confusion_cv.png)",
              "![test confusion](figures/confusion_test.png)", "![cascade](figures/cascade.png)", ""]
    (config.REPO_ROOT / "docs" / "results.md").write_text("\n".join(lines))
    log(f"results figures in {config.FIGURES_DIR}, summary in docs/results.md")
