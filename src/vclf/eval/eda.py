from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402

from .. import config, features  # noqa: E402
from ..taxonomy import CLASS_NAMES  # noqa: E402

PALETTE = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7", "#a463f2", "#97bbf5"]
COLOR = dict(zip(CLASS_NAMES, PALETTE))
KEY_FEATURES = {
    "pk_kf_per_min": "keyframes / min (cut proxy)",
    "pk_tiny_p_frac": "near-empty P-frames (static)",
    "pk_p_bpp_log": "P-frame bits/pixel (log, motion)",
    "sub_speech_cov": "subtitle speech coverage",
    "au_speech_frac": "VAD speech fraction",
    "lv_flat_frac": "flat-area fraction (slides/UI)",
    "emb_div": "frame embedding diversity",
    "lv_frame_diff": "frame-to-frame pixel change",
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#ddd", linewidth=0.6)
    ax.set_axisbelow(True)


def run(log=print) -> dict:
    from ..models.train import load_table

    df = load_table()
    vids = df.groupby("video_id").agg(label=("label", "first"), duration=("meta_log_duration", "first"))
    order = [c for c in CLASS_NAMES if c in set(vids.label)]
    counts = vids.label.value_counts().reindex(CLASS_NAMES, fill_value=0)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = {}

    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(CLASS_NAMES))
    ax.barh(y, counts.values, color=[COLOR[c] for c in CLASS_NAMES])
    ax.set_yticks(y, CLASS_NAMES)
    ax.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("videos")
    ax.set_title("Class balance (90 reviewed labels)", loc="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)
    nz = counts[counts > 0]
    out["imbalance_ratio"] = float(nz.max() / nz.min())

    fig, ax = plt.subplots(figsize=(7, 3.2))
    data = [10 ** vids.duration[vids.label == c] / 60 for c in order]
    ax.boxplot(data, vert=False, tick_labels=order, patch_artist=True,
               boxprops=dict(facecolor="#e8eefb", edgecolor="#4269d0"), medianprops=dict(color="#ff725c"))
    ax.set_xscale("log")
    ax.set_xlabel("duration (minutes, log)")
    ax.set_title("Duration by class", loc="left", fontsize=11)
    ax.invert_yaxis()
    _style(ax)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "duration_by_class.png", dpi=150)
    plt.close(fig)

    vmean = df.groupby("video_id")[list(KEY_FEATURES)].mean().join(vids.label)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for ax, (feat, title) in zip(axes.flat, KEY_FEATURES.items()):
        ax.boxplot([vmean[feat][vmean.label == c] for c in order], vert=False, tick_labels=order,
                   patch_artist=True, boxprops=dict(facecolor="#e8eefb", edgecolor="#4269d0"),
                   medianprops=dict(color="#ff725c"), flierprops=dict(markersize=3))
        ax.set_title(title, fontsize=10, loc="left")
        _style(ax)
    axes[0, 0].invert_yaxis()
    fig.suptitle("What the cheap features see, per class (one point per video)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "features_by_class.png", dpi=150)
    plt.close(fig)

    scalar = [c for c in features.columns("meta", "packet", "subs", "audio", "lowvis", "zs")]
    vm = df.groupby("video_id")[scalar].mean().join(vids.label)
    F, _ = f_classif(vm[scalar].fillna(0), vm.label)
    fscore = pd.Series(np.nan_to_num(F), index=scalar).sort_values(ascending=False)
    top = fscore.head(15)
    group_of = {c: g for g, cols in features.GROUPS.items() for c in cols}
    gcolor = {"meta": "#97bbf5", "packet": "#4269d0", "subs": "#efb118", "audio": "#ff725c",
              "lowvis": "#6cc5b0", "zs": "#a463f2"}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(range(len(top)), top.values, color=[gcolor.get(group_of.get(c, ""), "#999") for c in top.index])
    ax.set_yticks(range(len(top)), top.index)
    ax.invert_yaxis()
    ax.set_xlabel("ANOVA F (higher = separates classes better)")
    ax.set_title("Most separating scalar features", loc="left", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in gcolor.values()]
    ax.legend(handles, gcolor.keys(), fontsize=8, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_separability.png", dpi=150)
    plt.close(fig)
    out["top_features"] = top.round(1).to_dict()

    emb_cols = list(features.GROUPS["emb"])
    ve = df.groupby("video_id")[emb_cols].mean()
    ve = ve.div(np.linalg.norm(ve.values, axis=1), axis=0)
    pca = PCA(n_components=2, random_state=0).fit(ve.values)
    xy = pca.transform(ve.values)
    lab = vids.label.loc[ve.index]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for c in order:
        m = (lab == c).values
        ax.scatter(xy[m, 0], xy[m, 1], s=28, color=COLOR[c], label=c, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("MobileCLIP embedding, one point per video", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "embedding_pca.png", dpi=150)
    plt.close(fig)

    W = df[emb_cols].to_numpy()
    W = W / np.linalg.norm(W, axis=1, keepdims=True).clip(min=1e-8)
    sim = W @ W.T
    ids = np.asarray(df.video_id, dtype=object)
    same = ids[:, None] == ids[None, :]
    iu = np.triu_indices(len(df), 1)
    out["window_sim_same_video"] = float(sim[iu][same[iu]].mean())
    out["window_sim_other_video"] = float(sim[iu][~same[iu]].mean())

    zs = df.groupby("video_id")[[f"zs_{c}" for c in CLASS_NAMES]].mean()
    zs_pred = zs.idxmax(axis=1).str[3:]
    out["zero_shot_video_accuracy"] = float((zs_pred == vids.label.loc[zs.index]).mean())
    out["class_counts"] = counts.to_dict()
    out["missing_classes"] = [c for c in CLASS_NAMES if counts[c] == 0]
    out["n_subtitled"] = int(df.groupby("video_id")["meta_has_subs"].first().sum())

    _write_md(out, vids)
    log(f"EDA figures in {config.FIGURES_DIR}, summary in docs/eda.md")
    return out


def _write_md(out: dict, vids: pd.DataFrame) -> None:
    lines = [
        "# EDA summary",
        "",
        f"- **{len(vids)} videos**, {out['n_subtitled']} with subtitles. Classes with no examples: "
        f"{', '.join(out['missing_classes']) or 'none'} (covered only by zero-shot prompts).",
        f"- **Imbalance:** largest/smallest class = {out['imbalance_ratio']:.1f}x, so the headline metric is "
        "macro-F1 and classifiers use balanced class weights.",
        f"- **Window correlation:** windows of the same video have mean embedding cosine "
        f"{out['window_sim_same_video']:.2f} vs {out['window_sim_other_video']:.2f} across videos, so the 4 windows "
        "per video are far from independent; CV groups them by video to avoid leakage.",
        f"- **Zero-shot alone** gets {out['zero_shot_video_accuracy']:.0%} video accuracy, the bar a trained model must beat.",
        "- **Most separating scalar features (ANOVA F):** "
        + ", ".join(f"`{k}` {v}" for k, v in list(out["top_features"].items())[:8]) + ".",
        "",
        "![class balance](figures/class_balance.png)",
        "![features by class](figures/features_by_class.png)",
        "![separability](figures/feature_separability.png)",
        "![embedding](figures/embedding_pca.png)",
        "![duration](figures/duration_by_class.png)",
        "",
    ]
    (config.REPO_ROOT / "docs" / "eda.md").write_text("\n".join(lines))
