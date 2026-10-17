from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import config, dataset


def _records(args) -> list[dataset.VideoRecord]:
    recs = dataset.load_records(args.data_dir)
    if args.ids:
        wanted = set(args.ids.split(","))
        recs = [r for r in recs if r.video_id in wanted]
    if args.limit:
        recs = recs[: args.limit]
    missing = [r.video_id for r in recs if not r.path.exists()]
    if missing:
        sys.exit(f"{len(missing)} video files missing (e.g. {missing[0]}). Run `python src/main.py download`.")
    return recs


def cmd_download(args) -> None:
    path = dataset.download(args.data_dir)
    print(f"dataset ready at {path}")


def cmd_contact_sheets(args) -> None:
    from .labeling.pipeline import make_contexts

    recs = _records(args)
    t0 = time.perf_counter()
    make_contexts(recs, force=args.force, workers=args.workers)
    print(f"{len(recs)} contact sheets in {config.CONTACT_SHEETS_DIR} ({time.perf_counter() - t0:.1f}s)")


def cmd_draft_labels(args) -> None:
    from .labeling.draft import label_many
    from .labeling.pipeline import make_contexts

    recs = _records(args)
    contexts = make_contexts(recs, workers=args.workers)
    rows = label_many(contexts, config.DRAFT_LABELS_PATH, model=args.model, workers=args.workers)
    print(f"{len(rows)} draft labels in {config.DRAFT_LABELS_PATH}")


def cmd_review_page(args) -> None:
    from .labeling.draft import read_rows
    from .labeling.pipeline import make_contexts
    from .labeling.review import build_review_page

    rows = read_rows(config.DRAFT_LABELS_PATH)
    if not rows:
        sys.exit(f"no drafts at {config.DRAFT_LABELS_PATH}; run `draft-labels` first")
    recs = [r for r in dataset.load_records(args.data_dir) if r.video_id in rows]
    excerpts = {c.video_id: c.excerpts for c in make_contexts(recs, workers=args.workers)}
    out = build_review_page(rows, excerpts, config.CONTACT_SHEETS_DIR, config.ARTIFACTS_DIR / "review.html")
    print(f"review page: {out}")


def cmd_export_encoder(args) -> None:
    from .features.encoder import export

    t0 = time.perf_counter()
    path = export(force=args.force)
    print(f"encoder + VAD ready in {path} ({time.perf_counter() - t0:.1f}s)")


def cmd_features(args) -> None:
    from .features.extract import extract_many

    recs = _records(args)
    t0 = time.perf_counter()
    feats, timings = extract_many(recs, workers=args.workers)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(config.FEATURES_PATH, index=False)
    timings.to_csv(config.TIMINGS_PATH, index=False)
    wall = time.perf_counter() - t0
    print(f"{len(feats)} window rows from {len(recs)} videos -> {config.FEATURES_PATH} ({wall:.1f}s wall)")
    print(timings.drop(columns=["video_id"]).median().round(1).to_string())


def cmd_eda(args) -> None:
    from .eval.eda import run

    out = run()
    print(f"imbalance {out['imbalance_ratio']:.1f}x, zero-shot video accuracy {out['zero_shot_video_accuracy']:.0%}")


def cmd_train(args) -> None:
    from .models.train import run

    run(n_repeats=args.repeats)


def cmd_report(args) -> None:
    from .eval.report import run

    run()


def cmd_predict(args) -> None:
    import json

    from .models.predict import load_model, predict_file

    bundle = load_model()
    for p in args.videos:
        print(json.dumps(predict_file(Path(p), bundle), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vclf",
        description="Lightweight video classifier that routes each video to a chunking strategy.",
    )
    p.add_argument("--data-dir", type=Path, default=None, help=f"dataset root (default {config.DATA_DIR})")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download the VideoMME 10%% sample (skips complete files)").set_defaults(func=cmd_download)

    def with_selection(sp):
        sp.add_argument("--ids", help="comma-separated video ids to process")
        sp.add_argument("--limit", type=int, help="process only the first N videos")
        sp.add_argument("--workers", type=int, default=4)
        return sp

    sp = with_selection(sub.add_parser("contact-sheets", help="render 4x4 timestamped contact sheets"))
    sp.add_argument("--force", action="store_true", help="re-render existing sheets")
    sp.set_defaults(func=cmd_contact_sheets)

    sp = with_selection(sub.add_parser("draft-labels", help="teacher-label videos with `claude -p` (resumable)"))
    sp.add_argument("--model", default="sonnet", help="claude model alias for the teacher")
    sp.set_defaults(func=cmd_draft_labels)

    sp = with_selection(sub.add_parser("review-page", help="build artifacts/review.html for human review"))
    sp.set_defaults(func=cmd_review_page)

    sp = sub.add_parser("export-encoder", help="export MobileCLIP2-S0 to ONNX + class text embeddings + VAD (once)")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_export_encoder)

    sp = with_selection(sub.add_parser("features", help="extract per-window features for every video"))
    sp.set_defaults(func=cmd_features)

    sub.add_parser("eda", help="class balance + feature analysis -> docs/figures, docs/eda.md").set_defaults(func=cmd_eda)

    sp = sub.add_parser("train", help="tune, cross-validate all variants, score the locked test split, save model")
    sp.add_argument("--repeats", type=int, default=5, help="CV repeats (5 folds each)")
    sp.set_defaults(func=cmd_train)

    sub.add_parser("report", help="results figures + docs/results.md from the last train run").set_defaults(func=cmd_report)

    sp = sub.add_parser("predict", help="classify video files; prints JSON with route and latency")
    sp.add_argument("videos", nargs="+")
    sp.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
