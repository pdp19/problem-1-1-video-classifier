# Video classifier for chunking-strategy routing

Sorts incoming videos into **8 classes defined by how they should be chunked**
(not by topic), so a video index can apply the right chunker to each one.
Runs on a CPU at **~1.3 s per video regardless of length** (median, 1 worker),
with a 1 MB classifier on top of an 11M-parameter image encoder served through
ONNX Runtime.

| class | chunking strategy it triggers |
|---|---|
| `talking_head` | transcript topic segmentation |
| `screen_content` | slide/screen-change detection |
| `edited_narrative` | shots → scenes |
| `sports_broadcast` | plays/events (scoreboard, replays, crowd audio) |
| `egocentric_continuous` | activity-drift sliding windows |
| `static_camera` | motion-triggered events, idle dropped |
| `stage_performance` | audio structure (songs/acts), then shots |
| `gameplay` | game-state / HUD changes |

## Results on the VideoMME 10% sample (90 videos, 7 classes present)

| | macro-F1 [95% CI] | accuracy | top-2 acc. | routing regret |
|---|---|---|---|---|
| **locked test split** (23 videos, scored once) | 0.64 [0.38, 0.89] | 0.74 | 0.96 | 0.11 |
| repeated 5x5 CV, all 90 videos | 0.79 [0.64, 0.88] | 0.82 | 0.93 | 0.08 |
| zero-shot MobileCLIP only (no training) | 0.62 | | | |

Full tables, every variant and the latency breakdown: [docs/results.md](docs/results.md).
Data analysis: [docs/eda.md](docs/eda.md).

## Run it

```bash
pip install -r requirements.txt
python src/main.py download          # 11.2 GB, skips complete files
python src/main.py export-encoder    # once: MobileCLIP2-S0 -> ONNX + Silero VAD
python src/main.py features --workers 4
python src/main.py eda
python src/main.py train
python src/main.py report
python src/main.py predict path/to/video.mp4
python -m pytest -q
```

`run.yaml` has the exact commands the evaluator runs. CPU only; ffmpeg is
used if installed, otherwise downloaded automatically by `static-ffmpeg`.

## How it works

1. **Labels.** 90 videos labelled by a teacher model (Claude reading a
   16-frame contact sheet + subtitle excerpts, `draft-labels`), then reviewed by a
   human in a generated review page (`review-page`). Gold labels:
   [labels/labels.csv](labels/labels.csv).
2. **Features, cheapest first.** Each video is split into 4 windows:
   - container metadata (ffprobe, ~30 ms)
   - **packet sizes and keyframe flags without decoding** (motion, cut and static-scene proxies, ~80 ms)
   - subtitle speech coverage and pace
   - 4 x 8 s audio clips: Silero VAD speech ratio + loudness/spectral stats (~230 ms)
   - 16 seek-decoded frames: pixel statistics + MobileCLIP2-S0 embedding and zero-shot class scores (~900 ms)
3. **Model.** Soft-voting ensemble of a logistic regression (all features, embedding
   PCA-compressed) and gradient-boosted trees (scalar features), balanced class
   weights, trained on window rows, averaged per video. It was selected from 10
   variants by the one-standard-error rule on grouped CV over the training split.
4. **Routing.** `predict` returns the class, confidence, chunking strategy and
   per-stage latency; below 0.5 confidence it routes to a generic fallback chunker.

## Layout

```
src/vclf/taxonomy.py     classes, tie-break rules, CLIP prompts (single source of truth)
src/vclf/labeling/       contact sheets, Claude teacher labeller, review page
src/vclf/features/       metadata, packets, audio, visual, ONNX encoder, extraction
src/vclf/models/         variants, training protocol, prediction
src/vclf/eval/           metrics + bootstrap CIs, routing regret, EDA, report
labels/                  reviewed labels, teacher drafts, locked test split
tests/                   unit tests on synthetic ffmpeg clips
docs/                    write-up, EDA, results, figures
```
