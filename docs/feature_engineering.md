# Feature engineering

The rule I followed: compute the cheap signals first, and only decode pixels for the things that can't be read any other way. Every feature also has to say something about how a video should be chunked, not what it is about. A cooking show and a cooking vlog have the same topic but need different chunkers.

## Windows

Each video is cut into 4 equal windows and every window gets its own feature row. That gives 360 training rows instead of 90, and a window can disagree with the rest of the video (a news show that cuts to field footage). At prediction time the video's class is the average of its 4 window probabilities.

The rows are not independent, though. Windows from the same video have a mean embedding similarity of 0.79, against 0.41 between different videos. So cross-validation always keeps the 4 windows of a video in the same fold, otherwise the scores would be inflated.

## The groups, cheapest first

Times are medians per video on one M3 CPU worker.

| Group | What I measure | Why it matters for chunking | Cost |
|---|---|---|---|
| Container metadata (7) | duration, fps, aspect ratio, height, bits per pixel, audio and subtitle presence | Screen recordings and games have tell-tale fps, aspect and bitrate | 28 ms |
| Packet table (9) | per window: keyframes per minute and how irregular they are, P-frame size in bits per pixel, its spread and burstiness, share of near-empty P-frames, unkeyed size spikes per minute, keyframe/P-frame size ratio | The encoder already did the motion analysis. A P-frame's size is roughly how much the picture changed, and encoders insert extra keyframes at scene cuts. Static scenes give tiny P-frames, sports give big steady ones, edited video gives frequent cuts | 79 ms (396 ms p95, it grows with length) |
| Subtitles (2) | share of the window covered by speech cues, words per minute | Talking heads and lectures are wall-to-wall speech | ~0 ms |
| Audio (7) | one 8 s clip per window: Silero VAD speech fraction, loudness mean and spread, silence share, spectral flatness, zero-crossing rate, spectral centroid | Separates speech, music, crowd noise and silence, which lines up with podcasts, concerts, sports and CCTV | 232 ms |
| Pixel statistics (10) | on 16 sampled frames: brightness, saturation, colourfulness, edge density, flat-area share, dark frames, frame-to-frame change, near-duplicate share, embedding diversity within the window and across the video | Slides and UI have large flat areas; a fixed camera gives near-duplicate frames; edited video has high diversity | 29 ms (after decode) |
| Zero-shot scores (8) | MobileCLIP2-S0 similarity between the window's mean frame embedding and 3 text prompts per class | Visual semantics: HUDs, scoreboards, stages, slides | shared with below |
| Image embedding (512) | mean MobileCLIP2-S0 embedding of the window's 4 frames, compressed to 4 dimensions by PCA inside the model | Everything the hand-made features miss | 471 ms decode + 439 ms encode |

A few details that mattered:

- Frames are seeked, not decoded in sequence. Each of the 16 frames is a separate input seek, so ffmpeg jumps to the nearest keyframe. The cost stays almost flat with video length: a 59-minute video costs about the same as a 1-minute one.
- The encoder runs in ONNX Runtime. Eager PyTorch took 160 ms per frame on CPU. The exported ONNX graph takes 20–30 ms depending on thread count, gives identical embeddings, and means torch isn't needed at inference time.
- Letterboxing is trimmed before pixel statistics. Frames are padded to a square for the encoder, and the black bars were counting as "flat area", which would have made every 16:9 video look a bit like a slide deck.
- The VAD runs straight from its ONNX file with a small numpy loop, again so inference needs no torch.

## What the data said

The separability check ([eda.md](eda.md)) ranked the zero-shot scores and embedding diversity highest, with speech coverage, VAD speech fraction and the packet motion features next. The class-by-class plots show the expected pattern: speech features pick out talking heads and screen content, near-empty P-frames and bits per pixel pick out sports, flat area picks out slides.

Then the ablations, all on the same folds (macro-F1 on the training split):

| Features | Macro-F1 |
|---|---|
| cheap tier only (metadata, packets, subtitles, audio) | 0.34 |
| zero-shot prompts only, no training | 0.62 |
| frames only (pixel stats, zero-shot, embedding) | 0.67 |
| everything | 0.73 |
| everything, LR and boosted trees averaged | 0.76 |

So the cheap tier on its own is not good enough to route videos, but it adds about 6 points on top of the visual features. Per class, almost all of that comes from `egocentric_continuous`, which goes from 0.33 to 0.73 F1: handheld footage looks like ordinary edited video in single frames, and it is the camera motion in the packet table that gives it away. `talking_head` gains a little (0.67 to 0.73), and `screen_content` actually drops slightly (0.47 to 0.40), which is the class I'd target next. I had hoped the cheap tier would handle most videos alone and let a cascade skip the frames. It doesn't: at the threshold where quality holds, only 13–21% of videos can skip frame decoding.

## Things I would add next

- Read packet sizes and keyframe flags straight from the MP4 `moov` index (`stsz`/`stss` tables) instead of letting ffprobe demux the file. That turns the one O(duration) step into a fixed few-hundred-KB read.
- An OCR text-density score per frame, to separate slides from other flat graphics (`screen_content` is the weakest class in this ablation).
- A face-count feature for `talking_head`, which still loses half its videos to `edited_narrative`.
