from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..ffmpeg_bin import ensure as ensure_ffmpeg

SR = 16000
AUDIO_FEATURES = (
    "au_speech_frac", "au_rms_db", "au_rms_db_std", "au_silence_frac",
    "au_flatness", "au_zcr", "au_centroid_khz",
)


def decode_clip(path: str | Path, start: float, seconds: float) -> np.ndarray:
    ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{seconds:.3f}",
        "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


class SileroVad:
    CHUNK, CONTEXT = 512, 64

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio) // self.CHUNK
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, self.CONTEXT), dtype=np.float32)
        sr = np.array(SR, dtype=np.int64)
        probs = np.empty(n, dtype=np.float32)
        for i in range(n):
            chunk = audio[i * self.CHUNK:(i + 1) * self.CHUNK].reshape(1, -1).astype(np.float32)
            x = np.concatenate([context, chunk], axis=1)
            out, state = self.session.run(None, {"input": x, "state": state, "sr": sr})
            probs[i] = float(out.reshape(-1)[0])
            context = x[:, -self.CONTEXT:]
        return probs


@lru_cache(maxsize=1)
def get_vad() -> SileroVad:
    from .encoder import MODELS_DIR, VAD_PATH

    return SileroVad(MODELS_DIR / VAD_PATH.name)


def audio_features(audio: np.ndarray, vad: SileroVad | None = None) -> dict[str, float]:
    out = dict.fromkeys(AUDIO_FEATURES, 0.0)
    if len(audio) < SR // 2:
        out["au_rms_db"] = -90.0
        out["au_silence_frac"] = 1.0
        return out
    frame = 400
    n = len(audio) // frame
    frames = audio[: n * frame].reshape(n, frame)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-5)
    out["au_rms_db"] = float(rms_db.mean())
    out["au_rms_db_std"] = float(rms_db.std())
    out["au_silence_frac"] = float((rms_db < -45).mean())
    out["au_zcr"] = float((np.abs(np.diff(np.sign(frames), axis=1)) > 0).mean())
    spec = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1)) + 1e-10
    out["au_flatness"] = float((np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)).mean())
    freqs = np.fft.rfftfreq(frame, 1 / SR)
    out["au_centroid_khz"] = float(((spec * freqs).sum(axis=1) / spec.sum(axis=1)).mean() / 1000)
    if vad is not None:
        out["au_speech_frac"] = float((vad.probabilities(audio) > 0.5).mean())
    return out
