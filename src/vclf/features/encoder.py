from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config, taxonomy

MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"
INPUT_SIZE = 256

MODELS_DIR = config.ARTIFACTS_DIR / "models"
ONNX_PATH = MODELS_DIR / "mobileclip2_s0_image.onnx"
TEXT_EMB_PATH = MODELS_DIR / "class_text_embeddings.npy"
META_PATH = MODELS_DIR / "encoder_meta.json"
VAD_PATH = MODELS_DIR / "silero_vad.onnx"


def is_exported(models_dir: Path = MODELS_DIR) -> bool:
    return all((models_dir / p.name).exists() for p in (ONNX_PATH, TEXT_EMB_PATH, META_PATH, VAD_PATH))


def export(models_dir: Path = MODELS_DIR, force: bool = False) -> Path:
    if is_exported(models_dir) and not force:
        return models_dir
    import open_clip
    import silero_vad
    import torch
    from timm.utils.model import reparameterize_model

    models_dir.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = reparameterize_model(model.eval())
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    norm = next(t for t in preprocess.transforms if t.__class__.__name__ == "Normalize")
    meta = {
        "model": MODEL_NAME, "pretrained": PRETRAINED, "input_size": INPUT_SIZE,
        "mean": [float(v) for v in norm.mean], "std": [float(v) for v in norm.std],
        "classes": list(taxonomy.CLASS_NAMES),
        "logit_scale": float(model.logit_scale.exp().item()),
    }

    class ImageTower(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.encode_image(x)

    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        ImageTower(model), dummy, str(models_dir / ONNX_PATH.name),
        input_names=["pixels"], output_names=["embedding"],
        dynamic_axes={"pixels": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    with torch.inference_mode():
        rows = []
        for c in taxonomy.CLASSES:
            e = model.encode_text(tokenizer(list(c.clip_prompts)))
            e = e / e.norm(dim=-1, keepdim=True)
            rows.append(e.mean(0) / e.mean(0).norm())
    np.save(models_dir / TEXT_EMB_PATH.name, torch.stack(rows).numpy().astype(np.float32))

    vad_src = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
    shutil.copyfile(vad_src, models_dir / VAD_PATH.name)
    (models_dir / META_PATH.name).write_text(json.dumps(meta, indent=2))
    return models_dir


class Encoder:
    def __init__(self, models_dir: Path = MODELS_DIR, threads: int = 4):
        import onnxruntime as ort

        if not is_exported(models_dir):
            raise FileNotFoundError(f"encoder not exported to {models_dir}; run `python src/main.py export-encoder`")
        meta = json.loads((models_dir / META_PATH.name).read_text())
        if meta["classes"] != list(taxonomy.CLASS_NAMES):
            raise RuntimeError("taxonomy changed since export; re-run export-encoder --force")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(models_dir / ONNX_PATH.name), opts, providers=["CPUExecutionProvider"])
        self.mean = np.asarray(meta["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(meta["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.size = int(meta["input_size"])
        self.logit_scale = float(meta["logit_scale"])
        self.text = np.load(models_dir / TEXT_EMB_PATH.name)

    def preprocess(self, frames: list[np.ndarray]) -> np.ndarray:
        x = np.stack(frames).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        return (x - self.mean) / self.std

    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, self.text.shape[1]), dtype=np.float32)
        emb = self.session.run(None, {"pixels": self.preprocess(frames)})[0]
        return emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)

    def zero_shot(self, emb: np.ndarray) -> np.ndarray:
        logits = self.logit_scale * emb @ self.text.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


@lru_cache(maxsize=1)
def get_encoder() -> Encoder:
    return Encoder()
