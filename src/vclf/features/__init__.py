from __future__ import annotations

from ..taxonomy import CLASS_NAMES
from .audio import AUDIO_FEATURES
from .packets import PACKET_FEATURES
from .visual import LOWVIS_FEATURES

EMB_DIM = 512

META_FEATURES = (
    "meta_log_duration", "meta_fps", "meta_aspect", "meta_height", "meta_log_bpp",
    "meta_has_audio", "meta_has_subs", "vid_emb_div",
)
SUB_FEATURES = ("sub_speech_cov", "sub_wpm")

GROUPS: dict[str, tuple[str, ...]] = {
    "meta": tuple(f for f in META_FEATURES if f != "vid_emb_div"),
    "packet": PACKET_FEATURES,
    "subs": SUB_FEATURES,
    "audio": AUDIO_FEATURES,
    "lowvis": (*LOWVIS_FEATURES, "emb_div", "vid_emb_div"),
    "zs": tuple(f"zs_{c}" for c in CLASS_NAMES),
    "emb": tuple(f"emb_{i:03d}" for i in range(EMB_DIM)),
}


def columns(*groups: str) -> list[str]:
    cols: list[str] = []
    for g in groups:
        cols.extend(c for c in GROUPS[g] if c not in cols)
    return cols
