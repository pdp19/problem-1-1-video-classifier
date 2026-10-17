import numpy as np
from conftest import needs_ffmpeg

from vclf import features
from vclf.features.audio import SR, audio_features, decode_clip
from vclf.features.packets import PACKET_FEATURES, PacketTable, packet_features, read_packets
from vclf.features.visual import LOWVIS_FEATURES, embedding_diversity, lowvis_features, trim_letterbox


def test_feature_groups_are_disjoint_except_shared_video_diversity():
    seen = {}
    for g, cols in features.GROUPS.items():
        for c in cols:
            assert c not in seen or c == "vid_emb_div", f"{c} in {seen.get(c)} and {g}"
            seen[c] = g
    assert len(features.GROUPS["emb"]) == features.EMB_DIM


def _table(sizes, keys, fps=25.0):
    n = len(sizes)
    return PacketTable(np.arange(n) / fps, np.asarray(sizes, float), np.asarray(keys, bool))


def test_packet_features_static_vs_busy():
    n = 25 * 60
    keys = [i % 250 == 0 for i in range(n)]
    static = packet_features(_table([20_000 if k else 50 for k in keys], keys), 1280 * 720, 60.0)
    busy = packet_features(_table([60_000 if k else 15_000 for k in keys], keys), 1280 * 720, 60.0)
    assert static["pk_tiny_p_frac"] == 1.0 and busy["pk_tiny_p_frac"] == 0.0
    assert busy["pk_p_bpp_log"] > static["pk_p_bpp_log"]
    assert static["pk_key_p_ratio_log"] > busy["pk_key_p_ratio_log"]
    assert abs(static["pk_kf_per_min"] - 6.0) < 1e-9


def test_packet_features_count_unkeyed_cuts():
    n = 25 * 60
    sizes = [2_000.0] * n
    for i in range(100, n, 150):
        sizes[i] = 40_000.0
    out = packet_features(_table(sizes, [i == 0 for i in range(n)]), 640 * 360, 60.0)
    assert abs(out["pk_spike_per_min"] - 10.0) < 1e-9


def test_packet_features_empty_window_is_all_zero():
    out = packet_features(_table([], []), 100, 10.0)
    assert set(out) == set(PACKET_FEATURES) and not any(out.values())


@needs_ffmpeg
def test_read_packets_sorted_with_keyframes(synthetic_clip):
    tab = read_packets(synthetic_clip)
    assert len(tab.pts) == 150 and tab.key[0] and np.all(np.diff(tab.pts) > 0)
    assert len(tab.window(0, 2).pts) == 50


def test_audio_features_silence_vs_tone():
    silence = audio_features(np.zeros(SR * 2, dtype=np.float32))
    t = np.arange(SR * 2) / SR
    tone = audio_features((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    assert silence["au_silence_frac"] == 1.0 and tone["au_silence_frac"] == 0.0
    assert tone["au_rms_db"] > silence["au_rms_db"]
    assert tone["au_flatness"] < 0.1
    assert abs(tone["au_centroid_khz"] - 0.44) < 0.15
    assert audio_features(np.zeros(10, dtype=np.float32))["au_rms_db"] == -90.0


@needs_ffmpeg
def test_decode_clip_reads_16k_mono(synthetic_clip):
    audio = decode_clip(synthetic_clip, 1.0, 2.0)
    assert abs(len(audio) - 2 * SR) < SR * 0.05 and np.abs(audio).max() > 0.05


def test_trim_letterbox_and_lowvis():
    img = np.zeros((256, 256, 3), np.uint8)
    img[56:200] = 128
    assert trim_letterbox(img).shape == (144, 256, 3)
    out = lowvis_features([img, img])
    assert set(out) == set(LOWVIS_FEATURES)
    assert out["lv_flat_frac"] == 1.0
    assert out["lv_dup_frac"] == 1.0 and out["lv_frame_diff"] == 0.0


def test_embedding_diversity():
    same = np.tile(np.array([[1.0, 0.0]]), (4, 1))
    ortho = np.eye(2)
    assert embedding_diversity(same) == 0.0
    assert abs(embedding_diversity(ortho) - 1.0) < 1e-9
    assert embedding_diversity(same[:1]) == 0.0
