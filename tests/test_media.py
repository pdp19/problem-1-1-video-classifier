from conftest import needs_ffmpeg

from vclf.frames import grab_frames, sample_times
from vclf.labeling.contact_sheet import GRID, HEADER_H, TILE_H, TILE_W, build_contact_sheet
from vclf.probe import probe


def test_sample_times_are_slice_centres():
    assert sample_times(10.0, 4) == [1.25, 3.75, 6.25, 8.75]
    assert sample_times(0.0, 4) == []
    assert sample_times(10.0, 0) == []


@needs_ffmpeg
def test_probe_reads_streams(synthetic_clip):
    meta = probe(synthetic_clip)
    assert (meta.width, meta.height) == (320, 240)
    assert abs(meta.duration_s - 6.0) < 0.2
    assert abs(meta.fps - 25.0) < 0.01
    assert meta.has_audio and meta.audio_channels >= 1
    assert meta.video_codec == "h264"


@needs_ffmpeg
def test_grab_frames_letterboxes_to_requested_size(synthetic_clip):
    frames = grab_frames(synthetic_clip, [1.0, 3.0], 160, 160)
    assert all(f is not None and f.size == (160, 160) for f in frames)


@needs_ffmpeg
def test_grab_frame_past_end_returns_none(synthetic_clip):
    assert grab_frames(synthetic_clip, [60.0], 64, 64) == [None]


@needs_ffmpeg
def test_contact_sheet_geometry(synthetic_clip):
    sheet = build_contact_sheet(synthetic_clip, 6.0, "clip")
    assert sheet.size == (TILE_W * GRID, TILE_H * GRID + HEADER_H)
