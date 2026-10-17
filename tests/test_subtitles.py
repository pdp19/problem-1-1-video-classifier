from vclf import subtitles

SRT = """1
00:00:01,000 --> 00:00:03,000
Hello <i>world</i>

2
00:00:02,500 --> 00:00:05,000
overlapping cue here

3
00:00:08,000 --> 00:00:09,500
{\\an8}last words
"""


def test_parse_srt_strips_tags_and_reads_times():
    cues = subtitles.parse_srt(SRT)
    assert [c.text for c in cues] == ["Hello world", "overlapping cue here", "last words"]
    assert cues[0].start == 1.0 and cues[0].end == 3.0
    assert cues[2].end == 9.5


def test_parse_srt_handles_crlf_and_missing_index():
    text = "00:00:00,000 --> 00:00:01,000\r\nno index\r\n\r\n00:00:01,000 --> 00:00:02,000\r\nsecond\r\n"
    assert [c.text for c in subtitles.parse_srt(text)] == ["no index", "second"]


def test_speech_coverage_merges_overlaps():
    cues = subtitles.parse_srt(SRT)
    assert abs(subtitles.speech_coverage(cues, 10.0) - 0.55) < 1e-9


def test_speech_coverage_clips_to_duration_and_handles_empty():
    cues = subtitles.parse_srt(SRT)
    assert subtitles.speech_coverage(cues, 4.0) == 0.75
    assert subtitles.speech_coverage([], 10.0) == 0.0
    assert subtitles.speech_coverage(cues, 0.0) == 0.0


def test_words_per_minute():
    cues = subtitles.parse_srt(SRT)
    assert subtitles.words_per_minute(cues, 60.0) == 7.0


def test_excerpts_respect_positions_and_length():
    cues = subtitles.parse_srt(SRT)
    out = subtitles.excerpts(cues, 10.0, positions=(0.0, 0.85), max_words=3)
    assert out == ["Hello world overlapping", "last words"]
