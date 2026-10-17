import json
from pathlib import Path

import pytest

from vclf import taxonomy
from vclf.labeling.draft import (
    LABEL_SCHEMA,
    LabelContext,
    build_prompt,
    parse_claude_output,
    read_rows,
    write_rows,
)
from vclf.labeling.review import build_review_page


def _ctx(**kw) -> LabelContext:
    base = dict(
        video_id="abc", sheet_path=Path("/tmp/abc.jpg"), duration_s=120.0, width=1280, height=720,
        fps=25.0, has_audio=True, has_subtitles=True, speech_coverage=0.8, words_per_minute=150.0,
        excerpts=["hello there"], domain="Knowledge", sub_category="Geography",
    )
    base.update(kw)
    return LabelContext(**base)


def test_taxonomy_is_consistent():
    assert len(set(taxonomy.CLASS_NAMES)) == len(taxonomy.CLASS_NAMES) == 8
    for c in taxonomy.CLASSES:
        assert c.definition and c.chunking and len(c.clip_prompts) >= 2
    assert LABEL_SCHEMA["properties"]["label"]["enum"] == list(taxonomy.CLASS_NAMES)


def test_prompt_mentions_every_class_and_the_sheet():
    prompt = build_prompt(_ctx())
    for name in taxonomy.CLASS_NAMES:
        assert name in prompt
    assert "/tmp/abc.jpg" in prompt and "80%" in prompt


def test_prompt_without_subtitles():
    assert "no subtitles available" in build_prompt(_ctx(excerpts=[], has_subtitles=False))


def _envelope(**kw) -> str:
    return json.dumps({"type": "result", "is_error": False, "result": "", **kw})


def test_parse_structured_output():
    out = parse_claude_output(_envelope(structured_output={
        "label": "gameplay", "secondary": "nonsense", "confidence": 1.4, "rationale": "HUD visible",
    }))
    assert out["label"] == "gameplay"
    assert out["secondary"] == "none"
    assert out["confidence"] == 1.0


def test_parse_falls_back_to_json_in_text():
    text = 'Sure: {"label": "static_camera", "secondary": "none", "confidence": 0.7, "rationale": "fixed"}'
    assert parse_claude_output(_envelope(result=text))["label"] == "static_camera"


def test_parse_rejects_errors_and_bad_labels():
    with pytest.raises(RuntimeError):
        parse_claude_output(json.dumps({"is_error": True, "result": "auth expired"}))
    with pytest.raises(ValueError):
        parse_claude_output(_envelope(structured_output={"label": "podcast", "secondary": "none",
                                                         "confidence": 0.5, "rationale": ""}))


def test_csv_roundtrip(tmp_path):
    rows = {"b": {"video_id": "b", "label": "gameplay", "confidence": "0.9"},
            "a": {"video_id": "a", "label": "talking_head", "confidence": "0.5"}}
    path = tmp_path / "labels.csv"
    write_rows(path, rows)
    back = read_rows(path)
    assert list(back) == ["a", "b"] and back["b"]["label"] == "gameplay"


def test_review_page_embeds_rows_and_relative_sheet_paths(tmp_path):
    rows = {"vid1": {"video_id": "vid1", "label": "gameplay", "secondary": "none", "confidence": "0.4",
                     "rationale": "a </script> tag", "domain": "d", "sub_category": "s", "duration_s": "10"}}
    out = build_review_page(rows, {"vid1": ["x"]}, tmp_path / "contact_sheets", tmp_path / "review.html")
    page = out.read_text()
    assert "contact_sheets/vid1.jpg" in page
    assert "</script> tag" not in page
    assert "<title>Label Review</title>" in page
