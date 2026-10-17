from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .. import taxonomy

DRAFT_FIELDS = (
    "video_id", "label", "secondary", "confidence", "rationale",
    "domain", "sub_category", "duration_s", "teacher_model",
)

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(taxonomy.CLASS_NAMES)},
        "secondary": {"type": "string", "enum": [*taxonomy.CLASS_NAMES, "none"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 300},
    },
    "required": ["label", "secondary", "confidence", "rationale"],
    "additionalProperties": False,
}


@dataclass
class LabelContext:
    video_id: str
    sheet_path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    has_subtitles: bool
    speech_coverage: float
    words_per_minute: float
    excerpts: list[str]
    domain: str
    sub_category: str


def build_prompt(ctx: LabelContext) -> str:
    excerpt_text = "\n".join(f'  - "{e}"' for e in ctx.excerpts) or "  (no subtitles available)"
    return f"""You are labelling videos for a video-indexing pipeline. Each video must be assigned the
class whose CHUNKING STRATEGY suits it best. Classes describe how the footage is structured
(camera, editing, what carries the information), not the topic.

Classes:
{taxonomy.describe_for_prompt()}

Video {ctx.video_id}:
- Contact sheet (16 evenly spaced frames, timestamps in yellow): {ctx.sheet_path}
  Open it with the Read tool before answering.
- Duration {ctx.duration_s:.0f} s, {ctx.width}x{ctx.height} @ {ctx.fps:g} fps, audio: {"yes" if ctx.has_audio else "no"}
- Subtitles: {"yes" if ctx.has_subtitles else "no"}; speech covers {ctx.speech_coverage:.0%} of the running time; {ctx.words_per_minute:.0f} words/min
- Transcript excerpts (10%, 50%, 90% of the video):
{excerpt_text}
- Topic tags from the source benchmark (hint only, topic is NOT the class): {ctx.domain} / {ctx.sub_category}

Judge editing from the frames: many different shots across tiles suggests cuts; near-identical
tiles suggest a static camera; tilted or moving viewpoints suggest handheld footage.
Return the primary class, a secondary class (or "none"), your confidence 0-1 that the primary
class is right, and a one-sentence rationale citing what you saw."""


def _claude_cmd(prompt: str, sheet_dir: Path, model: str) -> list[str]:
    return [
        "claude", "-p", prompt,
        "--model", model,
        "--tools", "Read",
        "--allowedTools", "Read",
        "--add-dir", str(sheet_dir),
        "--output-format", "json",
        "--json-schema", json.dumps(LABEL_SCHEMA),
        "--no-session-persistence",
        "--strict-mcp-config",
    ]


def parse_claude_output(stdout: str) -> dict:
    envelope = json.loads(stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"claude returned an error: {envelope.get('result')}")
    payload = envelope.get("structured_output")
    if payload is None:
        text = envelope.get("result", "")
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in claude result: {text[:200]!r}")
        payload = json.loads(text[start:end + 1])
    if not taxonomy.is_valid(payload.get("label", "")):
        raise ValueError(f"invalid label {payload.get('label')!r}")
    if payload.get("secondary") not in (*taxonomy.CLASS_NAMES, "none"):
        payload["secondary"] = "none"
    payload["confidence"] = float(min(1.0, max(0.0, float(payload.get("confidence", 0)))))
    return payload


def _claude_env() -> dict[str, str]:
    env = dict(os.environ)
    if env.get("CLAUDE_KEY") and not env.get("CLAUDE_CODE_OAUTH_TOKEN"):
        env["CLAUDE_CODE_OAUTH_TOKEN"] = env["CLAUDE_KEY"]
    return env


def label_one(ctx: LabelContext, model: str, timeout_s: int = 300) -> dict:
    if shutil.which("claude") is None:
        raise RuntimeError("the `claude` CLI is not on PATH")
    cmd = _claude_cmd(build_prompt(ctx), ctx.sheet_path.parent, model)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=_claude_env())
    if proc.returncode != 0:
        detail = proc.stderr.strip()
        try:
            detail = json.loads(proc.stdout).get("result") or detail
        except (json.JSONDecodeError, AttributeError):
            detail = detail or proc.stdout.strip()
        raise RuntimeError(f"claude exited {proc.returncode}: {str(detail)[:300]}")
    payload = parse_claude_output(proc.stdout)
    return {
        "video_id": ctx.video_id,
        "label": payload["label"],
        "secondary": payload["secondary"],
        "confidence": round(payload["confidence"], 2),
        "rationale": " ".join(str(payload.get("rationale", "")).split()),
        "domain": ctx.domain,
        "sub_category": ctx.sub_category,
        "duration_s": round(ctx.duration_s, 1),
        "teacher_model": model,
    }


def read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {row["video_id"]: row for row in csv.DictReader(fh)}


def write_rows(path: Path, rows: dict[str, dict], fields=DRAFT_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for vid in sorted(rows):
            writer.writerow(rows[vid])


def label_many(
    contexts: list[LabelContext], out_path: Path, model: str, workers: int = 4, log=print
) -> dict[str, dict]:
    rows = read_rows(out_path)
    todo = [c for c in contexts if c.video_id not in rows]
    log(f"{len(rows)} already labelled, {len(todo)} to go (model={model}, workers={workers})")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(label_one, c, model): c for c in todo}
        for fut in as_completed(futures):
            ctx = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                log(f"  FAIL {ctx.video_id}: {exc}")
                continue
            rows[ctx.video_id] = row
            write_rows(out_path, rows)
            log(f"  {ctx.video_id}: {row['label']} ({row['confidence']}) {len(rows)}/{len(contexts)}")
    return rows
