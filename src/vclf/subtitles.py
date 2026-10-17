from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TIME = r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
_CUE_TIME_RE = re.compile(_TIME + r"\s*-->\s*" + _TIME)
_TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            m = _CUE_TIME_RE.search(line)
            if m:
                start = _to_seconds(*m.groups()[:4])
                end = _to_seconds(*m.groups()[4:])
                body = _TAG_RE.sub("", " ".join(lines[i + 1:])).strip()
                if end > start and body:
                    cues.append(Cue(start, end, body))
                break
    return cues


def load_cues(path: Path) -> list[Cue]:
    if not path.exists():
        return []
    return parse_srt(path.read_text(encoding="utf-8", errors="replace"))


def speech_coverage(cues: list[Cue], duration_s: float) -> float:
    if duration_s <= 0 or not cues:
        return 0.0
    covered = 0.0
    cur_start, cur_end = None, None
    for c in sorted(cues, key=lambda c: c.start):
        start, end = max(0.0, c.start), min(duration_s, c.end)
        if end <= start:
            continue
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                covered += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        covered += cur_end - cur_start
    return min(1.0, covered / duration_s)


def words_per_minute(cues: list[Cue], duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    words = sum(len(c.text.split()) for c in cues)
    return words / (duration_s / 60.0)


def excerpts(cues: list[Cue], duration_s: float, positions=(0.1, 0.5, 0.9), max_words: int = 45) -> list[str]:
    out = []
    for pos in positions:
        t = pos * duration_s
        after = [c for c in cues if c.end >= t]
        words: list[str] = []
        for c in after:
            words.extend(c.text.split())
            if len(words) >= max_words:
                break
        if words:
            out.append(" ".join(words[:max_words]))
    return out
