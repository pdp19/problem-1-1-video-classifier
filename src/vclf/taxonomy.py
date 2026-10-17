from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoClass:
    name: str
    definition: str
    examples: str
    chunking: str
    clip_prompts: tuple[str, ...]


CLASSES: tuple[VideoClass, ...] = (
    VideoClass(
        name="talking_head",
        definition=(
            "Speech carries the information and the picture is mostly one or a few "
            "people talking to camera or to each other, with few visual changes."
        ),
        examples="podcast, interview, lecture without dominant slides, news anchor desk, sit-down vlog",
        chunking="segment the transcript on topic shifts; visual boundaries are secondary",
        clip_prompts=(
            "a person talking to the camera",
            "a podcast with people sitting at microphones",
            "a news anchor at a desk",
        ),
    ),
    VideoClass(
        name="screen_content",
        definition=(
            "On-screen content dominates: slides, screen recordings, whiteboards, "
            "documents or text-heavy graphics, possibly with a small presenter inset."
        ),
        examples="slide lecture, software tutorial, screencast, whiteboard explainer, infographic video",
        chunking="split on slide/screen changes detected by frame differencing and OCR diff",
        clip_prompts=(
            "a presentation slide with text",
            "a computer screen recording",
            "a whiteboard with diagrams",
        ),
    ),
    VideoClass(
        name="edited_narrative",
        definition=(
            "Professionally edited footage with frequent cuts between shots that tell a "
            "story or explain a topic: shot boundaries are the natural unit."
        ),
        examples="film, TV episode, documentary, animation, music video, news package, edited cooking/fashion show",
        chunking="detect shots, then merge shots into scenes by visual and audio similarity",
        clip_prompts=(
            "a cinematic scene from a movie",
            "a documentary shot",
            "an animated cartoon scene",
        ),
    ),
    VideoClass(
        name="sports_broadcast",
        definition=(
            "Broadcast of a real-world sports competition: wide field views, scoreboards, "
            "replays and commentary structured around plays or events."
        ),
        examples="football, basketball, athletics, gymnastics, tennis broadcast",
        chunking="segment into plays/events using scoreboard changes, replay detection and crowd-audio peaks",
        clip_prompts=(
            "a sports game broadcast with a scoreboard",
            "athletes competing in a stadium",
            "a wide shot of a sports field",
        ),
    ),
    VideoClass(
        name="egocentric_continuous",
        definition=(
            "Handheld, first-person or body-worn footage following a person or activity "
            "in long continuous takes with camera motion and few hard cuts."
        ),
        examples="travel walk, vlog on the move, handheld cooking/DIY demo, GoPro, dashcam",
        chunking="sliding windows split where the activity or location changes (motion + scene embedding drift)",
        clip_prompts=(
            "a first-person view of hands doing a task",
            "a handheld camera walking through a street",
            "a point of view video from a head-mounted camera",
        ),
    ),
    VideoClass(
        name="static_camera",
        definition=(
            "A fixed camera watching a scene for long stretches with no cuts; most "
            "frames are near-identical and information arrives as sparse events."
        ),
        examples="surveillance/CCTV, wildlife trail cam, traffic cam, fixed livestream",
        chunking="event-triggered: cut on motion/activity onsets and drop idle stretches",
        clip_prompts=(
            "a security camera view",
            "a fixed camera overlooking a scene",
            "a CCTV footage frame",
        ),
    ),
    VideoClass(
        name="stage_performance",
        definition=(
            "A live performance in front of an audience: the structure follows the "
            "performance (songs, acts, routines) rather than editing."
        ),
        examples="concert, stage play, magic show, acrobatics show, variety show act",
        chunking="split on audio structure (song/act boundaries, applause) and then on shots",
        clip_prompts=(
            "a live concert on stage",
            "performers on a theater stage",
            "a magic show in front of an audience",
        ),
    ),
    VideoClass(
        name="gameplay",
        definition=(
            "Rendered video-game footage, with or without a HUD, facecam or casters; "
            "includes let's-plays and esports broadcasts."
        ),
        examples="let's play, speedrun, esports match broadcast, game trailer captured in-engine",
        chunking="split on game-state changes (HUD/minimap/score changes, menu vs play, round boundaries)",
        clip_prompts=(
            "a video game screenshot with a HUD",
            "an esports match being played",
            "3D rendered video game gameplay",
        ),
    ),
)

CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in CLASSES)
CLASS_BY_NAME: dict[str, VideoClass] = {c.name: c for c in CLASSES}

TIE_BREAKERS: tuple[str, ...] = (
    "Pick the class that covers the majority of the running time; name the runner-up as secondary.",
    "Live performance in front of an audience -> stage_performance; a produced music video -> edited_narrative.",
    "Esports or any rendered game footage -> gameplay, even with casters and replays.",
    "News: anchor on screen most of the time -> talking_head; mostly field footage with cuts -> edited_narrative.",
    "Cooking/DIY/fashion: handheld with few cuts -> egocentric_continuous; multi-camera with frequent cuts -> edited_narrative.",
    "Wildlife documentary with narration and cuts -> edited_narrative; fixed trail/nest cam -> static_camera.",
    "Lecture: slides fill the frame most of the time -> screen_content; speaker fills the frame -> talking_head.",
)


def is_valid(name: str) -> bool:
    return name in CLASS_BY_NAME


def describe_for_prompt() -> str:
    lines = []
    for c in CLASSES:
        lines.append(f"- {c.name}: {c.definition} Examples: {c.examples}.")
    lines.append("")
    lines.append("Tie-breakers:")
    lines.extend(f"- {rule}" for rule in TIE_BREAKERS)
    return "\n".join(lines)
