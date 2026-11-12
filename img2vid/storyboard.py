"""Rule-based storyboard: the editable JSON contract between analysis and rendering."""
from __future__ import annotations

import json
import os
import random

from .analyze import ImageInfo

VERSION = 1
ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5)}
MOTIONS = ("zoom_in", "zoom_out", "pan")
TRANSITIONS = ("crossfade", "fadeblack", "slide", "cut")
ORDERS = ("date", "name", "given")

# Aspect mismatch (>= 1) above which the photo is letterboxed over a blurred
# copy of itself instead of being cropped to fill the frame.
_CONTAIN_ABOVE = 1.5
# Mismatch range where the cropped-off strip is wide enough to pan across.
_PAN_RANGE = (1.2, _CONTAIN_ABOVE)


def aspect_ratio(aspect: str) -> float:
    w, h = ASPECTS[aspect]
    return w / h


def build(
    infos: list[ImageInfo],
    aspect: str = "16:9",
    scene_duration: float = 3.5,
    total_duration: float | None = None,
    transition: str = "crossfade",
    transition_duration: float = 0.8,
    order: str = "date",
    music: str | None = None,
    fps: int = 30,
    seed: int = 0,
) -> dict:
    if not infos:
        raise ValueError("no images to build a storyboard from")

    if order == "date":
        infos = sorted(infos, key=lambda i: i.taken_at)
    elif order == "name":
        infos = sorted(infos, key=lambda i: os.path.basename(i.path).lower())

    n = len(infos)
    overlap = 0.0 if transition == "cut" else transition_duration
    if total_duration:
        # Scenes overlap during transitions, so add the overlaps back in.
        scene_duration = (total_duration + (n - 1) * overlap) / n
    overlap = min(overlap, scene_duration / 2)

    rng = random.Random(seed)
    out_ar = aspect_ratio(aspect)
    scenes = []
    prev_motion = None
    for idx, info in enumerate(infos):
        img_ar = info.width / info.height
        mismatch = max(img_ar / out_ar, out_ar / img_ar)
        fit = "contain" if mismatch > _CONTAIN_ABOVE else "cover"

        if fit == "cover" and _PAN_RANGE[0] <= mismatch <= _PAN_RANGE[1]:
            motion, zoom = "pan", 1.05
        elif info.faces:
            motion, zoom = "zoom_in", 1.3
        else:
            choices = [m for m in ("zoom_in", "zoom_out") if m != prev_motion]
            motion, zoom = rng.choice(choices), 1.2
        prev_motion = motion

        scenes.append(
            {
                "image": info.path,
                "duration": round(scene_duration, 3),
                "fit": fit,
                "motion": motion,
                "zoom": zoom,
                "focus": [round(info.focus[0], 4), round(info.focus[1], 4)],
                "faces": len(info.faces),
                # How this scene hands over to the next one.
                "transition": transition if idx < n - 1 else "cut",
            }
        )

    return {
        "version": VERSION,
        "aspect": aspect,
        "fps": fps,
        "music": os.path.abspath(music) if music else None,
        "transition_duration": round(overlap, 3),
        "fade_in": 0.6,
        "fade_out": 1.0,
        "scenes": scenes,
    }


def save(storyboard: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(storyboard, f, indent=2)
        f.write("\n")


def load(path: str) -> dict:
    with open(path) as f:
        storyboard = json.load(f)
    if storyboard.get("aspect") not in ASPECTS:
        raise ValueError(f"unsupported aspect {storyboard.get('aspect')!r}")
    for i, scene in enumerate(storyboard.get("scenes", [])):
        if scene.get("motion") not in MOTIONS:
            raise ValueError(f"scene {i}: unknown motion {scene.get('motion')!r}")
        if scene.get("transition", "cut") not in TRANSITIONS:
            raise ValueError(f"scene {i}: unknown transition {scene.get('transition')!r}")
    return storyboard
