"""AI director: Claude looks at the images and the user's prompt and plans the edit."""
from __future__ import annotations

import base64
import io
import json
import os
import sys

from . import storyboard as sb
from .analyze import ImageInfo, load_image

DEFAULT_MODEL = "claude-opus-5-5"
_MAX_SIDE = 1024  # images are downscaled before upload to keep token cost down
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """You are a film editor turning a set of photos into a short animated video \
(Ken Burns style: slow zooms and pans over still images, with transitions and short on-screen text).

Look at every image, work out the story or occasion they share, and plan the edit:
- Order scenes so the video has a beginning, middle and end: establishing shots early, \
the strongest or most emotional image near the end. Capture times are a hint, not a rule.
- Leave out near-duplicates and clearly weak shots (blurry, accidental) when better ones exist. \
Use each image at most once and keep at least one scene.
- focus_x / focus_y (0-1, measured from the image's top-left) mark what the camera moves toward: \
a face, the main subject, the detail that matters.
- zoom_in reveals a detail, zoom_out reveals context, pan travels across a wide scene \
(only images marked pan-capable can pan).
- zoom is the tightest framing, from 1.0 to 1.5; go higher for small subjects in a big frame.
- Give important images more time. Match pacing and transitions to the mood that was asked for.
- Captions are optional: at most about six words, in the language of the request, "" for none. \
Don't caption every scene. Never invent names, dates or places you can't see or weren't told.
- title is shown over the first scene: short, or "" for none.
- motion_prompt feeds an image-to-video model that animates the photo. In 2-3 plain sentences, \
describe the scene as real footage and how its people and things move in a few seconds \
(e.g. "Women in colourful chaniya cholis dance garba, skirts swirling; lamps flicker"). \
Keep motion believable for what is in the photo; don't add new people or objects."""

_SCENE = {
    "type": "object",
    "properties": {
        "image_index": {"type": "integer"},
        "duration": {"type": "number"},
        "motion": {"type": "string", "enum": list(sb.MOTIONS)},
        "focus_x": {"type": "number"},
        "focus_y": {"type": "number"},
        "zoom": {"type": "number"},
        "transition": {"type": "string", "enum": list(sb.TRANSITIONS)},
        "caption": {"type": "string"},
        "motion_prompt": {"type": "string"},
    },
    "required": [
        "image_index", "duration", "motion", "focus_x", "focus_y", "zoom", "transition", "caption",
        "motion_prompt",
    ],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "mood": {"type": "string"},
        "scenes": {"type": "array", "items": _SCENE},
    },
    "required": ["title", "mood", "scenes"],
    "additionalProperties": False,
}


class DirectorError(Exception):
    """The AI plan could not be produced; callers fall back to the rule-based storyboard."""


def _encode(path: str) -> str:
    img = load_image(path)
    img.thumbnail((_MAX_SIDE, _MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _clamp(v, lo, hi):
    return min(max(float(v), lo), hi)


def _request(infos, base, prompt, target):
    content = []
    for i, (info, scene) in enumerate(zip(infos, base["scenes"])):
        pan = "pan-capable" if scene["motion"] == "pan" else "not pan-capable"
        when = f"taken {info.taken_at}" if info.exif_time else f"file date {info.taken_at} (no capture time)"
        content.append({
            "type": "text",
            "text": f"Image {i}: {os.path.basename(info.path)}, {when}, "
                    f"{info.width}x{info.height}, {len(info.faces)} face(s) detected, {pan}",
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode(info.path)},
        })
    transition = base["scenes"][0]["transition"] if len(base["scenes"]) > 1 else "crossfade"
    content.append({
        "type": "text",
        "text": f"Request: {prompt or 'No specific request. Make a pleasant montage.'}\n"
                f"Video: {base['aspect']} aspect, about {target:.0f}s in total. Transitions overlap "
                f"neighbouring scenes by {base['transition_duration']}s. Default transition: "
                f"{transition}; vary it only when the mood calls for it.",
    })
    return content


def _call(model, content):
    try:
        import anthropic
    except ImportError as e:
        raise DirectorError("the anthropic package is not installed") from e

    try:
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=[_FALLBACK_BETA],
            fallbacks="default",  # re-run on Anthropic's recommended model if this one declines
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError as e:
        raise DirectorError("no valid Claude credentials (set ANTHROPIC_API_KEY)") from e
    except anthropic.APIStatusError as e:
        raise DirectorError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise DirectorError("could not reach the Claude API") from e
    except TypeError as e:  # the SDK raises this when no credentials are configured at all
        raise DirectorError("no Claude credentials found (set ANTHROPIC_API_KEY)") from e

    if response.stop_reason == "refusal":
        raise DirectorError("Claude declined to plan this video")
    if response.stop_reason == "max_tokens":
        raise DirectorError("Claude's plan was cut off")
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as e:
        raise DirectorError("Claude returned an unreadable plan") from e
    u = response.usage
    print(f"director: {response.model}, {u.input_tokens} in / {u.output_tokens} out tokens",
          file=sys.stderr)
    return plan


def _apply(plan, base, total_duration):
    scenes, seen = [], set()
    for s in plan.get("scenes", []):
        i = s.get("image_index")
        if not isinstance(i, int) or not 0 <= i < len(base["scenes"]) or i in seen:
            continue
        seen.add(i)
        scene = dict(base["scenes"][i])
        motion = s["motion"]
        if motion == "pan" and scene["motion"] != "pan":
            motion = "zoom_in"  # no room to pan across this image in this aspect
        scene.update(
            motion=motion,
            zoom=1.05 if motion == "pan" else round(_clamp(s["zoom"], 1.0, 1.5), 3),
            focus=[round(_clamp(s["focus_x"], 0, 1), 4), round(_clamp(s["focus_y"], 0, 1), 4)],
            duration=round(_clamp(s["duration"], 1.5, 10.0), 3),
            transition=s["transition"],
        )
        if s.get("motion_prompt", "").strip():
            scene["motion_prompt"] = s["motion_prompt"].strip()
        caption = s.get("caption", "").strip()
        if caption:
            scene["caption"] = caption
        else:
            scene.pop("caption", None)
        scenes.append(scene)
    if not scenes:
        raise DirectorError("Claude's plan had no usable scenes")
    scenes[-1]["transition"] = "cut"

    if total_duration:
        td = base["transition_duration"]
        overlap = sum(td for s in scenes[:-1] if s["transition"] != "cut")
        k = (total_duration + overlap) / sum(s["duration"] for s in scenes)
        for s in scenes:
            s["duration"] = round(max(s["duration"] * k, 2 * td), 3)

    return {
        **base,
        "title": plan.get("title", "").strip() or None,
        "mood": plan.get("mood", "").strip() or None,
        "director": plan.get("_model"),
        "scenes": scenes,
    }


def direct(infos: list[ImageInfo], prompt: str | None, model: str = DEFAULT_MODEL,
           **build_kwargs) -> dict:
    """Return a storyboard planned by Claude, raising DirectorError on any failure."""
    # Claude sees images in capture order; it decides the final order itself.
    infos = sorted(infos, key=lambda i: i.taken_at)
    base = sb.build(infos, **{**build_kwargs, "order": "given"})
    n = len(base["scenes"])
    target = build_kwargs.get("total_duration") or (
        n * base["scenes"][0]["duration"] - (n - 1) * base["transition_duration"]
    )
    plan = _call(model, _request(infos, base, prompt, target))
    plan["_model"] = model
    return _apply(plan, base, build_kwargs.get("total_duration"))
