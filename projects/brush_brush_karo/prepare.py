"""Export image prompts and drive the existing img2vid planner for this song."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "scenes.json"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_brief():
    brief = json.loads(DATA.read_text(encoding="utf-8"))
    assert [s["id"] for s in brief["scenes"]] == list(range(1, 13))
    for scene in brief["scenes"]:
        assert len(motion_prompt(brief, scene).split()) <= 120
    return brief


def motion_prompt(brief, scene):
    return scene["motion_prompt"] + " " + brief["animation_lock"]


def image_prompt(brief, scene):
    parts = [brief["style"]]
    parts.extend(brief["characters"][name] for name in scene["cast"])
    if scene["bathroom"]:
        parts.append(brief["bathroom"])
    parts.append(scene["image_prompt"])
    return "\n\n".join(parts)


def export(brief):
    """These files are generated from scenes.json, so re-export after edits."""
    destination = HERE / "prompts"
    destination.mkdir(exist_ok=True)
    sections = [
        "# Brush Brush Karo — image and motion prompts, v1\n",
        "Generate and review the character references first, then scene 01. "
        "Approve the faces before proceeding. Scene 03 establishes the bathroom, "
        "scene 07 the mascot's size, and scene 09 its clean smile. "
        "Upload the listed references with each prompt. Paths below are relative "
        "to references/; ../images/ means an approved earlier scene. "
        "Save scenes as images/01.png through images/12.png. JPG or WebP is also accepted.\n",
        "If the image model has no negative-prompt field, append the AVOID paragraph "
        "to its main prompt. Repeating descriptions cannot guarantee identity; use "
        "actual image references or edit an approved frame.\n",
    ]
    for reference in brief["references"]:
        prompt = "\n\n".join([
            brief["style"],
            brief["characters"][reference["id"]],
            reference["prompt"],
            "AVOID: " + brief["negative_prompt"],
        ])
        (destination / f"reference_{reference['id']}.txt").write_text(
            prompt + "\n", encoding="utf-8"
        )
        sections.append(f"## Reference: {reference['file']}\n\n{prompt}\n")
    for scene in brief["scenes"]:
        stem = f"{scene['id']:02d}_{scene['slug']}"
        prompt = image_prompt(brief, scene) + "\n\nAVOID: " + brief["negative_prompt"]
        motion = motion_prompt(brief, scene)
        (destination / f"{stem}.image.txt").write_text(prompt + "\n", encoding="utf-8")
        (destination / f"{stem}.motion.txt").write_text(motion + "\n", encoding="utf-8")
        sections.append(
            f"## {scene['id']:02d} — {scene['title']}\n\n"
            f"Upload references: {', '.join(scene['references'])}\n\n"
            f"### Image prompt\n\n{prompt}\n\n### Motion prompt\n\n{motion}\n"
        )
    (HERE / "PROMPTS.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"Exported 4 reference prompts, 12 image prompts and 12 motion prompts to {destination}")


def check_images(brief):
    images = []
    problems = []
    folder = HERE / "images"
    for scene in brief["scenes"]:
        number = f"{scene['id']:02d}"
        matches = sorted(
            p for p in folder.glob(f"{number}.*") if p.suffix.lower() in EXTENSIONS
        )
        if len(matches) != 1:
            problems.append(
                f"{number}: expected exactly one images/{number}.png, .jpg, .jpeg or .webp; "
                f"found {len(matches)}"
            )
            continue
        path = matches[0]
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
            if abs(width / height - 16 / 9) > 0.02:
                problems.append(f"{number}: {width}x{height} is not 16:9; reframe before planning")
            elif width < 1920 or height < 1080:
                print(f"WARNING {number}: {width}x{height}; use 1920x1080 or higher artwork")
            images.append(path.resolve())
        except (OSError, ValueError) as error:
            problems.append(f"{number}: unreadable image: {error}")
    if problems:
        print("Artwork is not ready:\n  " + "\n  ".join(problems), file=sys.stderr)
        return None
    print(f"Verified {len(images)} readable 16:9 images; visual approval is still required.")
    return images


def plan(brief, args, images):
    from img2vid import storyboard

    output = HERE / "storyboard.json"
    if output.exists():
        raise ValueError(
            f"{output} already exists; keep it for cached clips or move it aside explicitly before replanning"
        )
    command = [
        sys.executable, "-m", "img2vid", "plan", *map(str, images),
        "--aspect", "16:9", "--order", "given", "--fps", "24",
        "--transition", "crossfade", "--transition-duration", "0.4",
        "--motion", "flat", "--title", "",
    ]
    if args.music:
        music = Path(args.music).expanduser().resolve()
        if not music.is_file():
            raise ValueError(f"Audio file does not exist: {music}")
        command += ["--music", str(music), "--fit-music"]
        if args.sync_beats:
            command.append("--sync-beats")
    else:
        if args.sync_beats:
            raise ValueError("--sync-beats requires --music")
        command += ["--duration", str(brief["target_duration"])]
    # Let img2vid analyze real images and calculate transition-aware timing.
    with tempfile.TemporaryDirectory(prefix="brush-plan-") as temporary:
        raw = Path(temporary) / "storyboard.json"
        subprocess.run(command + ["-o", str(raw)], cwd=ROOT, check=True)
        story = storyboard.load(str(raw))
    if len(story["scenes"]) != len(brief["scenes"]):
        raise ValueError("Planner scene count does not match the approved brief")
    for actual, scene, path in zip(story["scenes"], brief["scenes"], images):
        if Path(actual["image"]).resolve() != path:
            raise ValueError("Planner changed the expected image order")
        actual.update(
            motion_prompt=motion_prompt(brief, scene),
            motion=scene["camera"], zoom=scene["zoom"],
            focus=[0.5, 0.5], animate=True, depth=False,
        )
    storyboard.save(story, str(output))
    print(f"Prepared {output}")
    print("Next: review a flat preview, then animate scene 1 with img2vid animate-hq --only 1.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "check", "plan"])
    parser.add_argument("--music", help="song file, used with plan; fits duration to the track")
    parser.add_argument("--sync-beats", action="store_true", help="snap scene changes to song bars")
    args = parser.parse_args()
    brief = load_brief()
    if args.command == "export":
        export(brief)
        return
    images = check_images(brief)
    if images is None:
        sys.exit(2)
    if args.command == "plan":
        plan(brief, args, images)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"error: {error}")
