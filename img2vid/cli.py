"""Command line: `plan` writes an editable storyboard, `render` renders one, `make` does both."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import animate, depth, director, music
from . import storyboard as sb
from .analyze import SUPPORTED, analyze
from .render import render


def _collect(paths: list[str]) -> list[str]:
    found = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                full = os.path.join(p, name)
                if os.path.isfile(full) and os.path.splitext(name)[1].lower() in SUPPORTED:
                    found.append(full)
        elif os.path.splitext(p)[1].lower() in SUPPORTED:
            found.append(p)
        else:
            print(f"skipping unsupported file: {p}", file=sys.stderr)
    return found


def _music_length(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _plan(args) -> dict:
    paths = _collect(args.images)
    if not paths:
        sys.exit("no supported images found")
    total = args.duration
    if args.fit_music:
        if not args.music:
            sys.exit("--fit-music needs --music")
        total = _music_length(args.music)

    infos = []
    for i, p in enumerate(paths, 1):
        print(f"analyzing {i}/{len(paths)}: {os.path.basename(p)}", file=sys.stderr)
        infos.append(analyze(p))

    build_kwargs = dict(
        aspect=args.aspect,
        scene_duration=args.scene_duration,
        total_duration=total,
        transition=args.transition,
        transition_duration=args.transition_duration,
        music=args.music,
        fps=args.fps,
        seed=args.seed,
    )
    story = None
    if args.ai or args.prompt:
        print(f"asking {args.model} to plan the video...", file=sys.stderr)
        try:
            story = director.direct(infos, args.prompt, model=args.model, **build_kwargs)
        except director.DirectorError as e:
            print(f"AI director unavailable ({e}); using offline storyboard", file=sys.stderr)
    if story is None:
        story = sb.build(infos, order=args.order, **build_kwargs)

    if args.title is not None:
        story["title"] = args.title or None
    if args.font:
        story["font"] = os.path.abspath(args.font)
    if args.sync_beats:
        if not args.music:
            sys.exit("--sync-beats needs --music")
        music.snap_to_bars(story, args.music)
    _apply_motion(story, args)
    return story


def _apply_motion(story: dict, args) -> None:
    motion = args.motion
    if motion is None:
        motion = "3d" if depth.available() else "flat"
    if motion in ("3d", "ai"):
        if depth.available():
            for scene in story["scenes"]:
                scene["depth"] = True  # scenes that get an AI clip ignore this
        else:
            print("3D motion needs torch and transformers; using flat motion", file=sys.stderr)
    if motion == "ai":
        try:
            animate.generate(
                story, short=args.ai_res, seconds=args.ai_seconds, steps=args.ai_steps,
                seed=args.seed, prompt=args.motion_prompt, limit=args.ai_max,
                allow_people=args.ai_people,
            )
        except animate.AnimateUnavailable as e:
            print(f"AI motion unavailable ({e}); using 3D motion", file=sys.stderr)


def _attach_clips(story: dict, folder: str) -> None:
    """Use NN.* videos in folder (01.mp4 = scene 1); scenes without one fall back to 3D."""
    from .render import VIDEO_EXTS

    files = {}
    for name in sorted(os.listdir(folder)):
        num = name.split(".")[0].split("_")[0].split(" ")[0]
        if name.lower().endswith(VIDEO_EXTS) and num.isdigit():
            files.setdefault(int(num), os.path.abspath(os.path.join(folder, name)))
    used, missing = 0, []
    for i, scene in enumerate(story["scenes"], 1):
        if i in files:
            scene["clip"] = files[i]
            used += 1
        else:
            scene.pop("clip", None)
            missing.append(str(i))
    print(f"attached {used} clip(s)" +
          (f"; scenes {', '.join(missing)} use 3D motion" if missing else ""), file=sys.stderr)


def _render(story: dict, args) -> None:
    total = render(story, args.output, short_side=args.resolution, preview=args.preview)
    print(f"wrote {args.output} ({total:.1f}s, {len(story['scenes'])} scenes)")


def _add_plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("images", nargs="+", help="image files and/or folders")
    p.add_argument("--aspect", choices=sorted(sb.ASPECTS), default="16:9")
    p.add_argument("--music", help="audio track (looped/trimmed to fit)")
    p.add_argument("--scene-duration", type=float, default=3.5, help="seconds per image")
    p.add_argument("--duration", type=float, help="total video length; overrides --scene-duration")
    p.add_argument("--fit-music", action="store_true", help="make the video as long as the music")
    p.add_argument("--transition", choices=sb.TRANSITIONS, default="crossfade")
    p.add_argument("--transition-duration", type=float, default=0.8)
    p.add_argument("--order", choices=sb.ORDERS, default="date", help="date = EXIF capture time")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0, help="varies motion choices")
    p.add_argument("--ai", action="store_true", help="let Claude plan order, motion and captions")
    p.add_argument("--prompt", help="what the video is about / how it should feel (implies --ai)")
    p.add_argument("--model", default=director.DEFAULT_MODEL, help="Claude model for --ai")
    p.add_argument("--title", help='title over the first scene ("" removes an AI title)')
    p.add_argument("--font", help="TTF/OTF font for title and captions")
    p.add_argument("--sync-beats", action="store_true",
                   help="put scene changes on the music's bar lines and end with the track")
    p.add_argument("--motion", choices=("flat", "3d", "ai"),
                   help="flat = pan/zoom; 3d = depth parallax (default when available); "
                        "ai = local LTX-Video makes people and things move (slow)")
    _add_ai_args(p)


def _add_ai_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--motion-prompt", help="how things should move, for --motion ai")
    p.add_argument("--ai-people", action="store_true",
                   help="also AI-animate photos with people/idols (faces may distort)")
    p.add_argument("--ai-max", type=int, help="animate only the N longest scenes with --motion ai")
    p.add_argument("--ai-seconds", type=float, default=4.0, help="length of each AI clip")
    p.add_argument("--ai-steps", type=int, default=30, help="denoising steps (quality vs speed)")
    p.add_argument("--ai-res", type=int, default=480, help="short side of AI clips in pixels")


def _add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--output", default="out.mp4")
    p.add_argument("--resolution", type=int, default=1080, help="short side in pixels")
    p.add_argument("--preview", action="store_true", help="fast 480p/24fps draft")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="img2vid", description="Animate images into a video.")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make", help="analyze images and render a video")
    _add_plan_args(make)
    _add_render_args(make)
    make.add_argument("--storyboard-out", help="also save the storyboard JSON here")

    plan = sub.add_parser("plan", help="analyze images and write an editable storyboard")
    _add_plan_args(plan)
    plan.add_argument("-o", "--output", default="storyboard.json")

    anim = sub.add_parser("animate", help="add AI motion clips to an existing storyboard JSON")
    anim.add_argument("storyboard")
    anim.add_argument("-o", "--output", help="where to save it (default: overwrite)")
    anim.add_argument("--seed", type=int, default=0)
    _add_ai_args(anim)

    hq = sub.add_parser("animate-hq", help="sharper AI motion (LTX-Video 0.9.8 + 2x upscaler)")
    hq.add_argument("storyboard")
    hq.add_argument("-o", "--output", help="where to save it (default: overwrite)")
    hq.add_argument("--only", help="scene numbers to animate, e.g. 1,3,8 (default: all)")
    hq.add_argument("--seconds", type=float, default=5.0, help="length of each clip")
    hq.add_argument("--res", type=int, default=448, help="first-pass short side (doubled later)")
    hq.add_argument("--no-upscale", action="store_true", help="skip the 2x upscale pass")
    hq.add_argument("--seed", type=int, default=0)

    clips = sub.add_parser("clips",help="use your own video clips (01.mp4 = scene 1, ...)")
    clips.add_argument("storyboard")
    clips.add_argument("folder", help="folder of videos named by scene number")
    clips.add_argument("-o", "--output", help="where to save it (default: overwrite)")

    rend = sub.add_parser("render", help="render a storyboard JSON")
    rend.add_argument("storyboard")
    _add_render_args(rend)

    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            sb.save(_plan(args), args.output)
            print(f"wrote {args.output}")
        elif args.command == "make":
            story = _plan(args)
            if args.storyboard_out:
                sb.save(story, args.storyboard_out)
            _render(story, args)
        elif args.command == "animate":
            story = sb.load(args.storyboard)
            args.motion = "ai"
            _apply_motion(story, args)
            sb.save(story, args.output or args.storyboard)
            print(f"wrote {args.output or args.storyboard}")
        elif args.command == "animate-hq":
            from . import animate_hq

            story = sb.load(args.storyboard)
            only = [int(x) for x in args.only.split(",")] if args.only else None
            try:
                animate_hq.generate(story, short=args.res, seconds=args.seconds, seed=args.seed,
                                    only=only, upscale=not args.no_upscale)
            except animate.AnimateUnavailable as e:
                sys.exit(f"error: {e}")
            for scene in story["scenes"]:
                if "hq-" in os.path.basename(scene.get("clip") or ""):
                    scene["clip_loop"] = "forward"
            sb.save(story, args.output or args.storyboard)
            print(f"wrote {args.output or args.storyboard}")
        elif args.command == "clips":
            story = sb.load(args.storyboard)
            _attach_clips(story, args.folder)
            sb.save(story, args.output or args.storyboard)
            print(f"wrote {args.output or args.storyboard}")
        else:
            _render(sb.load(args.storyboard), args)
    except (RuntimeError, ValueError, OSError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
