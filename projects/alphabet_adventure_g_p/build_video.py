"""G–P lesson assembly: img2vid artwork plus deterministic, animated letter shapes."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FPS = 24
SIZE = (1920, 1080)
ART = (820, 238, 1844, 814)
INK = "#21334B"
FONT = "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"
REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
LETTERS = "GHIJKLMNOP"
VIDEO = HERE / "alphabet_adventure_G-P.mp4"


def run(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def read(name):
    return json.loads((HERE / name).read_text())


def write(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


@lru_cache(maxsize=80)
def font(size, regular=False):
    path = Path(REGULAR if regular else FONT)
    if path.exists():
        return ImageFont.truetype(str(path), size)
    sys.path.insert(0, str(ROOT))
    from img2vid.render import _font
    return _font("Alphabet", size, None)


def centered(draw, box, text, size, color=INK, regular=False):
    face = font(size, regular)
    while draw.textbbox((0, 0), text, font=face)[2] > box[2] - box[0] - 20:
        size -= 2
        face = font(size, regular)
    draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
              text, fill=color, font=face, anchor="mm")


def prepare():
    brief = read("lesson.json")
    audio = read("audio/g_p_lesson.timeline.json")
    assert "".join(x["letter"] for x in brief["letters"]) == LETTERS
    sections, events = [], []
    by_name = {}
    speech_count = {}
    group = {}
    for item in audio["segments"]:
        name = item["section"]
        if name not in by_name:
            index = len(sections)
            if "Verse" in name:
                n = int(name.rsplit(" ", 1)[1]) - 1
                section = {**brief["letters"][n], "kind": "letter", "image_index": n,
                           "heading": f"Meet the letter {LETTERS[n]}",
                           "badge": f"LETTER {n + 1} OF 10"}
            elif "Quiz" in name:
                n = int(name.rsplit(" ", 1)[1]) - 1
                quiz = brief["quizzes"][n]
                letter_index = LETTERS.index(quiz["letter"])
                section = {**brief["letters"][letter_index], **quiz, "kind": "quiz",
                           "image_index": letter_index,
                           "heading": "Let's play: find the letter!", "badge": f"GAME {n + 1} OF 3"}
            elif name == "Spoken Intro":
                section = {"kind": "intro", "heading": "Ten new letters. Let's explore!",
                           "badge": "G TO P", "image_index": 0}
            elif name == "Spoken Recap":
                section = {"kind": "recap", "heading": "Say five letters with me!",
                           "badge": "LET'S REMEMBER", "image_index": 6}
            elif name == "Spoken Outro":
                section = {"kind": "outro", "heading": "You are an alphabet explorer!",
                           "badge": "WELL DONE", "image_index": 9}
            else:
                raise ValueError(f"Unknown audio section {name!r}")
            section.update(name=name, index=index, start_frame=round(item["start"] * FPS))
            sections.append(section)
            by_name[name] = index
            speech_count[name] = 0
            group[name] = LETTERS
        index = by_name[name]
        section = sections[index]
        if item["type"] == "speech":
            speech_count[name] += 1
            if section["kind"] == "recap":
                group[name] = {1: LETTERS[:5], 2: LETTERS[5:], 3: LETTERS}[speech_count[name]]
                phase = "talk" if speech_count[name] < 3 else "answer"
            else:
                phase = "answer" if speech_count[name] > 1 or section["kind"] == "outro" else "talk"
        elif item["seconds"] >= 0.75 and section["kind"] in ("letter", "quiz", "recap"):
            phase = "wait"
        else:
            phase = events[-1]["phase"]
        start, end = round(item["start"] * FPS), round(item["end"] * FPS)
        if events and all(events[-1][key] == value for key, value in
                          (("section", index), ("phase", phase), ("group", group[name]))):
            events[-1]["end_frame"] = end
        else:
            events.append({"section": index, "phase": phase, "group": group[name],
                           "start_frame": start, "end_frame": end})
        section["end_frame"] = end
    total = math.ceil(audio["duration"] * FPS)
    sections[-1]["end_frame"] = events[-1]["end_frame"] = total
    for left, right in zip(events, events[1:]):
        assert left["end_frame"] == right["start_frame"]
    for s in sections:
        s.update(start=s["start_frame"] / FPS, end=s["end_frame"] / FPS,
                 duration=(s["end_frame"] - s["start_frame"]) / FPS)
    write("timeline.json", {"fps": FPS, "duration": total / FPS, "frames": total,
                            "audio_duration": audio["duration"],
                            "sections": sections, "events": events})
    print(f"Prepared {len(sections)} sections, {len(events)} states, {total / FPS:.3f}s.")


def plan():
    brief = read("lesson.json")
    path = HERE / "motion.json"
    if path.exists():
        raise ValueError("motion.json already exists; retain its generated clip attachments")
    images = [HERE / "images" / f"{i:02d}.png" for i in range(1, 11)]
    if not all(p.exists() for p in images):
        raise ValueError("All ten illustrations must exist before planning")
    run(sys.executable, "-m", "img2vid", "plan", *images, "-o", path,
        "--aspect", "16:9", "--order", "given", "--scene-duration", 4,
        "--transition", "cut", "--fps", FPS, "--motion", "flat", "--title", "")
    story = read("motion.json")
    for scene, item in zip(story["scenes"], brief["letters"]):
        scene.update(motion_prompt=item["motion_prompt"], motion="zoom_in",
                     zoom=1.0, focus=[0.5, 0.5], animate=True, depth=False)
    write("motion.json", story)


def edit():
    source = read("motion.json")["scenes"]
    timeline = read("timeline.json")
    scenes = []
    for section in timeline["sections"]:
        scene = dict(source[section["image_index"]])
        if not scene.get("clip") or not Path(scene["clip"]).exists():
            raise ValueError(f"Missing animation for image {section['image_index'] + 1}")
        # Each source is a reversible idle action; use the renderer's ping-pong loop.
        scene.pop("clip_loop", None)
        scene.update(duration=section["duration"], clip_speed=0.85,
                     transition="cut", motion="zoom_in", zoom=1.0, depth=False)
        scenes.append(scene)
    write("storyboard.json", {"version": 1, "aspect": "16:9", "fps": FPS,
                              "music": None, "transition_duration": 0,
                              "fade_in": 0, "fade_out": 0, "scenes": scenes})


def base_card(section, phase, group):
    image = Image.new("RGBA", SIZE, "#FFF9ED")
    draw = ImageDraw.Draw(image)
    color = section.get("color", "#1764A7")
    draw.ellipse((1720, -190, 2050, 140), fill="#E8F1EF")
    draw.ellipse((-120, 930, 120, 1170), fill="#FCE6D5")
    draw.text((80, 52), "ALPHABET ADVENTURE", font=font(30), fill="#496176")
    draw.text((80, 112), section["heading"], font=font(61), fill=INK)
    draw.rounded_rectangle((1590, 62, 1840, 125), radius=30, fill="#E8F1EF")
    centered(draw, (1590, 62, 1840, 125), section["badge"], 26)
    draw.rounded_rectangle((76, 234, 776, 818), radius=40, fill="#E5E1D9")
    draw.rounded_rectangle((80, 238, 772, 814), radius=38, fill="white")
    draw.rounded_rectangle((812, 230, 1852, 822), radius=28, fill="white")
    draw.rectangle(ART, fill="#F3EFE8")
    kind = section["kind"]
    if kind == "letter":
        centered(draw, (110, 580, 742, 640), "is for", 44, "#526175", True)
        centered(draw, (105, 655, 747, 765), section["word"], 82, color)
        message = {"talk": f"Can you say {section['word'].lower()}?",
                   "wait": f"Your turn! Say {section['word'].lower()}.",
                   "answer": f"{section['letter']} is for {section['word'].lower()}!"}[phase]
    elif kind == "quiz":
        centered(draw, (110, 265, 742, 350), section["word"], 68, color)
        centered(draw, (110, 365, 742, 435), "Which letter?", 45)
        for i, letter in enumerate(section["options"]):
            x = 125 + i * 205
            selected = phase == "answer" and letter == section["letter"]
            draw.rounded_rectangle((x, 480, x + 175, 670), radius=30,
                                   fill="#D5F2E4" if selected else "#EFF3F7")
        centered(draw, (110, 702, 742, 780),
                 "You found it!" if phase == "answer" else "Point to your answer",
                 42 if phase == "answer" else 34, "#176145" if phase == "answer" else "#526175")
        message = {"talk": "Look at the picture. Choose a letter.",
                   "wait": "Point and say your answer!",
                   "answer": f"{section['letter']} is for {section['word'].lower()}!"}[phase]
    else:
        title = {"intro": "Hello, explorer!", "outro": "You did it!",
                 "recap": "Your turn!" if phase == "wait" else "Let's remember!"}[kind]
        centered(draw, (110, 270, 742, 350), title, 59, color)
        message = {"intro": "Look, listen, and say it with me!",
                   "outro": "Tell a grown-up your favorite word!",
                   "recap": "Your turn! Say these five letters." if phase == "wait" else
                            "Ready for a guessing game?" if len(group) == 10 else
                            "   ".join(group)}[kind]
    draw.rounded_rectangle((80, 862, 1840, 1000), radius=32,
                           fill="#E0F1E9" if phase == "answer" else "#EAF0F8")
    tag = {"talk": "LISTEN & SAY", "wait": "YOUR TURN", "answer": "SAY IT TOGETHER"}[phase]
    draw.text((115, 880), tag, font=font(23), fill="#496176")
    centered(draw, (110, 915, 1620 if phase == "wait" else 1810, 984), message, 48)
    for i, letter in enumerate(LETTERS):
        x = 533 + i * 95
        active = letter == section.get("letter") and (kind != "quiz" or phase == "answer")
        draw.text((x, 1034), letter, font=font(27), anchor="mm",
                  fill=color if active else "#8390A0")
        if active:
            draw.ellipse((x - 4, 1055, x + 4, 1063), fill=color)
    return image


@lru_cache(maxsize=100)
def glyph(letter, size, color):
    image = Image.new("RGBA", (size + 80, int(size * 1.5) + 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # A barred capital I is easier to distinguish from lowercase l in early reading.
    barred_i = Path("/System/Library/Fonts/Supplemental/Verdana Bold.ttf")
    face = ImageFont.truetype(str(barred_i), size) if letter == "I" and barred_i.exists() else font(size)
    draw.text((image.width / 2 + 2, image.height / 2 + 5), letter,
              font=face, anchor="mm", fill="#E8E5E5")
    draw.text((image.width / 2, image.height / 2), letter,
              font=face, anchor="mm", fill=color)
    return image


def animated_letter(canvas, letter, center, size, color, local, delay=0, pulse=0):
    t = max(0, local - delay)
    progress = min(1.0, t / 0.8)
    scale = 0.68 + 0.32 * (1 - (1 - progress) ** 3) + 0.055 * math.sin(math.pi * progress)
    scale += progress * (0.014 * math.sin(t * math.tau / 4.6) + pulse)
    y = center[1] + 35 * (1 - progress) ** 2 - 8 * math.sin(math.pi * progress)
    y += progress * 4 * math.sin(t * math.tau / 4.6)
    source = glyph(letter, size, color)
    layer = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS)
    canvas.alpha_composite(layer, (round(center[0] - layer.width / 2), round(y - layer.height / 2)))


def draw_frame(section, event, frame, base):
    image = base.copy()
    phase = event["phase"]
    local = frame / FPS - section["start"]
    phase_time = (frame - event["start_frame"]) / FPS
    color = section.get("color", "#1764A7")
    kind = section["kind"]
    if kind == "letter":
        pulse = 0.055 * math.exp(-phase_time * 1.6) * abs(math.sin(phase_time * math.pi * 1.5)) if phase == "answer" else 0
        animated_letter(image, section["letter"], (278, 452), 230, color, local, pulse=pulse)
        animated_letter(image, section["letter"].lower(), (575, 452), 224, color, local, 0.16, pulse)
    elif kind == "quiz":
        for i, letter in enumerate(section["options"]):
            selected = phase == "answer" and letter == section["letter"]
            pulse = 0.08 * math.exp(-phase_time * 1.4) * abs(math.sin(phase_time * math.pi * 1.5)) if selected else 0
            animated_letter(image, letter, (212 + i * 205, 573), 116,
                            "#176145" if selected else INK, local, i * 0.09, pulse)
    else:
        group = event["group"]
        colors = ("#BC343E", "#1764A7", "#8C4CAD", "#217562", "#A96916")
        for i, letter in enumerate(group):
            if len(group) == 10:
                center = (176 + (i % 5) * 125, 465 + (i // 5) * 185)
                size = 94
                x, y = center
                ImageDraw.Draw(image).rounded_rectangle((x - 52, y - 66, x + 52, y + 66), radius=24, fill="#F1F5F8")
            else:
                center = (218 + i * 207, 465) if i < 3 else (320 + (i - 3) * 207, 650)
                size = 116
                x, y = center
                ImageDraw.Draw(image).rounded_rectangle((x - 76, y - 75, x + 76, y + 75), radius=24, fill="#F1F5F8")
            cue = (2.0 if group == LETTERS[:5] and phase == "talk" else 0.6) + i * 0.45
            bounce = 0.07 * math.exp(-((phase_time - cue) / 0.28) ** 2) if kind == "recap" else 0
            # A new recap group has its own entrance; other grids enter once per section.
            animation_time = phase_time if kind == "recap" and phase != "wait" else local
            animated_letter(image, letter, center, size, colors[i % 5],
                            animation_time, i * 0.07, bounce)
    if phase == "wait":
        remaining = max(0, (event["end_frame"] - frame) / FPS)
        total = round((event["end_frame"] - event["start_frame"]) / FPS)
        draw = ImageDraw.Draw(image)
        for i in range(total):
            x = 1640 + i * 47
            draw.ellipse((x, 934, x + 28, 962),
                         fill=color if i < math.ceil(remaining) else "#CBD7E6")
    return image.convert("RGB")


def boards(preview=False):
    timeline = read("timeline.json")
    sections = timeline["sections"]
    events = timeline["events"]
    limit = min(timeline["frames"], 26 * FPS) if preview else timeline["frames"]
    output = HERE / ("letters_preview.mp4" if preview else "letters.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    event_index = 0
    base = None
    last_key = None
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for frame in range(limit):
            while frame >= events[event_index]["end_frame"]:
                event_index += 1
            event = events[event_index]
            section = sections[event["section"]]
            key = (event["section"], event["phase"], event["group"])
            if key != last_key:
                base = base_card(section, event["phase"], event["group"])
                last_key = key
            image = draw_frame(section, event, frame, base)
            encoder.stdin.write(image.tobytes())
            if frame % (30 * FPS) == 0:
                print(f"Letter animation: {frame / FPS:.0f}/{limit / FPS:.0f}s", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Letter animation encoding failed")
    print(f"Wrote {output}")


def finish():
    timeline = read("timeline.json")
    run(sys.executable, "-m", "img2vid", "render", HERE / "storyboard.json",
        "-o", HERE / "artwork.mp4", "--resolution", "576")
    run("ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", HERE / "letters.mp4", "-i", HERE / "artwork.mp4",
        "-i", HERE / "audio/g_p_lesson.mp3", "-i", HERE / "audio/g_p_lesson.srt",
        "-filter_complex",
        "[0:v]setpts=PTS-STARTPTS[board];"
        "[1:v]scale=1024:576,setsar=1,setpts=PTS-STARTPTS[art];"
        "[board][art]overlay=820:238:shortest=1,format=yuv420p[v]",
        "-map", "[v]", "-map", "2:a:0", "-map", "3:s:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
        "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng",
        "-metadata", "title=Alphabet Adventure: G to P",
        "-movflags", "+faststart", "-t", timeline["duration"], VIDEO)
    print(f"Finished {VIDEO}")


def review():
    import cv2
    timeline = read("timeline.json")
    video = cv2.VideoCapture(str(VIDEO))
    review_dir = HERE / "review"
    review_dir.mkdir(exist_ok=True)
    sheet = Image.new("RGB", (1920, 360 * math.ceil(len(timeline["sections"]) / 3)), "white")
    for i, section in enumerate(timeline["sections"]):
        video.set(cv2.CAP_PROP_POS_MSEC, (section["start"] + 2) * 1000)
        ok, frame = video.read()
        if not ok:
            raise ValueError(f"Cannot decode {section['name']}")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.save(review_dir / f"section-{i:02d}.jpg")
        sheet.paste(image.resize((640, 360)), ((i % 3) * 640, (i // 3) * 360))
    video.release()
    sheet.save(review_dir / "contact-sheet.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "plan", "edit", "boards", "preview", "finish", "review"])
    args = parser.parse_args()
    if args.stage in ("boards", "preview"):
        boards(args.stage == "preview")
    else:
        {"prepare": prepare, "plan": plan, "edit": edit, "finish": finish, "review": review}[args.stage]()


if __name__ == "__main__":
    main()
