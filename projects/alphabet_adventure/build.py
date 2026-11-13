"""Build a narrated A–F participation video with the existing img2vid CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FPS = 24
RATE = 24000
SIZE = (1920, 1080)
FONT = Path("/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf")
REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
INK = "#21334B"
CREAM = "#FFF9ED"
ART = (820, 238, 1844, 814)


def run(*args):
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT)


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def font(size, regular=False):
    return ImageFont.truetype(str(REGULAR if regular else FONT), size)


def centered(draw, box, text, size, color=INK, regular=False):
    face = font(size, regular)
    while draw.textbbox((0, 0), text, font=face)[2] > box[2] - box[0] - 20:
        size -= 2
        face = font(size, regular)
    draw.text(
        ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        text, fill=color, font=face, anchor="mm",
    )


def card(section, phase, text, counter=None):
    """Typography is added AFTER AI generation, so letters cannot morph."""
    image = Image.new("RGB", SIZE, CREAM)
    d = ImageDraw.Draw(image)
    color = section.get("color", "#1764A7")
    d.ellipse((1720, -190, 2050, 140), fill="#E8F1EF")
    d.ellipse((-120, 930, 120, 1170), fill="#FCE6D5")
    d.text((80, 52), "ALPHABET ADVENTURE", font=font(30), fill="#496176")
    d.text((80, 112), section["heading"], font=font(64), fill=INK)
    d.rounded_rectangle((1590, 62, 1840, 125), radius=30, fill="#E8F1EF")
    centered(d, (1590, 62, 1840, 125), section["badge"], 26)
    d.rounded_rectangle((76, 234, 776, 818), radius=40, fill="#E5E1D9")
    d.rounded_rectangle((80, 238, 772, 814), radius=38, fill="white")
    d.rounded_rectangle((812, 230, 1852, 822), radius=28, fill="white")
    d.rectangle(ART, fill="#F3EFE8")
    kind = section["kind"]
    if kind == "letter":
        d.text((426, 455), section["letter"] + " " + section["letter"].lower(),
               font=font(246), fill=color, anchor="mm")
        centered(d, (120, 580, 732, 640), "is for", 44, "#526175", True)
        centered(d, (110, 650, 742, 765), section["word"], 84, color)
    elif kind == "quiz":
        centered(d, (110, 265, 742, 350), section["word"], 70, color)
        centered(d, (110, 365, 742, 435), "Which letter?", 45)
        for index, letter in enumerate(section["options"]):
            x = 125 + index * 205
            selected = phase == "answer" and letter == section["letter"]
            fill = "#D5F2E4" if selected else "#EFF3F7"
            d.rounded_rectangle((x, 480, x + 175, 670), radius=30, fill=fill)
            centered(d, (x, 480, x + 175, 670), letter, 118,
                     "#176145" if selected else INK)
        if phase == "answer":
            centered(d, (110, 702, 742, 780), "You found it!", 45, "#176145")
        else:
            centered(d, (110, 702, 742, 780), "Point to your answer", 34, "#526175")
    else:
        centered(d, (110, 270, 742, 350), section.get("card_title", "Let's learn!"), 60, color)
        for i, letter in enumerate("ABCDEF"):
            x = 140 + (i % 3) * 195
            y = 395 + (i // 3) * 170
            d.rounded_rectangle((x, y, x + 160, y + 140), radius=25,
                                fill=("#FCE4DD", "#E1EFF9", "#EDE3F5",
                                      "#DDF0E8", "#FFF0D4", "#DFEFF4")[i])
            centered(d, (x, y, x + 160, y + 140), letter, 96)
    d.rounded_rectangle((80, 862, 1840, 1000), radius=32,
                        fill="#E0F1E9" if phase == "answer" else "#EAF0F8")
    tag = {"talk": "LISTEN & SAY", "wait": "YOUR TURN", "answer": "SAY IT TOGETHER"}[phase]
    d.text((115, 880), tag, font=font(23), fill="#496176")
    centered(d, (110, 915, 1640 if counter else 1810, 984), text, 48)
    if counter:
        for i in range(counter[1]):
            x = 1640 + i * 40
            d.ellipse((x, 935, x + 24, 959),
                      fill=color if i < counter[0] else "#CBD7E6")
    for i, letter in enumerate("ABCDEF"):
        x = 702 + i * 95
        current = letter == section.get("letter") and (kind != "quiz" or phase == "answer")
        d.text((x, 1034), letter, font=font(27), anchor="mm",
               fill=color if current else "#8390A0")
        if current:
            d.ellipse((x - 4, 1055, x + 4, 1063), fill=color)
    return image


def speech(text, brief):
    key = hashlib.sha1(
        f"{brief['voice']}|{brief['speech_rate']}|{brief['speech_speed']}|{text}".encode()
    ).hexdigest()[:14]
    wav = HERE / "audio" / f"{key}.wav"
    if not wav.exists():
        aiff = wav.with_suffix(".aiff")
        run("say", "-v", brief["voice"], "-r", brief["speech_rate"], "-o", aiff, text)
        run("ffmpeg", "-v", "error", "-y", "-i", aiff,
            "-af", f"atempo={brief['speech_speed']}", "-ac", "1", "-ar", RATE, wav)
        aiff.unlink()
    with wave.open(str(wav)) as stream:
        assert stream.getnchannels() == 1 and stream.getsampwidth() == 2
        values = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    values = values.astype(np.float32) / 32768
    # Level-normalize the cached speech after the pitch-preserving tempo adjustment.
    peak = np.max(np.abs(values), initial=0)
    if peak > 0:
        values *= 0.72 / peak
    return values


def timestamp(seconds, separator=","):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{separator}{ms % 1000:03d}"


def prepare(brief):
    letters = brief["letters"]
    sections = [{
        "kind": "intro", "heading": "Six letters. A big adventure!",
        "badge": "AGES 3–5", "card_title": "Hello, explorer!", "image_index": 0,
        "parts": [
            ("talk", "Hello, little explorer! Let's learn the first six letters. "
             "Look, listen, and say the words with me.", "Look, listen, and say it with me!"),
        ],
    }]
    for i, item in enumerate(letters):
        letter, word = item["letter"], item["word"]
        sections.append({
            **item, "kind": "letter", "heading": f"Meet the letter {letter}",
            "badge": f"LETTER {i + 1} OF 6", "image_index": i,
            "parts": [
                ("talk", f"This is the letter {letter}. Capital {letter}, and lowercase {letter}. "
                 f"{letter} is for {word.lower()}. Can you say {word.lower()}?",
                 f"Can you say {word.lower()}?"),
                ("wait", 3, f"Your turn! Say {word.lower()}."),
                ("answer", f"{word}! {letter} is for {word.lower()}. Let's say {letter} together!",
                 f"{letter} is for {word.lower()}!"),
            ],
        })
    sections.append({
        "kind": "recap", "heading": "Say the letters with me!",
        "badge": "LET'S REMEMBER", "image_index": 2, "card_title": "All six letters",
        "parts": [
            ("talk", "Let's say our six letters together. A. B. C. D. E. F. "
             "Now it's your turn!", "A   B   C   D   E   F"),
            ("wait", 4, "Your turn! Say all six letters."),
            ("answer", "Great practicing! Now let's play a little guessing game.",
             "Ready for a guessing game?"),
        ],
    })
    for n, quiz in enumerate(brief["quizzes"], 1):
        i = next(i for i, item in enumerate(letters) if item["letter"] == quiz["letter"])
        item = letters[i]
        options = ", ".join(quiz["options"][:-1]) + ", or " + quiz["options"][-1]
        sections.append({
            **item, **quiz, "kind": "quiz", "heading": "Let's play: find the letter!",
            "badge": f"GAME {n} OF 3", "image_index": i,
            "parts": [
                ("talk", f"Which letter starts {item['word'].lower()}? {options}? "
                 "Point to it, and say the letter!", "Look at the picture. Choose a letter."),
                ("wait", 4, "Point and say your answer!"),
                ("answer", f"It's {item['letter']}! {item['letter']} is for "
                 f"{item['word'].lower()}. Well done for trying!",
                 f"{item['letter']} is for {item['word'].lower()}!"),
            ],
        })
    sections.append({
        "kind": "outro", "heading": "You are an alphabet explorer!",
        "badge": "WELL DONE", "image_index": 5, "card_title": "Keep exploring!",
        "parts": [
            ("talk", "You practiced A, B, C, D, E, and F! "
             "Tell a grown-up your favorite word. See you next time!",
             "Tell a grown-up your favorite word!"),
        ],
    })

    audio_parts, events, subtitles = [], [], []
    frame = 0
    transcript = ["# Alphabet Adventure — narration and timing\n",
                  "English letter names and word associations. Pause-and-answer participation video.\n"]
    for section_index, section in enumerate(sections):
        section["start"] = frame / FPS
        transcript.append(f"\n## {section_index + 1}. {section['heading']}\n")
        for phase, content, on_screen in section.pop("parts"):
            if phase == "wait":
                count = int(content)
                transcript.append(f"\n[Wait {count} seconds for the child.]\n")
                for left in range(count, 0, -1):
                    name = f"overlays/{len(events):03d}.png"
                    card(section, phase, on_screen, (left, count)).save(HERE / name)
                    events.append({"image": name, "start": frame / FPS,
                                   "duration": 1, "phase": phase, "section": section_index})
                    audio_parts.append(np.zeros(RATE, dtype=np.float32))
                    frame += FPS
                continue
            spoken = speech(content, brief)
            padding = 0.45 if phase == "talk" else 0.7
            count_frames = math.ceil((len(spoken) / RATE + padding) * FPS)
            samples = count_frames * RATE // FPS
            audio_parts.append(np.pad(spoken, (0, samples - len(spoken))))
            name = f"overlays/{len(events):03d}.png"
            card(section, phase, on_screen).save(HERE / name)
            events.append({"image": name, "start": frame / FPS,
                           "duration": count_frames / FPS, "phase": phase,
                           "section": section_index})
            subtitles.append((frame / FPS, frame / FPS + len(spoken) / RATE, content))
            transcript.append(f"\n{content}\n")
            frame += count_frames
        section["duration"] = frame / FPS - section["start"]
        section["end"] = frame / FPS

    audio = np.concatenate(audio_parts)
    # Quiet original three-note chimes introduce the lesson and each answer.
    for start in [0.0] + [e["start"] for e in events if e["phase"] == "answer"]:
        for n, frequency in enumerate([523.25, 659.25, 783.99]):
            t = np.arange(int(0.22 * RATE)) / RATE
            note = 0.035 * np.sin(2 * np.pi * frequency * t) * np.exp(-t * 18)
            index = round((start + n * 0.13) * RATE)
            end = min(index + len(note), len(audio))
            audio[index:end] += note[:end - index]
    assert np.max(np.abs(audio)) < 1, "Audio would clip"
    with wave.open(str(HERE / "audio" / "narration.wav"), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes((audio * 32767).astype("<i2").tobytes())
    captions = []
    webcaptions = ["WEBVTT\n"]
    for i, (start, end, text) in enumerate(subtitles, 1):
        captions.append(f"{i}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n")
        webcaptions.append(f"{timestamp(start, '.')} --> {timestamp(end, '.')}\n{text}\n")
    (HERE / "captions.srt").write_text("\n".join(captions))
    (HERE / "captions.vtt").write_text("\n".join(webcaptions))
    (HERE / "SCRIPT.md").write_text("\n".join(transcript))
    write_json("timeline.json", {"fps": FPS, "duration": frame / FPS,
                                "sections": sections, "events": events})
    # PNG concat is a presentation track, independent of AI video motion.
    lines = ["ffconcat version 1.0"]
    for event in events:
        lines += [f"file '{event['image']}'", f"duration {event['duration']:.9f}"]
    lines.append(f"file '{events[-1]['image']}'")
    (HERE / "overlays.ffconcat").write_text("\n".join(lines) + "\n")
    print(f"Prepared {len(sections)} sections, {frame / FPS:.2f}s, with narration and answer pauses.")


def plan(brief):
    output = HERE / "motion.json"
    if output.exists():
        raise ValueError("motion.json already exists; do not discard generated clip attachments")
    paths = [HERE / "images" / f"{i:02d}.png" for i in range(1, 7)]
    if not all(path.exists() for path in paths):
        raise ValueError("All six generated images must exist before planning")
    run(sys.executable, "-m", "img2vid", "plan", *paths, "-o", output,
        "--aspect", "16:9", "--order", "given", "--scene-duration", "4",
        "--transition", "cut", "--fps", FPS, "--motion", "flat", "--title", "")
    story = json.loads(output.read_text())
    for scene, item in zip(story["scenes"], brief["letters"]):
        scene.update(motion_prompt=item["motion_prompt"], motion="zoom_in",
                     zoom=1.0, focus=[0.5, 0.5], animate=True, depth=False)
    write_json("motion.json", story)


def edit():
    sources = json.loads((HERE / "motion.json").read_text())["scenes"]
    timeline = json.loads((HERE / "timeline.json").read_text())
    scenes = []
    for section in timeline["sections"]:
        scene = dict(sources[section["image_index"]])
        # These are reversible idle motions, so slow ping-pong keeps the animal
        # moving during the child's response pause without stretching a 4s clip to 20s.
        scene.pop("clip_loop", None)
        scene.update(duration=section["duration"], clip_speed=0.65, transition="cut",
                     motion="zoom_in", zoom=1.0, depth=False)
        scenes.append(scene)
    write_json("storyboard.json", {
        "version": 1, "aspect": "16:9", "fps": FPS, "music": None,
        "transition_duration": 0, "fade_in": 0, "fade_out": 0, "scenes": scenes,
    })
    print("Wrote storyboard.json with narration-aligned sections.")


def finish():
    timeline = json.loads((HERE / "timeline.json").read_text())
    run(sys.executable, "-m", "img2vid", "render", HERE / "storyboard.json",
        "-o", HERE / "artwork.mp4", "--resolution", "576")
    # Artwork occupies a 1024x576 panel; the letter and response cards remain 1080p.
    run("ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "concat", "-safe", "0", "-i", HERE / "overlays.ffconcat",
        "-i", HERE / "artwork.mp4", "-i", HERE / "audio" / "narration.wav",
        "-i", HERE / "captions.srt",
        "-filter_complex",
        f"[0:v]fps={FPS},setpts=PTS-STARTPTS[board];"
        "[1:v]scale=1024:576,setsar=1,setpts=PTS-STARTPTS[art];"
        "[board][art]overlay=820:238:shortest=1,format=yuv420p[v]",
        "-map", "[v]", "-map", "2:a", "-map", "3:s",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text",
        "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng",
        "-metadata", "title=Alphabet Adventure: A to F",
        "-movflags", "+faststart", "-t", timeline["duration"],
        HERE / "alphabet_adventure_A-F.mp4")
    print(f"Finished {HERE / 'alphabet_adventure_A-F.mp4'}")


def review():
    import cv2

    timeline = json.loads((HERE / "timeline.json").read_text())
    video = cv2.VideoCapture(str(HERE / "alphabet_adventure_A-F.mp4"))
    thumbs = []
    for section in timeline["sections"]:
        video.set(cv2.CAP_PROP_POS_MSEC, (section["start"] + 2) * 1000)
        ok, frame = video.read()
        if not ok:
            raise ValueError(f"Cannot decode section {section['heading']}")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.save(HERE / "review" / f"{len(thumbs):02d}.jpg")
        image.thumbnail((640, 360))
        thumbs.append(image)
    video.release()
    sheet = Image.new("RGB", (640 * 3, 360 * math.ceil(len(thumbs) / 3)), "white")
    for i, image in enumerate(thumbs):
        sheet.paste(image, ((i % 3) * 640, (i // 3) * 360))
    sheet.save(HERE / "review" / "contact-sheet.jpg")
    print("Extracted review frames from the final video.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "plan", "edit", "finish", "review"])
    args = parser.parse_args()
    brief = json.loads((HERE / "lesson.json").read_text())
    if args.stage == "prepare":
        prepare(brief)
    elif args.stage == "plan":
        plan(brief)
    else:
        {"edit": edit, "finish": finish, "review": review}[args.stage]()


if __name__ == "__main__":
    main()
