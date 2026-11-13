"""Retiming for the supplied Suno lesson: preserve its voice, speed and complete audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

from build import HERE, FPS, card, run, timestamp, write_json


def frame_time(seconds):
    return round(seconds * FPS) / FPS


def prepare(source):
    timing = json.loads((HERE / "suno_timing.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != timing["source_sha256"]:
        raise ValueError("The supplied audio differs from the recording used for these cue timings")
    audio = HERE / "audio" / "suno_A_is_for_Apple.mp3"
    if audio.exists() and hashlib.sha256(audio.read_bytes()).hexdigest() != timing["source_sha256"]:
        raise ValueError("A different audio file already exists at the destination")
    if source.resolve() != audio.resolve():
        shutil.copy2(source, audio)
    previous = json.loads((HERE / "timeline.json").read_text())
    motion = json.loads((HERE / "motion.json").read_text())["scenes"]
    cues = timing["sections"]
    assert len(previous["sections"]) == len(cues) == 12
    duration = math.ceil(timing["audio_duration"] * FPS) / FPS
    overlays = HERE / "suno_overlays"
    overlays.mkdir(exist_ok=True)
    captions = iter([
        "\n".join(block.splitlines()[2:])
        for block in re.split(r"\n\s*\n", (HERE / "captions.srt").read_text().strip())
    ])
    sections, events, subtitles, scenes = [], [], [], []

    def visual(section, index, phase, begin, end, text):
        begin, end = frame_time(begin), frame_time(end)
        if end <= begin:
            return
        name = f"suno_overlays/{len(events):03d}.png"
        card(section, phase, text).save(HERE / name)
        events.append({"image": name, "start": begin, "duration": end - begin,
                       "section": index, "phase": phase})

    for index, (old, cue) in enumerate(zip(previous["sections"], cues)):
        end = frame_time(cues[index + 1]["start"]) if index + 1 < len(cues) else duration
        section = dict(old)
        section.update(start=frame_time(cue["start"]), end=end,
                       duration=end - frame_time(cue["start"]))
        sections.append(section)
        old_events = [e for e in previous["events"] if e["section"] == index]
        # Retain existing educational wording, with timing taken from the new recording.
        if section["kind"] == "letter":
            talk_text = f"Can you say {section['word'].lower()}?"
            answer_text = f"{section['letter']} is for {section['word'].lower()}!"
        elif section["kind"] == "quiz":
            talk_text = "Look at the picture. Choose a letter."
            answer_text = f"{section['letter']} is for {section['word'].lower()}!"
        elif section["kind"] == "recap":
            talk_text = "A   B   C   D   E   F"
            answer_text = "Ready for a guessing game?"
        elif section["kind"] == "intro":
            talk_text = "Look, listen, and say it with me!"
        else:
            talk_text = "Tell a grown-up your favorite word!"

        if "response_start" in cue:
            question_end = cue["question_end"]
            response = cue["response_start"]
            reveal = cue.get("reveal_start", response)
            # Avoid flashing a "your turn" card for subsecond natural pauses.
            show_wait = reveal - question_end >= 0.75
            talk_end = question_end if show_wait else reveal
            visual(section, index, "talk", section["start"], talk_end, talk_text)
            if show_wait:
                visual(section, index, "wait", question_end, reveal,
                       "Point and say your answer!")
            visual(section, index, "answer", reveal, end, answer_text)
            subtitles.append((cue["start"], question_end, next(captions)))
            subtitles.append((response, end, next(captions)))
        else:
            assert len([e for e in old_events if e["phase"] != "wait"]) == 1
            visual(section, index, "talk", section["start"], end, talk_text)
            subtitles.append((cue["start"], min(end, timing["audio_duration"]), next(captions)))

        scene = dict(motion[section["image_index"]])
        scene.pop("clip_loop", None)
        scene.update(duration=section["duration"], clip_speed=0.65, transition="cut",
                     motion="zoom_in", zoom=1.0, depth=False)
        scenes.append(scene)
    assert next(captions, None) is None
    for left, right in zip(events, events[1:]):
        assert abs(left["start"] + left["duration"] - right["start"]) < 1e-6
    assert abs(sum(e["duration"] for e in events) - duration) < 1e-6
    write_json("suno_timeline.json", {"fps": FPS, "duration": duration,
                                     "audio_duration": timing["audio_duration"],
                                     "sections": sections, "events": events})
    write_json("suno_storyboard.json", {
        "version": 1, "aspect": "16:9", "fps": FPS, "music": None,
        "transition_duration": 0, "fade_in": 0, "fade_out": 0, "scenes": scenes,
    })
    concat = ["ffconcat version 1.0"]
    for event in events:
        concat += [f"file '{event['image']}'", f"duration {event['duration']:.9f}"]
    concat += [f"file '{events[-1]['image']}'"]
    (HERE / "suno_overlays.ffconcat").write_text("\n".join(concat) + "\n")
    srt, vtt = [], ["WEBVTT\n"]
    for i, (start, end, text) in enumerate(subtitles, 1):
        srt.append(f"{i}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n")
        vtt.append(f"{timestamp(start, '.')} --> {timestamp(end, '.')}\n{text}\n")
    (HERE / "suno_captions.srt").write_text("\n".join(srt))
    (HERE / "suno_captions.vtt").write_text("\n".join(vtt))
    print(f"Prepared Suno edit: {duration:.3f}s, {len(sections)} sections, {len(events)} visual states.")


def render():
    timeline = json.loads((HERE / "suno_timeline.json").read_text())
    run(sys.executable, "-m", "img2vid", "render", HERE / "suno_storyboard.json",
        "-o", HERE / "suno_artwork.mp4", "--resolution", "576")
    run("ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "concat", "-safe", "0", "-i", HERE / "suno_overlays.ffconcat",
        "-i", HERE / "suno_artwork.mp4",
        "-i", HERE / "audio" / "suno_A_is_for_Apple.mp3",
        "-i", HERE / "suno_captions.srt",
        "-filter_complex",
        f"[0:v]fps={FPS},setpts=PTS-STARTPTS[board];"
        "[1:v]scale=1024:576,setsar=1,setpts=PTS-STARTPTS[art];"
        "[board][art]overlay=820:238:shortest=1,format=yuv420p[v]",
        "-map", "[v]", "-map", "2:a:0", "-map", "3:s:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
        "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng",
        "-metadata", "title=Alphabet Adventure: A to F — Suno Audio",
        "-movflags", "+faststart", "-t", timeline["duration"],
        HERE / "alphabet_adventure_A-F_suno.mp4")
    print(f"Finished {HERE / 'alphabet_adventure_A-F_suno.mp4'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="the supplied A is for Apple.mp3")
    args = parser.parse_args()
    prepare(args.audio.expanduser().resolve())
    render()
