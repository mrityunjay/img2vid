"""Beat-synced dance edit: one shot per musical bar, hard cuts on the downbeat, the 12
scenes cycled in story order, and a zoom kick on every beat.

Run from the repo root:
  .venv/bin/python projects/brush_brush_karo/dance.py
  .venv/bin/img2vid render projects/brush_brush_karo/dance.json -o projects/brush_brush_karo/brush_brush_karo.mp4
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from img2vid import music  # noqa: E402

TITLE = "ब्रश ब्रश करो"
PULSE = 0.035          # 3.5% zoom kick
KICKS_PER_BEAT = 2     # this song: 67.5 BPM beat grid; kicking on half-beats feels like 135 BPM
MOTIONS = ["zoom_in", "pan", "zoom_out"]


def main():
    song = glob.glob(os.path.join(HERE, "audio", "*.mp3"))[0]
    images = [os.path.join(HERE, "images", f"{n:02d}.png") for n in range(1, 13)]
    rel = lambda p: os.path.relpath(p, ROOT)  # paths relative to the repo root (run from there)
    song_rel, images_rel = rel(song), [rel(p) for p in images]
    period, bars, length = music.beat_grid(song)
    # Shot boundaries on bar lines; the first shot starts at 0, the last runs to the end.
    cuts = [0.0] + [float(b) for b in bars[1:] if b < length - 1.0] + [length]
    scenes = []
    for i in range(len(cuts) - 1):
        n = i % len(images)
        lap = i // len(images)
        scenes.append({
            "image": images_rel[n],
            "duration": round(cuts[i + 1] - cuts[i], 3),
            "fit": "cover",
            "motion": MOTIONS[(n + lap) % 3],
            "zoom": 1.15,
            "focus": [0.5, 0.5],
            "transition": "cut",
        })
    scenes[0]["title_y"] = 0.15
    story = {
        "version": 1, "aspect": "16:9", "fps": 30, "music": song_rel, "title": TITLE,
        "transition_duration": 0.0, "fade_in": 0.3, "fade_out": 2.0,
        "beat_pulse": PULSE, "beat_pulse_period": round(period / KICKS_PER_BEAT, 4),
        "beat_offset": round(float(bars[0]), 4),
        "scenes": scenes,
    }
    out = os.path.join(HERE, "dance.json")
    with open(out, "w") as f:
        json.dump(story, f, indent=2, ensure_ascii=False)
        f.write("\n")
    total = sum(s["duration"] for s in scenes)
    print(f"wrote {out}: {len(scenes)} shots, {total:.1f}s (song {length:.1f}s), "
          f"bar {period * 4:.2f}s, kick every {period / KICKS_PER_BEAT:.2f}s")


if __name__ == "__main__":
    main()
