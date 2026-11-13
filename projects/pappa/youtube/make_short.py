"""Build a 9:16 YouTube Short storyboard from the main video's scenes and clips.

Run:  .venv/bin/python projects/pappa/youtube/make_short.py
Then: .venv/bin/img2vid render projects/pappa/youtube/short.json -o projects/pappa/youtube/short.mp4
"""
import copy
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from img2vid import music  # noqa: E402

LENGTH = 52.0  # seconds of the song to use (Shorts must be under 60 s)
# (scene number in the main storyboard, Hindi caption in the space below the picture)
PICKS = [
    (1, "पापा दूर जा रहे थे..."),
    (2, "और मैं बस देखती रह गई"),
    (12, "हर रात उनकी फोटो से बातें करती हूँ"),
    (14, "माँ कहती है, पापा जल्दी आएँगे"),
    (15, "सपनों में रोज़ आते हैं पापा"),
    (20, "पापा, जल्दी आ जाना"),
]


def main():
    with open(os.path.join(HERE, "..", "storyboard.json")) as f:
        main_story = json.load(f)
    song = main_story["music"]
    clip_music = os.path.join(HERE, "short_music.m4a")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", song, "-vn", "-t", str(LENGTH),
                    "-c:a", "aac", "-b:a", "192k", clip_music], check=True)

    scenes = []
    for n, caption in PICKS:
        s = copy.deepcopy(main_story["scenes"][n - 1])
        s["fit"] = "contain"  # whole 16:9 picture in the middle, blurred fill above/below
        s["caption"] = caption
        s["duration"] = LENGTH / len(PICKS)
        s["transition"] = "crossfade"
        scenes.append(s)
    scenes[0]["title_y"] = 0.2  # title in the blurred space above the picture

    story = {
        "aspect": "9:16", "fps": 30, "music": clip_music,
        "title": main_story.get("title"), "font": main_story.get("font"),
        "transition_duration": 1.0, "fade_in": 0.3, "fade_out": 1.5,
        "scenes": scenes,
    }
    music.snap_to_bars(story, clip_music)
    out = os.path.join(HERE, "short.json")
    with open(out, "w") as f:
        json.dump(story, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {out}: {len(scenes)} scenes, durations "
          f"{[round(s['duration'], 1) for s in scenes]}")


if __name__ == "__main__":
    main()
