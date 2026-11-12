"""Put the action prompts (shared with online_prompts.py) on storyboard.json for animate-hq.

Run:  .venv/bin/python projects/pappa/hq_prompts.py
"""
import json
import os

from online_prompts import ACTIONS, SKIP, STYLE

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = os.path.join(HERE, "storyboard.json")
    with open(path) as f:
        story = json.load(f)
    assert len(story["scenes"]) == len(ACTIONS), "scene count changed; update ACTIONS"
    for n, (scene, action) in enumerate(zip(story["scenes"], ACTIONS), 1):
        scene["motion_prompt"] = f"{STYLE}. {action}"
        scene["animate"] = n not in SKIP
        scene.pop("clip", None)  # old 0.9.5 clips are replaced; skipped scenes use 3D
        scene.pop("clip_loop", None)
    with open(path, "w") as f:
        json.dump(story, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"updated {len(ACTIONS)} scenes in {path}")


if __name__ == "__main__":
    main()
