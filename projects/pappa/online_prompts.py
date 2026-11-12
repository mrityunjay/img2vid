"""Make online_clips/: numbered images + PROMPTS.md for a free online image-to-video service.

Run:  .venv/bin/python projects/pappa/online_prompts.py
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "online_clips")

STYLE = ("Pixar-style 3D animation, keep characters' faces and outfits exactly as in the image, "
         "smooth natural motion, natural colors and soft lighting as in the image")

# One clear action per scene; services animate best with a single, concrete motion.
ACTIONS = [
    "The father kneels down and hugs his little daughter tightly; she wraps her small arms "
    "around his neck. Sea waves roll gently, seagulls glide. Slow camera push in.",
    "The little girl hugs her white bunny tighter as she watches her father walk away in the "
    "distance. Sunset clouds drift. Slow camera push in.",
    "The father lifts the pink doll from the shelf and smiles warmly at it. Fairy lights "
    "twinkle softly. Slow camera push in.",
    "The baby smiles wider and gently sways her head while dreaming of her father with the "
    "doll. Sparkles float upward. Slow camera push in.",
    "The little girl lying on the rug happily swings her feet while she dreams of the doll. "
    "Small hearts float upward. Slow camera pull back.",
    "The toddler looks up at the dream bubble and smiles; inside it her father opens his arms "
    "wide. Lamp light glows. Slow camera push in.",
    "The little girl hugs her knees and slowly rests her head on them. Rain falls outside. "
    "Static camera.",
    "The little girl slowly wipes the foggy window glass with her hand and looks outside. "
    "Raindrops run down the glass. Slow camera push in.",
    "A quiet empty living room at night; the curtains blow gently and moonlight shifts "
    "slowly across the floor. Slow camera pan.",
    "The little girl hugs her teddy bear and slowly turns her head toward the door. "
    "Curtains drift, stars twinkle. Slow camera push in.",
    "The little girl slowly opens the door a little and peeks outside, holding her teddy. "
    "Moonlight spills in. Slow camera push in.",
    "The little girl lifts her father's photo frame and hugs it to her chest. Lamp light "
    "flickers softly. Slow camera push in.",
    "The toddler slowly lifts her head and looks up at the bright moon. The curtain sways. "
    "Slow camera push in.",
    "The mother gently strokes her daughter's hair while the girl leans on her shoulder. "
    "Warm lamp light. Slow camera push in.",
    "The little girl in bed slowly closes her eyes and falls asleep hugging her teddy. "
    "Moonlight on the blanket. Slow camera pull back.",
    "The little girl picks up a broken toy and looks at it sadly. Lamp light flickers gently. "
    "Slow camera push in.",
    "The paper letter on the table flutters in a soft breeze while dust drifts in the "
    "moonlight; nobody is in the room. Slow camera pan.",
    "The family turns their heads together toward the open door with hopeful smiles. Diya "
    "flames flicker, festive lights twinkle. Slow camera push in.",
    "The little girl turns toward the door and her face lights up with a big smile. Warm "
    "lamp light. Slow camera push in.",
    "The father gently kisses his daughter's forehead while the mother smiles beside them. "
    "Warm night lamp glows. Slow camera pull back.",
]

# Scene 7 has words painted into it; online models usually garble text.
SKIP = {7: "has words painted in the image; AI video usually scrambles text, so skip it "
           "(it keeps the 3D motion)"}


def main():
    with open(os.path.join(HERE, "storyboard.json")) as f:
        story = json.load(f)
    scenes = story["scenes"]
    assert len(scenes) == len(ACTIONS), "scene count changed; update ACTIONS"
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)

    lines = [
        "# Online AI clips for पप्पा जल्दी आ जाना",
        "",
        "For each scene: upload `images/NN.png` to the image-to-video tool, paste the prompt,",
        "pick **16:9** and **10 seconds** (or the longest length offered), generate, and save the",
        "video into this folder as `NN.mp4` (for example `01.mp4`). Skip any scene you like;",
        "skipped scenes keep the 3D depth motion.",
        "",
        "Free tools to try (free daily/sign-up credits; limits change): Kling AI, Hailuo AI",
        "(MiniMax), Google Veo in the Gemini app, Pika.",
        "",
        "Negative prompt (if the tool has one): `blurry, distorted face, extra fingers, morphing, "
        "text, watermark, flicker`",
        "",
        "When done, run from the project folder:",
        "",
        "```bash",
        ".venv/bin/img2vid clips projects/pappa/storyboard.json projects/pappa/online_clips",
        "```",
        "",
        "```bash",
        '.venv/bin/img2vid render projects/pappa/storyboard.json -o projects/pappa/pappa_final.mp4',
        "```",
        "",
    ]
    for i, (scene, action) in enumerate(zip(scenes, ACTIONS), 1):
        ext = os.path.splitext(scene["image"])[1].lower()
        shutil.copyfile(scene["image"], os.path.join(OUT, "images", f"{i:02d}{ext}"))
        lines += [f"## {i:02d}  ({scene['duration']:.0f}s in the song)", ""]
        if i in SKIP:
            lines += [f"Skip: {SKIP[i]}.", ""]
            continue
        lines += [f"Image: `images/{i:02d}{ext}`", "", "```", f"{action} {STYLE}.", "```", ""]

    with open(os.path.join(OUT, "PROMPTS.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}/PROMPTS.md and {len(scenes)} images")


if __name__ == "__main__":
    main()
