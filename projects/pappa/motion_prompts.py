"""Per-scene I2V motion prompts for "पप्पा जल्दी आ जाना"; applied onto storyboard.json.

Each prompt layers subject micro-motion, environment motion and camera, kept slow and stable.
Run:  .venv/bin/python projects/pappa/motion_prompts.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = ("Pixar-style 3D animated film, soft cinematic lighting, slow tender melancholic pacing, "
         "gentle natural micro-movements, stable consistent faces and hands, no morphing")
CAMERA_STILL = "Camera: locked-off tripod shot, very subtle depth parallax"

# (subject action, environment motion, renderer move, animate with AI?)
SCENES = [
    ("A little girl in a pink dress slowly waves her small hand goodbye; her father with a "
     "suitcase raises his hand in a slow wave and gives a sad, gentle smile, both blink softly",
     "golden sunset light shimmers on gently rolling sea waves, seagulls glide slowly, the dress "
     "and his shirt flutter in a soft sea breeze", "zoom_in", True),
    ("The little girl hugs her white bunny closer and breathes slowly, eyes following the plane, "
     "a slight sad tilt of her head; her father in the distance walks slowly away with his bag",
     "an airplane drifts slowly across the pink sunset sky, clouds drift, warm light glows on the "
     "window glass", "zoom_in", True),
    ("The father holds the pink doll up and gently turns it, smiling warmly as he admires it, "
     "a slow blink, his shoulders relax", "fairy lights twinkle softly over shelves of dolls, warm "
     "shop light glows, tiny sparkles drift", "zoom_in", True),
    ("The baby rests her cheeks on her hands, rocks very slightly and smiles dreamily, slow "
     "blinks; inside the glowing dream bubble the father gently offers the doll",
     "the dream bubble glows and softly pulses, sparkles float upward, warm afternoon light",
     "zoom_in", True),
    ("The girl lying on the rug slowly swings her feet, blinks dreamily and smiles at the doll in "
     "her dream", "the dream cloud glows, small hearts float slowly upward, the lamp light "
     "flickers warmly, curtains sway slightly", "zoom_out", True),
    ("The toddler sits holding her little card, gazes up at the dream bubble, breathes gently and "
     "blinks; in the bubble her father opens his arms wide in a slow welcome",
     "the bubble shimmers, warm lamp light glows, dust motes drift in the light", "zoom_in", True),
    # Has words painted into the image; AI motion would garble them, so it keeps 3D motion.
    ("The little girl sits hugging her knees, a slow sad breath and a gentle blink",
     "rain streaks down the window, lamp light glows softly", "zoom_in", False),
    ("The little girl sits by the window with her chin on her hands, a slow sad blink and a "
     "small sigh", "rain droplets run down the glass, blurred city lights twinkle behind the rain, "
     "the lamp glow flickers softly", "zoom_in", True),
    ("An empty, quiet living room at night, nobody moves",
     "pale moonlight slowly shifts across the floor, the curtains sway gently, stars twinkle "
     "outside the window, the small lamp glows", "pan", True),
    ("The little girl sits on the floor hugging her teddy, looking up at the door, breathing "
     "slowly with a hopeful blink", "moonlight glows through the window, stars twinkle, curtains "
     "drift gently, shadows shift softly", "zoom_in", True),
    ("The little girl stands at the door hugging her teddy, leans gently on it and peeks out "
     "hopefully, a slow breath", "moonlight spills through the windows, curtains sway, stars "
     "twinkle softly", "zoom_in", True),
    ("The little girl holds her teddy, her eyes glisten, she blinks slowly and lets out a small "
     "sad sigh beside her father's photo", "the table lamp glows warmly and flickers softly, "
     "moonlight glows in the window, dust motes float", "zoom_in", True),
    ("The toddler rests her chin on her hands and slowly looks up at the moon, blinks, breathes "
     "softly", "the full moon glows, stars twinkle, the curtain drifts in a light breeze",
     "zoom_in", True),
    ("The mother gently rocks her little daughter and slowly strokes her hair; the girl hugs her "
     "teddy and blinks sleepily", "warm lamp light glows, moonlight on the bedsheets, soft "
     "shadows", "zoom_in", True),
    ("The little girl lies in bed hugging her teddy, eyes half open, a slow blink, the blanket "
     "rising and falling with her breathing, looking at her father's photo", "moonlight glows "
     "through the window, the lamp glows softly", "zoom_out", True),
    ("The little girl rests her cheeks on her hands, sighs and blinks slowly, looking down at "
     "her broken toys and doll", "warm lamp light flickers gently, soft shadows", "zoom_in", True),
    ("A quiet room with an old wooden chest and a letter on the table, nobody moves",
     "the edge of the paper letter flutters slightly, moonlight slowly shifts, dust motes drift "
     "in the lamp beam", "pan", True),
    ("The father and children sit together looking toward the open door, a slow breath, a "
     "gentle blink, the baby hugs the teddy", "diya flames flicker, the string of lights twinkles "
     "softly, warm festive glow", "zoom_in", True),
    ("The little girl sits on the rug hugging her teddy, gazing at the door with hope, breathing "
     "slowly, a soft blink", "warm lamp light glows, moonlight through the window, stars "
     "twinkle", "zoom_in", True),
    ("The father gently strokes his daughter's head while the mother smiles softly beside them; "
     "the girl hugs her bunny, blinks and slowly relaxes", "night lamp glows warmly, stars on the "
     "bedsheet, the curtain sways softly", "zoom_out", True),
]


def main():
    path = os.path.join(HERE, "storyboard.json")
    with open(path) as f:
        story = json.load(f)
    assert len(story["scenes"]) == len(SCENES), "scene count changed; update SCENES"
    for scene, (subject, env, move, animate) in zip(story["scenes"], SCENES):
        scene["motion_prompt"] = f"{STYLE}. Subject: {subject}. Environment: {env}. {CAMERA_STILL}."
        if scene["motion"] != "pan" or move == "pan":
            scene["motion"] = move if move != "pan" or scene["motion"] == "pan" else "zoom_in"
        scene["zoom"] = 1.12  # slow, gentle push for an emotional pace
        scene["animate"] = animate
        scene["clip_speed"] = 0.8
    with open(path, "w") as f:
        json.dump(story, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"updated {len(SCENES)} scenes in {path}")


if __name__ == "__main__":
    main()
