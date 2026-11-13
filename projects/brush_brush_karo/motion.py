"""Write anim.json: the 12 scene images with one motion prompt each, for animate-hq.

The prompts describe what is really in the chosen images (some differ from the brief,
e.g. 06 reuses a brushing shot). One small action per scene keeps faces and hands stable;
the dance feel comes from the cuts and beat kicks in dance.py.

Run from the repo root (the AI run takes ~6 min per scene on an 18 GB Mac):
  .venv/bin/python projects/brush_brush_karo/motion.py
  HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/img2vid animate-hq projects/brush_brush_karo/anim.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

LOCK = ("3D animated cartoon, cheerful and bouncy, smooth natural motion, faces, hands and "
        "toothbrushes stay stable, same characters and colors as the image, no new people.")

ACTIONS = {
    1: "The little girl on the bed stretches her arms up and smiles; the boy at the door "
       "waves his toothbrush happily, bouncing on his toes. Slow camera push in.",
    2: "The girl and the boy sway side to side happily in the hallway, smiling, the boy "
       "holding the toothbrushes. Slow camera push in.",
    3: "The girl on the step lifts her pink toothbrush and smiles; the boy nods happily; the "
       "mother smiles and holds up a toothbrush. Static camera.",
    4: "The boy brushes his teeth with small gentle strokes; the girl on the step bobs her "
       "head happily; the mother smiles. Static camera.",
    5: "The girl on the step brushes her teeth gently; the boy behind the basin smiles and "
       "nods; the mother watches warmly. Static camera.",
    6: "The children brush their teeth with small gentle strokes and sway a little to the "
       "music while the mother smiles. Static camera.",
    7: "The tooth mascot on the basin waves its tiny arm; the children laugh and lean toward "
       "it; the mother smiles. Slow camera push in.",
    8: "The tooth mascot wiggles happily on the basin; the boy holds his toothbrush up; the "
       "mother makes one small brushing motion. Static camera.",
    9: "The tooth mascot bounces gently and sparkles; the boy gives a thumbs-up; the girl "
       "claps her hands once; the mother smiles. Static camera.",
    10: "The boy pats the tooth mascot; the girl on the step touches her chin and giggles; "
        "the mother smiles. Static camera.",
    11: "The girl and the boy clap their hands and bounce happily, dancing; the tooth mascot "
        "sways side to side between them. Slow camera push in.",
    12: "The girl and the boy wave goodbye and bounce happily; the tooth mascot sways. Slow "
        "camera pull back.",
}


def main():
    scenes = []
    for n in range(1, 13):
        scenes.append({
            "image": os.path.relpath(os.path.join(HERE, "images", f"{n:02d}.png"), ROOT),
            "duration": 5.0, "fit": "cover", "motion": "zoom_in", "zoom": 1.05,
            "focus": [0.5, 0.5], "transition": "cut", "animate": True,
            "motion_prompt": f"{ACTIONS[n]} {LOCK}",
        })
    story = {"version": 1, "aspect": "16:9", "fps": 30, "scenes": scenes}
    out = os.path.join(HERE, "anim.json")
    with open(out, "w") as f:
        json.dump(story, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
