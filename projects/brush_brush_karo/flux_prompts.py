"""Condensed prompts for local FLUX.1-schnell (`img2vid imagine`).

PROMPTS.md is written for reference-image models and runs ~600 words per prompt. FLUX
reads only the first 256 T5 tokens, has no negative prompt and no reference upload, so
these versions put the shot first, then short fixed character descriptions, then style.
Identity comes from repeating the same wording and seed; review every image.

Run from the repo root:
  .venv/bin/python projects/brush_brush_karo/flux_prompts.py            # write + check lengths
  .venv/bin/python projects/brush_brush_karo/flux_prompts.py refs       # generate references
  .venv/bin/python projects/brush_brush_karo/flux_prompts.py scenes 1 2 # generate scenes
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "flux")
SEED = 7
MAX_TOKENS = 256

STYLE = ("Premium 3D animated family film still, soft global illumination, warm morning "
         "light, vibrant pastel colors, sharp faces and hands, single image, no text.")

# Kept short so a 4-character bathroom shot still fits FLUX's 256-token limit.
CHAR = {
    "meera": ("Meera: 2-year-old Indian toddler girl, fair skin, chubby cheeks, big brown eyes, "
              "short dark-brown curls, one small pink bow on her left, pink frilly frock, pink "
              "shoes; much shorter than Bunny."),
    "bunny": ("Bunny: her 8-year-old brother, a human boy, fair skin, short neat black hair, "
              "big brown eyes, plain yellow T-shirt, blue shorts, blue sneakers."),
    "parent": ("Mother: Indian woman, fair skin, black hair in a low bun, plain teal kurta, "
               "cream trousers, flat slippers."),
    "toothy": ("Toothy: small cute ivory-white tooth mascot with short root-like feet, tiny "
               "arms, big brown eyes."),
}
BATH = ("Bathroom: pastel-blue tiles, ivory walls, white basin on a pale wood vanity, plain "
        "oval mirror, pink and blue towels.")
AT_SINK = "Meera on a sturdy step at the basin's left, Mother behind her, Bunny at the right."

REFS = {
    "meera": ("Full-body character reference portrait of Meera alone, front-facing, relaxed "
              "smile, arms slightly away from her body, both hands and shoes visible, plain "
              "warm-ivory studio background, even soft light."),
    "bunny": ("Full-body character reference portrait of Bunny alone, front-facing, friendly "
              "smile, hands relaxed, both sneakers visible, plain warm-ivory studio "
              "background, even soft light."),
    "toothy": ("Full-body character reference of Toothy alone on a plain warm-ivory studio "
               "background, shy friendly smile showing slightly yellow little teeth, clean "
               "ivory outer body, both tiny arms and root feet visible."),
    "parent": ("One adult Indian mother standing alone in an empty studio, full body from "
               "head to slippers, front-facing, calm encouraging smile, arms relaxed at her "
               "sides, plain warm-ivory background, even soft light."),
}

# (cast, bathroom?, shot) per scene, condensed from scenes.json
SCENES = {
    1: (["meera", "bunny"], False,
        "Wide shot of a cheerful Indian children's bedroom. Meera has just woken and sits up "
        "on a low toddler bed, sleepy curious half-open eyes. Bunny stands by the open door "
        "at right, smiling, holding one pink and one blue toothbrush at chest height. Soft "
        "toys, pastel pillows, light curtains, tidy floor. Exactly two children."),
    2: (["meera", "bunny"], False,
        "Full-body three-quarter view in a bright home hallway: Meera at left and Bunny at "
        "right stand side by side, about to walk toward an open bathroom door at right. "
        "Bunny holds the pink and blue toothbrushes low in his right hand. Both look at the "
        "bathroom, cheerful. Through the door: pastel-blue tiles and a white basin. Exactly "
        "two children."),
    3: (["meera", "bunny", "parent"], True,
        f"Wide bathroom shot. {AT_SINK} Meera holds her pink toothbrush, Bunny his blue one, "
        "both at chest height, looking at them curiously. Mother's hand on Meera's shoulder. "
        "Three people."),
    4: (["meera", "bunny", "parent"], True,
        f"Medium shot at the basin. {AT_SINK} Bunny gently brushes his front teeth with his "
        "blue toothbrush. Mother guides Meera's small hand brushing with the pink toothbrush. "
        "A trace of foam. Three people."),
    5: (["meera", "bunny", "parent"], True,
        f"Waist-up shot at the basin. {AT_SINK} Bunny, in focus, shows gentle brushing with his "
        "blue toothbrush. Meera watches, pink toothbrush lowered. Three people."),
    6: (["meera", "bunny", "parent"], True,
        f"Medium-wide shot. {AT_SINK} Bunny brushes with his blue brush, Mother guides Meera's "
        "pink brush. A small colorful sand timer on the counter. Cheerful faces. Three people."),
    7: (["meera", "bunny", "toothy", "parent"], True,
        f"{AT_SINK} Toothy stands on the counter between the children, one tiny arm raised to "
        "wave, slightly yellow teeth in its smile. Meera looks at it delighted, Bunny smiles, "
        "toothbrushes lowered."),
    8: (["meera", "bunny", "toothy", "parent"], True,
        f"{AT_SINK} Toothy on the counter brushes its little teeth with a tiny mint-green "
        "toothbrush, copying Bunny brushing with his blue toothbrush. Meera smiles, pink brush "
        "lowered."),
    9: (["meera", "bunny", "toothy", "parent"], True,
        f"{AT_SINK} Toothy on the counter beams a smile with clean white teeth and two tiny "
        "sparkles. Bunny gives a thumbs-up, Meera smiles at "
        "Toothy."),
    10: (["meera", "bunny", "toothy", "parent"], True,
         f"{AT_SINK} Bunny leans over the sink, spitting a little "
         "toothpaste. Mother holds a washcloth near Meera's chin. Toothy watches from the "
         "counter."),
    11: (["meera", "bunny", "toothy"], False,
         "Wide full-body shot in a cheerful pastel playroom with wall stars and paper "
         "garlands. Meera at left, Bunny at right, Toothy on the floor between them. The "
         "children hold empty hands apart at chest height, about to clap, big happy smiles. "
         "Two children and one mascot."),
    12: (["meera", "bunny", "toothy", "parent"], True,
         "Wide closing shot. Meera at left and Bunny at right stand on the bathroom floor, "
         "Toothy on the floor between them, Mother behind Meera. All smile at the camera, the "
         "children waving. Empty space at the top."),
}


# v2 for crowded shots (3-12). In v1 every "Name: description" block became one more
# figure, the name "Bunny" grew rabbit ears and the mirror added reflected faces. So: no
# names, everyone described inline in one sentence, head count first, no mirror.
GIRL = ("a 2-year-old fair-skinned toddler girl with short dark-brown curls, a small pink "
        "bow and a pink frilly frock")
BOY = ("a fair-skinned 8-year-old boy, her brother, with short boyish black hair, a plain "
       "yellow T-shirt and blue shorts")
MOM = ("their mother, a tall fair-skinned adult Indian woman in her thirties with a black "
       "low bun and a plain teal kurta")
TOOTH = "one small cute ivory-white tooth mascot with tiny arms and root-like feet"
ROOM = ("A bright bathroom with pastel-blue tiles, ivory walls and a white basin on a pale "
        "wood vanity, no mirror.")
SINK = ("the toddler stands on a sturdy step at the left of the basin with the mother right "
        "behind her, and the boy stands at the right")
THREE = f"Exactly three people: {GIRL}, {BOY}, and {MOM}."
FOUR = f"Exactly three people and one toy: {GIRL}, {BOY}, {MOM}, and {TOOTH}."
SCENES2 = {
    3: f"{THREE} {ROOM} At the basin {SINK}. The girl holds a pink toothbrush and the boy a "
       "blue toothbrush at chest height, both looking at them curiously; the mother's hand "
       "rests on the girl's shoulder.",
    4: f"{THREE} {ROOM} At the basin {SINK}. The boy gently brushes his front teeth with a "
       "blue toothbrush while the mother guides the girl's small hand brushing with a pink "
       "toothbrush.",
    5: f"{THREE} {ROOM} Waist-up shot at the basin: {SINK}. The boy, in focus, shows gentle "
       "brushing with his blue toothbrush; the girl watches him with her pink toothbrush "
       "lowered.",
    6: f"{THREE} {ROOM} At the basin {SINK}. The boy brushes with a blue toothbrush and the "
       "mother guides the girl's pink toothbrush. A small colorful sand timer on the counter.",
    7: f"{FOUR} {ROOM} At the basin {SINK}. The tooth mascot stands on the counter between "
       "the children, waving one tiny arm. The girl looks at it delighted, the boy smiles.",
    8: f"{FOUR} {ROOM} At the basin {SINK}. The tooth mascot on the counter brushes its "
       "little teeth with a tiny mint-green toothbrush, copying the boy, who brushes with a "
       "blue toothbrush.",
    9: f"{FOUR} {ROOM} At the basin {SINK}. The tooth mascot on the counter beams with clean "
       "white teeth and two tiny sparkles. The boy gives a thumbs-up; the girl smiles at the "
       "mascot.",
    10: f"{FOUR} {ROOM} At the basin {SINK}. The boy leans over the sink, spitting a little "
        "toothpaste. The mother wipes the girl's chin with a soft washcloth. The tooth mascot "
        "watches from the counter.",
    11: f"Exactly two children and one toy: {GIRL}, {BOY}, and {TOOTH} standing on the floor "
        "between them. A cheerful pastel playroom with paper stars. Wide full-body shot, the "
        "children about to clap with empty hands, big happy smiles.",
    12: f"{FOUR} {ROOM} Wide closing shot: the girl at the left and the boy at the right stand "
        "on the floor, the tooth mascot on the floor between them, the mother behind the girl. "
        "All smile at the camera, the children waving.",
}


def ref_prompt(name):
    return " ".join([REFS[name], CHAR[name], STYLE])


# Mother is a supporting character; a shorter description leaves room in crowded shots.
PARENT_SHORT = "Mother: fair-skinned Indian woman, black low bun, plain teal kurta."
LEAD = "3D animated family film still."


def scene_prompt(n):
    """Shot first (FLUX weighs the start most), style last; the full style line is
    dropped when it would not fit in MAX_TOKENS."""
    if n in SCENES2:
        full = " ".join([LEAD, SCENES2[n], STYLE])
        return full if _tokens(full) <= MAX_TOKENS else " ".join([LEAD, SCENES2[n]])
    cast, bath, shot = SCENES[n]
    chars = [PARENT_SHORT if c == "parent" else CHAR[c] for c in cast]
    parts = [LEAD, shot] + chars + ([BATH] if bath else [])
    full = " ".join(parts + [STYLE])
    return full if _tokens(full) <= MAX_TOKENS else " ".join(parts)


_TOK = []


def _tokens(text):
    if not _TOK:
        from transformers import T5TokenizerFast

        _TOK.append(T5TokenizerFast.from_pretrained("Lightricks/LTX-Video-0.9.5",
                                                    subfolder="tokenizer"))
    return len(_TOK[0](text).input_ids)


def token_counts(prompts):
    return [_tokens(p) for p in prompts]


def imagine(prompts, outputs, seeds=None):
    """One model load for all images; every image uses SEED unless `seeds` is given."""
    sys.path.insert(0, ROOT)
    from img2vid import imagine as im

    w, h = im.size_for("16:9")
    for p, o in zip(prompts, outputs):
        if os.path.exists(o):
            print(f"skip {o} (exists; delete it to regenerate)")
    todo = [(p, o) for p, o in zip(prompts, outputs) if not os.path.exists(o)]
    if not todo:
        return
    if seeds:
        keep = [o for o in outputs if not os.path.exists(o)]
        seeds = [sd for o, sd in zip(outputs, seeds) if o in keep]
        im.generate([p for p, _ in todo], [o for _, o in todo], width=w, height=h,
                    seeds=seeds)
        return
    im.generate([p for p, _ in todo], [o for _, o in todo], width=w, height=h, seed=SEED,
                same_seed=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    os.makedirs(OUT, exist_ok=True)
    if cmd == "refs":
        names = sys.argv[2:] or list(REFS)
        imagine([ref_prompt(n) for n in names],
                 [os.path.join(HERE, "references", f"{n}.png") for n in names])
    elif cmd == "scenes":
        nums = [int(x) for x in sys.argv[2:]] or list(SCENES)
        imagine([scene_prompt(n) for n in nums],
                [os.path.join(HERE, "images", f"{n:02d}.png") for n in nums])
    elif cmd == "try":  # try N SEED SEED ... -> flux/try/NN_sSEED.png candidates
        n, seeds = int(sys.argv[2]), [int(x) for x in sys.argv[3:]] or [1, 2, 3]
        imagine([scene_prompt(n)] * len(seeds),
                [os.path.join(OUT, "try", f"{n:02d}_s{sd}.png") for sd in seeds], seeds)
    else:
        items = [(f"reference_{n}", ref_prompt(n)) for n in REFS] + \
                [(f"{n:02d}", scene_prompt(n)) for n in SCENES]
        for (name, p), count in zip(items, token_counts([p for _, p in items])):
            with open(os.path.join(OUT, name + ".txt"), "w") as f:
                f.write(p + "\n")
            flag = "" if count <= MAX_TOKENS else f"  OVER {MAX_TOKENS}: end is cut off"
            print(f"{name:16s} {count:4d} tokens{flag}")


if __name__ == "__main__":
    main()
