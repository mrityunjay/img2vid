# Brush Brush Karo

A 12-scene production pack for Meera, Bunny and the friendly tooth mascot Toothy.
Artwork is generated locally with `img2vid imagine` (FLUX.1-schnell) via `flux_prompts.py`;
the song goes in `audio/`. Images, audio and video are not stored in git.

Quick path used for the first cut (beat-synced dance edit, no AI motion):

```bash
.venv/bin/python projects/brush_brush_karo/flux_prompts.py refs        # 4 character portraits
.venv/bin/python projects/brush_brush_karo/flux_prompts.py scenes      # images/01-12.png
.venv/bin/python projects/brush_brush_karo/flux_prompts.py try 7 1 2 3 # extra candidates for one scene
.venv/bin/python projects/brush_brush_karo/dance.py                    # dance.json: a cut per bar, beat kicks
.venv/bin/img2vid render projects/brush_brush_karo/dance.json -o projects/brush_brush_karo/brush_brush_karo.mp4
```

FLUX has no reference-image input, so `flux_prompts.py` condenses the PROMPTS.md text to
fit its 256-token limit and avoids character names (they caused duplicate figures and
rabbit ears on "Bunny"). Review every image; crowded shots often need a few seeds.

## Files and input contract

- `PROMPTS.md`: all copy-ready reference, image and animation prompts.
- `prompts/`: one text file per prompt for pasting into an image or video model.
- `scenes.json`: source brief, character definitions, environment, reference dependencies
  and the 12 scenes. Edit this file and re-export to revise the prompts.
- `references/`: save approved `meera.png`, `bunny.png`, `toothy.png`, `parent.png` here.
- `images/`: save approved scenes as `01.png` through `12.png`.
  `.jpg`, `.jpeg` and `.webp` also work; use exactly one file per scene number.
- `audio/`: put the original song here when available.
- `prepare.py`: exports prompts, checks input images, and invokes the existing
  `img2vid plan` CLI before adding scene motion prompts to its real storyboard.

The image prompts request native 3840×2160 where the chosen generator supports it.
At least 1920×1080 is preferable. A prompt cannot force unsupported resolution or
guarantee identity; use reference-image conditioning and review the actual artwork.
The checker verifies file readability and aspect ratio, not character identity.

## Generate and approve artwork

1. Generate the four individual reference portraits. Keep the best single image for each.
2. Generate scene 01 using Meera and Bunny's reference images. Approve their faces,
   proportions, clothing and style before continuing.
3. Generate scenes in number order. Scene 03 establishes the bathroom layout and
   Mother, scene 07 establishes Toothy's counter scale, scene 09 its clean smile.
   Each scene prompt lists the reference images to upload.
4. Check hands and brush ownership, number of characters, consistency, floor contact,
   clear silhouettes and absence of mirror duplicates. Reject identity changes before animation.
5. Save approved scene files in `images/`. Reference portraits are kept separately
   so they cannot accidentally become video scenes.

`../images/01.png` in a prompt's reference list means the approved scene 01 image,
relative to `references/`. If using JPG/WebP, upload the corresponding file instead.
If a model limits reference count, prioritize the characters visible in that shot
and its approved bathroom scene. An edit-based generation mode can preserve the set.

## Changes from the supplied draft

| Draft issue | Prepared version and reason |
|---|---|
| Generic “exact same characters” instruction | Fixed skin, shoe, bow, height and bathroom details plus reference dependencies reduce redesign ambiguity. |
| Full-body rule conflicts with scene 05 waist-up framing | Foot visibility is required only for full-body shots. |
| Large mirror behind children | Angled mirror shows no reflected faces, reducing duplicate-character errors. |
| Several actions per clip | One principal action; scene 08 animates Toothy while Bunny holds the demonstration pose. |
| Meera claps while holding a brush | Brushes are put away before the scene 11 clap. |
| Mascot must appear from nowhere | Toothy starts visible and waves, avoiding an unnecessary object-generation transition. |
| Toddler brushing alone / dancing with brush in mouth | Mother guides Meera; brushing shots keep feet planted. Celebration happens after brushes are put away. |
| “Rinse and spit” with a toddler's water cup | Scene 10 shows Bunny spitting and Mother helping Meera clean up, without teaching a toddler to rinse or swallow toothpaste. |
| 12 short clips presented as 2–3 minutes of motion | The 150-second edit is a timing draft; 12 five-second clips supply only about 60 seconds of generated motion before transitions. |

The brushing shots illustrate a routine; a short sand-timer shot does not demonstrate
two elapsed minutes. Keep detailed dental instructions out of unreviewed generated
lyrics; Mother supplies the toddler's brushing assistance.

## Run with the installed tool

Run these commands from the `plan_project` repository root. No new packages are needed
for prompt export, input checks or a flat preview.

```bash
.venv/bin/python projects/brush_brush_karo/prepare.py export
.venv/bin/python projects/brush_brush_karo/prepare.py check

# Once all twelve images are present:
.venv/bin/python projects/brush_brush_karo/prepare.py plan

# Or, INSTEAD of the preceding plan command, fit the edit to your actual song:
.venv/bin/python projects/brush_brush_karo/prepare.py plan \
  --music projects/brush_brush_karo/audio/song.mp3 --sync-beats

# Check framing and order before spending time on AI clips:
.venv/bin/img2vid render projects/brush_brush_karo/storyboard.json \
  -o projects/brush_brush_karo/preview.mp4 --preview
```

Without audio, planning produces a silent 150-second draft including transition overlap.
With audio, `--fit-music` uses the actual track duration. Beat snapping is optional;
it does not understand lyrics or synchronize mouths. Review the scene timing against
the song before the expensive animation pass. An existing `storyboard.json` is never
overwritten by the preparation script; preserve it or move it aside before replanning.

## Animate and render

Start with scene 01 after the artwork is approved. These commands run in offline mode
so missing model files cause an error instead of an unrequested multi-GB download.
Existing model-cache directories were found during setup; that alone does not prove
every required model file is present. Ask before downloading missing weights.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 caffeinate -i -s \
  .venv/bin/img2vid animate-hq projects/brush_brush_karo/storyboard.json \
  --only 1 --seconds 5 --res 448 --seed 0 \
  > projects/brush_brush_karo/animate-scene-01.log 2>&1

# Review scene 01 before doing the remaining scenes:
.venv/bin/img2vid render projects/brush_brush_karo/storyboard.json \
  -o projects/brush_brush_karo/preview.mp4 --preview

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 caffeinate -i -s \
  .venv/bin/img2vid animate-hq projects/brush_brush_karo/storyboard.json \
  --only 2,3,4,5,6,7,8,9,10,11,12 --seconds 5 --res 448 --seed 0 \
  > projects/brush_brush_karo/animate-remaining.log 2>&1

.venv/bin/img2vid render projects/brush_brush_karo/storyboard.json \
  -o projects/brush_brush_karo/brush_brush_karo_1080p.mp4 --resolution 1080
```

On macOS, stay plugged in and leave the lid open. Run only one GPU job at a time.
Long jobs should run in a background terminal/session; the log records progress.
`--res 448` plus the tool's latent upscaler yields approximately 1600×896 motion clips;
a 1080p or 4K export scales those clips and does not make them native 4K animation.

The tool plays HQ clips forward once, slowed to fill each scene. For 150 seconds,
five-second clips would be stretched to roughly 12.9 seconds each, which can look
too slow for a children's song. Treat this as a first edit; for a lively final song,
use additional approved takes/cutaways and chorus reprises in the editor, or a provider
capable of longer stable clips. Do not run long local clips without checking memory.

Inspect each clip's beginning, middle and end, especially hands, mouths and the
guided-brushing shots. If a scene deforms, simplify its motion or use the flat fallback:
remove that scene's `clip`, set `animate` to `false`, and keep `depth` false.
The project uses flat fallback deliberately so it does not silently download a depth model.

Alternatively, generate clips elsewhere using the same `.motion.txt` files. Save them
as `01.mp4` through `12.mp4` in an `online_clips/` directory, then:

```bash
.venv/bin/img2vid clips projects/brush_brush_karo/storyboard.json \
  projects/brush_brush_karo/online_clips \
  -o projects/brush_brush_karo/storyboard-online.json
.venv/bin/img2vid render projects/brush_brush_karo/storyboard-online.json \
  -o projects/brush_brush_karo/brush_brush_karo_online_1080p.mp4 --resolution 1080
```

## Prompt review

RASCEF brief audit, scored manually 0–2: role 1→2, audience 2→2, subject 2→2,
constraints 1→2, expected result 2→2, format 2→2. This checks whether the brief
communicates the intent; it is not a measured image-quality result.

Manual review, 1–5: clarity 4→5 (single actions), specificity 4→5 (fixed identity/set),
completeness 4→4 (artwork/audio still missing), consistency 3→4 (references needed),
actionability 3→5 (files and CLI preparation). Mean 3.6→4.6. Model output quality and
character preservation remain untested until images and clips are produced.

Recommended: reference-conditioned scene generation plus restrained local HQ motion.
Alternative: edit the approved scene image to preserve the bathroom and change poses.
Alternative: use the same source images with an external video model, then attach clips.
All retain the same creative brief. Review visual results before batch generation.
