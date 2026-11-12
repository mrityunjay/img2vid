# img2vid

Turn one image or a folder of images into an animated video with music.

`img2vid` looks at each picture, works out where the faces and the subject are, and plans a
video: the order, how long each shot lasts, where the camera moves, transitions, a title and
captions. It then renders the frames itself and muxes in your song with `ffmpeg`.

There are four levels of motion, from fastest to most lifelike:

| Level | What moves | Needs | Speed |
|---|---|---|---|
| `flat` | Camera only: Ken Burns zoom and pan toward faces | nothing extra | real time |
| `3d` | Camera with depth parallax: near things move more than far things | PyTorch + a 100 MB depth model | ~1 min per image |
| `ai` | People and things move (LTX-Video 0.9.5, 480p) | PyTorch + ~24 GB of models | ~5 min per clip on a Mac |
| `animate-hq` | Real actions (hugs, walking), 1600×896 (LTX-Video 0.9.8 + 2× upscaler) | the above + ~6.5 GB more | ~6 min per clip on a Mac |

You can also make clips with any online image-to-video service and drop them in
(`img2vid clips`). Everything runs locally and free. The only optional cloud piece is the
Claude "AI director", which needs an Anthropic API key.

---

## Contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Install](#install): [macOS](#macos), [Linux](#linux-ubuntu--debian), [Windows](#windows)
4. [Models](#models): what gets downloaded, sizes, where it is stored
5. [Quick start](#quick-start)
6. [Commands](#commands)
7. [Workflows](#workflows): photo slideshow, music video, AI motion, online clips
8. [Storyboard JSON reference](#storyboard-json-reference)
9. [Performance and memory](#performance-and-memory)
10. [Troubleshooting](#troubleshooting)
11. [Project layout](#project-layout)
12. [Notes for LLM agents and other AI tools](#notes-for-llm-agents-and-other-ai-tools)

---

## How it works

```
images ─▶ analyze ─▶ storyboard.json ─▶ (optional) motion clips ─▶ render ─▶ video.mp4
          faces,        editable plan     animate / animate-hq /      frames piped
          focus,        (the contract     clips                       to ffmpeg,
          EXIF date     between steps)                                + music
```

- **analyze** (`img2vid/analyze.py`): reads each image, fixes EXIF rotation, finds faces
  (OpenCV Haar cascade) and a focus point, and reads the capture date.
- **storyboard** (`storyboard.py`, or `director.py` when Claude plans it): a JSON file with
  one entry per scene. Every later step reads and writes this file, so you can edit it by
  hand between steps.
- **music** (`music.py`): optional beat tracking (numpy only). It moves scene changes onto the
  song's bar lines and makes the video end with the track.
- **depth** (`depth.py`): Depth-Anything-V2 depth maps for 2.5D parallax.
- **people** (`people.py`): YOLOS person detector, used to keep the old AI model away from
  faces unless you allow it.
- **animate** / **animate_hq** (`animate.py`, `animate_hq.py`): local image-to-video with
  LTX-Video. Each finished clip is cached on disk and attached to its scene as `"clip"`.
- **render** (`render.py`): draws every frame with numpy and OpenCV, including camera moves,
  parallax, clip playback, transitions, the title and captions (with Indic script shaping).
  It pipes the frames to `ffmpeg` (H.264, yuv420p, AAC audio).

---

## Requirements

| | Minimum | Tested with |
|---|---|---|
| OS | macOS 13+, Linux, Windows 10/11 (WSL2 recommended) | macOS 26.7 on Apple M3 Pro |
| Python | 3.10+ | 3.12.15 |
| ffmpeg / ffprobe | any recent build, on `PATH` | 8.1.1 |
| RAM | 8 GB for `flat`/`3d`; **16 GB+** for `ai`/`animate-hq` | 18 GB unified memory |
| GPU for AI motion | Apple Silicon (MPS) or NVIDIA with CUDA and 12 GB+ VRAM; CPU works but is far too slow for AI clips | Apple M3 Pro (MPS) |
| Disk | ~1 GB for `flat`/`3d`; **~35 GB** for all AI models + clip cache | |

Python packages (pinned in `pyproject.toml`):

| Extra | Packages | Used for |
|---|---|---|
| core | `numpy`, `pillow`, `opencv-python-headless>=4.9,<5` | everything; OpenCV 5 drops the Haar face detector, so stay below 5 |
| `[ai]` | `anthropic>=2.8` | the Claude AI director (`--ai`, `--prompt`) |
| `[motion]` | `torch`, `diffusers>=0.43`, `transformers`, `accelerate`, `sentencepiece`, `protobuf`, `huggingface_hub`, `safetensors` | `3d` depth motion, people detection, `ai` and `animate-hq` |

Versions known to work together: torch 2.15.0, diffusers 0.43.0, transformers 5.22.2,
accelerate 1.15.0, huggingface_hub 2.4.0, numpy 2.5.4, pillow 13.0.0, opencv-python-headless
4.14.0.94, anthropic 2.8.0.

---

## Install

Clone or copy this folder, then follow the section for your system. All commands are run
from the project folder (the one containing `pyproject.toml`).

### macOS

```bash
# 1. Tools (Homebrew: https://brew.sh)
brew install python@3.12 ffmpeg libraqm

# 2. Virtual environment and the core tool
python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .

# 3. Optional: Claude director and local AI motion
.venv/bin/pip install -e '.[ai,motion]'

# 4. Optional but needed for Hindi/Indic titles: rebuild Pillow against libraqm
.venv/bin/pip install --force-reinstall --no-deps --no-binary pillow "pillow==13.0.0"
.venv/bin/python -c "from PIL import features; print(features.check('raqm'))"   # must print True
```

The default PyTorch wheel on macOS already supports Apple Silicon GPUs (MPS).

### Linux (Ubuntu / Debian)

```bash
# 1. Tools
sudo apt update
sudo apt install -y python3 python3-venv python3-dev ffmpeg libraqm-dev \
                    libfreetype6-dev libharfbuzz-dev libfribidi-dev \
                    fonts-dejavu fonts-noto-core build-essential

# 2. Virtual environment and the core tool
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .

# 3. Optional: NVIDIA GPU. Install the CUDA build of PyTorch FIRST
#    (pick the command for your CUDA version at https://pytorch.org/get-started/locally/)
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -e '.[ai,motion]'

# 4. Optional: Indic text shaping
.venv/bin/pip install --force-reinstall --no-deps --no-binary pillow "pillow==13.0.0"
.venv/bin/python -c "from PIL import features; print(features.check('raqm'))"   # True
```

Check that the GPU is visible: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.
The tool picks CUDA first, then Apple MPS, then CPU.

### Windows

**Recommended: WSL2 (Ubuntu).** Run `wsl --install` in an admin PowerShell, reboot, open
Ubuntu, and follow the [Linux](#linux-ubuntu--debian) steps. NVIDIA GPUs work inside WSL2
with a current NVIDIA Windows driver.

**Native Windows** also works for `flat`, `3d` and AI motion. Indic text shaping is not
supported natively (Pillow + libraqm is hard to build there), so use WSL2 if you need
Hindi titles.

```powershell
# Tools
winget install Python.Python.3.12
winget install Gyan.FFmpeg        # then open a new terminal so ffmpeg is on PATH

# Virtual environment
py -3.12 -m venv .venv
.venv\Scripts\pip install -U pip
.venv\Scripts\pip install -e .

# NVIDIA GPU: CUDA PyTorch first (see pytorch.org for your CUDA version)
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install -e ".[ai,motion]"
```

Use `.venv\Scripts\img2vid` wherever this README says `.venv/bin/img2vid`. The built-in font
paths are macOS/Linux paths, so on Windows pass a font explicitly, for example
`--font C:\Windows\Fonts\arialbd.ttf` (or `Nirmala.ttc` for Hindi without shaping).

### Check the install

```bash
.venv/bin/img2vid --help
ffmpeg -version
.venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"
```

---

## Models

Models download automatically from Hugging Face the first time a feature needs them. They
are cached in `~/.cache/huggingface/hub`, or under `$HF_HOME` if you set it.

| Model (Hugging Face id) | Used by | Download size | When |
|---|---|---|---|
| `depth-anything/Depth-Anything-V2-Small-hf` | `--motion 3d` (default when torch is installed) | ~100 MB | first 3D render |
| `hustvl/yolos-small` | `--motion ai` people check | ~120 MB | first `ai` run |
| `Lightricks/LTX-Video-0.9.5` (full repo: T5 text encoder ~18 GB, transformer ~3.6 GB, VAE ~2.3 GB) | `--motion ai`, `animate`; its **text encoder is also used by `animate-hq`** | ~24 GB | first `ai` or `animate-hq` run |
| `Lightricks/LTX-Video` → file `ltxv-2b-0.9.8-distilled.safetensors` | `animate-hq` | 6.34 GB | first `animate-hq` run |
| `Lightricks/ltxv-spatial-upscaler-0.9.7` → folder `latent_upsampler/` | `animate-hq` 2× upscale | 0.51 GB | first `animate-hq` run |
| Claude (`claude-opus-5-5` by default) | `--ai` / `--prompt` | cloud API, nothing downloaded | when used |

To download everything ahead of time (useful on slow or metered networks):

```bash
.venv/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("depth-anything/Depth-Anything-V2-Small-hf")
snapshot_download("hustvl/yolos-small")
snapshot_download("Lightricks/LTX-Video-0.9.5")
hf_hub_download("Lightricks/LTX-Video", "ltxv-2b-0.9.8-distilled.safetensors")
snapshot_download("Lightricks/ltxv-spatial-upscaler-0.9.7",
                  allow_patterns=["latent_upsampler/*", "model_index.json"])
EOF
```

Setting `HF_TOKEN` (a free Hugging Face token) gives faster downloads. It isn't required.

**Model licences.** The LTX-Video weights come with Lightricks' own licence (see the
`*.license.txt` in the model repo). Read it before any commercial use.

### img2vid's own caches

| Path | Contents | Safe to delete? |
|---|---|---|
| `~/.cache/img2vid/depth/` | depth maps per image (`.npy`) | yes, recomputed |
| `~/.cache/img2vid/people/` | person counts per image | yes |
| `~/.cache/img2vid/clips/*.npz` | generated AI clips (frames), several hundred MB each at HQ | yes, but regenerating costs minutes per clip |
| `~/.cache/img2vid/clips/*.lat.pt` | HQ latents waiting to be decoded (interrupted run) | keep them to resume; delete to start over |
| `~/.cache/img2vid/clips/embeds/` | cached text-encoder outputs per prompt | yes |

A clip's cache key includes the image bytes, prompt, size, frame count and seed. Changing
any of them makes a new clip, and re-running with the same values reuses the cached one.

---

## Quick start

```bash
# Folder of photos -> 16:9 video with music, as long as the song, scene changes on the beat
.venv/bin/img2vid make ~/Pictures/trip/ -o trip.mp4 \
    --music song.mp3 --fit-music --sync-beats --title "Goa 2026"

# Fast draft (480p, 24 fps) to check timing before the full render
.venv/bin/img2vid make ~/Pictures/trip/ -o draft.mp4 --music song.mp3 --fit-music --preview
```

For anything more than a quick slideshow, use the two-step flow: `plan` writes
`storyboard.json`, you review or edit it, then `render`.

---

## Commands

All commands: `.venv/bin/img2vid <command> --help`.

### `plan`: analyze images and write a storyboard

```bash
img2vid plan IMAGES... [-o storyboard.json] [options]
```

| Option | Default | Meaning |
|---|---|---|
| `IMAGES` | | files and/or folders (`.jpg .jpeg .png .webp .bmp .tif .tiff`; folders are read in name order) |
| `--aspect` | `16:9` | `16:9`, `9:16`, `1:1`, `4:5` |
| `--music FILE` | | audio track; looped or trimmed to fit, faded in and out |
| `--scene-duration S` | `3.5` | seconds per image |
| `--duration S` | | total length; overrides `--scene-duration` |
| `--fit-music` | | make the video exactly as long as `--music` |
| `--sync-beats` | | put scene changes on the song's bar lines (needs `--music`) |
| `--transition` | `crossfade` | `crossfade`, `fadeblack`, `slide`, `cut` |
| `--transition-duration S` | `0.8` | |
| `--order` | `date` | `date` (EXIF capture time), `name`, `given` (command-line order) |
| `--fps` | `30` | output frame rate |
| `--seed N` | `0` | varies motion choices; also the AI seed |
| `--title TEXT` | | title over the first scene (`""` removes one Claude wrote) |
| `--font FILE` | auto per script | TTF/OTF/TTC for title and captions |
| `--ai` | | let Claude plan order, motion, captions and title |
| `--prompt TEXT` | | what the video is about or how it should feel (implies `--ai`) |
| `--model ID` | `claude-opus-5-5` | Claude model for `--ai` |
| `--motion` | `3d` if torch is installed, else `flat` | `flat`, `3d`, `ai` |
| `--motion-prompt TEXT` | built-in "subtle natural motion" | default prompt for `--motion ai` scenes without their own |
| `--ai-people` | off | also AI-animate photos with people (the 0.9.5 model can distort faces) |
| `--ai-max N` | all | AI-animate only the N longest scenes |
| `--ai-seconds S` | `4.0` | AI clip length |
| `--ai-steps N` | `30` | denoising steps (quality vs speed) |
| `--ai-res PX` | `480` | short side of AI clips |

### `render`: render a storyboard to video

```bash
img2vid render storyboard.json -o out.mp4 [--resolution 1080] [--preview]
```

`--resolution` is the short side in pixels (1080 gives 1920×1080 at 16:9). `--preview`
renders at 480p and 24 fps for a fast check.

### `make`: plan and render in one step

Takes all `plan` options plus `-o`, `--resolution`, `--preview`, and `--storyboard-out FILE`
(also save the storyboard it used).

### `animate`: add LTX-Video 0.9.5 clips to a storyboard

```bash
img2vid animate storyboard.json [-o new.json] [--ai-people] [--ai-max N] \
    [--ai-seconds 4] [--ai-steps 30] [--ai-res 480] [--motion-prompt TEXT] [--seed 0]
```

This makes 480p clips with subtle motion (blinks, breathing, lights, water). By default,
scenes with people keep 3D motion. It's the older, lower-quality engine; prefer `animate-hq`.

### `animate-hq`: sharper clips with real actions (recommended AI engine)

```bash
img2vid animate-hq storyboard.json [-o new.json] [--only 1,3,8] \
    [--seconds 5] [--res 448] [--no-upscale] [--seed 0]
```

| Option | Default | Meaning |
|---|---|---|
| `--only` | all scenes | comma-separated scene numbers (1-based) |
| `--seconds` | `5.0` | clip length (121 frames at 24 fps); longer needs more memory and time |
| `--res` | `448` | first-pass short side; the upscaler doubles it (448 → 896, so 1600×896 at 16:9) |
| `--no-upscale` | | skip the 2× upscale and refine pass (faster, softer) |
| `--seed` | `0` | try another seed if one scene comes out badly |

How it works: an 8-step distilled pass at low resolution, a 2× latent upscale, a 4-step
refine pass, then a tiled VAE decode. Every clip is colour-matched back to its source image
(the model drifts cooler and more saturated). Clips play forwards once; when a clip is
shorter than its scene it is slowed to at most 0.75× and then holds its last frame. Scenes
with `"animate": false` are skipped and keep 3D motion.

Each scene's `motion_prompt` drives its clip. Prompts work best with one concrete action,
for example *"The father kneels down and hugs his little daughter tightly. Sea waves roll
gently. Slow camera push in."*

### `clips`: use videos you made elsewhere (online AI services, phone videos)

```bash
img2vid clips storyboard.json FOLDER [-o new.json]
```

Name files by scene number: `01.mp4` (or `01_hug.mp4`, `1.mov`) is scene 1. Accepted types are
`.mp4 .mov .m4v .webm .mkv`. Scenes without a file fall back to 3D motion. Audio in the
clips is ignored; the storyboard's `music` is used. Then run `render` as usual.

---

## Workflows

### 1. Photo slideshow (fast, no AI)

```bash
img2vid plan photos/ -o story.json --aspect 9:16 --music song.mp3 --fit-music --motion flat
img2vid render story.json -o reel.mp4
```

### 2. Let Claude plan it (needs an Anthropic API key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
img2vid plan photos/ -o story.json --prompt "Navaratri with family, warm and festive" \
    --music song.mp3 --fit-music --sync-beats
img2vid render story.json -o festive.mp4
```

Claude sees each photo downscaled to 1024 px. It drops duplicates, orders the photos, picks
each shot's zoom or pan target, writes a title, captions and a per-scene `motion_prompt`. If
the key is missing or the call fails, the offline planner is used instead.

### 3. Music video with real character motion (local, free)

This is the flow used for `projects/pappa/`:

```bash
# a) Plan: 16:9, scenes synced to the song's bars, 3D motion as the fallback
img2vid plan pics/ -o projects/song/storyboard.json --order given \
    --music song.mp3 --fit-music --sync-beats --transition crossfade \
    --transition-duration 1.4 --title "My title"

# b) Edit storyboard.json: remove duplicates or grids, reorder scenes, and write one
#    "motion_prompt" per scene (one clear action). Set "animate": false on images with
#    painted-in text, because AI video scrambles text.

# c) Generate clips. Long job; see "Performance". Run it in the background and keep the
#    machine awake (macOS: caffeinate).
caffeinate -i -s .venv/bin/img2vid animate-hq projects/song/storyboard.json

# d) Render with the song
.venv/bin/img2vid render projects/song/storyboard.json -o projects/song/video.mp4
```

To redo one bad scene, change its prompt or `--seed` and run `animate-hq --only N`, then
`render` again. Other clips come from the cache.

### 4. Clips from an online service (Kling, Hailuo, Google Flow/Veo, Pika, …)

1. For each scene, upload its image and paste its prompt. Choose 16:9 and the longest free
   length.
2. Save the results as `01.mp4`, `02.mp4`, … in one folder.
3. Run `img2vid clips storyboard.json that_folder/`, then `img2vid render ...`.

`projects/pappa/online_prompts.py` shows how to generate a numbered image folder plus a
`PROMPTS.md` sheet from a storyboard.

---

## Storyboard JSON reference

Top level:

| Field | Example | Meaning |
|---|---|---|
| `aspect` | `"16:9"` | `16:9`, `9:16`, `1:1`, `4:5` |
| `fps` | `30` | output frame rate |
| `music` | `"/path/song.mp3"` | audio track or `null` |
| `title` | `"पप्पा जल्दी आ जाना"` | shown over scene 1 |
| `font` | `"/path/font.ttf"` | optional font override |
| `transition_duration` | `1.4` | seconds of overlap between scenes |
| `fade_in`, `fade_out` | `0.6`, `1.0` | video fade at start and end, in seconds |
| `beat_period` | `0.6966` | written by `--sync-beats` (seconds per beat), informational |
| `scenes` | `[...]` | list, in play order |

Per scene:

| Field | Values | Meaning |
|---|---|---|
| `image` | path | source image |
| `duration` | seconds | includes the transition overlap |
| `motion` | `zoom_in`, `zoom_out`, `pan` | camera move |
| `zoom` | `1.0`–`1.5` | tightest zoom (capped at 1.08 with an AI clip, 1.04 with a video file) |
| `focus` | `[x, y]` in 0..1 | point the camera moves toward (faces by default) |
| `fit` | `cover`, `contain` | crop to fill, or show whole on a blurred copy (auto when the aspect differs a lot) |
| `transition` | `crossfade`, `fadeblack`, `slide`, `cut` | hand-over to the next scene |
| `caption` | text | lower-third caption |
| `faces` | int | faces found by `analyze` |
| `depth` | `true`/`false` | use 3D depth parallax when there is no clip |
| `motion_prompt` | text | what should move, for `animate` / `animate-hq` |
| `animate` | `true` / `false` / absent | force AI motion on or off for this scene |
| `clip` | path | attached motion clip: `.npz` (generated) or a video file |
| `clip_speed` | `0.8` | playback speed for 0.9.5 `.npz` clips (ping-pong loop) |
| `clip_loop` | `"forward"` | play once forwards then hold the last frame (set by `animate-hq`) |
| `video_speed` | `0.5`–`1.0` | override the automatic slow-down for video-file clips |

---

## Performance and memory

Times are measured on an Apple M3 Pro with 18 GB of memory, for 16:9 clips:

| Step | Time |
|---|---|
| `flat` render, 4 min of 1080p video | ~10 min |
| 3D depth map | a few seconds per image (cached) |
| `ai` (0.9.5) clip, 864×480, 97 frames, 30 steps | 3–6 min denoise + ~25 s decode |
| `animate-hq` clip, 1600×896, 121 frames | ~4 min denoise + ~1.7 min decode |
| T5 text-encoder load (once per run, only for new prompts) | 3–5 min |

Memory safeguards built in:

- The 18 GB T5 text encoder runs in a **separate process**, so its memory goes back to the
  OS before video generation starts. Prompt embeddings are cached.
- Denoising runs for all clips first, then the transformer is freed, then decoding happens
  in time and space tiles. Decoding with the transformer still loaded caused heavy swapping.
- On Apple MPS, memory is capped at 90% of the recommended working set, so a too-large job
  fails with an out-of-memory error instead of freezing the Mac. Without the cap, one run
  ran out of memory and the Mac **rebooted**.
- `animate-hq` saves each clip's latents to disk as soon as it is denoised, so an
  interrupted run resumes and only loses the clip in progress.

Keep the machine awake and on power during long runs. A sleeping laptop pauses the job:
one clip took 100 minutes instead of 4. On macOS, `caffeinate -i -s <command>` prevents
sleep **only while on the charger**; closing the lid can still sleep the Mac.

On NVIDIA GPUs with 24 GB of VRAM you can raise `--res` (for example 576) or `--seconds`.
With less than about 12 GB, use `--no-upscale`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ffmpeg not found on PATH` | install ffmpeg (see Install) and open a new terminal |
| `AttributeError: ... CascadeClassifier` | OpenCV 5 is installed: `pip install "opencv-python-headless>=4.9,<5"` |
| Hindi or other Indic text has misplaced vowel signs | Pillow lacks libraqm; do the Pillow rebuild step and check `features.check('raqm')` is `True` |
| `AI director unavailable (no Claude credentials found ...)` | set `ANTHROPIC_API_KEY`, or drop `--ai`/`--prompt` |
| `3D motion needs torch and transformers` | `pip install -e '.[motion]'` |
| Tokenizer error mentioning protobuf | `pip install protobuf sentencepiece` (both are in `[motion]`) |
| `MPS backend out of memory` while decoding | close other apps; lower `--res` or `--seconds`; latents are kept, so re-run the same command to retry only the decode |
| `MPS backend out of memory` while denoising | lower `--res`/`--seconds`, or `--no-upscale` |
| A job is extremely slow | the machine slept, or it's swapping; check Activity Monitor → Memory → Swap Used and close other apps |
| AI clip distorts a face | change that scene's `motion_prompt` to a smaller action, try `--seed 1`, or set `"animate": false` for 3D motion |
| Words painted in an image get scrambled | set `"animate": false` for that scene |
| Video is longer or shorter than the song | plan with `--fit-music` (and `--sync-beats`); `render` never changes durations |
| CUDA machine runs on CPU | install the CUDA build of torch *before* `.[motion]`; check `torch.cuda.is_available()` |

---

## Project layout

```
img2vid/
  cli.py          command line (plan, make, animate, animate-hq, clips, render)
  analyze.py      image loading, EXIF, faces, focus point
  storyboard.py   offline planner, validation, load/save
  director.py     Claude planner (structured JSON output)
  music.py        beat tracking and bar snapping
  depth.py        Depth-Anything-V2 depth maps
  people.py       YOLOS person detection
  animate.py      LTX-Video 0.9.5 image-to-video
  animate_hq.py   LTX-Video 0.9.8 distilled + 2x upscaler, colour matching
  render.py       frame renderer, text, transitions, ffmpeg muxing
projects/pappa/   example music-video project (storyboard + prompt scripts)
pyproject.toml    package definition and dependency extras
```

The `projects/pappa/` scripts are examples of driving the tool from Python:

- `motion_prompts.py` / `hq_prompts.py` write per-scene prompts into `storyboard.json`.
- `online_prompts.py` makes `online_clips/` (numbered images + `PROMPTS.md`) for online services.

---

## Notes for LLM agents and other AI tools

If you are an AI coding assistant (Claude Code, Cursor, Copilot, Codex, Gemini CLI, …)
operating this tool for a user:

- **Use the CLI.** Don't reimplement the pipeline. The flow is
  `plan → (edit JSON) → animate-hq / clips → render`. `storyboard.json` is the contract;
  edit it with a JSON library and keep scene order and field names exactly as documented
  above. Run `img2vid <cmd> --help` for the current options rather than relying on memory.
- **Long jobs:** `animate`, `animate-hq` and full `render` runs take minutes to hours. Run
  them in the background, log to a file, and check the log. Lines `clip N/M ... denoised in
  Xs` and `scene N: decoded in Xs` show progress, and `wrote <file>` means done. Don't start
  a second GPU job while one is running: they compete for memory.
- **Don't kill a run to "retry faster".** Finished `animate-hq` latents are on disk
  (`*.lat.pt`), but 0.9.5 `animate` keeps latents in memory until the end.
- **Ask the user before downloading models** (sizes are in [Models](#models)) or installing
  packages. Downloads are tens of GB.
- **Keep the machine awake** (`caffeinate` on macOS) and tell the user to stay on power.
- **Reviewing results:** pull frames with
  `ffmpeg -ss <t> -i out.mp4 -frames:v 1 frame.jpg` at the middle of each scene (scene start
  times follow from `duration` minus `transition_duration`). For a generated clip, load it
  with `numpy.load(path)["frames"]` (BGR uint8, N×H×W×3). Check faces for drift and painted
  text for scrambling.
- **Writing prompts:** one concrete action per scene, plus environment motion and a camera
  hint, in English, under ~120 words. Avoid two simultaneous actions for one person. Use
  `"animate": false` for images with text in them.
- **Secrets:** the only credential is `ANTHROPIC_API_KEY`, read from the environment. Never
  write it into files.
