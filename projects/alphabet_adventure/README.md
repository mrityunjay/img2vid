# Alphabet Adventure — A to F

An English preschool participation video: look, listen, repeat, then point to the
answer. Suitable for approximately ages 3–5 with a caregiver.

The lesson teaches **letter names, uppercase/lowercase recognition, and familiar
word associations**. It does not teach isolated phonemes.

| Letter | Word | Motion |
|---|---|---|
| A a | Apple | Small leaf sway |
| B b | Ball | Gentle rocking |
| C c | Cat | Blink and tail movement |
| D d | Dog | Tail wag |
| E e | Elephant | Small trunk sway |
| F f | Fish | Gentle fin and tail movement |

Each letter scene asks the child to say the word, waits three seconds, and repeats
the answer. A recap gives four seconds for saying all six letters. Three review
questions ask the child to point to a letter, wait four seconds, then reveal and
speak the answer. The waiting dots show time remaining; there is no penalty for
needing longer. Pause playback whenever the child wants more time.

## Deliverables

- `alphabet_adventure_A-F.mp4`: final 1920×1080, 24 fps video with English narration.
- `captions.srt` / `captions.vtt`: optional English captions; the MP4 also includes
  a selectable English subtitle track.
- `SCRIPT.md`: exact narration with response pauses.
- `timeline.json`: exact timing for each section and visual card.
- `review/contact-sheet.jpg`: frames extracted from the finished video.
- `images/01.png`–`06.png`: original generated source artwork.
- `lesson.json`, `image_prompts.txt`, `build.py`: editable lesson and reproducible workflow.
- `youtube/YOUTUBE.md`: title, description, tags and upload settings for the Suno version.
- `youtube/thumbnail.jpg`: bright 1280×720 upload-ready thumbnail; PNG and a small
  readability preview are included alongside it.

Open the MP4 in QuickTime, VLC or another local video player. Children answer aloud
and point at the screen. The MP4 has fixed answer pauses; it does not listen to or
grade the child, and it has no clickable answers.

## Completed output and verification

The generated MP4 is **173.83 seconds**, **1920×1080 at 24 fps**, approximately
**21.9 MB**, with H.264 video, AAC English audio and selectable English subtitles.
All six source images and opening/middle/ending animation frames were visually
reviewed. The final video decodes without reported FFmpeg errors. All 56 visual
states were checked against their expected cards and timing, including the three
quiz reveals after the response pauses. The source audio has no clipped samples,
and the response pauses contain silence.

## Production

### Version using the supplied Suno audio

`alphabet_adventure_A-F_suno.mp4` uses the complete supplied **A is for Apple.mp3**
at its original speed. This version is **133.54 seconds**, **1920×1080 at 24 fps**,
approximately **18.8 MB**, with stereo English audio and re-timed subtitles.
The original narrated video remains available under its original filename.

Scene boundaries and quiz reveals follow the recording's word timestamps, obtained
with the already-installed macOS SpeechTranscriber. `song_transcript.json` contains
the automatic recognition output, which has recognition errors; it is used for
timing, while caption wording comes from the lesson script and the audio's embedded
lyrics. `suno_timing.json` records the selected cues and source-file hash.

Suno did not retain the original three-/four-second response pauses. This version
preserves the supplied audio without adding gaps or changing its speed; pause
playback when a child needs more time. Subsecond pauses do not flash a countdown.

All 25 visual states were checked against their new cues. The full video decodes
without reported FFmpeg errors, and the decoded soundtrack correlates 0.999893
with the supplied audio, confirming that the earlier synthetic narration was replaced.

Rebuild this version with:

```bash
caffeinate -i -s .venv/bin/python projects/alphabet_adventure/use_suno_audio.py \
  "$HOME/Downloads/A is for Apple.mp3"
```

`use_suno_audio.py` verifies the audio hash before applying these recording-specific
timings. A different recording requires new cue timings.

### Original narrated version

All image generation and animation use the existing `img2vid` CLI. The original
artwork is generated locally with FLUX.1-schnell at 1280×720. Approved art is animated
with LTX-Video. Final letters, words and quiz choices are drawn with Pillow **after**
AI generation, ensuring correct spelling and stable typography.

The final artwork panel is 1024×576 inside a 1080p lesson layout. Export resolution
does not imply native 1080p generated artwork. Speech uses the installed macOS
Samantha voice with a requested rate of 145 words per minute and an additional
pitch-preserving 0.8× tempo adjustment for a gentler preschool pace. Short, quiet three-note chimes are synthesized
locally; there is no third-party music. Model files are used in offline mode.

The model's idle motions are deliberately small and reversible; the editor uses
gentle ping-pong playback to keep movement going during questions without stretching
a four-second clip across the entire scene. Typographic changes and audio stay
aligned to 24 fps boundaries.

## Rebuild

From the `plan_project` repository root:

```bash
# Generate new artwork only when needed; this overwrites numbered image files.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 caffeinate -i -s \
  .venv/bin/img2vid imagine \
  -f projects/alphabet_adventure/image_prompts.txt \
  -o projects/alphabet_adventure/images --size 1280x720 --steps 4 --quant q4 --seed 81

# Generate narration, the exact timeline, captions and typography cards.
.venv/bin/python projects/alphabet_adventure/build.py prepare

# Review the six images before planning and animating.
# plan refuses to overwrite an existing motion.json containing clip attachments.
.venv/bin/python projects/alphabet_adventure/build.py plan
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 caffeinate -i -s \
  .venv/bin/img2vid animate-hq projects/alphabet_adventure/motion.json \
  --seconds 4 --res 384 --seed 21

.venv/bin/python projects/alphabet_adventure/build.py edit
caffeinate -i -s .venv/bin/python projects/alphabet_adventure/build.py finish
.venv/bin/python projects/alphabet_adventure/build.py review
```

Keep the Mac plugged in with its lid open. Do not run image and video generation
simultaneously; both use GPU memory. No model downloads or package installation are
required when the existing caches are complete.

This project uses English macOS speech and system fonts. To rebuild on another OS,
adapt the speech command and font paths in `build.py`.

## Creative choices

The recommended design uses one recognizable object per image and accurate text
overlays. This avoids unnecessary character-consistency problems and keeps the
learning target clear. Alternatives for a future version are a recurring child
presenter, or a phonics lesson with reviewed human-recorded phonemes. Neither is
required for this letter-name lesson.
