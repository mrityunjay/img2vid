# Alphabet Adventure — G to P

The next lesson covers **Giraffe, Hat, Ice cream, Juice, Kite, Lion, Monkey, Nest,
Orange and Penguin**. Its script introduces uppercase/lowercase pairs, two
five-letter recaps, and three picture quizzes with answers **M, K, P**.

The full narration is now generated locally with Kokoro-82M, without Suno or a
hosted voice service. The current recording is approximately **3 minutes 26 seconds**.

## Audio files

- `audio/g_p_lesson.mp3`: complete narration with quiet original instrumental backing.
- `audio/g_p_lesson.voice.wav`: narration alone.
- `audio/g_p_lesson.timeline.json`: exact speech/pause timestamps and section labels.
- `audio/g_p_lesson.srt`: timed English captions.
- `audio/voice_preview.wav`: short sample of the selected voice.

The lesson uses the `af_heart` voice at speed `0.86`. Repeat/recap pauses are two
seconds; quiz pauses are three seconds. The clean voice track contains real silence
in these gaps; gentle background music continues in the mixed MP3.

## Rebuild

From the repository root:

```bash
.venv/bin/python projects/alphabet_adventure_g_p/prepare_audio.py --force
```

The helper reads the approved script from `SUNO_SCRIPT.md`, writes the structured
`local_script.json`, and runs `img2vid speak`. Omit `--force` for the first run.
Model installation and all voice commands are documented in the main README.

## Verification

The clean narration has no clipped samples, all response gaps contain exact silence,
and its duration matches the generated timeline. FFmpeg decoded the full MP3 without
reported errors. A fresh utterance was also synthesized with outbound socket
connections blocked, verifying offline inference with the installed model.
The audio module's 10 focused tests passed.

## Video status

The script and local soundtrack are prepared. G–P illustrations, animations and
final video assembly are the next production stage; no G–P video has been rendered yet.
