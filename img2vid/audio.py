"""Offline Kokoro narration, explicit pauses, timestamps, and optional procedural music."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave

import numpy as np

RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
FILES = {
    "kokoro-v1.0.onnx": (325505369, "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a"),
    "voices-v1.0.bin": (28214398, "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
}
RATE = 24000
STYLES = {"neutral": 1.0, "gentle": 0.9, "bright": 1.04}
LETTER_NAMES = dict(zip(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ay bee see dee ee eff gee aitch eye jay kay ell em en oh pee cue ar ess tee you vee double-you ex why zee".split(),
))


def cache_dir():
    return Path(os.environ.get("IMG2VID_AUDIO_CACHE", Path.home() / ".cache/img2vid/audio")).expanduser()


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download_models():
    """Explicit, checksum-verified download. Ordinary inference never calls this."""
    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name, (size, sha256) in FILES.items():
        target = root / name
        if target.exists() and target.stat().st_size == size and digest(target) == sha256:
            print(f"cached: {target}", file=sys.stderr)
            continue
        print(f"downloading {name} ({size / 1e6:.1f} MB)...", file=sys.stderr)
        with tempfile.NamedTemporaryFile(dir=root, prefix=name + ".", suffix=".part", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with urllib.request.urlopen(f"{RELEASE}/{name}", timeout=90) as source, temporary.open("wb") as out:
                total = 0
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > size:
                        raise ValueError(f"unexpected download size for {name}")
                    out.write(chunk)
            if temporary.stat().st_size != size or digest(temporary) != sha256:
                raise ValueError(f"download integrity check failed for {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Audio models ready: {root}", file=sys.stderr)


def model_files():
    root = cache_dir()
    for name, (size, checksum) in FILES.items():
        path = root / name
        if not path.exists() or path.stat().st_size != size or digest(path) != checksum:
            raise RuntimeError(
                f"Missing or damaged audio model: {path}. "
                "Run 'img2vid speak --download-models' once (354 MB)."
            )
    return root / "kokoro-v1.0.onnx", root / "voices-v1.0.bin"


def voices():
    _, path = model_files()
    with np.load(path, allow_pickle=False) as bank:
        return sorted(name for name in bank.files if name.startswith(("af_", "am_", "bf_", "bm_")))


def pause(seconds, section=""):
    seconds = float(seconds)
    if not math.isfinite(seconds) or not 0 <= seconds <= 30:
        raise ValueError("pause duration must be a finite value from 0 to 30 seconds")
    return {"type": "pause", "seconds": seconds, "section": section}


def parse_script(content, default_pause=2.0, is_json=False):
    """Read plain text, one fenced Suno script, or structured JSON; never read stage directions."""
    pause(default_pause)
    if is_json:
        items = json.loads(content)
        if not isinstance(items, list):
            raise ValueError("JSON scripts must be a list of text/pause objects")
        result = []
        for item in items:
            if not isinstance(item, dict) or ("text" in item) == ("pause" in item):
                raise ValueError("each JSON segment needs exactly one of 'text' or 'pause'")
            section = str(item.get("section", ""))
            if "pause" in item:
                result.append(pause(item["pause"], section))
            elif isinstance(item["text"], str) and item["text"].strip():
                result.append({"type": "speech", "text": item["text"].strip(), "section": section})
            else:
                raise ValueError("speech segments need nonempty text strings")
    else:
        fences = re.findall(r"```(?:text|plaintext)?[ \t]*\n(.*?)```", content, re.S)
        if fences:
            if len(fences) != 1:
                raise ValueError("provide one fenced narration script, or a plain text file")
            content = fences[0]
        result, lines, section = [], [], ""

        def flush():
            if lines:
                result.append({"type": "speech", "text": " ".join(lines), "section": section})
                lines.clear()

        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                flush()
            elif line.startswith("#"):
                flush()
                section = line.lstrip("# ").strip()
            elif line.startswith("[") and line.endswith("]"):
                flush()
                directive = line[1:-1].strip()
                explicit = re.fullmatch(r"pause\s*:?\s*(\d+(?:\.\d+)?)\s*(?:s|seconds?)?", directive, re.I)
                if explicit:
                    result.append(pause(explicit[1], section))
                elif directive.lower() in ("short instrumental break", "instrumental break"):
                    result.append(pause(default_pause, section))
                elif directive.lower() == "gentle instrumental ending":
                    result.append(pause(2, section))
                elif re.fullmatch(r"(?:spoken\s+)?(?:intro|verse(?:\s+\d+)?|recap|quiz(?:\s+\d+)?|outro|chorus)", directive, re.I):
                    section = directive
                else:
                    raise ValueError(f"unknown stage direction [{directive}]; use [pause 2]")
            else:
                lines.append(line)
        flush()
    if not any(item["type"] == "speech" for item in result):
        raise ValueError("the script contains no spoken text")
    return result


def pronounce_letters(text, british=False):
    """Opt-in for alphabet lessons: uppercase single letters are names, not words."""
    names = dict(LETTER_NAMES)
    if british:
        names["Z"] = "zed"
    text = re.sub(r"\blowercase\s+([a-z])\b",
                  lambda m: "lowercase " + names[m[1].upper()], text, flags=re.I)
    return re.sub(r"\b[A-Z]\b", lambda m: names[m[0]], text)


def load_engine():
    try:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
    except ImportError as error:
        raise RuntimeError("Local speech needs: pip install -e '.[audio]'") from error
    model, bank = model_files()
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session, str(bank))


def synthesize(items, engine, voice, speed, spell_letters=False):
    parts, timeline = [], []
    samples = 0
    spoken_count = 0
    total = sum(item["type"] == "speech" for item in items)
    language = "en-gb" if voice.startswith("b") else "en-us"
    clip_cache = cache_dir() / "utterances"
    clip_cache.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(items):
        entry = dict(item)
        if item["type"] == "pause":
            sound = np.zeros(round(item["seconds"] * RATE), dtype=np.float32)
        else:
            text = pronounce_letters(item["text"], language == "en-gb") if spell_letters else item["text"]
            entry["spoken_text"] = text
            key = hashlib.sha256(json.dumps(
                ["kokoro-onnx-0.6.1", FILES["kokoro-v1.0.onnx"][1], text, voice, speed, language]
            ).encode()).hexdigest()
            cached = clip_cache / f"{key}.npz"
            if cached.exists():
                with np.load(cached, allow_pickle=False) as data:
                    sound = data["samples"]
            else:
                sound, sample_rate = engine.create(text, voice=voice, speed=speed, lang=language)
                if sample_rate != RATE:
                    raise RuntimeError(f"unexpected speech sample rate: {sample_rate}")
                sound = np.asarray(sound, dtype=np.float32)
                if sound.ndim != 1 or not len(sound) or not np.isfinite(sound).all():
                    raise RuntimeError("speech model returned invalid audio")
                # Atomic writes allow safe resume after interruption.
                with tempfile.NamedTemporaryFile(dir=clip_cache, suffix=".npz", delete=False) as temp:
                    temporary = Path(temp.name)
                try:
                    np.savez(temporary, samples=sound)
                    temporary.replace(cached)
                finally:
                    temporary.unlink(missing_ok=True)
            spoken_count += 1
            print(f"speech {spoken_count}/{total}: {len(sound) / RATE:.1f}s — {item['text'][:65]}", file=sys.stderr)
        entry.update(start=round(samples / RATE, 6), end=round((samples + len(sound)) / RATE, 6))
        timeline.append(entry)
        parts.append(sound)
        samples += len(sound)
        if item["type"] == "speech" and i + 1 < len(items) and items[i + 1]["type"] == "speech":
            gap = np.zeros(round(0.2 * RATE), dtype=np.float32)
            timeline.append({"type": "pause", "section": item["section"], "seconds": 0.2,
                             "start": round(samples / RATE, 6),
                             "end": round((samples + len(gap)) / RATE, 6)})
            parts.append(gap)
            samples += len(gap)
    audio = np.concatenate(parts)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-5:
        raise RuntimeError("speech model produced silent audio")
    audio *= 0.82 / peak
    return audio, timeline


def music_bed(length, timeline, volume=0.07):
    """An original, deterministic instrumental bed—not a prompt-to-song AI model."""
    bed = np.zeros(length, dtype=np.float32)
    beat = 60 / 88
    chords = [(48, 52, 55), (45, 48, 52), (41, 45, 48), (43, 47, 50)]

    def note(start, midi, seconds, gain, pad=False):
        offset = round(start * RATE)
        n = min(round(seconds * RATE), length - offset)
        if n <= 0:
            return
        t = np.arange(n, dtype=np.float32) / RATE
        hz = 440 * 2 ** ((midi - 69) / 12)
        signal = np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(4 * np.pi * hz * t)
        envelope = np.minimum(t / 0.02, 1) * np.exp(-t * (1.0 if pad else 4.5))
        envelope *= np.minimum((n / RATE - t) / 0.08, 1)
        bed[offset:offset + n] += gain * signal * envelope

    for bar in range(math.ceil(length / RATE / (4 * beat))):
        chord = chords[bar % len(chords)]
        start = bar * 4 * beat
        for midi in chord:
            note(start, midi, 4 * beat, 0.10, pad=True)
        for step in (0, 2):
            note(start + step * beat, chord[(bar + step) % 3] + 24, 1.3, 0.6)
    duck = np.ones(length, dtype=np.float32)
    ramp = round(0.12 * RATE)
    for segment in timeline:
        if segment["type"] != "speech":
            continue
        a, b = round(segment["start"] * RATE), round(segment["end"] * RATE)
        duck[a:b] = 0.25
        left, right = max(0, a - ramp), min(length, b + ramp)
        duck[left:a] = np.minimum(duck[left:a], np.linspace(1, 0.25, a - left))
        duck[b:right] = np.minimum(duck[b:right], np.linspace(0.25, 1, right - b))
    fade = min(RATE, length // 2)
    bed[:fade] *= np.linspace(0, 1, fade)
    bed[-fade:] *= np.linspace(1, 0, fade)
    return bed * duck * volume


def write_wav(path, samples):
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def timestamp(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def generate(items, output, voice="af_heart", speed=0.9, music=False,
             music_volume=0.07, spell_letters=False, force=False):
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() not in (".wav", ".mp3"):
        raise ValueError("audio output must end in .wav or .mp3")
    if not math.isfinite(speed) or not 0.5 <= speed <= 1.5:
        raise ValueError("speed must be between 0.5 and 1.5")
    if not math.isfinite(music_volume) or not 0 <= music_volume <= 0.3:
        raise ValueError("music volume must be between 0 and 0.3")
    destinations = [output, output.with_suffix(".voice.wav"),
                    output.with_suffix(".timeline.json"), output.with_suffix(".srt")]
    if not force and any(path.exists() for path in destinations):
        raise ValueError("an output or sidecar already exists; choose a new name or pass --force")
    if output.suffix.lower() == ".mp3" and not shutil.which("ffmpeg"):
        raise RuntimeError("MP3 export needs ffmpeg; use .wav or install ffmpeg")
    available = voices()
    if voice not in available:
        raise ValueError(f"unknown English voice {voice!r}; run 'img2vid speak --list-voices'")
    engine = load_engine()
    narration, segments = synthesize(items, engine, voice, speed, spell_letters)
    mixed = narration + music_bed(len(narration), segments, music_volume) if music else narration
    if np.max(np.abs(mixed)) > 0.99:
        mixed *= 0.99 / np.max(np.abs(mixed))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".img2vid-audio-") as temp:
        temp = Path(temp)
        final = temp / output.name
        wave_path = temp / "mixed.wav"
        voice_path = temp / "voice.wav"
        write_wav(wave_path, mixed)
        write_wav(voice_path, narration)
        if output.suffix.lower() == ".mp3":
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wave_path),
                            "-c:a", "libmp3lame", "-b:a", "192k", str(final)], check=True)
        else:
            wave_path.replace(final)
        data = {
            "version": 1, "engine": "Kokoro-82M v1.0 / kokoro-onnx",
            "voice": voice, "speed": speed, "sample_rate": RATE,
            "duration": len(narration) / RATE, "music": "procedural gentle" if music else None,
            "spell_letters": spell_letters, "segments": segments,
        }
        timeline_path = temp / "timeline.json"
        timeline_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        captions = [s for s in segments if s["type"] == "speech"]
        subtitle_path = temp / "captions.srt"
        subtitle_path.write_text("\n".join(
            f"{i}\n{timestamp(s['start'])} --> {timestamp(s['end'])}\n{s['text']}\n"
            for i, s in enumerate(captions, 1)
        ), encoding="utf-8")
        for source, target in zip([final, voice_path, timeline_path, subtitle_path], destinations):
            source.replace(target)
    print(f"wrote {output} ({data['duration']:.1f}s, {voice}, fully local)")
    return data


def command(args):
    if args.download_models:
        download_models()
    if args.list_voices:
        print("\n".join(voices()))
        return
    if args.file and args.text:
        raise ValueError("give either spoken text or --file, not both")
    if args.file:
        source = Path(args.file)
        content = source.read_text(encoding="utf-8")
        structured = source.suffix.lower() == ".json"
    else:
        content, structured = args.text or "", False
    if not content:
        if args.download_models:
            return
        raise ValueError("give text to speak, or --file with a lesson script")
    items = parse_script(content, args.pause, structured)
    generate(items, args.output, args.voice,
             args.speed if args.speed is not None else STYLES[args.style],
             args.music == "gentle", args.music_volume, args.spell_letters, args.force)
