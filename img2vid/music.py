"""Beat tracking (numpy only) and snapping scene changes to the music's bars."""
from __future__ import annotations

import subprocess

import numpy as np

_SR = 22050
_HOP = 512


def _decode(path: str) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(_SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, np.float32)


def _onset_envelope(x: np.ndarray) -> np.ndarray:
    win = np.hanning(2048).astype(np.float32)
    n = (len(x) - 2048) // _HOP
    idx = np.arange(2048)[None, :] + _HOP * np.arange(n)[:, None]
    spec = np.log1p(np.abs(np.fft.rfft(x[idx] * win, axis=1)))
    flux = np.maximum(np.diff(spec, axis=0), 0).sum(1)
    return (flux - flux.mean()) / (flux.std() + 1e-9)


def beat_grid(path: str, beats_per_bar: int = 4) -> tuple[float, np.ndarray, float]:
    """Return (beat period seconds, bar start times, track length seconds)."""
    x = _decode(path)
    env = _onset_envelope(x)
    fps = _SR / _HOP

    # Tempo: strongest autocorrelation lag between 60 and 180 BPM.
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    lags = np.arange(1, len(ac))
    bpm = 60 * fps / lags
    ok = (bpm >= 60) & (bpm <= 180)
    lag = int(lags[ok][np.argmax(ac[1:][ok])])

    # Phase: offset whose beat positions collect the most onset energy; bars start on
    # the beat (of beats_per_bar candidates) where onsets are strongest.
    beats_per = [env[o::lag].sum() for o in range(lag)]
    phase = int(np.argmax(beats_per))
    bar_lag = lag * beats_per_bar
    bar_phase = max(range(phase, phase + bar_lag, lag), key=lambda o: env[o % bar_lag::bar_lag].sum())
    length = len(x) / _SR
    bars = np.arange(bar_phase % bar_lag, len(env), bar_lag) / fps
    return lag / fps, bars, length


def snap_to_bars(storyboard: dict, music: str, min_bars: int = 2) -> None:
    """Move each scene change onto the nearest bar line and end with the music."""
    period, bars, length = beat_grid(music)
    scenes = storyboard["scenes"]
    td = float(storyboard.get("transition_duration", 0.8))
    overlaps = [0.0 if i == len(scenes) - 1 or s.get("transition") == "cut" else td
                for i, s in enumerate(scenes)]

    # Scene changes (middle of each transition), scaled so the video spans the track.
    total = sum(s["duration"] for s in scenes) - sum(overlaps)
    k = length / total
    t, changes = 0.0, []
    for s, ov in zip(scenes[:-1], overlaps):
        t += (s["duration"] - ov) * k
        changes.append(t + ov / 2)

    bar_len = period * 4
    snapped, last = [], 0.0
    for c in changes:
        candidates = bars[bars >= last + min_bars * bar_len - 1e-6]
        if not len(candidates):
            break
        b = float(candidates[np.argmin(np.abs(candidates - c))])
        snapped.append(b)
        last = b
    if len(snapped) < len(changes) or length - snapped[-1] < min_bars * bar_len:
        return  # track too short for this many scenes; keep proportional timing

    start = 0.0
    for i, s in enumerate(scenes):
        if i < len(snapped):
            end = snapped[i] + overlaps[i] / 2
        else:
            end = length
        s["duration"] = round(end - start, 3)
        start = end - overlaps[i]
    storyboard["beat_period"] = round(period, 4)
