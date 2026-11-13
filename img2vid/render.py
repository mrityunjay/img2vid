"""Render a storyboard: per-frame camera moves in numpy, encoded by an ffmpeg pipe."""
from __future__ import annotations

import math
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from .analyze import load_image
from .storyboard import aspect_ratio

_LATIN_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
_UNICODE_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_SYS = "/System/Library/Fonts/"
_SUP = _SYS + "Supplemental/"
_NOTO = "/usr/share/fonts/truetype/noto/"
# Unicode block -> fonts that cover it (macOS first, then common Linux Noto paths).
# The Kohinoor families also carry Latin glyphs, so mixed-script text stays in one font.
_SCRIPT_FONTS = (
    ((0x0900, 0x097F), (_SYS + "Kohinoor.ttc", _SUP + "Devanagari Sangam MN.ttc",
                        _NOTO + "NotoSansDevanagari-Bold.ttf")),
    ((0x0980, 0x09FF), (_SYS + "KohinoorBangla.ttc", _SUP + "Bangla Sangam MN.ttc",
                        _NOTO + "NotoSansBengali-Bold.ttf")),
    ((0x0A00, 0x0A7F), (_SUP + "Gurmukhi Sangam MN.ttc", _NOTO + "NotoSansGurmukhi-Bold.ttf")),
    ((0x0A80, 0x0AFF), (_SYS + "KohinoorGujarati.ttc", _SUP + "Gujarati Sangam MN.ttc",
                        _NOTO + "NotoSansGujarati-Bold.ttf")),
    ((0x0B00, 0x0B7F), (_SUP + "Oriya Sangam MN.ttc", _NOTO + "NotoSansOriya-Bold.ttf")),
    ((0x0B80, 0x0BFF), (_SUP + "Tamil Sangam MN.ttc", _NOTO + "NotoSansTamil-Bold.ttf")),
    ((0x0C00, 0x0C7F), (_SYS + "KohinoorTelugu.ttc", _SUP + "Telugu Sangam MN.ttc",
                        _NOTO + "NotoSansTelugu-Bold.ttf")),
    ((0x0C80, 0x0CFF), (_SUP + "Kannada Sangam MN.ttc", _NOTO + "NotoSansKannada-Bold.ttf")),
    ((0x0D00, 0x0D7F), (_SUP + "Malayalam Sangam MN.ttc", _NOTO + "NotoSansMalayalam-Bold.ttf")),
)
_BOLD_STYLES = ("Bold", "Semibold", "SemiBold", "Medium")


def output_size(aspect: str, short_side: int) -> tuple[int, int]:
    ar = aspect_ratio(aspect)
    if ar >= 1:
        return _even(short_side * ar), _even(short_side)
    return _even(short_side), _even(short_side / ar)


def _even(v: float) -> int:
    return int(round(v / 2)) * 2


def _ease(u: float) -> float:
    # Half linear, half smoothstep: keeps the camera moving through
    # crossfades instead of visibly stopping at each scene edge.
    u = min(max(u, 0.0), 1.0)
    return 0.5 * u + 0.5 * u * u * (3 - 2 * u)


def _open_boldest(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Open a font file; for collections (.ttc) pick the boldest face available."""
    first = ImageFont.truetype(path, size)
    if not path.lower().endswith(".ttc"):
        return first
    faces, i = {}, 0
    while True:
        try:
            face = ImageFont.truetype(path, size, index=i)
        except OSError:
            break
        faces.setdefault(face.getname()[1], face)
        i += 1
    return next((faces[s] for s in _BOLD_STYLES if s in faces), first)


def _font_candidates(text: str) -> list[str]:
    for (lo, hi), paths in _SCRIPT_FONTS:
        if any(lo <= ord(c) <= hi for c in text):
            return [*paths, *_UNICODE_FONTS]
    if any(ord(c) > 0x24F for c in text):
        return list(_UNICODE_FONTS)
    return list(_LATIN_FONTS)


def _font(text: str, size: int, path: str | None) -> ImageFont.FreeTypeFont:
    candidates = [path] if path else []
    for p in candidates + _font_candidates(text):
        try:
            return _open_boldest(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _needs_shaping(text: str) -> bool:
    return any(lo <= ord(c) <= hi for (lo, hi), _ in _SCRIPT_FONTS for c in text)


def shaping_available() -> bool:
    """True when Pillow can shape complex scripts (Indic, Arabic) via libraqm."""
    return features.check("raqm")


def _wrap(text, font, draw, max_w):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=font) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    return "\n".join(lines + [line])


class _Text:
    """Pre-rendered text with a soft shadow, alpha-composited onto frames."""

    def __init__(self, text, size, max_w, font_path):
        font = _font(text, size, font_path)
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        text = _wrap(text, font, probe, max_w)
        stroke = max(1, size // 18)
        l, t, r, b = probe.multiline_textbbox(
            (0, 0), text, font=font, align="center", stroke_width=stroke
        )
        l, t, r, b = math.floor(l), math.floor(t), math.ceil(r), math.ceil(b)  # may be floats
        pad = size // 2
        w, h = r - l + 2 * pad, b - t + 2 * pad

        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shadow = Image.new("L", (w, h), 0)
        ImageDraw.Draw(shadow).multiline_text(
            (pad - l, pad - t + stroke * 2), text, font=font, fill=200,
            align="center", stroke_width=stroke * 2,
        )
        layer.putalpha(shadow.filter(ImageFilter.GaussianBlur(size / 8)))
        ImageDraw.Draw(layer).multiline_text(
            (pad - l, pad - t), text, font=font, fill=(255, 255, 255, 255),
            align="center", stroke_width=stroke, stroke_fill=(20, 20, 20, 255),
        )
        rgba = np.asarray(layer).astype(np.float32)
        self.bgr = rgba[..., 2::-1]
        self.alpha = rgba[..., 3:] / 255.0
        self.w, self.h = w, h

    def draw(self, frame, cx, cy, opacity):
        if opacity <= 0:
            return
        x, y = int(cx - self.w / 2), int(cy - self.h / 2)
        fh, fw = frame.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + self.w, fw), min(y + self.h, fh)
        if x0 >= x1 or y0 >= y1:
            return
        a = self.alpha[y0 - y:y1 - y, x0 - x:x1 - x] * opacity
        region = frame[y0:y1, x0:x1].astype(np.float32)
        src = self.bgr[y0 - y:y1 - y, x0 - x:x1 - x]
        frame[y0:y1, x0:x1] = (region * (1 - a) + src * a).astype(np.uint8)


def _text_opacity(ts, duration, delay):
    # Fade in after `delay`, fade out before the scene starts handing over.
    fade = 0.5
    return min(max((ts - delay) / fade, 0.0), 1.0) * min(max((duration - ts - 0.9) / fade, 0.0), 1.0)


def _contain_canvas(arr, ar, fx, fy):
    """Place the photo on an output-aspect canvas filled with a blurred copy of itself."""
    h, w = arr.shape[:2]
    if w / h > ar:
        cw, ch = w, int(round(w / ar))
    else:
        cw, ch = int(round(h * ar)), h

    # Cover-crop the photo to the canvas, then blur at low resolution (cheap).
    scale = max(cw / w, ch / h)
    bg = cv2.resize(arr, (int(w * scale) + 1, int(h * scale) + 1), interpolation=cv2.INTER_AREA)
    by, bx = (bg.shape[0] - ch) // 2, (bg.shape[1] - cw) // 2
    bg = bg[by:by + ch, bx:bx + cw]
    small = cv2.resize(bg, (max(1, cw // 8), max(1, ch // 8)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6)
    canvas = (cv2.resize(small, (cw, ch), interpolation=cv2.INTER_LINEAR) * 0.55).astype(np.uint8)

    ox, oy = (cw - w) // 2, (ch - h) // 2
    canvas[oy:oy + h, ox:ox + w] = arr
    return canvas, ((ox + fx * w) / cw, (oy + fy * h) / ch), (ox, oy)


def _rect(src_w, src_h, ar, zoom, cx, cy):
    """Largest output-aspect rect in the source, shrunk by zoom, centered near (cx, cy)."""
    if src_w / src_h > ar:
        bw, bh = src_h * ar, src_h
    else:
        bw, bh = src_w, src_w / ar
    w, h = bw / zoom, bh / zoom
    x = min(max(cx * src_w - w / 2, 0.0), src_w - w)
    y = min(max(cy * src_h - h / 2, 0.0), src_h - h)
    return x, y, w


VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv")


def _read_video(path: str) -> tuple[list[np.ndarray], float]:
    """All frames (BGR uint8) of a video file and its frame rate."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open clip {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise ValueError(f"clip has no frames: {path}")
    return frames, fps


class _Scene:
    """A scene's prepared source image, start/end camera rects, and text overlays."""

    def __init__(self, scene: dict, out_w: int, out_h: int, title: str | None = None,
                 font: str | None = None):
        self.out_w, self.out_h = out_w, out_h
        self.duration = float(scene["duration"])
        short = min(out_w, out_h)
        max_w = out_w * 0.86
        self.title = _Text(title, int(short * 0.09), max_w, font) if title else None
        self.title_y = float(scene.get("title_y", 0.45))  # vertical position, 0..1
        caption = (scene.get("caption") or "").strip()
        self.caption = _Text(caption, int(short * 0.055), max_w, font) if caption else None
        ar = out_w / out_h
        fx, fy = scene.get("focus", (0.5, 0.5))
        zoom = max(1.0, float(scene.get("zoom", 1.2)))
        motion = scene.get("motion", "zoom_in")

        # Source frames: the AI-generated clip when there is one, else the still photo.
        clip = scene.get("clip")
        self.clip_fps = 24.0  # LTX clips are generated at 24 fps
        self.clip_speed = float(scene.get("clip_speed", 0.8))  # a touch of slow motion
        self.ping_pong = True
        self._flow_pair = (-1, None)  # cached optical flow for one frame pair
        if clip and clip.lower().endswith(VIDEO_EXTS):
            # A finished video (e.g. from an online AI): play it forwards once, slowed
            # (with optical-flow in-betweens) to fill the whole scene.
            frames, self.clip_fps = _read_video(clip)
            self.ping_pong = False
            length = len(frames) / self.clip_fps
            self.clip_speed = float(scene.get("video_speed")
                                    or min(1.0, length / self.duration))
            zoom = min(zoom, 1.04)  # the video has its own camera; barely add to it
        elif clip:
            frames = list(np.load(clip)["frames"])  # BGR uint8, N x H x W x 3
            zoom = min(zoom, 1.08)  # the clip already moves; keep the camera gentle
            if scene.get("clip_loop") == "forward":
                self.ping_pong = False
                length = len(frames) / self.clip_fps
                self.clip_speed = min(1.0, length / self.duration)
        else:
            frames = [cv2.cvtColor(np.asarray(load_image(scene["image"])), cv2.COLOR_RGB2BGR)]
        depth_map = None
        if scene.get("depth") and not clip:
            from .depth import estimate
            depth_map = estimate(scene["image"])

        h0, w0 = frames[0].shape[:2]
        if scene.get("fit") == "contain":
            canvas, (fx, fy), (ox, oy) = _contain_canvas(frames[0], ar, *scene.get("focus", (0.5, 0.5)))
            composed = []
            for f in frames:
                c = canvas.copy()
                c[oy:oy + h0, ox:ox + w0] = f
                composed.append(c)
            frames = composed
            if depth_map is not None:
                full = np.zeros(canvas.shape[:2], np.float32)  # blurred fill sits far back
                full[oy:oy + h0, ox:ox + w0] = cv2.resize(depth_map, (w0, h0))
                depth_map = full

        # Downscale once so the tightest crop lands near 1:1 with the output;
        # never upscale here (warpAffine handles that per frame).
        h, w = frames[0].shape[:2]
        base_w = min(w, h * ar)
        scale = min(1.0, out_w * zoom / base_w)
        if scale < 1.0:
            size = (int(w * scale), int(h * scale))
            frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
        self.frames = frames
        h, w = frames[0].shape[:2]

        self.depth = None
        if depth_map is not None:
            from .depth import prepare_for_warp
            self.depth = prepare_for_warp(depth_map, (w, h))
            px, py = int(fx * (w - 1)), int(fy * (h - 1))
            r = max(2, min(w, h) // 40)
            # Depth at the focus point stays put; nearer things swing one way, farther the other.
            self.d_ref = float(np.median(self.depth[max(py - r, 0):py + r, max(px - r, 0):px + r]))
            sway_dir = (1.0, 0.0) if w / h >= ar or motion != "pan" else (0.0, 1.0)
            if motion == "pan":
                self.dolly, self.sway = 0.0, 0.035
            else:
                self.dolly, self.sway = (0.12 if motion == "zoom_in" else -0.12), 0.015
            self.sway_dir = sway_dir if fx < 0.5 else (-sway_dir[0], -sway_dir[1])
            self.grid = _grid(out_w, out_h)

        if motion == "pan":
            wide = w / h > ar
            # Travel from the far side toward the side holding the focus point.
            toward = fx if wide else fy
            a, b = (0.0, 1.0) if toward >= 0.5 else (1.0, 0.0)
            if wide:
                start, end = _rect(w, h, ar, zoom, a, fy), _rect(w, h, ar, zoom, b, fy)
            else:
                start, end = _rect(w, h, ar, zoom, fx, a), _rect(w, h, ar, zoom, fx, b)
        else:
            wide_rect = _rect(w, h, ar, 1.0, 0.5, 0.5)
            tight_rect = _rect(w, h, ar, zoom, fx, fy)
            start, end = (wide_rect, tight_rect) if motion == "zoom_in" else (tight_rect, wide_rect)
        self.start, self.end = start, end

    def frame(self, u: float) -> np.ndarray:
        e = _ease(u)
        (x0, y0, w0), (x1, y1, w1) = self.start, self.end
        x = x0 + (x1 - x0) * e
        y = y0 + (y1 - y0) * e
        w = math.exp(math.log(w0) + (math.log(w1) - math.log(w0)) * e)  # constant-speed zoom
        s = self.out_w / w
        src = self._clip_frame(u) if len(self.frames) > 1 else self.frames[0]
        m = np.array([[s, 0, -x * s], [0, s, -y * s]], dtype=np.float32)
        if self.depth is None:
            frame = cv2.warpAffine(
                src, m, (self.out_w, self.out_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )
        else:
            frame = self._parallax(src, m, x, y, s, e)

        ts = u * self.duration
        rise = self.out_h * 0.02  # text drifts up slightly as it fades in
        if self.title:
            op = _text_opacity(ts, self.duration, 0.3)
            self.title.draw(frame, self.out_w / 2, self.out_h * self.title_y + rise * (1 - op), op)
        if self.caption:
            op = _text_opacity(ts, self.duration, 0.6 if self.title else 0.4)
            cy = self.out_h * 0.86 - self.caption.h / 2 + rise * (1 - op)
            self.caption.draw(frame, self.out_w / 2, cy, op)
        return frame

    def _clip_frame(self, u):
        """Clip frame for scene time u: slowed to fill the scene (forward clips) or
        ping-ponged (0.9.5 clips), with in-between frames.

        Ping-pong keeps every frame within one clip length of the source image, so the
        model's drift (faces, shapes) never builds up over a long scene.
        """
        n = len(self.frames)
        pos = u * self.duration * self.clip_fps * self.clip_speed
        if self.ping_pong:
            period = 2 * (n - 1)
            pos %= period
            if pos > n - 1:
                pos = period - pos
        else:
            pos = min(pos, n - 1)
        i = int(pos)
        frac = pos - i
        if frac < 0.02 or i >= n - 1:
            return self.frames[min(i, n - 1)]
        if self.ping_pong:
            return cv2.addWeighted(self.frames[i], 1 - frac, self.frames[i + 1], frac, 0)
        return self._interpolate(i, frac)

    def _interpolate(self, i, t):
        """In-between frame from optical flow, so slowed clips move smoothly instead of
        stuttering (repeated frames) or ghosting (cross-blended frames)."""
        a, b = self.frames[i], self.frames[i + 1]
        if self._flow_pair[0] != i:
            h, w = a.shape[:2]
            small = (w // 2, h // 2)
            ga = cv2.cvtColor(cv2.resize(a, small), cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(cv2.resize(b, small), cv2.COLOR_BGR2GRAY)
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            flow = cv2.resize(dis.calc(ga, gb, None), (w, h)) * 2.0
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._flow_pair = (i, (flow, gx, gy))
        flow, gx, gy = self._flow_pair[1]
        fx, fy = flow[..., 0], flow[..., 1]
        wa = cv2.remap(a, gx - t * fx, gy - t * fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wb = cv2.remap(b, gx + (1 - t) * fx, gy + (1 - t) * fy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
        return cv2.addWeighted(wa, 1 - t, wb, t, 0)

    def _parallax(self, src, m, x, y, s, e):
        """Backward-warp with per-pixel depth: near pixels zoom and sway more than far ones."""
        d = cv2.warpAffine(self.depth, m, (self.out_w, self.out_h),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dd = d - self.d_ref
        gx, gy = self.grid
        bx, by = x + gx / s, y + gy / s  # flat Ken Burns sample positions
        cx, cy = x + self.out_w / (2 * s), y + self.out_h / (2 * s)
        grow = self.dolly * (e if self.dolly > 0 else e - 1)  # dolly-in grows near, -out shrinks
        f = 1.0 + grow * dd
        sway = (2 * e - 1) * self.sway * self.out_w / s
        mx = cx + (bx - cx) / f - sway * self.sway_dir[0] * dd
        my = cy + (by - cy) / f - sway * self.sway_dir[1] * dd
        return cv2.remap(src, mx.astype(np.float32), my.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


_grids: dict = {}


def _grid(w: int, h: int):
    if (w, h) not in _grids:
        _grids[(w, h)] = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return _grids[(w, h)]


def _blend(a, b, p, kind):
    if kind == "fadeblack":
        return (a * (1 - 2 * p)).astype(np.uint8) if p < 0.5 else (b * (2 * p - 1)).astype(np.uint8)
    if kind == "slide":
        off = int(round(_ease(p) * a.shape[1]))
        if off <= 0:
            return a
        out = np.empty_like(a)
        out[:, :a.shape[1] - off] = a[:, off:]
        out[:, a.shape[1] - off:] = b[:, :off]
        return out
    return cv2.addWeighted(a, 1 - p, b, p, 0)


def _timeline(storyboard):
    scenes = storyboard["scenes"]
    td = float(storyboard.get("transition_duration", 0.8))
    starts, t = [], 0.0
    for i, scene in enumerate(scenes):
        starts.append(t)
        last = i == len(scenes) - 1
        overlap = 0.0 if last or scene.get("transition", "cut") == "cut" else min(td, scene["duration"] / 2)
        t += scene["duration"] - overlap
    return starts, starts[-1] + scenes[-1]["duration"]


def _ffmpeg_cmd(out_path, w, h, fps, total, music, preview):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
    ]
    if music:
        cmd += ["-stream_loop", "-1", "-i", music, "-map", "0:v", "-map", "1:a"]
        fade = min(2.5, total / 3)
        cmd += ["-af", f"afade=t=in:d=0.5,afade=t=out:st={total - fade:.3f}:d={fade:.3f}",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast" if preview else "medium",
        "-crf", "26" if preview else "18",
        "-movflags", "+faststart", "-t", f"{total:.3f}", out_path,
    ]
    return cmd


def render(storyboard: dict, out_path: str, short_side: int = 1080, preview: bool = False) -> float:
    """Render to out_path and return the video duration in seconds."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    scenes = storyboard["scenes"]
    if not scenes:
        raise ValueError("storyboard has no scenes")

    texts = [storyboard.get("title") or ""] + [s.get("caption") or "" for s in scenes]
    if not shaping_available() and any(_needs_shaping(t) for t in texts):
        print("warning: Pillow lacks libraqm, so Indic text will render with misplaced "
              "vowel signs (see README)", file=sys.stderr)

    fps = 24 if preview else int(storyboard.get("fps", 30))
    w, h = output_size(storyboard["aspect"], 480 if preview else short_side)
    starts, total = _timeline(storyboard)
    n_frames = int(round(total * fps))
    fade_in = float(storyboard.get("fade_in", 0.0))
    fade_out = float(storyboard.get("fade_out", 0.0))

    proc = subprocess.Popen(
        _ffmpeg_cmd(out_path, w, h, fps, total, storyboard.get("music"), preview),
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    loaded: dict[int, _Scene] = {}
    try:
        for f in range(n_frames):
            t = f / fps
            active = [i for i, s in enumerate(scenes) if starts[i] <= t < starts[i] + s["duration"]]
            if not active:  # float edge at the very end
                active = [len(scenes) - 1]
            for i in list(loaded):  # free scenes that are over
                if i < active[0]:
                    del loaded[i]

            frames = []
            for i in active[:2]:
                if i not in loaded:
                    loaded[i] = _Scene(
                        scenes[i], w, h,
                        title=storyboard.get("title") if i == 0 else None,
                        font=storyboard.get("font"),
                    )
                frames.append(loaded[i].frame((t - starts[i]) / scenes[i]["duration"]))

            if len(frames) == 2:
                a, b = active[0], active[1]
                p = (t - starts[b]) / (starts[a] + scenes[a]["duration"] - starts[b])
                frame = _blend(frames[0], frames[1], p, scenes[a].get("transition", "crossfade"))
            else:
                frame = frames[0]

            gain = 1.0
            if fade_in and t < fade_in:
                gain = t / fade_in
            if fade_out and t > total - fade_out:
                gain = min(gain, (total - t) / fade_out)
            if gain < 1.0:
                frame = (frame * max(gain, 0.0)).astype(np.uint8)

            proc.stdin.write(frame.tobytes())
            if f % fps == 0 or f == n_frames - 1:
                sys.stderr.write(f"\rrendering {100 * (f + 1) // n_frames:3d}%")
                sys.stderr.flush()
        proc.stdin.close()
    except BrokenPipeError:
        pass
    finally:
        sys.stderr.write("\n")
        if not proc.stdin.closed:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        err = proc.stderr.read().decode(errors="replace")
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}): {err.strip()}")
    return total
