"""Per-image analysis: capture time, faces, and a focus point for camera motion."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 36867
_DATETIME = 306
_ANALYSIS_SIDE = 800  # analysis runs on a downscaled copy for speed

_cascade = None


@dataclass
class ImageInfo:
    path: str
    width: int
    height: int
    taken_at: str  # ISO timestamp (EXIF capture time, else file mtime)
    exif_time: bool = False  # True when taken_at came from EXIF, not the file date
    faces: list = field(default_factory=list)  # [x, y, w, h], normalized 0..1
    focus: tuple = (0.5, 0.5)  # normalized point the camera should move toward


def load_image(path: str) -> Image.Image:
    """Open an image upright (EXIF orientation applied) as RGB."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _taken_at(img: Image.Image, path: str) -> tuple[str, bool]:
    try:
        exif = img.getexif()
        value = exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
        if value:
            stamp = str(value).strip("\x00 ")[:19]
            return datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S").isoformat(), True
    except (ValueError, KeyError, AttributeError):
        pass
    return datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"), False


def _detect_faces(gray: np.ndarray) -> list:
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
    h, w = gray.shape
    min_side = max(24, int(min(h, w) * 0.04))
    boxes = _cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side)
    )
    return [[x / w, y / h, bw / w, bh / h] for x, y, bw, bh in boxes]


def _focus(gray: np.ndarray, faces: list) -> tuple:
    if faces:
        # Area-weighted center of all faces.
        areas = np.array([f[2] * f[3] for f in faces])
        cx = np.array([f[0] + f[2] / 2 for f in faces])
        cy = np.array([f[1] + f[3] / 2 for f in faces])
        return float((cx * areas).sum() / areas.sum()), float((cy * areas).sum() / areas.sum())

    # No faces: use the centroid of edge energy as a cheap saliency proxy,
    # pulled toward the center so motion stays gentle.
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160).astype(np.float32)
    total = edges.sum()
    if total == 0:
        return 0.5, 0.5
    ys, xs = np.indices(edges.shape)
    fx = (xs * edges).sum() / total / edges.shape[1]
    fy = (ys * edges).sum() / total / edges.shape[0]
    return float(0.5 + (fx - 0.5) * 0.6), float(0.5 + (fy - 0.5) * 0.6)


def analyze(path: str) -> ImageInfo:
    with Image.open(path) as raw:
        taken_at, exif_time = _taken_at(raw, path)
        img = ImageOps.exif_transpose(raw).convert("RGB")

    width, height = img.size
    small = img.copy()
    small.thumbnail((_ANALYSIS_SIDE, _ANALYSIS_SIDE))
    gray = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)

    faces = _detect_faces(gray)
    return ImageInfo(
        path=os.path.abspath(path),
        width=width,
        height=height,
        taken_at=taken_at,
        exif_time=exif_time,
        faces=faces,
        focus=_focus(gray, faces),
    )
