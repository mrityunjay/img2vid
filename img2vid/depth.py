"""Monocular depth (Depth Anything V2) for 2.5D parallax, cached per image."""
from __future__ import annotations

import hashlib
import os
import sys

import cv2
import numpy as np

from .analyze import load_image

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
CACHE_DIR = os.path.expanduser("~/.cache/img2vid/depth")
_INPUT_SIDE = 768  # depth runs on a downscaled copy; the map is upsampled afterwards

_pipe = None


class DepthUnavailable(Exception):
    """torch/transformers or the model are missing; callers fall back to flat motion."""


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _pipeline():
    global _pipe
    if _pipe is None:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise DepthUnavailable("install torch and transformers for 3D motion") from e
        print(f"loading depth model {MODEL}...", file=sys.stderr)
        _pipe = pipeline("depth-estimation", model=MODEL, device=_device())
    return _pipe


def _cache_path(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(MODEL.encode())
    return os.path.join(CACHE_DIR, h.hexdigest() + ".npy")


def estimate(path: str) -> np.ndarray:
    """Depth for an (upright) image as float32 HxW in 0..1, where 1 is nearest."""
    cache = _cache_path(path)
    if os.path.exists(cache):
        return np.load(cache)

    img = load_image(path)
    small = img.copy()
    small.thumbnail((_INPUT_SIDE, _INPUT_SIDE))
    out = _pipeline()(small)
    raw = out["predicted_depth"].squeeze().float().cpu().numpy()  # relative inverse depth

    lo, hi = np.percentile(raw, (2, 98))
    d = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    d = cv2.resize(d, small.size, interpolation=cv2.INTER_CUBIC)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache, d)
    return d


def prepare_for_warp(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to the warp source and soften edges so parallax stretches instead of tearing."""
    d = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
    k = max(3, int(min(size) * 0.012) | 1)
    # Grow near regions slightly so foreground outlines carry a margin of background.
    d = cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.GaussianBlur(d, (0, 0), k / 2)
