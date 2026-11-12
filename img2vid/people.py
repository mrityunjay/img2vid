"""Does a photo show people (or idols)? Local video models distort faces, so those skip AI motion."""
from __future__ import annotations

import hashlib
import json
import os

from .analyze import load_image

MODEL = "hustvl/yolos-small"
CACHE_DIR = os.path.expanduser("~/.cache/img2vid/people")
# Low on purpose: idols and half-hidden people score 0.3-0.4. A false "yes" only costs
# the scene its AI motion (it gets 3D instead); a false "no" lets a face get distorted.
THRESHOLD = 0.3

_det = None


def _detector():
    global _det
    if _det is None:
        import torch
        from transformers import pipeline

        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        _det = pipeline("object-detection", model=MODEL, device=device)
    return _det


def _cache_path(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(f"{MODEL}|{THRESHOLD}".encode())
    return os.path.join(CACHE_DIR, h.hexdigest() + ".json")


def count(path: str) -> int:
    """Number of people the detector sees (cached per image)."""
    cache = _cache_path(path)
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)["people"]
    img = load_image(path)
    img.thumbnail((800, 800))
    n = sum(1 for d in _detector()(img, threshold=THRESHOLD) if d["label"] == "person")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"people": n}, f)
    return n


def has_people(scene: dict) -> bool:
    # Either signal counts: the person detector misses many idols, which the face cascade
    # catches. Scenes from storyboards older than the "faces" field count as unknown (yes).
    return scene.get("faces", 1) > 0 or count(scene["image"]) > 0
