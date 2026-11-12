"""Local image-to-video (LTX-Video) so people and things in each photo actually move."""
from __future__ import annotations

import gc
import hashlib
import multiprocessing
import os
import sys
import time

import numpy as np

from .analyze import load_image

MODEL = "Lightricks/LTX-Video-0.9.5"
CACHE_DIR = os.path.expanduser("~/.cache/img2vid/clips")
FPS = 24
MPS_MEMORY_FRACTION = 0.9  # of macOS's recommended GPU working set (~14 GB on 18 GB RAM)
NEGATIVE = ("worst quality, inconsistent motion, blurry, jittery, distorted, deformed face, "
            "morphing, extra limbs, flickering, static image")
DEFAULT_MOTION = ("Subtle, natural motion in a still scene: candle and lamp flames flicker, "
                  "flower petals and fabric sway gently in a light breeze, light shimmers and "
                  "smoke drifts. The camera is steady with a slow push in. Realistic, "
                  "real-life footage.")


class AnimateUnavailable(Exception):
    """torch/diffusers or the model are missing; callers fall back to 3D or flat motion."""


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _gen_size(ar: float, short: int) -> tuple[int, int]:
    ar = min(max(ar, 9 / 16), 16 / 9)
    if ar >= 1:
        return int(round(short * ar / 32)) * 32, short
    return short, int(round(short / ar / 32)) * 32


def _cover(img, w, h, fx, fy):
    """Crop a PIL image to w:h around the focus point, then resize to exactly w x h."""
    iw, ih = img.size
    if iw / ih > w / h:
        cw, ch = ih * w / h, ih
    else:
        cw, ch = iw, iw * h / w
    x = min(max(fx * iw - cw / 2, 0), iw - cw)
    y = min(max(fy * ih - ch / 2, 0), ih - ch)
    return img.crop((int(x), int(y), int(x + cw), int(y + ch))).resize((w, h))


def _key(path, prompt, w, h, frames, steps, seed):
    hsh = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hsh.update(chunk)
    hsh.update(f"{MODEL}|{prompt}|{w}x{h}|{frames}|{steps}|{seed}".encode())
    return os.path.join(CACHE_DIR, hsh.hexdigest() + ".npz")


def _jobs(scenes, aspect, short, frames, steps, seed, default_prompt):
    from .storyboard import aspect_ratio

    out_ar = aspect_ratio(aspect)
    jobs = []
    for scene in scenes:
        img = load_image(scene["image"])
        # Contained photos keep their own shape; the renderer adds the blurred fill later.
        ar = img.width / img.height if scene.get("fit") == "contain" else out_ar
        w, h = _gen_size(ar, short)
        prompt = scene.get("motion_prompt") or default_prompt
        fx, fy = scene.get("focus", (0.5, 0.5))
        jobs.append({
            "scene": scene, "prompt": prompt, "w": w, "h": h,
            "image": _cover(img, w, h, fx, fy) if scene.get("fit") != "contain" else img.resize((w, h)),
            "path": _key(scene["image"], prompt, w, h, frames, steps, seed),
        })
    return jobs


def _embed_path(prompt):
    key = hashlib.sha1(f"{MODEL}|{prompt}|{NEGATIVE}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, "embeds", key + ".pt")


def _encode_prompts(prompts, device):
    """Run the (large, slow to load) T5 text encoder only for prompts not cached on disk."""
    import torch
    from diffusers import LTXImageToVideoPipeline

    embeds = {p: torch.load(_embed_path(p), weights_only=True)
              for p in prompts if os.path.exists(_embed_path(p))}
    missing = [p for p in prompts if p not in embeds]
    if not missing:
        return embeds

    print(f"loading text encoder ({MODEL}); this takes a few minutes the first time...",
          file=sys.stderr)
    pipe = LTXImageToVideoPipeline.from_pretrained(
        MODEL, transformer=None, vae=None, torch_dtype=torch.bfloat16
    )
    pipe.text_encoder.to(device)
    os.makedirs(os.path.dirname(_embed_path("")), exist_ok=True)
    with torch.no_grad():
        for p in missing:
            pe, pm, npe, npm = pipe.encode_prompt(
                p, NEGATIVE, do_classifier_free_guidance=True, device=device, dtype=torch.bfloat16
            )
            embeds[p] = tuple(t.to("cpu") for t in (pe, pm, npe, npm))
            torch.save(embeds[p], _embed_path(p))
    del pipe
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return embeds


def generate(storyboard: dict, short: int = 480, seconds: float = 4.0, steps: int = 30,
             seed: int = 0, prompt: str | None = None, limit: int | None = None,
             allow_people: bool = False) -> int:
    """Attach a generated clip ("clip": path) to scenes; returns how many scenes got one."""
    try:
        import torch
        from diffusers import LTXImageToVideoPipeline
    except ImportError as e:
        raise AnimateUnavailable("install torch and diffusers for AI motion") from e

    from .people import has_people

    # A scene's "animate" field forces the choice (true/false); otherwise photos with
    # people or idols keep 3D motion, because this model distorts faces.
    scenes, kept_3d = [], []
    for scene in storyboard["scenes"]:
        want = scene.get("animate")
        if want is False or (want is None and not allow_people and has_people(scene)):
            kept_3d.append(os.path.basename(scene["image"]))
        else:
            scenes.append(scene)
    if kept_3d:
        print(f"AI motion: {len(kept_3d)} photo(s) with people/idols keep 3D motion "
              f"({', '.join(kept_3d)})", file=sys.stderr)

    frames = int(round(seconds * FPS / 8)) * 8 + 1  # LTX needs 8k+1 frames
    jobs = _jobs(scenes, storyboard["aspect"], short, frames, steps, seed,
                 prompt or DEFAULT_MOTION)
    if limit is not None:
        # Animate the scenes that stay on screen longest; the rest keep 3D motion.
        keep = sorted(jobs, key=lambda j: -j["scene"]["duration"])[:limit]
        jobs = [j for j in jobs if j in keep]

    for j in jobs:
        if os.path.exists(j["path"]):
            j["scene"]["clip"] = j["path"]
    todo = [j for j in jobs if "clip" not in j["scene"]]
    print(f"AI motion: {len(jobs) - len(todo)} cached, {len(todo)} to generate "
          f"({frames} frames, {steps} steps each)", file=sys.stderr)
    if not todo:
        return len(jobs)

    device = _device()
    if device == "mps":
        # Fail with an out-of-memory error instead of pushing macOS into swap (or a reboot).
        torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)
    try:
        prompts = sorted({j["prompt"] for j in todo})
        if any(not os.path.exists(_embed_path(p)) for p in prompts):
            # The 19 GB text encoder runs in its own process: exiting hands all of its
            # memory back to macOS. In-process, its leftovers pushed the system into swap
            # and made denoising ~20x slower.
            ctx = multiprocessing.get_context("spawn")
            proc = ctx.Process(target=_encode_prompts, args=(prompts, device))
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                raise AnimateUnavailable("the text encoder failed")
        embeds = _encode_prompts(prompts, device)  # all cached on disk now
        print("loading video model...", file=sys.stderr)
        pipe = LTXImageToVideoPipeline.from_pretrained(
            MODEL, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16
        ).to(device)
    except (OSError, ValueError) as e:
        raise AnimateUnavailable(f"could not load {MODEL}: {e}") from e

    # Phase 1: denoise every clip to latents while the transformer is loaded. Decoding
    # with the transformer still in memory was ~20x slower on 18 GB, so it waits.
    os.makedirs(CACHE_DIR, exist_ok=True)
    latents = []
    for n, j in enumerate(todo, 1):
        pe, pm, npe, npm = (t.to(device) for t in embeds[j["prompt"]])
        started = time.time()

        def step_cb(_pipe, i, _t, kwargs, _n=n, _started=started):
            sys.stderr.write(f"\rclip {_n}/{len(todo)}: step {i + 1}/{steps} "
                             f"({time.time() - _started:.0f}s)")
            sys.stderr.flush()
            return kwargs

        try:
            packed = pipe(
                image=j["image"], prompt_embeds=pe, prompt_attention_mask=pm,
                negative_prompt_embeds=npe, negative_prompt_attention_mask=npm,
                width=j["w"], height=j["h"], num_frames=frames, frame_rate=FPS,
                num_inference_steps=steps, guidance_scale=3.0,
                generator=torch.Generator(device).manual_seed(seed),
                output_type="latent", callback_on_step_end=step_cb,
            ).frames
        except RuntimeError as e:  # e.g. out of memory; this scene keeps 3D motion
            sys.stderr.write(f"\nclip {n}/{len(todo)} failed ({e}); using 3D motion for it\n")
            continue
        lat = pipe._unpack_latents(
            packed, (frames - 1) // pipe.vae_temporal_compression_ratio + 1,
            j["h"] // pipe.vae_spatial_compression_ratio,
            j["w"] // pipe.vae_spatial_compression_ratio,
            pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size,
        )
        lat = pipe._denormalize_latents(lat, pipe.vae.latents_mean, pipe.vae.latents_std,
                                        pipe.vae.config.scaling_factor)
        latents.append((j, lat.to("cpu")))
        sys.stderr.write(f"\rclip {n}/{len(todo)}: denoised in {time.time() - started:.0f}s\n")

    # Phase 2: free the transformer, then decode each clip in short time chunks.
    vae, processor = pipe.vae, pipe.video_processor
    del pipe
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    vae.enable_tiling(
        tile_sample_min_height=4096, tile_sample_min_width=4096,  # no spatial tiles
        tile_sample_min_num_frames=16, tile_sample_stride_num_frames=12,
    )
    timestep_cond = getattr(vae.config, "timestep_conditioning", False)
    done = 0
    for n, (j, lat) in enumerate(latents, 1):
        started = time.time()
        lat = lat.to(device, torch.bfloat16)
        ts = None
        if timestep_cond:
            # 0.9.5's VAE denoises slightly while decoding; the model card's values.
            noise = torch.randn(lat.shape, generator=torch.Generator(device).manual_seed(seed),
                                device=device, dtype=lat.dtype)
            lat = 0.975 * lat + 0.025 * noise
            ts = torch.tensor([0.05], device=device, dtype=lat.dtype)
        try:
            with torch.no_grad():
                video = vae.decode(lat, ts, return_dict=False)[0]
        except RuntimeError as e:
            sys.stderr.write(f"clip {n}/{len(latents)} decode failed ({e}); using 3D motion\n")
            continue
        out = processor.postprocess_video(video, output_type="np")[0]
        clip = (np.clip(out, 0, 1) * 255).astype(np.uint8)[..., ::-1]  # RGB float -> BGR uint8
        np.savez_compressed(j["path"], frames=np.ascontiguousarray(clip))
        j["scene"]["clip"] = j["path"]
        done += 1
        sys.stderr.write(f"clip {n}/{len(latents)}: decoded in {time.time() - started:.0f}s\n")
        del video, lat
        if device == "mps":
            torch.mps.empty_cache()
    return len(jobs) - len(todo) + done
