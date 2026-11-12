"""Sharper, livelier AI motion: LTX-Video 0.9.8 2B distilled + 2x latent upscaler.

Each clip is generated at low resolution in 8 distilled steps, its latents are upscaled 2x
and refined in a few more steps, then decoded. Same memory discipline as animate.py: the
text encoder runs in its own process and decoding waits until the transformer is freed.
"""
from __future__ import annotations

import gc
import hashlib
import multiprocessing
import os
import sys
import time

import numpy as np

from . import animate
from .analyze import load_image

REPO = "Lightricks/LTX-Video"
CKPT = "ltxv-2b-0.9.8-distilled.safetensors"
UPSAMPLER = "Lightricks/ltxv-spatial-upscaler-0.9.7"
BASE = "Lightricks/LTX-Video-0.9.5"  # T5 text encoder, tokenizer and scheduler config
ENGINE = f"{CKPT}+{UPSAMPLER}"
FPS = 24
MAX_TOKENS = 256
# Timesteps from Lightricks' distilled-model recipe.
FIRST_PASS = [1000, 993, 987, 981, 975, 909, 725, 0.03]
REFINE_PASS = [1000, 909, 725, 421, 0]


def _base_size(ar: float, short: int) -> tuple[int, int]:
    """First-pass size (multiples of 32); the upscaler doubles it."""
    w, h = animate._gen_size(ar, short)
    return w, h


def match_colors(frames: np.ndarray, ref_bgr: np.ndarray, window: int = 25) -> np.ndarray:
    """Pull a clip's colours back to its source image (the distilled model drifts cooler and
    more saturated as the clip goes on).

    Each frame's LAB mean/std is mapped onto the image's; the per-frame statistics are
    averaged over `window` frames so the correction itself never flickers.
    """
    import cv2

    ref = cv2.cvtColor(cv2.resize(ref_bgr, frames.shape[2:0:-1]), cv2.COLOR_BGR2LAB)
    rm, rs = ref.reshape(-1, 3).mean(0), ref.reshape(-1, 3).std(0) + 1e-6
    labs = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in frames]
    means = np.array([x.reshape(-1, 3).mean(0) for x in labs])
    stds = np.array([x.reshape(-1, 3).std(0) for x in labs]) + 1e-6
    k = np.ones(window) / window
    pad = window // 2

    def smooth(a):
        a = np.pad(a, ((pad, window - 1 - pad), (0, 0)), mode="edge")
        return np.stack([np.convolve(a[:, c], k, "valid") for c in range(3)], 1)

    means, stds = smooth(means), smooth(stds)
    out = np.empty_like(frames)
    for i, lab in enumerate(labs):
        lab = (lab - means[i]) * np.clip(rs / stds[i], 0.6, 1.4) + rm
        out[i] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def _key(path, prompt, w, h, frames, seed):
    hsh = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hsh.update(chunk)
    hsh.update(f"{ENGINE}|{prompt}|{w}x{h}|{frames}|{seed}".encode())
    return os.path.join(animate.CACHE_DIR, "hq-" + hsh.hexdigest() + ".npz")


def _embed_path(prompt):
    key = hashlib.sha1(f"{BASE}|{MAX_TOKENS}|{prompt}".encode()).hexdigest()
    return os.path.join(animate.CACHE_DIR, "embeds", "hq-" + key + ".pt")


def _encode_prompts(prompts, device):
    import torch
    from diffusers import LTXConditionPipeline

    embeds = {p: torch.load(_embed_path(p), weights_only=True)
              for p in prompts if os.path.exists(_embed_path(p))}
    missing = [p for p in prompts if p not in embeds]
    if not missing:
        return embeds
    print("loading text encoder; this takes a few minutes...", file=sys.stderr)
    pipe = LTXConditionPipeline.from_pretrained(
        BASE, transformer=None, vae=None, torch_dtype=torch.bfloat16
    )
    pipe.text_encoder.to(device)
    os.makedirs(os.path.dirname(_embed_path("")), exist_ok=True)
    with torch.no_grad():
        for p in missing:
            pe, pm, _, _ = pipe.encode_prompt(
                p, do_classifier_free_guidance=False, device=device,
                dtype=torch.bfloat16, max_sequence_length=MAX_TOKENS,
            )
            embeds[p] = (pe.to("cpu"), pm.to("cpu"))
            torch.save(embeds[p], _embed_path(p))
    return embeds


def generate(storyboard: dict, short: int = 448, seconds: float = 5.0, seed: int = 0,
             only: list[int] | None = None, upscale: bool = True) -> int:
    """Attach HQ clips to scenes (all scenes unless `only` lists 1-based numbers)."""
    try:
        import torch
        from diffusers import (AutoencoderKLLTXVideo, LTXConditionPipeline,
                               LTXLatentUpsamplePipeline, LTXLatentUpsamplerModel,
                               LTXVideoTransformer3DModel)
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise animate.AnimateUnavailable("install torch and diffusers for AI motion") from e
    from .storyboard import aspect_ratio

    out_ar = aspect_ratio(storyboard["aspect"])
    frames = int(round(seconds * FPS / 8)) * 8 + 1
    jobs = []
    for n, scene in enumerate(storyboard["scenes"], 1):
        if (only and n not in only) or scene.get("animate") is False:
            continue
        img = load_image(scene["image"])
        ar = img.width / img.height if scene.get("fit") == "contain" else out_ar
        w, h = _base_size(ar, short)
        prompt = scene.get("motion_prompt") or animate.DEFAULT_MOTION
        fx, fy = scene.get("focus", (0.5, 0.5))
        big = (w * 2, h * 2) if upscale else (w, h)
        cond = (animate._cover(img, *big, fx, fy) if scene.get("fit") != "contain"
                else img.resize(big))
        path = _key(scene["image"], prompt, w, h, frames, seed) if upscale else \
            _key(scene["image"], prompt + "|noup", w, h, frames, seed)
        jobs.append({"n": n, "scene": scene, "prompt": prompt, "w": w, "h": h,
                     "image": cond, "path": path})

    for j in jobs:
        if os.path.exists(j["path"]):
            j["scene"]["clip"] = j["path"]
    todo = [j for j in jobs if not os.path.exists(j["path"])]
    print(f"HQ motion: {len(jobs) - len(todo)} cached, {len(todo)} to generate "
          f"({frames} frames)", file=sys.stderr)
    if not todo:
        return len(jobs)

    device = animate._device()
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(animate.MPS_MEMORY_FRACTION)
    prompts = sorted({j["prompt"] for j in todo})
    if any(not os.path.exists(_embed_path(p)) for p in prompts):
        proc = multiprocessing.get_context("spawn").Process(
            target=_encode_prompts, args=(prompts, device))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise animate.AnimateUnavailable("the text encoder failed")
    embeds = _encode_prompts(prompts, device)

    print("loading video model...", file=sys.stderr)
    ckpt = hf_hub_download(REPO, CKPT)
    dt = torch.bfloat16
    transformer = LTXVideoTransformer3DModel.from_single_file(ckpt, torch_dtype=dt)
    vae = AutoencoderKLLTXVideo.from_single_file(ckpt, torch_dtype=dt)
    pipe = LTXConditionPipeline.from_pretrained(
        BASE, transformer=transformer, vae=vae, text_encoder=None, tokenizer=None,
        torch_dtype=dt,
    ).to(device)
    upsampler = None
    if upscale:
        up_dir = snapshot_download(UPSAMPLER, allow_patterns=["latent_upsampler/*",
                                                              "model_index.json"])
        upsampler = LTXLatentUpsamplePipeline(
            vae=pipe.vae,
            latent_upsampler=LTXLatentUpsamplerModel.from_pretrained(
                up_dir, subfolder="latent_upsampler", torch_dtype=dt),
        ).to(device)

    common = dict(num_frames=frames, frame_rate=FPS, guidance_scale=1.0,
                  guidance_rescale=0.7, image_cond_noise_scale=0.0, output_type="latent")
    os.makedirs(animate.CACHE_DIR, exist_ok=True)
    # Latents saved by an earlier, interrupted run only need decoding.
    finished = [j for j in todo if os.path.exists(j["path"] + ".lat.pt")]
    for k, j in enumerate(todo, 1):
        if j in finished:
            continue
        pe, pm = (t.to(device) for t in embeds[j["prompt"]])
        started = time.time()
        gen = torch.Generator(device).manual_seed(seed)
        small = j["image"].resize((j["w"], j["h"]))
        try:
            lat = pipe(conditions=[LTXVideoCondition(image=small, frame_index=0)],
                       prompt_embeds=pe, prompt_attention_mask=pm,
                       width=j["w"], height=j["h"], timesteps=FIRST_PASS,
                       generator=gen, **common).frames
            if upsampler is not None:
                lat = upsampler(latents=lat, adain_factor=1.0, output_type="latent").frames
                lat = pipe(conditions=[LTXVideoCondition(image=j["image"], frame_index=0)],
                           prompt_embeds=pe, prompt_attention_mask=pm,
                           width=j["w"] * 2, height=j["h"] * 2, latents=lat,
                           denoise_strength=0.999, timesteps=REFINE_PASS,
                           generator=gen, **common).frames
            lat = pipe._denormalize_latents(lat, pipe.vae.latents_mean, pipe.vae.latents_std,
                                            pipe.vae.config.scaling_factor)
            lat = lat.to("cpu")
        except RuntimeError as e:
            sys.stderr.write(f"\nclip {k}/{len(todo)} (scene {j['n']}) failed ({e})\n")
            continue
        # Save latents right away so a crash or sleep loses at most one clip.
        torch.save(lat, j["path"] + ".lat.pt")
        finished.append(j)
        sys.stderr.write(f"clip {k}/{len(todo)} (scene {j['n']}): denoised in "
                         f"{time.time() - started:.0f}s\n")

    vae, processor = pipe.vae, pipe.video_processor
    del pipe, upsampler, transformer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    # Upscaled clips are too big to decode whole on 18 GB: decode in space/time tiles.
    vae.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512,
                      tile_sample_stride_height=448, tile_sample_stride_width=448,
                      tile_sample_min_num_frames=16, tile_sample_stride_num_frames=12)
    done = 0
    for k, j in enumerate(finished, 1):
        started = time.time()
        lat = torch.load(j["path"] + ".lat.pt", weights_only=True).to(device, dt)
        ts = None
        if getattr(vae.config, "timestep_conditioning", False):
            noise = torch.randn(lat.shape, generator=torch.Generator(device).manual_seed(seed),
                                device=device, dtype=lat.dtype)
            lat = 0.975 * lat + 0.025 * noise
            ts = torch.tensor([0.05], device=device, dtype=lat.dtype)
        try:
            with torch.no_grad():
                video = vae.decode(lat, ts, return_dict=False)[0]
        except RuntimeError as e:
            sys.stderr.write(f"scene {j['n']} decode failed ({e})\n")
            continue
        out = processor.postprocess_video(video, output_type="np")[0]
        clip = (np.clip(out, 0, 1) * 255).astype(np.uint8)[..., ::-1]
        clip = match_colors(clip, np.asarray(j["image"])[..., ::-1])
        np.savez_compressed(j["path"], frames=np.ascontiguousarray(clip))
        os.remove(j["path"] + ".lat.pt")
        j["scene"]["clip"] = j["path"]
        done += 1
        sys.stderr.write(f"scene {j['n']}: decoded in {time.time() - started:.0f}s\n")
        del video, lat
        if device == "mps":
            torch.mps.empty_cache()
    return len(jobs) - len(todo) + done
