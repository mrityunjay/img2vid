"""Text-to-image with FLUX.1-schnell (Apache-2.0), run locally from a 4-bit GGUF file.

Memory plan for an 18 GB Mac: the T5 text encoder (~9.5 GB, the same weights LTX-Video
uses) runs in its own process and its prompt embeddings are cached on disk; only then is
the 4-bit transformer (~7 GB) loaded, so the two never share memory.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import sys
import time

from . import animate

GGUF_REPO = "city96/FLUX.1-schnell-gguf"
QUANTS = {  # name -> file; bigger = closer to the full model, more memory
    "q4": "flux1-schnell-Q4_K_S.gguf",   # 6.8 GB
    "q5": "flux1-schnell-Q5_K_S.gguf",   # 8.3 GB
    "q8": "flux1-schnell-Q8_0.gguf",     # 12.7 GB
}
CLIP_REPO, CLIP_FILE = "comfyanonymous/flux_text_encoders", "clip_l.safetensors"
CLIP_CONFIG = "openai/clip-vit-large-patch14"  # config + tokenizer only
VAE_REPO, VAE_FILE = "ffxvs/vae-flux", "ae.safetensors"
T5_REPO = "Lightricks/LTX-Video-0.9.5"  # google/t5-v1_1-xxl encoder, shared with animate-hq
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "flux_config")
MAX_TOKENS = 256  # schnell's T5 limit
ASPECTS = {  # ~1 megapixel, multiples of 16
    "1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344),
    "4:3": (1152, 864), "3:4": (864, 1152), "3:2": (1216, 832), "2:3": (832, 1216),
}


class ImagineUnavailable(Exception):
    pass


def _embed_path(prompt: str) -> str:
    key = hashlib.sha1(f"flux-t5|{T5_REPO}|{MAX_TOKENS}|{prompt}".encode()).hexdigest()
    return os.path.join(animate.CACHE_DIR, "embeds", "flux-" + key + ".pt")


def _encode_t5(prompts: list[str], device: str) -> None:
    """Runs in a child process: cache T5 embeddings for prompts not cached yet."""
    import torch
    from transformers import T5EncoderModel, T5TokenizerFast

    missing = [p for p in prompts if not os.path.exists(_embed_path(p))]
    if not missing:
        return
    print("loading T5 text encoder; this takes a minute or two...", file=sys.stderr)
    tok = T5TokenizerFast.from_pretrained(T5_REPO, subfolder="tokenizer")
    enc = T5EncoderModel.from_pretrained(T5_REPO, subfolder="text_encoder",
                                         torch_dtype=torch.bfloat16).to(device)
    os.makedirs(os.path.dirname(_embed_path("")), exist_ok=True)
    with torch.no_grad():
        for p in missing:
            ids = tok(p, padding="max_length", max_length=MAX_TOKENS, truncation=True,
                      return_tensors="pt").input_ids.to(device)
            torch.save(enc(ids)[0].to("cpu"), _embed_path(p))


def _clip_pooled(prompts: list[str], device: str) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(CLIP_CONFIG)
    model = CLIPTextModel(CLIPTextConfig.from_pretrained(CLIP_CONFIG))
    state = load_file(hf_hub_download(CLIP_REPO, CLIP_FILE))
    if not any(k.startswith("text_model.") for k in model.state_dict()):
        # transformers 5 dropped the text_model. prefix
        state = {k.removeprefix("text_model."): v for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    if [k for k in missing if "position_ids" not in k]:
        raise ImagineUnavailable(f"CLIP weights don't match: missing {missing[:3]}")
    model = model.to(device, torch.bfloat16).eval()
    out = {}
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").input_ids.to(device)
            out[p] = model(ids).pooler_output.to("cpu")
    del model
    return out


def size_for(aspect: str | None = None, size: str | None = None) -> tuple[int, int]:
    if size:
        w, h = (int(x) for x in size.lower().split("x"))
    else:
        if aspect not in ASPECTS:
            raise ValueError(f"aspect must be one of {', '.join(ASPECTS)}")
        w, h = ASPECTS[aspect]
    return max(256, w // 16 * 16), max(256, h // 16 * 16)


def generate(prompts: list[str], outputs: list[str], width: int = 1024, height: int = 1024,
             steps: int = 4, seed: int = 0, quant: str = "q4",
             same_seed: bool = False, seeds: list[int] | None = None) -> list[str]:
    """Render prompts[i] to outputs[i] (PNG/JPG). Seed i is `seed + i`, or `seed` for
    every image with same_seed (keeps a recurring character's look more stable), or
    seeds[i] when a list is given."""
    try:
        import torch
        from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline,
                               FluxTransformer2DModel, GGUFQuantizationConfig)
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImagineUnavailable("install the motion extra and gguf: "
                                 "pip install -e '.[motion,image]'") from e
    if quant not in QUANTS:
        raise ValueError(f"quant must be one of {', '.join(QUANTS)}")

    device = animate._device()
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(animate.MPS_MEMORY_FRACTION)
    unique = sorted(set(prompts))
    if any(not os.path.exists(_embed_path(p)) for p in unique):
        proc = multiprocessing.get_context("spawn").Process(target=_encode_t5,
                                                            args=(unique, device))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise ImagineUnavailable("the T5 text encoder failed")
    t5 = {p: torch.load(_embed_path(p), weights_only=True) for p in unique}
    pooled = _clip_pooled(unique, device)

    print("loading FLUX.1-schnell...", file=sys.stderr)
    dt = torch.bfloat16
    transformer = FluxTransformer2DModel.from_single_file(
        hf_hub_download(GGUF_REPO, QUANTS[quant]),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dt),
        config=CONFIG_DIR, subfolder="transformer", torch_dtype=dt)
    vae = AutoencoderKL.from_single_file(hf_hub_download(VAE_REPO, VAE_FILE),
                                         config=CONFIG_DIR, subfolder="vae", torch_dtype=dt)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(CONFIG_DIR,
                                                                subfolder="scheduler")
    pipe = FluxPipeline(scheduler=scheduler, vae=vae, transformer=transformer,
                        text_encoder=None, tokenizer=None,
                        text_encoder_2=None, tokenizer_2=None).to(device)
    # Decode in overlapping 512px tiles: a full 1 MP decode next to the transformer
    # needs more memory than an 18 GB Mac lets the GPU have.
    vae.enable_tiling()
    vae.tile_sample_min_size = 512
    vae.tile_latent_min_size = 512 // 8

    for i, (prompt, out) in enumerate(zip(prompts, outputs)):
        started = time.time()
        image = pipe(prompt_embeds=t5[prompt].to(device, dt),
                     pooled_prompt_embeds=pooled[prompt].to(device, dt),
                     width=width, height=height, num_inference_steps=steps,
                     guidance_scale=0.0, max_sequence_length=MAX_TOKENS,
                     generator=torch.Generator("cpu").manual_seed(
                         seeds[i] if seeds else seed if same_seed else seed + i)).images[0]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        image.save(out)
        print(f"[{i + 1}/{len(prompts)}] {out} ({time.time() - started:.0f}s)", file=sys.stderr)
        if device == "mps":
            torch.mps.empty_cache()
    return outputs
