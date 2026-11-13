#!/usr/bin/env bash
# One-step setup for img2vid on macOS (Apple Silicon or Intel) and Linux.
#
#   ./install.sh               core tool; on Apple Silicon / NVIDIA also the local AI parts
#   ./install.sh --core        core tool only (slideshows, beat sync, titles, your own clips)
#   ./install.sh --with-3d     Intel Mac: also try CPU 3D depth motion (old PyTorch, slow, untested)
#   ./install.sh --models      also download the AI models now (~37 GB) instead of on first use
#   ./install.sh --core --with-audio --models  local narration and its ~354 MB models only
#
# Nothing here needs a paid service. The optional Claude "director" needs ANTHROPIC_API_KEY.
set -euo pipefail

cd "$(dirname "$0")"

CORE_ONLY=0
WITH_3D=0
PREFETCH=0
WITH_AUDIO=0
for arg in "$@"; do
  case "$arg" in
    --core) CORE_ONLY=1 ;;
    --with-3d) WITH_3D=1 ;;
    --models) PREFETCH=1 ;;
    --with-audio) WITH_AUDIO=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

if [[ "$WITH_AUDIO" == 1 && "$WITH_3D" == 1 ]]; then
  echo "Audio requires NumPy 2; experimental Intel --with-3d uses NumPy 1. Use separate environments." >&2
  exit 2
fi

say() { printf '\n==> %s\n' "$*"; }
OS="$(uname -s)"
ARCH="$(uname -m)"

# ---- 1. System tools: Python 3.10+, ffmpeg, libraqm (Hindi/Indic text shaping) ----
if [[ "$OS" == "Darwin" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required: install it from https://brew.sh, then re-run ./install.sh" >&2
    exit 1
  fi
  say "Installing python@3.12, ffmpeg and libraqm with Homebrew"
  brew install python@3.12 ffmpeg libraqm
  PY="$(brew --prefix)/bin/python3.12"
elif [[ "$OS" == "Linux" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    say "Installing python3, ffmpeg and libraqm with apt (needs sudo)"
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-dev ffmpeg libraqm-dev \
      libfreetype6-dev libjpeg-dev zlib1g-dev fonts-noto-core build-essential
  else
    echo "Install python3 (3.10+), ffmpeg and libraqm with your package manager first." >&2
  fi
  PY="$(command -v python3)"
else
  echo "Unsupported system $OS; on Windows use WSL2 (see README)." >&2
  exit 1
fi

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "Python 3.10 or newer is required" >&2; exit 1; }

# ---- 2. Virtual environment and the core tool ----
say "Creating .venv and installing the core tool"
[[ -d .venv ]] || "$PY" -m venv .venv
VPIP=".venv/bin/pip"
VPY=".venv/bin/python"
"$VPIP" install -q -U pip
"$VPIP" install -q -e .

# Pillow built against libraqm so Devanagari vowel signs render correctly.
say "Rebuilding Pillow with libraqm for Hindi/Indic titles"
if [[ "$OS" == "Darwin" ]]; then
  export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi
if "$VPIP" install -q --force-reinstall --no-deps --no-binary pillow "pillow==13.0.0"; then
  "$VPY" -c "from PIL import features; print('  libraqm text shaping:', features.check('raqm'))"
else
  echo "  Pillow rebuild failed; English text is fine, Hindi vowel signs may misplace."
fi

# ---- 3. Local AI (PyTorch) where the hardware can run it ----
HAS_GPU=0
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  HAS_GPU=1                                   # Apple Silicon: MPS
elif [[ "$OS" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1                                   # NVIDIA: CUDA
fi

if [[ "$CORE_ONLY" == 1 ]]; then
  say "Skipping AI parts (--core)"
elif [[ "$HAS_GPU" == 1 ]]; then
  if [[ "$OS" == "Linux" ]]; then
    say "Installing CUDA PyTorch (change cu124 in install.sh to match your CUDA if needed)"
    "$VPIP" install -q torch --index-url https://download.pytorch.org/whl/cu124
  fi
  say "Installing local AI extras (3D depth, AI motion, text-to-image)"
  "$VPIP" install -q -e '.[ai,motion,image]'
elif [[ "$OS" == "Darwin" && "$ARCH" == "x86_64" ]]; then
  echo
  echo "Intel Mac: PyTorch has no GPU support here and stopped Intel-Mac releases at 2.2.2,"
  echo "so AI motion (animate, animate-hq) and imagine are not installed. The core tool,"
  echo "beat sync and your own clips (img2vid clips) all work."
  if [[ "$WITH_3D" == 1 ]]; then
    say "Trying CPU 3D depth motion with PyTorch 2.2.2 (experimental)"
    "$VPIP" install -q "torch==2.2.2" "numpy<2" "transformers>=4.45,<4.50" \
      "huggingface_hub>=0.23" safetensors sentencepiece protobuf \
      || echo "  3D depth install failed; use --motion flat."
  fi
else
  say "No supported GPU found; installing only the Claude director extra (optional)"
  "$VPIP" install -q -e '.[ai]'
fi

if [[ "$WITH_AUDIO" == 1 ]]; then
  say "Installing local CPU narration (Kokoro; no GPU needed)"
  "$VPIP" install -q -e '.[audio]'
  if [[ "$PREFETCH" == 1 ]]; then
    "$VPY" -m img2vid speak --download-models
  fi
fi

# ---- 4. Optional: download image/video models now ----
if [[ "$PREFETCH" == 1 && "$HAS_GPU" == 1 && "$CORE_ONLY" == 0 ]]; then
  say "Downloading AI models (~37 GB) into ~/.cache/huggingface"
  "$VPY" - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("depth-anything/Depth-Anything-V2-Small-hf")
snapshot_download("hustvl/yolos-small")
snapshot_download("Lightricks/LTX-Video-0.9.5")
hf_hub_download("Lightricks/LTX-Video", "ltxv-2b-0.9.8-distilled.safetensors")
snapshot_download("Lightricks/ltxv-spatial-upscaler-0.9.7",
                  allow_patterns=["latent_upsampler/*", "model_index.json"])
hf_hub_download("city96/FLUX.1-schnell-gguf", "flux1-schnell-Q4_K_S.gguf")
hf_hub_download("comfyanonymous/flux_text_encoders", "clip_l.safetensors")
hf_hub_download("ffxvs/vae-flux", "ae.safetensors")
snapshot_download("openai/clip-vit-large-patch14", allow_patterns=["*.json", "*.txt"])
EOF
fi

# ---- 5. Check ----
say "Checking the install"
command -v ffmpeg >/dev/null && echo "  ffmpeg: ok"
.venv/bin/img2vid --help >/dev/null && echo "  img2vid: ok"
"$VPY" - <<'EOF'
try:
    import torch
    dev = "cuda" if torch.cuda.is_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  torch {torch.__version__}: {dev}")
except ImportError:
    print("  torch: not installed (flat motion only)")
EOF

cat <<'EOF'

Done. Try:
  .venv/bin/img2vid make path/to/photos/ -o video.mp4 --music song.mp3 --fit-music --sync-beats
See README.md for every command.
EOF
