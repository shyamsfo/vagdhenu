#!/usr/bin/env bash
set -e
# Python 3.10. Linux needs a CUDA 12.1 GPU; macOS runs on Apple Metal (MPS) or CPU.

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" = "Darwin" ]; then
  if [ "$ARCH" != "arm64" ]; then
    echo "✗ Intel Macs are not supported: PyTorch stopped publishing macOS x86_64 wheels after 2.2.2," >&2
    echo "  and there is no Metal backend for them. Apple Silicon or a CUDA box only." >&2
    exit 1
  fi
  # PyPI ships the macOS arm64 wheels; the cu121 index has no Darwin builds at all.
  TORCH_INSTALL="pip install torch==2.4.1 torchaudio==2.4.1"
  command -v ffmpeg >/dev/null || echo "⚠ ffmpeg not found — 'brew install ffmpeg' (pydub decodes the reference clips)"
else
  TORCH_INSTALL="pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121"
fi

$TORCH_INSTALL
pip install -r requirements.txt
# IndicF5's setup.py asks for torch>=2.0.0, which lets a resolver upgrade the 2.4.1 just pinned
# (uv silently took it to 2.14.0). Re-assert the validated version, then fail loudly if it slipped.
$TORCH_INSTALL
python -c "import torch,sys; v=torch.__version__; sys.exit(0) if v.startswith('2.4.1') else sys.exit('torch is '+v+', expected 2.4.1 - a dependency overrode the pin')"

# NVIDIA BigVGAN is a repo, not a pip package (no setup.py). Clone it and add to PYTHONPATH so
# `import bigvgan` resolves (we use the torch path / use_cuda_kernel=False, so nothing is compiled).
[ -d BigVGAN/.git ] || git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git BigVGAN
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"   # add this to your shell rc for persistent use
python scripts/download_weights.py             # our weights -> models/ + IndicF5 base (vocab)

python - <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from device import pick_device, describe
print(f"✓ inference backend: {describe(pick_device())}")
PY
echo "✓ setup complete — see Quickstart in README.md"
