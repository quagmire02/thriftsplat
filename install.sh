#!/usr/bin/env bash
# ThriftSplat installer.
#
# Every step here exists because the obvious version of it fails. Read the
# comments before changing the order.
set -euo pipefail

VENV="${VENV:-.venv}"
PY="${PY:-python3}"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    warning: %s\033[0m\n' "$*"; }
die()  { printf '\033[31m    error: %s\033[0m\n' "$*" >&2; exit 1; }

say "Checking prerequisites"

command -v ffmpeg >/dev/null || die "ffmpeg not found. Install it: sudo apt install -y ffmpeg"
echo "    ffmpeg      ok"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found. On WSL2 the driver comes from Windows, not Linux."
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "    GPU         $GPUNAME (${VRAM} MiB)"
[ "$VRAM" -lt 5500 ] && warn "under 6 GB of VRAM. Reconstruction needs about 3 GB, training about 3.4 GB."

# gsplat JIT compiles its CUDA kernels on first use, so it needs a real nvcc.
# PyTorch ships only the CUDA runtime, and the pip packaged nvidia-cuda-nvcc
# wheels contain ptxas and nvvm but NOT the nvcc frontend.
if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda-12.4/bin/nvcc ]; then
  cat <<'EOF'
    nvcc not found. gsplat cannot build its kernels without it.

    On WSL2 (use the wsl-ubuntu repo, it omits the Linux GPU driver):
      wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
      sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
      sudo apt-get -y install cuda-nvcc-12-4 cuda-cudart-dev-12-4 cuda-cccl-12-4

    That is 10 packages rather than the 95 in the full cuda-toolkit-12-4.
    It must be 12-4 to match the PyTorch build below.
EOF
  die "install nvcc, then rerun"
fi
echo "    nvcc        ok"

say "Creating virtualenv at $VENV"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install -q --upgrade pip

say "Installing PyTorch (cu124)"
"$PIP" install -q torch==2.6.0 torchvision==0.21.0 --index-url "$TORCH_INDEX"

# MapAnything depends on uniception, which requires torchaudio UNPINNED. torchaudio
# publishes no torch pin of its own, so a plain install grabs the newest build and
# silently pairs it with your torch. It installs fine and then dies at import with
# an ABI error. Pin the matching version before anything can pull a newer one.
say "Installing torchaudio (version matched, see comment in this script)"
"$PIP" install -q "torchaudio==2.6.0+cu124" --index-url "$TORCH_INDEX"

say "Installing MapAnything"
if [ ! -d third_party/map-anything ]; then
  mkdir -p third_party
  git clone --depth 1 https://github.com/facebookresearch/map-anything.git third_party/map-anything
fi
# The README asks for conda and Python 3.12, but pyproject declares >=3.10.0.
"$PIP" install -q -e third_party/map-anything

say "Installing gsplat"
"$PIP" install -q gsplat --index-url https://docs.gsplat.studio/whl/pt26cu124 \
                       --extra-index-url https://pypi.org/simple
"$PIP" install -q scipy plyfile

say "Verifying"
"$VENV/bin/python" - <<'PY'
import torch, torchaudio, torchvision
assert torch.cuda.is_available(), "CUDA not visible to torch"
print(f"    torch       {torch.__version__}")
print(f"    torchvision {torchvision.__version__}")
print(f"    torchaudio  {torchaudio.__version__}  (imported, so no ABI mismatch)")
print(f"    device      {torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
if torch.cuda.get_device_capability(0)[0] < 8:
    print("    note        pre-Ampere GPU: weights run in bf16 via emulation. fp16 produces NaN, see README.")
import mapanything, gsplat
print(f"    mapanything ok")
print(f"    gsplat      {gsplat.__version__}")
PY

cat <<EOF

$(printf '\033[1mReady.\033[0m')

  # the Xet download backend deadlocks on some WSL2 network setups
  export HF_HUB_DISABLE_XET=1
  export CUDA_HOME=/usr/local/cuda-12.4
  export PATH="\$PWD/$VENV/bin:\$CUDA_HOME/bin:\$PATH"

  python -m thriftsplat yourclip.mp4 -o myscene

First run downloads about 4.6 GB of MapAnything weights, and gsplat spends
10 to 20 minutes compiling CUDA kernels. Both are one time costs.
EOF
