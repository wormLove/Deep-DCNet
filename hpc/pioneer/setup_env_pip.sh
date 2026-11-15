#!/bin/bash
# =============================================================================
# One-time setup of the pip PyTorch environment on Pioneer - run on the
# LOGIN NODE, from the repo root:
#   bash hpc/pioneer/setup_env_pip.sh
#
# Why: the PyTorch-bundle module is built for sm_75 only, so it runs only on
# the RTX 2080 Ti nodes (gpu2080). The pip wheels below are the same torch
# version (2.1.2) built for CUDA 12.1 with kernels for every current GPU
# (L40S, V100, H100, 4090, ...), so only the GPU support changes.
#
# What it does:
#   - module purge, then loads a plain Python module (no PyTorch module!);
#   - creates a plain venv at ~/dcnet_pioneer_pip_venv (NOT
#     --system-site-packages, so nothing leaks in from a module);
#   - pip-installs torch 2.1.2 + torchvision 0.16.2 (cu121), numpy<2,
#     tqdm and matplotlib;
#   - prints the versions and the GPU architectures torch was built for.
#
# Overrides (optional): PY_MODULE=<module>   DCNET_PIP_VENV=<path>
#
# Safe to re-run: an existing venv is reused and pip skips anything already
# installed at the right version.
# =============================================================================

set -e

# Run from the repo root no matter where the script was called from.
cd "$(dirname "$0")/../.." || exit 1

PY_MODULE="${PY_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
DCNET_PIP_VENV="${DCNET_PIP_VENV:-$HOME/dcnet_pioneer_pip_venv}"
TORCH_VERSION="2.1.2"
TORCHVISION_VERSION="0.16.2"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# `module` is normally already defined; fall back to the Lmod init script
# for shells where it is not.
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
        [ -f "$f" ] && . "$f" && break
    done
fi

module purge
if ! module load "$PY_MODULE"; then
    echo "" >&2
    echo "setup_env_pip.sh: could not load module '$PY_MODULE'." >&2
    echo "Find the available Python 3.11 modules with:" >&2
    echo "    module spider Python/3.11" >&2
    echo "then re-run with e.g.  PY_MODULE=<name from that list> bash hpc/pioneer/setup_env_pip.sh" >&2
    echo "(and use the same PY_MODULE when submitting jobs, or change the default in hpc/pioneer/activate.sh)." >&2
    exit 1
fi
echo "Loaded $PY_MODULE -> $(command -v python) ($(python --version 2>&1))"

if [ ! -f "$DCNET_PIP_VENV/bin/activate" ]; then
    echo "Creating venv at $DCNET_PIP_VENV ..."
    python -m venv "$DCNET_PIP_VENV"
else
    echo "Reusing existing venv at $DCNET_PIP_VENV"
fi
source "$DCNET_PIP_VENV/bin/activate"

pip install --upgrade pip
# --no-cache-dir: the cu121 wheels are ~2.5 GB; don't keep a second copy in ~/.cache/pip.
pip install --no-cache-dir "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"
# torch 2.1.2 wheels are built against numpy 1.x.
pip install "numpy<2" tqdm matplotlib

echo ""
echo "=== Environment check ==="
echo "venv        : $DCNET_PIP_VENV"
echo "python      : $(command -v python)"
python -c "
import sys, torch, torchvision, numpy
print(f'Python      : {sys.version.split()[0]}')
print(f'PyTorch     : {torch.__version__}')
print(f'torchvision : {torchvision.__version__}')
print(f'torch CUDA  : {torch.version.cuda}')
print(f'numpy       : {numpy.__version__}')
print(f'CUDA available (expected False on the login node): {torch.cuda.is_available()}')
# get_arch_list() is empty when no GPU is visible (login node); the compiled
# arch flags below are what the wheel was built for and work without a GPU.
print(f'cuda.get_arch_list() : {torch.cuda.get_arch_list()}')
print(f'compiled arch flags  : {torch._C._cuda_getArchFlags()}')
import tqdm, matplotlib
print(f'tqdm        : {tqdm.__version__}')
print(f'matplotlib  : {matplotlib.__version__}')
"
echo "=== Setup complete ==="
