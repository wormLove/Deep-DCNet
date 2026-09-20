#!/bin/bash
# =============================================================================
# One-time dependency setup - run this from the HPC LOGIN NODE only.
#
# Markov has no pre-built PyTorch module, so we build our own virtual
# environment and install PyTorch from PyPI. Its wheels bundle their own
# CUDA runtime libraries, so no system CUDA module is needed at import
# time - just a GPU driver new enough for CUDA 12.6, which this cluster
# should have since it offers the CUDA/12.6.0 module.
#
# Usage (from the project root, on the login node):
#   bash hpc/setup_env.sh
#
# The venv lives at ~/dcnet_venv and is reused by every job - compute nodes
# don't need internet because everything is already installed there.
# =============================================================================

set -e

module purge
module load Python/3.13.5-GCCcore-14.3.0

if [ ! -d "$HOME/dcnet_venv" ]; then
    echo "Creating virtual environment at ~/dcnet_venv ..."
    python -m venv "$HOME/dcnet_venv"
fi

source "$HOME/dcnet_venv/bin/activate"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tqdm matplotlib

echo ""
echo "=== Smoke test ==="
python -c "
import torch, torchvision, tqdm, matplotlib
print(f'PyTorch     : {torch.__version__}')
print(f'torchvision : {torchvision.__version__}')
print(f'tqdm        : {tqdm.__version__}')
print(f'matplotlib  : {matplotlib.__version__}')
print(f'CUDA available (expected False on the login node) : {torch.cuda.is_available()}')
"
echo "=== Setup complete ==="