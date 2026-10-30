#!/bin/bash
# =============================================================================
# One-time environment setup for Pioneer - run on the LOGIN NODE.
#
# Pioneer already has a PyTorch module (torch + torchvision + CUDA), so no
# PyTorch install is needed. This script only makes sure tqdm and
# matplotlib are importable on top of it:
#   - if the module's python already has both, nothing is installed;
#   - otherwise a venv is created at ~/dcnet_pioneer_venv with
#     --system-site-packages (so it keeps the module's torch/torchvision)
#     and tqdm + matplotlib are pip-installed into it.
#
# Usage (from the repo root):
#   bash hpc/pioneer/setup_env.sh
#
# Safe to re-run: an existing venv is reused, and nothing is installed
# when the imports already work.
# =============================================================================

set -e

# Run from the repo root no matter where the script was called from.
cd "$(dirname "$0")/../.." || exit 1

# Module load + venv (if it already exists from an earlier run).
source hpc/pioneer/activate.sh

if python -c "import tqdm, matplotlib" >/dev/null 2>&1; then
    echo "tqdm and matplotlib already import with $(command -v python) - nothing to install."
else
    if [ ! -f "$DCNET_VENV/bin/activate" ]; then
        echo "Creating venv at $DCNET_VENV (with the module's site-packages) ..."
        python -m venv --system-site-packages "$DCNET_VENV"
    fi
    source "$DCNET_VENV/bin/activate"
    pip install tqdm matplotlib
fi

echo ""
echo "=== Environment check ==="
echo "python      : $(command -v python)"
python -c "
import sys, torch, torchvision
print(f'Python      : {sys.version.split()[0]}')
print(f'PyTorch     : {torch.__version__}')
print(f'torchvision : {torchvision.__version__}')
print(f'torch CUDA  : {torch.version.cuda}')
print(f'CUDA available (expected False on the login node): {torch.cuda.is_available()}')
import tqdm, matplotlib
print(f'tqdm        : {tqdm.__version__}')
print(f'matplotlib  : {matplotlib.__version__}')
"
echo "=== Setup complete ==="
