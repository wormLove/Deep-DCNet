#!/bin/bash
# =============================================================================
# One-time dependency setup — run this from the HPC LOGIN NODE only.
#
# PyTorch 2.1.2, torchvision 0.16.2, and CUDA 12.1.1 are already bundled in
# the cluster module PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1.
# We only need to pip-install the two small packages that aren't included.
#
# Usage (from the project root):
#   bash hpc/setup_env.sh
#
# Packages are installed to ~/.local and are picked up automatically whenever
# the PyTorch-bundle module is loaded — no conda environment needed.
# =============================================================================

set -e

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo "Installing tqdm and matplotlib to ~/.local ..."
pip install --user tqdm matplotlib

# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
echo ""
echo "=== Smoke test ==="
python -c "
import torch, torchvision, tqdm, matplotlib
print(f'PyTorch     : {torch.__version__}')
print(f'torchvision : {torchvision.__version__}')
print(f'tqdm        : {tqdm.__version__}')
print(f'matplotlib  : {matplotlib.__version__}')
print(f'CUDA available : {torch.cuda.is_available()}')
"
echo "=== Setup complete ==="
