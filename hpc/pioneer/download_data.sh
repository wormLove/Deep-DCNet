#!/bin/bash
# =============================================================================
# Pre-download datasets on Pioneer - run on the LOGIN NODE only
# (compute nodes may have no internet access).
#
# Usage (from the repo root):
#   bash hpc/pioneer/download_data.sh                      # mnist fashion_mnist kmnist cifar10
#   bash hpc/pioneer/download_data.sh all                  # everything registered
#   bash hpc/pioneer/download_data.sh kmnist cifar10_gray  # any registered names
#
# Safe to re-run - files already present are skipped.
# =============================================================================

set -e

# Run from the repo root no matter where the script was called from.
cd "$(dirname "$0")/../.." || exit 1

source hpc/pioneer/activate.sh

if [ $# -eq 0 ]; then
    set -- mnist fashion_mnist kmnist cifar10
fi

python training/datasets.py download "$@"
echo
python training/datasets.py list
