#!/bin/bash
# =============================================================================
# Pre-download datasets - run this from the HPC LOGIN NODE only.
#
# Compute nodes on Markov do NOT have outbound internet access, so datasets
# must be downloaded before a job starts. Every dataset the code knows
# about is registered in training/datasets.py; this script calls its
# downloader inside the venv from hpc/setup_env.sh.
#
# Usage (from the project root):
#   bash hpc/download_data.sh                      # mnist fashion_mnist cifar10
#   bash hpc/download_data.sh all                  # everything registered
#   bash hpc/download_data.sh kmnist cifar10_gray  # any registered names
#
# Safe to run multiple times - files already present are skipped.
# =============================================================================

set -e

module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$HOME/dcnet_venv/bin/activate"

if [ $# -eq 0 ]; then
    set -- mnist fashion_mnist cifar10
fi

python training/datasets.py download "$@"
echo
python training/datasets.py list