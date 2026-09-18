#!/bin/bash
# =============================================================================
# Pre-download datasets - run this from the HPC LOGIN NODE only.
#
# Compute nodes on CWRU HPC do NOT have outbound internet access, so datasets
# must be downloaded before a job starts. Every dataset the code knows about
# is registered in training/datasets.py; this script just calls its
# downloader.
#
# Usage (from the project root):
#   bash hpc/download_data.sh                      # mnist fashion_mnist cifar10
#   bash hpc/download_data.sh all                  # everything registered
#   bash hpc/download_data.sh kmnist cifar10_gray  # any registered names
#
# Safe to run multiple times - files already present are skipped.
# =============================================================================

set -e

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

if [ $# -eq 0 ]; then
    set -- mnist fashion_mnist cifar10
fi

python training/datasets.py download "$@"
echo
python training/datasets.py list
