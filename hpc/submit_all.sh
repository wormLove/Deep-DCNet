#!/bin/bash
# =============================================================================
# Submit the full experiment set as independent SLURM jobs (one GPU each).
#
# Usage (from the project root, after hpc/download_data.sh on the login node):
#   bash hpc/submit_all.sh            # submit everything below
#   bash hpc/submit_all.sh mnist      # only the three MNIST baselines
#   bash hpc/submit_all.sh point6     # only the Fashion-MNIST + CIFAR-10 runs
#
# The point #4 robustness sweep is NOT submitted here: it needs the trained
# runs to exist first. Once the jobs above have finished, run it as one job:
#   sbatch hpc/run_experiment.slurm 2026-09-18_point4_robustness.py
#   sbatch hpc/run_experiment.slurm 2026-09-18_point4_robustness.py --runs \
#       point6_fashion_mnist_single_layer_readout \
#       point6_fashion_mnist_single_layer_traditional_mlp \
#       point6_fashion_mnist_stacked_readout
# =============================================================================

set -e
WHAT="${1:-all}"

submit() {
    echo "sbatch hpc/run_experiment.slurm $*"
    sbatch hpc/run_experiment.slurm "$@"
}

if [ "$WHAT" = "all" ] || [ "$WHAT" = "mnist" ]; then
    submit 2026-09-17_single_layer_baseline.py
    submit 2026-09-17_point5_traditional_head.py
    submit 2026-09-17_stacked_2layer_baseline.py
fi

if [ "$WHAT" = "all" ] || [ "$WHAT" = "point6" ]; then
    for ds in fashion_mnist cifar10; do
        submit 2026-09-18_point6_${ds}.py --arch single_layer --head readout
        submit 2026-09-18_point6_${ds}.py --arch single_layer --head traditional_mlp
        submit 2026-09-18_point6_${ds}.py --arch stacked --head readout
    done
fi

echo
squeue -u "$USER"
