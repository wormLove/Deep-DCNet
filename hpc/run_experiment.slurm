#!/bin/bash
# =============================================================================
# SLURM job script - run ONE dated experiment script at full scale.
#
# Submit from the project root, passing the script name:
#   sbatch hpc/run_experiment.slurm 2026-09-17_single_layer_baseline.py
#   sbatch hpc/run_experiment.slurm 2026-09-17_point5_traditional_head.py
#   sbatch hpc/run_experiment.slurm 2026-09-17_stacked_2layer_baseline.py
#
# The three jobs are independent, so submit all three at once. Same module /
# partition as hpc/run_train.slurm, but only 1 GPU requested (training uses 1).
# Wall-clock for a full run has not been measured yet; the stacked run does
# roughly 2x the discrimination-layer work per step plus review replay, so
# every job just requests the full 50 h limit.
# =============================================================================
#SBATCH --job-name=dcnet-exp
#SBATCH --partition=markov_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --mem=32G
#SBATCH --output=hpc/logs/slurm-%j.out
#SBATCH --error=hpc/logs/slurm-%j.err

if [ -z "$1" ]; then
    echo "usage: sbatch hpc/run_experiment.slurm <script in experiments/> [extra args]"
    exit 1
fi
SCRIPT="$1"; shift

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p hpc/logs DATA RESULT checkpoints

# Fail fast with a clear message if a dataset is missing (compute nodes cannot download).
python training/datasets.py list

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Experiment : $SCRIPT $*"
echo "Commit     : $(git rev-parse --short HEAD 2>/dev/null)"
echo "Start time : $(date)"
python -c "import torch; print(f'GPU        : {torch.cuda.get_device_name(0)}')"
echo "=========================================="

# Every experiment script accepts --device; the point #4 sweep and point #6 scripts
# also take their own flags (--runs, --arch, --head, ...), passed through via "$@".
python "experiments/$SCRIPT" --device cuda "$@"

echo "=========================================="
echo "Job finished : $(date)"
echo "=========================================="
