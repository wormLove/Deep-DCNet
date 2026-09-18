#!/bin/bash
# =============================================================================
# SLURM job script — Deep-DCNet classifier training
#
# Resources requested:  2 GPUs  |  11 CPUs  |  50 hours
# Training uses:        1 GPU   (GPU 1 is allocated but left idle)
#
# Submit from the project root:
#   sbatch hpc/run_train.slurm
# =============================================================================

# --- Resource requests -------------------------------------------------------
#SBATCH --job-name=deep-dcnet
#SBATCH --partition=markov_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:2
#SBATCH --time=50:00:00
#SBATCH --mem=32G

# --- Output logs -------------------------------------------------------------
#SBATCH --output=hpc/logs/slurm-%j.out
#SBATCH --error=hpc/logs/slurm-%j.err

# Optional: uncomment if your HPC allocation requires an account name
# #SBATCH --account=YOUR_ACCOUNT_NAME

# =============================================================================
# Environment
# PyTorch 2.1.2 + torchvision 0.16.2 + CUDA 12.1.1 — all in one module load.
# tqdm and matplotlib were installed to ~/.local via hpc/setup_env.sh.
# =============================================================================

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Move to the project root (where you ran sbatch from)
cd "$SLURM_SUBMIT_DIR"

# Create required directories if they don't exist
mkdir -p hpc/logs DATA RESULT

# =============================================================================
# GPU setup
# =============================================================================

# We use only GPU 0 even though 2 are allocated.
# GPU 1 stays idle — remove this line if you add a second experiment below.
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "GPU in use : $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'GPU        : {torch.cuda.get_device_name(0)}')"
echo "=========================================="

# =============================================================================
# Training
#
# Flags you are most likely to tune between runs:
#   --training-mode       plain | review
#   --integration-dim     256 (omit entirely to disable the integration layer)
#   --num-train-samples   up to 60000 (full MNIST train split)
#   --batch-size          32 is a good GPU default; increase if VRAM allows
#   --organize-interval-samples   how often Hebb/Anti-Hebb weights update
#   --eval-interval-samples       0 = evaluate only at organize boundaries
# =============================================================================

python training/train_classifier.py \
    --device cuda \
    --training-mode plain \
    --integration-dim 256 \
    --integration-activation relu \
    --num-train-samples 60000 \
    --num-test-samples 10000 \
    --batch-size 32 \
    --organize-interval-samples 1000 \
    --eval-interval-samples 5000 \
    --show-train-log

echo "=========================================="
echo "Job finished : $(date)"
echo "=========================================="
