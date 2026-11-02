#!/bin/bash
# =============================================================================
# SLURM job script - run ONE dated experiment script at full scale.
#
# Submit from the project root, passing the script name:
#   sbatch hpc/run_experiment.slurm 2026-09-17_single_layer_baseline.py
#
# The jobs are independent, so submit several at once. Only 1 GPU requested
# (training uses 1). Wall-clock for a full run has not been measured yet;
# the stacked run does roughly 2x the discrimination-layer work per step
# plus review replay, so every job just requests the full 50 h limit.
# =============================================================================
#SBATCH --job-name=dcnet-exp
#SBATCH --partition=markov_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --mem=32G
#SBATCH --account=wids
#SBATCH --qos=l1
#SBATCH --output=hpc/logs/slurm-%j.out
#SBATCH --error=hpc/logs/slurm-%j.err

if [ -z "$1" ]; then
    echo "usage: sbatch hpc/run_experiment.slurm <script in experiments/> [extra args]"
    exit 1
fi
SCRIPT="${1#experiments/}"; shift

module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$HOME/dcnet_venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p hpc/logs DATA RESULT checkpoints

# Fail fast with a clear message if a dataset is missing (compute nodes cannot download).
python training/datasets.py list

# Every experiment script accepts --device; the point #4 sweep and point #6 scripts
# also take their own flags (--runs, --arch, --head, ...), passed through via "$@".
CMD=(python "experiments/$SCRIPT" --device cuda "$@")

# Same header as hpc/pioneer/run_experiment.slurm. Commit shows "unknown" if this
# copy of the repo is not a git clone (the old hand-uploaded Markov copy).
echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Host       : $(hostname)"
echo "Start time : $(date)"
echo "Commit     : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Git status (empty = clean tree):"
git status --short 2>/dev/null | head
echo "GPU:"
nvidia-smi -L
echo "Python     : $(command -v python)"
echo "Command    : ${CMD[*]}"
echo "=========================================="

START=$SECONDS
"${CMD[@]}"
STATUS=$?
ELAPSED=$((SECONDS - START))

echo "=========================================="
echo "End time   : $(date)"
printf 'Elapsed    : %02d:%02d:%02d\n' $((ELAPSED / 3600)) $((ELAPSED % 3600 / 60)) $((ELAPSED % 60))
echo "Exit code  : $STATUS"
echo "=========================================="
exit $STATUS