#!/bin/bash
# =============================================================================
# Pioneer SLURM job - run ONE experiment script from experiments/.
#
# Submit from the repo root:
#   sbatch hpc/pioneer/run_experiment.slurm <script in experiments/> [script flags]
# e.g.
#   sbatch hpc/pioneer/run_experiment.slurm 2026-09-17_single_layer_baseline.py --smoke
#
# --device cuda is added automatically; everything after the script name
# is passed through to it. Logs go to hpc/logs/slurm-<jobid>.out/.err.
# =============================================================================
#SBATCH --job-name=dcnet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/slurm-%j.out
#SBATCH --error=hpc/logs/slurm-%j.err
# Fill in ONLY if jobs are rejected for a missing/invalid account (find yours with
# `sacctmgr -nP show assoc user=$USER format=account,partition`), then remove the
# extra leading '#':
##SBATCH --account=

if [ -z "$1" ]; then
    echo "usage: sbatch hpc/pioneer/run_experiment.slurm <script in experiments/> [script flags]"
    exit 1
fi
SCRIPT="${1#experiments/}"; shift

cd "$SLURM_SUBMIT_DIR" || { echo "cannot cd to SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"; exit 1; }
source hpc/pioneer/activate.sh || { echo "environment setup failed"; exit 1; }

if [ ! -f "experiments/$SCRIPT" ]; then
    echo "no such script: experiments/$SCRIPT"
    exit 1
fi

CMD=(python "experiments/$SCRIPT" --device cuda "$@")

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
