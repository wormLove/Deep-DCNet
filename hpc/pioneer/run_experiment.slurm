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
#
# Environment: DCNET_TORCH=pip (default; pip torch 2.1.2+cu121 venv) or
# DCNET_TORCH=module (PyTorch-bundle module, RTX 2080 Ti only), e.g.
#   sbatch --constraint=gpu2080 --export=ALL,DCNET_TORCH=module hpc/pioneer/run_experiment.slurm <script> [flags]
# =============================================================================
#SBATCH --job-name=dcnet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Pinned GPU type so experiments are comparable (all arms on the same GPU).
# Override per job with `sbatch --constraint=<feature> ...` (features: gpul40s,
# gpu2080, gpu4v100, gpu2v100, gpu2h100, gpu4090, gpup100). With
# DCNET_TORCH=module you MUST use --constraint=gpu2080 (the module is sm_75 only).
#SBATCH --constraint=gpul40s
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
echo "DCNET_TORCH: ${DCNET_TORCH:-pip}"
# SLURM_JOB_CONSTRAINTS is not set on Pioneer; read the job's Features instead.
FEATURES=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'Features=[^ ]*' | head -1 | cut -d= -f2)
echo "Constraint : ${FEATURES:-unknown}"
echo "Python     : $(command -v python)"
python -c "
import torch
print(f'torch      : {torch.__version__} (CUDA {torch.version.cuda})')
print(f'arch list  : {torch.cuda.get_arch_list()}')
print(f'torch GPU  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none visible\"}')
" || echo "torch      : could not import torch"
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
