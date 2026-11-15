#!/bin/bash
# =============================================================================
# Pioneer environment - meant to be SOURCED, not run:
#   source hpc/pioneer/activate.sh
#
# Two environments, picked with DCNET_TORCH (default: pip):
#   DCNET_TORCH=pip     plain Python module + the venv that
#                       hpc/pioneer/setup_env_pip.sh creates
#                       (~/dcnet_pioneer_pip_venv, pip torch 2.1.2 + cu121).
#                       Runs on L40S / V100 / H100 / 4090 / 2080 nodes.
#   DCNET_TORCH=module  the cluster's PyTorch-bundle module (+ the small
#                       venv from hpc/pioneer/setup_env.sh if it exists).
#                       Built for sm_75 only -> RTX 2080 Ti (gpu2080) nodes only.
#
# Used interactively and by hpc/pioneer/run_experiment.slurm, e.g.
#   source hpc/pioneer/activate.sh                       # pip env
#   DCNET_TORCH=module source hpc/pioneer/activate.sh    # module env
#
# No `set -e` here on purpose: this file runs inside the caller's shell.
# =============================================================================

DCNET_TORCH="${DCNET_TORCH:-pip}"

# pip route
PY_MODULE="${PY_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
DCNET_PIP_VENV="${DCNET_PIP_VENV:-$HOME/dcnet_pioneer_pip_venv}"

# module route
DCNET_MODULE="PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1"
DCNET_VENV="${DCNET_VENV:-$HOME/dcnet_pioneer_venv}"

# `module` is normally already defined; fall back to the Lmod init script
# for shells where it is not (some non-interactive shells).
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
        [ -f "$f" ] && . "$f" && break
    done
fi

case "$DCNET_TORCH" in
    pip)
        module purge && module load "$PY_MODULE" || {
            echo "activate.sh: could not load module $PY_MODULE (find one with: module spider Python/3.11)" >&2
            return 1 2>/dev/null || exit 1
        }
        if [ ! -f "$DCNET_PIP_VENV/bin/activate" ]; then
            echo "activate.sh: no pip venv at $DCNET_PIP_VENV." >&2
            echo "  Create it once on the login node:  bash hpc/pioneer/setup_env_pip.sh" >&2
            echo "  (or use the module env:  DCNET_TORCH=module, GPU constraint gpu2080)" >&2
            return 1 2>/dev/null || exit 1
        fi
        . "$DCNET_PIP_VENV/bin/activate"
        echo "activate.sh: DCNET_TORCH=pip ($PY_MODULE + venv $DCNET_PIP_VENV)"
        ;;
    module)
        module purge && module load "$DCNET_MODULE" || {
            echo "activate.sh: could not load module $DCNET_MODULE" >&2
            return 1 2>/dev/null || exit 1
        }
        if [ -f "$DCNET_VENV/bin/activate" ]; then
            . "$DCNET_VENV/bin/activate"
            echo "activate.sh: DCNET_TORCH=module ($DCNET_MODULE + venv $DCNET_VENV)"
        else
            echo "activate.sh: DCNET_TORCH=module ($DCNET_MODULE)"
        fi
        ;;
    *)
        echo "activate.sh: unknown DCNET_TORCH='$DCNET_TORCH' (use 'pip' or 'module')" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
