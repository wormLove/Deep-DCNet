#!/bin/bash
# =============================================================================
# Pioneer environment - meant to be SOURCED, not run:
#   source hpc/pioneer/activate.sh
#
# Loads the cluster's PyTorch module, then activates the small venv that
# hpc/pioneer/setup_env.sh creates (only if it exists - it is only needed
# when the module is missing tqdm/matplotlib). Used interactively and by
# hpc/pioneer/run_experiment.slurm.
#
# No `set -e` here on purpose: this file runs inside the caller's shell.
# =============================================================================

DCNET_MODULE="PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1"
DCNET_VENV="${DCNET_VENV:-$HOME/dcnet_pioneer_venv}"

# `module` is normally already defined; fall back to the Lmod init script
# for shells where it is not (some non-interactive shells).
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
        [ -f "$f" ] && . "$f" && break
    done
fi

module purge && module load "$DCNET_MODULE" || {
    echo "activate.sh: could not load module $DCNET_MODULE" >&2
    return 1 2>/dev/null || exit 1
}

if [ -f "$DCNET_VENV/bin/activate" ]; then
    . "$DCNET_VENV/bin/activate"
fi
