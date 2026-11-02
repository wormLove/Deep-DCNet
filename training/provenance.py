"""
Run provenance: seeding and the "where/what/when" fields recorded in every
run's run_config.json, so a result folder says on its own which code,
machine and command produced it.

    from training.provenance import set_seed, run_provenance
    set_seed(33)                       # first thing in a run
    cfg.update(run_provenance(device)) # git_commit, git_dirty, env, argv, created_at
"""
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    """Seed torch (CPU + every CUDA device), python `random` and numpy (if installed)."""
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def now_iso() -> str:
    """Local time with UTC offset, e.g. 2026-11-02T09:15:03-06:00."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git(args, cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(repo_root: Path = REPO_ROOT) -> Tuple[str, Any]:
    """
    (short commit hash, dirty?) for the repo this code lives in. Outside a
    git repo (e.g. the hand-uploaded Markov copy) or without git installed
    both come back as "unknown" instead of raising.
    """
    commit = _git(["rev-parse", "--short", "HEAD"], repo_root)
    if not commit:
        return "unknown", "unknown"
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo_root)
    return commit, ("unknown" if status is None else bool(status))


def env_info(device) -> Dict[str, Any]:
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = None
    device = torch.device(device)
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device.index if device.index is not None else torch.cuda.current_device())
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
    }


def run_provenance(device) -> Dict[str, Any]:
    """The provenance block merged into run_config.json."""
    commit, dirty = git_info()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "env": env_info(device),
        "argv": [sys.executable, *sys.argv],
        "created_at": now_iso(),
    }


def peak_gpu_mem_mb(device) -> Optional[float]:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)


def reset_peak_gpu_mem(device) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
