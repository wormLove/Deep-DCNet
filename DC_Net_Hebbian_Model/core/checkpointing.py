"""
Checkpointing for individual reusable layers (not whole models).

Why this exists separately from RESULT/<experiment>/<timestamp>/checkpoints/:
that folder is a per-run audit trail (best.pth / final.pth for one specific
experiment, reproducing one specific result). This module is for the
opposite case: a layer that trained well and you now want to plug into
OTHER experiments (a parallel branch, a deeper stack, a fuse-expand block).
Those live in a separate top-level checkpoints/ folder, indexed by name in
checkpoints/manifest.json, so they can be found and loaded without knowing
which experiment or timestamp originally produced them.

Usage:
    from core.checkpointing import save_layer, load_layer

    # after training + organizing a DiscriminationModule inside some model:
    save_layer(
        model.discrimination_layer,
        name="single_layer_mnist_h300",
        checkpoints_dir="checkpoints",
        meta={
            "architecture": "single_layer",
            "in_dim": 784, "out_dim": 300,
            "activity_optimizer": "least_squares",
            "source_experiment": "2026-09-10_single_layer_baseline",
            "dataset": "mnist",
            "notes": "trained to stable, test acc 95.2%",
        },
    )

    # later, in a different experiment file:
    dl = load_layer("single_layer_mnist_h300", checkpoints_dir="checkpoints")
    # dl is a ready-to-use DiscriminationModule with frozen weights
"""
import json
import os
import subprocess
from datetime import datetime

import torch

from modules.discrimination import DiscriminationModule


def _git_commit(repo_dir: str = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _manifest_path(checkpoints_dir: str) -> str:
    return os.path.join(checkpoints_dir, "manifest.json")


def _load_manifest(checkpoints_dir: str) -> dict:
    path = _manifest_path(checkpoints_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _write_manifest(checkpoints_dir: str, manifest: dict) -> None:
    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(_manifest_path(checkpoints_dir), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def save_layer(
    module: DiscriminationModule,
    name: str,
    checkpoints_dir: str = "checkpoints",
    meta: dict | None = None,
    include_training_state: bool = True,
    overwrite: bool = False,
) -> str:
    """
    Save a DiscriminationModule's weights under a stable, human-chosen name.

    Returns the path written. Raises if `name` already exists and
    overwrite=False, since silently clobbering a checkpoint another
    experiment depends on is the main failure mode this is meant to avoid.
    """
    os.makedirs(checkpoints_dir, exist_ok=True)
    weights_path = os.path.join(checkpoints_dir, f"{name}.pt")

    manifest = _load_manifest(checkpoints_dir)
    if name in manifest and not overwrite:
        raise FileExistsError(
            f"Checkpoint '{name}' already exists in {checkpoints_dir}/manifest.json. "
            "Pass overwrite=True if you really mean to replace it (this will break "
            "any experiment that loads it expecting the old weights)."
        )

    payload = {
        "state": module.save_state(include_training_state=include_training_state),
        "in_dim": module.neuron_weights.shape[0],
        "out_dim": module.neuron_weights.shape[1],
    }
    torch.save(payload, weights_path)

    entry = dict(meta or {})
    entry.update({
        "name": name,
        "path": weights_path,
        "in_dim": payload["in_dim"],
        "out_dim": payload["out_dim"],
        "saved_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "git_commit": _git_commit(),
    })
    manifest[name] = entry
    _write_manifest(checkpoints_dir, manifest)

    return weights_path


def load_layer(
    name: str,
    checkpoints_dir: str = "checkpoints",
    freeze: bool = True,
    activity_optimizer: str | None = None,
) -> DiscriminationModule:
    """
    Reconstruct a DiscriminationModule from a saved checkpoint, ready to
    plug into a new architecture. Frozen (requires_grad=False) by default,
    matching how these layers are used everywhere else in DCNet (no
    gradients through the discrimination layer).
    """
    manifest = _load_manifest(checkpoints_dir)
    if name not in manifest:
        raise KeyError(
            f"No checkpoint named '{name}' in {checkpoints_dir}/manifest.json. "
            f"Available: {sorted(manifest)}"
        )
    entry = manifest[name]
    payload = torch.load(entry["path"], weights_only=False)

    optimizer_name = activity_optimizer or entry.get("activity_optimizer", "least_squares")
    module = DiscriminationModule(
        payload["in_dim"], payload["out_dim"], activity_optimizer=optimizer_name
    )
    module.load_state(payload["state"], include_training_state=True)

    if freeze:
        for p in module.parameters():
            p.requires_grad = False

    return module


def list_checkpoints(checkpoints_dir: str = "checkpoints") -> dict:
    """Return the manifest dict (name -> metadata) for quick inspection."""
    return _load_manifest(checkpoints_dir)
