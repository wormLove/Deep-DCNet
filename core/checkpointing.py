"""
Checkpointing for individual reusable discrimination layers (not whole
models, and not the same thing as AnalysisEngine's per-run
RESULT/<run_name>/<timestamp>/checkpoints/ audit trail).

Why this exists separately from that per-run audit trail: that folder
records one specific run's progress (model_step_*.pth / final.pth,
reproducing one specific result). This module is for the opposite case: a
discrimination layer that trained well and you now want to reuse as a
frozen feature extractor in a DIFFERENT experiment (e.g. comparing
classifier heads without re-running the multi-hour Hebbian training every
time - see experiments/2026-09-17_point5_traditional_head.py). Named
checkpoints live in a separate top-level checkpoints/ folder, indexed by
name in checkpoints/manifest.json, independent of which run or timestamp
originally produced them.

Usage:
    from core.checkpointing import save_layer, load_layer

    # after training a BiologicalClassifier:
    save_layer(model, name="single_layer_h2000_readout", checkpoints_dir="checkpoints",
               meta={"notes": "trained to stable, test acc ~XX%"})

    # later, in a different experiment file:
    dl = load_layer("single_layer_h2000_readout", checkpoints_dir="checkpoints")
    # dl is a ready-to-use, frozen DiscriminationLayer

Caveat: this saves/restores the discrimination layer's state_dict() (weights,
correlation matrix, and any buffers PyTorch tracks on it) plus the
discrimination_config needed to reconstruct it. It is meant for reuse as a
FROZEN feature extractor (call .eval() before use, same as
load_layer(freeze=True) already does for requires_grad) - it has not been
verified to round-trip every last piece of internal optimizer/stats state
needed to correctly RESUME Hebbian training on a loaded layer, since that
depends on whether core/organizer.py, core/stats.py and core/lr_policy.py's
helper classes register their internal tensors as proper nn.Module
buffers. If you need to resume training (not just read out frozen
features), verify that first rather than assuming it works.
"""
import json
import os
import subprocess
from datetime import datetime
from typing import Optional

import torch

from modules.discrimination import DiscriminationLayer


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
    classifier,
    name: str,
    checkpoints_dir: str = "checkpoints",
    meta: Optional[dict] = None,
    overwrite: bool = False,
) -> str:
    """
    Save a BiologicalClassifier's trained discrimination_layer under a
    stable, human-chosen name, plus the discrimination_config used to
    build it (DiscriminationLayer takes ~20 hyperparameters, so the config
    travels with the checkpoint rather than requiring load_layer's caller
    to remember/repeat it exactly).

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

    dl = classifier.discrimination_layer
    payload = {
        "state_dict": dl.state_dict(),
        "in_dim": dl.in_dim,
        "out_dim": dl.out_dim,
        "discrimination_config": getattr(classifier, "discrimination_config", {}) or {},
    }
    torch.save(payload, weights_path)

    entry = dict(meta or {})
    entry.update({
        "name": name,
        "path": weights_path,
        "in_dim": dl.in_dim,
        "out_dim": dl.out_dim,
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
) -> DiscriminationLayer:
    """
    Reconstruct a DiscriminationLayer from a saved checkpoint, ready to
    plug into a new BiologicalClassifier as its .discrimination_layer.
    Frozen (requires_grad=False) by default. Call .eval() on it before use
    so forward() skips the organizer.step() Hebbian update entirely - this
    is meant to be reused as a fixed feature extractor, not retrained.
    """
    manifest = _load_manifest(checkpoints_dir)
    if name not in manifest:
        raise KeyError(
            f"No checkpoint named '{name}' in {checkpoints_dir}/manifest.json. "
            f"Available: {sorted(manifest)}"
        )
    entry = manifest[name]
    payload = torch.load(entry["path"], weights_only=False)

    module = DiscriminationLayer(
        in_dim=payload["in_dim"],
        out_dim=payload["out_dim"],
        **(payload.get("discrimination_config") or {}),
    )
    module.load_state_dict(payload["state_dict"])

    # Defensive: re-derive the activity optimizer's cached gain from the
    # loaded correlation matrix, in case that cache isn't part of
    # state_dict() (depends on IterativeActivityOptimizer's internals).
    # Cheap, and guarantees forward() is correct regardless.
    module.activity_optimizer.update_cached_gain(module.neuron_correlation_matrix)

    if freeze:
        for p in module.parameters():
            p.requires_grad = False

    return module


def list_checkpoints(checkpoints_dir: str = "checkpoints") -> dict:
    """Return the manifest dict (name -> metadata) for quick inspection."""
    return _load_manifest(checkpoints_dir)


def save_stack(
    classifier,
    name: str,
    checkpoints_dir: str = "checkpoints",
    meta: Optional[dict] = None,
    overwrite: bool = False,
) -> str:
    """
    Multi-layer counterpart to save_layer(), for a
    architectures.stacked.StackedBiologicalClassifier. Saves ALL of its
    discrimination_layers (in order) plus layer_dims and the shared
    discrimination_config as one named checkpoint bundle, so a stacked
    model can be reused as a frozen multi-layer feature extractor without
    retraining, same motivation as save_layer() for the single-layer case.

    Stored in the same checkpoints/manifest.json as single-layer
    checkpoints, tagged "kind": "stack" so load_stack()/load_layer() can
    each reject a name that was saved by the other one instead of silently
    misconstructing a model.

    Raises if `name` already exists and overwrite=False.
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

    dls = classifier.discrimination_layers
    payload = {
        "kind": "stack",
        "layer_dims": list(classifier.layer_dims),
        "state_dicts": [dl.state_dict() for dl in dls],
        "discrimination_config": getattr(classifier, "discrimination_config", {}) or {},
    }
    torch.save(payload, weights_path)

    entry = dict(meta or {})
    entry.update({
        "kind": "stack",
        "name": name,
        "path": weights_path,
        "layer_dims": list(classifier.layer_dims),
        "num_dl_layers": len(dls),
        "saved_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "git_commit": _git_commit(),
    })
    manifest[name] = entry
    _write_manifest(checkpoints_dir, manifest)

    return weights_path


def load_stack(
    name: str,
    checkpoints_dir: str = "checkpoints",
    freeze: bool = True,
):
    """
    Reconstruct the ordered list of frozen DiscriminationLayers from a
    checkpoint saved by save_stack(), ready to assign to a
    StackedBiologicalClassifier.discrimination_layers (wrap in
    nn.ModuleList(...)). Raises if `name` refers to a single-layer
    checkpoint saved by save_layer() instead - use load_layer() for those.
    """
    manifest = _load_manifest(checkpoints_dir)
    if name not in manifest:
        raise KeyError(
            f"No checkpoint named '{name}' in {checkpoints_dir}/manifest.json. "
            f"Available: {sorted(manifest)}"
        )
    entry = manifest[name]
    if entry.get("kind") != "stack":
        raise ValueError(
            f"Checkpoint '{name}' is not a stack checkpoint (kind={entry.get('kind', 'layer')!r}). "
            "Use load_layer() for single-layer checkpoints."
        )
    payload = torch.load(entry["path"], weights_only=False)

    layer_dims = payload["layer_dims"]
    config = payload.get("discrimination_config") or {}
    layers = []
    for i, sd in enumerate(payload["state_dicts"]):
        in_d, out_d = layer_dims[i], layer_dims[i + 1]
        module = DiscriminationLayer(in_dim=in_d, out_dim=out_d, **config)
        module.load_state_dict(sd)
        module.activity_optimizer.update_cached_gain(module.neuron_correlation_matrix)
        if freeze:
            for p in module.parameters():
                p.requires_grad = False
        layers.append(module)
    return layers
