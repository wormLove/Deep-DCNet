"""
Run configs: every training run writes a `run_config.json` next to its logs
in RESULT/<run_name>/<timestamp>/, recording exactly what was built
(architecture, dataset, head, layer sizes, discrimination_config). Anything
that needs the trained model back later - the point #4 robustness sweep
today, whatever comes next - rebuilds it from that file plus the run's
checkpoints/final.pth, instead of re-typing the config by hand.

    from training.run_config import load_trained_run
    model, cfg, spec = load_trained_run("single_layer_baseline", result_root="RESULT", device=device)
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from architectures.single_layer import BiologicalClassifier
from architectures.stacked import StackedBiologicalClassifier
from training.datasets import get_spec

RUN_CONFIG_FILENAME = "run_config.json"


def write_run_config(run_dir, cfg: Dict[str, Any]) -> Path:
    path = Path(run_dir) / RUN_CONFIG_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)
    return path


def find_run_dir(run_name: str, result_root, timestamp: Optional[str] = None) -> Path:
    """
    RESULT/<run_name>/<timestamp>/ - the given timestamp, or the newest
    folder that has a run_config.json AND a checkpoints/final.pth.
    """
    base = Path(result_root) / run_name
    if not base.is_dir():
        raise FileNotFoundError(f"No results folder for run '{run_name}' under {Path(result_root).resolve()}")
    if timestamp is not None:
        run_dir = base / timestamp
        if not run_dir.is_dir():
            raise FileNotFoundError(f"{run_dir} does not exist")
        return run_dir
    candidates = sorted(
        d for d in base.iterdir()
        if d.is_dir() and (d / RUN_CONFIG_FILENAME).exists() and (d / "checkpoints" / "final.pth").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No finished run (run_config.json + checkpoints/final.pth) under {base}. "
            "Either the run has not finished or it was trained before run_config.json existed."
        )
    return candidates[-1]


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1)


def build_model_from_config(cfg: Dict[str, Any]):
    """Rebuild an untrained model with exactly the shape recorded in cfg."""
    from training.train_classifier import resolve_head  # local import: avoids a circular import at module load

    head_cls, head_kwargs = resolve_head(cfg["head"], tuple(cfg.get("head_hidden_dims", (256,))), cfg.get("head_dropout", 0.2))
    arch = cfg["arch"]
    if arch == "single_layer":
        return BiologicalClassifier(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            output_dim=cfg["num_classes"],
            transform=_flatten,
            discrimination_config=cfg["discrimination_config"],
            integration_dim=cfg.get("integration_dim"),
            integration_activation=cfg.get("integration_activation", "relu"),
            head_cls=head_cls,
            head_kwargs=head_kwargs,
        )
    if arch == "stacked":
        return StackedBiologicalClassifier(
            layer_dims=list(cfg["layer_dims"]),
            transform=_flatten,
            discrimination_config=cfg["discrimination_config"],
            head_cls=head_cls,
            head_kwargs=head_kwargs,
        )
    raise ValueError(f"Unknown arch '{arch}' in run config")


def load_trained_run(run_name: str, result_root, device, timestamp: Optional[str] = None) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Returns (model in eval mode on `device`, cfg dict, DatasetSpec) for a
    finished run. The model's weights come from that run's
    checkpoints/final.pth (whole model: discrimination layer(s) + head).
    """
    run_dir = find_run_dir(run_name, result_root, timestamp)
    with open(run_dir / RUN_CONFIG_FILENAME, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_run_dir"] = str(run_dir)
    model = build_model_from_config(cfg)
    model.load_model(str(run_dir / "checkpoints" / "final.pth"), map_location=device)
    model.to(device)
    model.eval()
    return model, cfg, get_spec(cfg["dataset"])
