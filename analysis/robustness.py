"""
Robustness evaluation (Prof. Yu's point #4): score an already-trained
model on a perturbed copy of its test set, for every (perturbation, level)
in a grid, and write one tidy TSV.

Evaluation only - nothing is trained or organized here. The model is put
in eval() mode, so the discrimination layer's forward() does not touch its
Hebbian potentials or activity stats.
"""
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

from core.perturbations import PERTURBATIONS, make_generator


@torch.no_grad()
def evaluate_perturbed(model, loader, device, perturb_fn: Callable, level: float, seed: int) -> float:
    """Accuracy (%) of `model` on `loader` after applying perturb_fn(x, level) to every batch."""
    was_training = model.training
    model.eval()
    gen = make_generator(seed, device)
    total = 0
    correct = 0
    for x, y in loader:
        x = perturb_fn(x.to(device), level, gen)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    if was_training:
        model.train()
    return 100.0 * correct / max(total, 1)


def run_robustness_sweep(
    models: Dict[str, torch.nn.Module],
    loader,
    device,
    grid: Iterable[Tuple[str, float]],
    out_dir,
    seed: int = 0,
    dataset_name: str = "",
    log: Optional[Callable[[str], None]] = print,
) -> List[Dict]:
    """
    models: {label: trained model}. Every model sees the identical corrupted
    test set (same seed per grid cell). Writes <out_dir>/robustness.tsv with
    columns dataset, model, perturbation, level, acc, drop_vs_clean and
    returns the rows.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = list(grid)
    rows: List[Dict] = []
    clean_acc: Dict[str, float] = {}

    tsv_path = out_dir / "robustness.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("dataset\tmodel\tperturbation\tlevel\tacc\tdrop_vs_clean\n")

    for cell_idx, (name, level) in enumerate(tqdm(grid, desc="Robustness grid", leave=False)):
        fn = PERTURBATIONS[name]
        for label, model in models.items():
            acc = evaluate_perturbed(model, loader, device, fn, level, seed=seed + cell_idx)
            if name == "clean":
                clean_acc[label] = acc
            drop = clean_acc.get(label, acc) - acc
            row = dict(dataset=dataset_name, model=label, perturbation=name, level=level, acc=acc, drop_vs_clean=drop)
            rows.append(row)
            with open(tsv_path, "a", encoding="utf-8") as f:
                f.write(f"{dataset_name}\t{label}\t{name}\t{level}\t{acc:.4f}\t{drop:.4f}\n")
            if log is not None:
                log(f"[{name:<11} {str(level):>5}] {label:<28} acc={acc:6.2f}%  drop={drop:6.2f}")

    return rows


def format_table(rows: List[Dict]) -> str:
    """Human-readable pivot: one row per (perturbation, level), one column per model."""
    labels = list(dict.fromkeys(r["model"] for r in rows))
    cells = list(dict.fromkeys((r["perturbation"], r["level"]) for r in rows))
    lookup = {(r["model"], r["perturbation"], r["level"]): r["acc"] for r in rows}
    w = max(12, max(len(l) for l in labels) + 2)
    lines = [f"{'perturbation':<14}{'level':>6}  " + "".join(f"{l:>{w}}" for l in labels)]
    for name, level in cells:
        lines.append(f"{name:<14}{str(level):>6}  " + "".join(f"{lookup[(l, name, level)]:>{w}.2f}" for l in labels))
    return "\n".join(lines)
