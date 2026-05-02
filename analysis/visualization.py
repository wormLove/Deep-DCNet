from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _per_patch_minmax(images: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = images.view(images.shape[0], -1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    denom = (maxs - mins).clamp_min(eps)
    return (images - mins) / denom


def save_patch_grid(
    column_tensor: torch.Tensor,
    save_path: Path,
    image_hw: Tuple[int, int] = (28, 28),
    grid_rows: int = 25,
    grid_cols: int = 40,
    per_patch_minmax: bool = True,
    pad_value: float = 0.0,
) -> None:
    save_path = Path(save_path)
    num_slots = grid_rows * grid_cols
    num_features, num_neurons = column_tensor.shape
    if num_features != image_hw[0] * image_hw[1]:
        raise ValueError("column_tensor does not match requested image_hw.")

    cols = column_tensor.detach().to(torch.float32).cpu().transpose(0, 1)
    cols = cols[:num_slots]
    if cols.shape[0] < num_slots:
        pad = torch.zeros(num_slots - cols.shape[0], cols.shape[1], dtype=cols.dtype)
        cols = torch.cat([cols, pad], dim=0)

    images = cols.view(num_slots, 1, image_hw[0], image_hw[1])
    if per_patch_minmax:
        images = _per_patch_minmax(images)

    grid = make_grid(images, nrow=grid_cols, padding=1, pad_value=pad_value)
    if grid.shape[0] == 1:
        grid_np = grid.squeeze(0).numpy()
        imshow_kwargs = {"cmap": "gray"}
    else:
        grid_np = grid.permute(1, 2, 0).numpy()
        imshow_kwargs = {}

    fig_w = max(10, grid_cols * 0.45)
    fig_h = max(8, grid_rows * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(grid_np, interpolation="nearest", **imshow_kwargs)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_heatmap(
    tensor_2d: torch.Tensor,
    save_path: Path,
    cmap: str = "viridis",
    percentile_clip: float = 99.0,
) -> None:
    save_path = Path(save_path)
    arr = tensor_2d.detach().to(torch.float32).cpu()
    if percentile_clip is not None:
        vmax = torch.quantile(arr.abs().flatten(), percentile_clip / 100.0).item()
        vmin = -vmax if arr.min().item() < 0 else 0.0
    else:
        vmin = None
        vmax = None

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(arr.numpy(), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_sorted_curve(
    vector: torch.Tensor,
    save_path: Path,
    title: str = "Sorted Values",
    ylabel: str = "value",
) -> None:
    save_path = Path(save_path)
    arr = vector.detach().to(torch.float32).cpu().flatten()
    sorted_arr, _ = torch.sort(arr, descending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sorted_arr.numpy(), linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("sorted neuron index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
