"""
Input perturbations for robustness testing (Prof. Yu's point #4).

Every function takes an image batch `x` of shape [B, C, H, W] with values in
[0, 1] (exactly what the dataset transforms in training/datasets.py
produce, BEFORE the model's own flatten transform) and returns a tensor of
the same shape, again in [0, 1]. Staying non-negative matters: the
discrimination layer is built on non-negative inputs and weights.

All randomness goes through an explicit torch.Generator so the *same*
corrupted test set can be shown to every model being compared - otherwise
a difference between two models could just be two different noise draws.

    from core.perturbations import PERTURBATIONS, make_generator
    fn = PERTURBATIONS["gaussian"]
    x_noisy = fn(x, 0.2, make_generator(seed=0, device=x.device))

The grid used by experiments/2026-09-18_point4_robustness.py lives in
DEFAULT_GRID below.
"""
from typing import Callable, Dict, List, Tuple

import torch


def make_generator(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def identity(x: torch.Tensor, level: float, gen: torch.Generator) -> torch.Tensor:
    return x


def gaussian_noise(x: torch.Tensor, sigma: float, gen: torch.Generator) -> torch.Tensor:
    """Additive N(0, sigma^2) noise per pixel, clamped back into [0, 1]."""
    if sigma <= 0:
        return x
    noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * float(sigma)
    return (x + noise).clamp(0.0, 1.0)


def salt_and_pepper(x: torch.Tensor, p: float, gen: torch.Generator) -> torch.Tensor:
    """Each pixel independently set to 1 ("salt") or 0 ("pepper") with total probability p."""
    if p <= 0:
        return x
    u = torch.rand(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    out = x.clone()
    out[u < p / 2] = 0.0
    out[(u >= p / 2) & (u < p)] = 1.0
    return out


def occlusion(x: torch.Tensor, size: float, gen: torch.Generator) -> torch.Tensor:
    """One random k x k square per image set to 0 (k = int(size)), position uniform over the image."""
    k = int(size)
    if k <= 0:
        return x
    b, c, h, w = x.shape
    k = min(k, h, w)
    out = x.clone()
    tops = torch.randint(0, h - k + 1, (b,), generator=gen, device=x.device)
    lefts = torch.randint(0, w - k + 1, (b,), generator=gen, device=x.device)
    for i in range(b):
        t, l = int(tops[i]), int(lefts[i])
        out[i, :, t:t + k, l:l + k] = 0.0
    return out


def shift(x: torch.Tensor, max_shift: float, gen: torch.Generator) -> torch.Tensor:
    """Translate each image by an integer offset drawn uniformly from [-d, d] in both axes, zero fill."""
    d = int(max_shift)
    if d <= 0:
        return x
    b, c, h, w = x.shape
    out = torch.zeros_like(x)
    dys = torch.randint(-d, d + 1, (b,), generator=gen, device=x.device)
    dxs = torch.randint(-d, d + 1, (b,), generator=gen, device=x.device)
    for i in range(b):
        dy, dx = int(dys[i]), int(dxs[i])
        src_y = slice(max(0, -dy), h - max(0, dy))
        src_x = slice(max(0, -dx), w - max(0, dx))
        dst_y = slice(max(0, dy), h - max(0, -dy))
        dst_x = slice(max(0, dx), w - max(0, -dx))
        out[i, :, dst_y, dst_x] = x[i, :, src_y, src_x]
    return out


PERTURBATIONS: Dict[str, Callable[[torch.Tensor, float, torch.Generator], torch.Tensor]] = {
    "clean": identity,
    "gaussian": gaussian_noise,
    "salt_pepper": salt_and_pepper,
    "occlusion": occlusion,
    "shift": shift,
}

# (perturbation name, level) pairs swept by the point #4 experiment.
# Levels: gaussian = sigma; salt_pepper = flip probability; occlusion = square
# side in pixels; shift = max translation in pixels.
DEFAULT_GRID: List[Tuple[str, float]] = [
    ("clean", 0.0),
    ("gaussian", 0.1), ("gaussian", 0.2), ("gaussian", 0.3), ("gaussian", 0.5),
    ("salt_pepper", 0.05), ("salt_pepper", 0.1), ("salt_pepper", 0.2),
    ("occlusion", 6), ("occlusion", 10), ("occlusion", 14),
    ("shift", 2), ("shift", 4),
]
