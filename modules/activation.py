import torch
from torch import nn


class LayerThresholding(nn.Module):
    """
    Minimal sparse activation head:
    1. clamp to non-negative
    2. threshold by per-sample std
    3. keep only top-k% activations
    """

    def __init__(self, threshold_factor: float = 1.0, sparsity: float = 0.05, soft: bool = False):
        super().__init__()
        if not (0.0 < sparsity <= 1.0):
            raise ValueError("sparsity must be in (0, 1].")
        self.threshold_factor = float(threshold_factor)
        self.sparsity = float(sparsity)
        self.soft = bool(soft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=0.0)

        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        thr = self.threshold_factor * std

        if self.soft:
            y = torch.clamp(x - thr, min=0.0)
        else:
            y = x * (x > thr)

        if not (y > 0).any():
            return y

        num_neurons = y.size(-1)
        k = max(1, int(round(self.sparsity * num_neurons)))
        _, topk_idx = torch.topk(y, k, dim=-1)
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        return y * mask
