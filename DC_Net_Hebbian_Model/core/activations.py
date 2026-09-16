import torch
import torch.nn as nn

class LayerThresholding(nn.Module):
    """Stateless: non-neg -> std-threshold -> hard sparsity cap (top-k%)."""
    def __init__(self, threshold_factor: float = 1.0, sparsity: float = 0.05, soft: bool = False):
        super().__init__()
        assert 0 < sparsity <= 1.0
        self.threshold_factor = float(threshold_factor)
        self.sparsity = float(sparsity)
        self.soft = bool(soft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) non-neg
        x = torch.clamp(x, min=0.0)

        # 2) threshold by std (per-sample, along last dim)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        thr = self.threshold_factor * std

        if self.soft:
            y = torch.clamp(x - thr, min=0.0)
        else:
            y = x * (x > thr)

        # 3) enforce sparsity cap per sample (keep top-k%)
        N = y.size(-1)
        k = max(1, int(round(self.sparsity * N)))

        _, topk_idx = torch.topk(y, k, dim=-1)
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        y = y * mask

        return y




