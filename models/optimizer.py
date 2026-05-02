from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class IterativeActivityOptimizer(nn.Module):
    """
    Batch-friendly iterative optimizer for neuron activities.

    This keeps the main idea of the CPU reference implementation:
    optimize activity under lateral interaction with a non-negative
    shrinkage step, while avoiding CPU-specific control flow.
    """

    def __init__(
        self,
        max_iters: int = 50,
        lambda_: float = 0.1,
        gain_factor: float = 10.0,
        estimate_steps: int = 20,
        warmup_iters: int = 2,
        step_decay_interval: Optional[int] = None,
        step_decay_factor: float = 0.5,
    ):
        super().__init__()
        self.max_iters = int(max_iters)
        self.lambda_ = float(lambda_)
        self.gain_factor = float(gain_factor)
        self.estimate_steps = int(estimate_steps)
        self.warmup_iters = int(warmup_iters)
        self.step_decay_interval = step_decay_interval
        self.step_decay_factor = float(step_decay_factor)
        self.register_buffer("cached_max_eigenvalue", torch.tensor(1.0))

    @torch.no_grad()
    def _estimate_max_eigenvalue(self, matrix: torch.Tensor) -> torch.Tensor:
        n = matrix.shape[0]
        vec = torch.randn(n, device=matrix.device, dtype=matrix.dtype)
        vec = vec / vec.norm().clamp_min(1e-12)
        for _ in range(self.estimate_steps):
            vec = torch.matmul(matrix, vec)
            vec = vec / vec.norm().clamp_min(1e-12)
        eig = torch.dot(vec, torch.matmul(matrix, vec))
        return eig.clamp_min(1e-6)

    @torch.no_grad()
    def update_cached_gain(self, matrix: torch.Tensor) -> torch.Tensor:
        eig = self._estimate_max_eigenvalue(matrix)
        self.cached_max_eigenvalue.copy_(eig.to(self.cached_max_eigenvalue.device, dtype=self.cached_max_eigenvalue.dtype))
        return eig

    def forward(self, raw_activity: torch.Tensor, neuron_correlation_matrix: torch.Tensor) -> torch.Tensor:
        if raw_activity.dim() != 2:
            raise ValueError("raw_activity must be [batch, out_dim].")

        x = raw_activity
        corr = neuron_correlation_matrix

        max_eig = self.cached_max_eigenvalue.to(device=corr.device, dtype=corr.dtype).clamp_min(1e-6)
        step_size = 1.0 / (self.gain_factor * max_eig)

        y = x.clone()
        for iter_idx in range(self.max_iters):
            lam = 0.0 if iter_idx < self.warmup_iters else self.lambda_
            if self.step_decay_interval and iter_idx > 0 and iter_idx % self.step_decay_interval == 0:
                step_size = step_size * self.step_decay_factor

            update = x - torch.matmul(y, corr)
            y = y + step_size * update
            y = F.relu(y - lam * step_size)

        return y
