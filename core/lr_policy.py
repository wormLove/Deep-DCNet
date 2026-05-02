from typing import Optional

import torch
from torch import nn


class ProtectWithRecoveryLR(nn.Module):
    """
    Per-neuron learning-rate policy:
    - protect consolidated neurons with low lr
    - recover inactive neurons with gated nonlinear boost
    """

    def __init__(
        self,
        lr_init: float = 0.99,
        min_lr: float = 1e-3,
        max_lr: float = 0.99,
        recover_step: float = 0.05,
        recover_alpha: float = 0.2,
        strength_gate_k: float = 1.0,
        lr_gate_n: float = 0.5,
    ):
        super().__init__()
        self.lr_init = float(lr_init)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.recover_step = float(recover_step)
        self.recover_alpha = float(recover_alpha)
        self.strength_gate_k = float(strength_gate_k)
        self.lr_gate_n = float(lr_gate_n)

    @torch.no_grad()
    def compute(
        self,
        strength_scores: torch.Tensor,
        inactive_mask: Optional[torch.Tensor] = None,
        cycles_since_active: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        strength = strength_scores.clamp(0.0, 1.0)
        lr = self.lr_init * (1.0 - strength)
        lr = lr.clamp(min=self.min_lr, max=self.max_lr)

        if inactive_mask is None:
            return lr

        if cycles_since_active is None:
            cycles_since_active = torch.zeros_like(lr, dtype=torch.long)

        cycles = cycles_since_active.to(lr.dtype).clamp_min(0.0)
        base_recovery = self.recover_step * (1.0 - torch.exp(-self.recover_alpha * cycles))
        strength_gate = torch.pow((1.0 - strength).clamp(0.0, 1.0), self.strength_gate_k)

        lr_span = max(self.max_lr - self.min_lr, 1e-12)
        lr_gate_input = ((lr - self.min_lr) / lr_span).clamp(0.0, 1.0)
        lr_gate = torch.pow(lr_gate_input, self.lr_gate_n)

        boost = base_recovery * strength_gate * lr_gate
        updated_lr = torch.clamp(lr + boost, max=self.max_lr)
        return torch.where(inactive_mask.to(torch.bool), updated_lr, lr)
