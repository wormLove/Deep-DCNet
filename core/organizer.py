import torch
from torch import nn


class DiscriminationOrganizer(nn.Module):
    """
    Minimal organizer for Hebbian / Anti-Hebbian weight accumulation and update.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        default_lr: float = 0.99,
        beta: float = 0.99,
        use_cooldown: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.beta = float(beta)
        self.use_cooldown = bool(use_cooldown)
        self.default_lr = float(default_lr)

        self.register_buffer("potential_hebb", torch.zeros(in_dim, out_dim))
        self.register_buffer("potential_antihebb", torch.zeros(out_dim, out_dim))
        self.register_buffer("cooldown_state", torch.zeros(out_dim, dtype=torch.bool))
        self.register_buffer("lr_vec", torch.full((out_dim,), self.default_lr))
        self.register_buffer("beta_accum", torch.tensor(float(beta)))
        self.register_buffer("zero_norm_rows_total", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def set_learning_rates(self, lr_vec: torch.Tensor) -> None:
        if lr_vec.dim() != 1 or lr_vec.shape[0] != self.out_dim:
            raise ValueError("lr_vec must be [out_dim].")
        self.lr_vec.copy_(lr_vec.to(self.lr_vec.device, dtype=self.lr_vec.dtype))

    @torch.no_grad()
    def step(self, input_signal: torch.Tensor, activity_signal: torch.Tensor) -> None:
        activity = activity_signal
        if self.use_cooldown:
            activity = self._apply_cooldown(activity)
        activity = self._normalize(activity)

        batch_size = max(int(input_signal.shape[0]), 1)
        hebb_update = (1.0 - self.beta) * (self._potential(input_signal, activity) / batch_size)
        antihebb_update = (1.0 - self.beta) * (self._potential(activity, activity) / batch_size)

        self.potential_hebb.mul_(self.beta).add_(hebb_update)
        self.potential_antihebb.mul_(self.beta).add_(antihebb_update)

    @torch.no_grad()
    def organize(self, weights: torch.Tensor) -> torch.Tensor:
        correction_factor = 1.0 / (1.0 - self.beta_accum.clamp_max(1.0 - 1e-12))
        potential_diff = self.potential_hebb - torch.matmul(weights, self.potential_antihebb)
        lr_col = self.lr_vec.unsqueeze(0)
        updated_weights = (1.0 - lr_col) * weights + (correction_factor * lr_col) * potential_diff
        self.beta_accum.mul_(self.beta)
        return updated_weights

    @staticmethod
    def _potential(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        return torch.matmul(input_tensor.transpose(0, 1), target_tensor)

    def _apply_cooldown(self, activity_signal: torch.Tensor) -> torch.Tensor:
        masked = (~self.cooldown_state).unsqueeze(0).to(activity_signal.dtype) * activity_signal
        self.cooldown_state.copy_(masked.any(dim=0))
        return masked

    def _normalize(self, activity_signal: torch.Tensor) -> torch.Tensor:
        if activity_signal.dim() != 2:
            raise ValueError("activity_signal must be [batch, out_dim].")

        norm = torch.norm(activity_signal, dim=1, keepdim=True)
        zero_rows = norm.squeeze(1) == 0
        if zero_rows.any():
            self.zero_norm_rows_total.add_(zero_rows.sum().to(self.zero_norm_rows_total.dtype))
        norm = norm.clamp_min(1e-12)
        return activity_signal / norm
