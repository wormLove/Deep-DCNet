from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

ORGANIZE_STEPS = ("direction", "correction_factor")


class DiscriminationOrganizer(nn.Module):
    """
    Minimal organizer for Hebbian / Anti-Hebbian weight accumulation and update.

    ``organize_step`` selects how far organize() moves each neuron along its residual:
    ``"direction"`` uses the residual's direction only (step set by the learning rate);
    ``"correction_factor"`` is the previous rule, the raw residual scaled by
    ``1 / (1 - beta_accum)``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        default_lr: float = 0.99,
        beta: float = 0.99,
        correction_beta: Optional[float] = None,
        organize_step: str = "direction",
        weight_norm_p: int = 2,
        use_cooldown: bool = False,
    ):
        super().__init__()
        if organize_step not in ORGANIZE_STEPS:
            raise ValueError(f"organize_step must be one of {ORGANIZE_STEPS}.")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        # Decay of the potential EMA per training sample, so its memory is counted in
        # data seen, like the organize interval, and does not depend on the batch size.
        self.beta = float(beta)
        self.organize_step = str(organize_step)
        # The norm the neuron weights are kept at; the "direction" step measures the
        # update direction in the same norm so the two mix as unit vectors.
        self.weight_norm_p = int(weight_norm_p)
        # Used by organize_step="correction_factor" only. correction_factor runs on its
        # own clock: beta_accum decays once per organize. It shared beta's value while
        # beta was applied per batch; keeping it separate leaves that schedule unchanged
        # now that beta is per sample.
        self.correction_beta = float(beta if correction_beta is None else correction_beta)
        self.use_cooldown = bool(use_cooldown)
        self.default_lr = float(default_lr)

        self.register_buffer("potential_hebb", torch.zeros(in_dim, out_dim))
        self.register_buffer("potential_antihebb", torch.zeros(out_dim, out_dim))
        self.register_buffer("cooldown_state", torch.zeros(out_dim, dtype=torch.bool))
        self.register_buffer("lr_vec", torch.full((out_dim,), self.default_lr))
        self.register_buffer("beta_accum", torch.tensor(self.correction_beta))
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
        # One decay per sample. The batch decays the potential by beta**batch_size and
        # carries the total weight its samples would have had one at a time, shared
        # equally because they were presented together.
        batch_beta = self.beta ** batch_size
        hebb_update = (1.0 - batch_beta) * (self._potential(input_signal, activity) / batch_size)
        antihebb_update = (1.0 - batch_beta) * (self._potential(activity, activity) / batch_size)

        self.potential_hebb.mul_(batch_beta).add_(hebb_update)
        self.potential_antihebb.mul_(batch_beta).add_(antihebb_update)

    @torch.no_grad()
    def organize(self, weights: torch.Tensor) -> torch.Tensor:
        potential_diff = self.potential_hebb - torch.matmul(weights, self.potential_antihebb)
        lr_col = self.lr_vec.unsqueeze(0)
        if self.organize_step == "direction":
            # Only the direction of the residual is used. Its length depends on input
            # scale, beta, the interval and how recently the neuron fired, so it would
            # act as a global, history-dependent step size; how far a neuron moves is
            # the learning rate's alone. A neuron with no potential has no direction and
            # keeps its weights. The fixed point (residual parallel to the weights) is
            # the same as with the unnormalized residual.
            direction = F.normalize(potential_diff, p=self.weight_norm_p, dim=0)
            return (1.0 - lr_col) * weights + lr_col * direction

        correction_factor = 1.0 / (1.0 - self.beta_accum.clamp_max(1.0 - 1e-12))
        updated_weights = (1.0 - lr_col) * weights + (correction_factor * lr_col) * potential_diff
        self.beta_accum.mul_(self.correction_beta)
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
