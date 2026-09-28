from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class IterativeActivityOptimizer(nn.Module):
    """Optimize non-negative neuron activity under lateral interaction.

    The optimizer uses the exact lateral-matrix scale, a canonical g=1 step,
    ReLU projection, and the variance turning point plus an activity-support
    constraint as its stopping mechanism.
    """

    def __init__(
        self,
        max_iters: int = 1000,
        variance_stop_window: int = 20,
        variance_stop_nonzero_ratio: float = 0.08,
        variance_stop_initial_nonzero_ratio: Optional[float] = 0.05,
        variance_stop_initial_max_iters: int = 20000,
    ):
        super().__init__()
        self.max_iters = int(max_iters)
        self.variance_stop_window = int(variance_stop_window)
        self.variance_stop_nonzero_ratio = float(variance_stop_nonzero_ratio)
        self.variance_stop_initial_nonzero_ratio = float(
            variance_stop_nonzero_ratio
            if variance_stop_initial_nonzero_ratio is None
            else variance_stop_initial_nonzero_ratio
        )
        self.variance_stop_initial_max_iters = int(variance_stop_initial_max_iters)
        self._variance_stop_initial_phase = True

        if self.max_iters < 1 or self.variance_stop_initial_max_iters < 1:
            raise ValueError("Iteration limits must be positive.")
        if self.variance_stop_window < 2:
            raise ValueError("variance_stop_window must be at least 2.")
        if not 0.0 < self.variance_stop_nonzero_ratio <= 1.0:
            raise ValueError("variance_stop_nonzero_ratio must be inside (0, 1].")
        if not 0.0 < self.variance_stop_initial_nonzero_ratio <= 1.0:
            raise ValueError(
                "variance_stop_initial_nonzero_ratio must be inside (0, 1]."
            )

        self.last_completed_iters = 0
        self.last_variance_stop_triggered = False
        self.last_variance_stop_resolved_count = 0
        self.last_variance_turn_count = 0
        self.last_variance_stop_completed_iters = torch.empty(0, dtype=torch.long)
        self.register_buffer("cached_max_eigenvalue", torch.tensor(1.0))

    def set_variance_stop_initial_phase(self, enabled: bool) -> None:
        self._variance_stop_initial_phase = bool(enabled)

    def _iteration_limit(self) -> int:
        if self._variance_stop_initial_phase:
            return self.variance_stop_initial_max_iters
        return self.max_iters

    def _variance_nonzero_ratio_limit(self) -> float:
        if self._variance_stop_initial_phase:
            return self.variance_stop_initial_nonzero_ratio
        return self.variance_stop_nonzero_ratio

    @staticmethod
    def _linear_slopes(values: torch.Tensor) -> torch.Tensor:
        x = torch.arange(values.shape[0], device=values.device, dtype=values.dtype)
        x = x - x.mean()
        return (values * x[:, None]).sum(dim=0) / x.square().sum().clamp_min(1e-12)

    def _variance_stop_update(
        self,
        y: torch.Tensor,
        variance_history: deque[torch.Tensor],
        turn_seen: torch.Tensor,
        finished: torch.Tensor,
        completed_per_case: torch.Tensor,
        completed_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(turn_seen.all().item()):
            variance_history.append(y.var(dim=1, unbiased=False).detach())
            if len(variance_history) == 2 * self.variance_stop_window:
                history = torch.stack(list(variance_history), dim=0)
                previous = history[: self.variance_stop_window]
                current = history[self.variance_stop_window :]
                turn_seen = turn_seen | (
                    (self._linear_slopes(previous) < 0.0)
                    & (self._linear_slopes(current) > 0.0)
                )

        nonzero_ratio = (y > 0.0).to(y.dtype).mean(dim=1)
        newly_finished = (
            (~finished)
            & turn_seen
            & (nonzero_ratio <= self._variance_nonzero_ratio_limit())
        )
        completed_per_case = torch.where(
            newly_finished,
            torch.full_like(completed_per_case, completed_iters),
            completed_per_case,
        )
        return turn_seen, finished | newly_finished, completed_per_case

    @torch.no_grad()
    def _compute_max_eigenvalue(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square.")

        # MPS does not currently implement eigvalsh. This calculation runs only
        # after context organization, so a CPU fallback has negligible cost.
        eig_matrix = matrix.cpu() if matrix.device.type == "mps" else matrix
        return torch.linalg.eigvalsh(eig_matrix)[-1].clamp_min(1e-6).to(matrix.device)

    @torch.no_grad()
    def update_cached_gain(self, matrix: torch.Tensor) -> torch.Tensor:
        eigenvalue = self._compute_max_eigenvalue(matrix)
        self.cached_max_eigenvalue.copy_(
            eigenvalue.to(
                self.cached_max_eigenvalue.device,
                dtype=self.cached_max_eigenvalue.dtype,
            )
        )
        return eigenvalue

    def forward(
        self,
        raw_activity: torch.Tensor,
        neuron_correlation_matrix: torch.Tensor,
    ) -> torch.Tensor:
        if raw_activity.dim() != 2:
            raise ValueError("raw_activity must be [batch, out_dim].")

        x = raw_activity
        correlation = neuron_correlation_matrix
        max_eigenvalue = self.cached_max_eigenvalue.to(
            device=correlation.device, dtype=correlation.dtype
        ).clamp_min(1e-6)
        step_size = 1.0 / max_eigenvalue

        y = x.clone()
        variance_history: deque[torch.Tensor] = deque(
            [y.var(dim=1, unbiased=False).detach()],
            maxlen=2 * self.variance_stop_window,
        )
        turn_seen = torch.zeros(y.shape[0], dtype=torch.bool, device=y.device)
        finished = torch.zeros_like(turn_seen)
        completed_per_case = torch.full(
            (y.shape[0],), -1, dtype=torch.long, device=y.device
        )

        for iteration in range(self._iteration_limit()):
            update = x - torch.matmul(y, correlation)
            candidate = F.relu(y + step_size * update)
            y_next = torch.where(finished[:, None], y, candidate)
            completed_iters = iteration + 1

            turn_seen, finished, completed_per_case = self._variance_stop_update(
                y_next,
                variance_history,
                turn_seen,
                finished,
                completed_per_case,
                completed_iters,
            )

            y = y_next
            if bool(finished.all().item()):
                break

        self.last_completed_iters = completed_iters
        self.last_variance_stop_triggered = bool(finished.all().item())
        self.last_variance_stop_resolved_count = int(finished.sum().item())
        self.last_variance_turn_count = int(turn_seen.sum().item())
        self.last_variance_stop_completed_iters = completed_per_case.detach().cpu().clone()
        return y
