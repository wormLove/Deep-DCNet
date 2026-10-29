from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from core.initializers import RandomInitializer
from core.lr_policy import ProtectWithRecoveryLR
from core.organizer import DiscriminationOrganizer
from core.stats import NeuronStateTracker
from models.optimizer import L1NormalizedActivityOptimizer, IterativeActivityOptimizer


DEFAULT_CONNECTION_RATIO_SCHEDULE = {
    0: 0.50,
    4: 0.40,
    5: 0.30,
    6: 0.20,
}


class DiscriminationLayer(nn.Module):
    """
    Minimal GPU-ready discrimination layer:
    - neuron context weights
    - iterative activity optimization
    - sparse activation
    - Hebbian / Anti-Hebbian organization
    - neuron-level strength tracking and learning-rate control

    This intentionally excludes readout, review, and context forgetting.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        initializer=None,
        non_negative: bool = True,
        non_negative_strategy: str = "abs",
        weight_norm_p: int = 2,
        optimizer_profile: str = "l1_sparse",
        optimizer_max_iters: int = 1000,
        optimizer_support_threshold: float = 0.08,
        optimizer_initial_support_threshold: float | None = None,
        optimizer_initial_max_iters: int | None = None,
        optimizer_variance_stop_window: int = 20,
        optimizer_y0_divide_by_diagonal: bool | None = None,
        connection_ratio_schedule: Optional[Mapping[int, float]] = None,
        lr_init: float = 0.99,
        min_lr: float = 1e-3,
        max_lr: float = 0.99,
        beta: float = 0.99,
        recover_step: float = 0.05,
        recover_alpha: float = 0.2,
        strength_gate_k: float = 1.0,
        lr_gate_n: float = 0.5,
        half_count: float = 100.0,
        strong_bonus: float = 0.5,
        std_scale: float = 1.5,
        strong_bonus_power: float = 2.0,
        strong_bonus_cap: float = 3.0,
        min_active_for_std: int = 2,
        std_eps: float = 1e-8,
        state_cache_capacity: int = 256,
        strength_decay_start: int = 5,
        strength_decay_power: float = 2.0,
        strength_decay_scale: float = 1.0,
        use_cooldown: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.non_negative = bool(non_negative)
        self.non_negative_strategy = str(non_negative_strategy)
        # Which norm fixes each neuron's total synaptic resource. p=2 is the default
        # and keeps diag(W^T W) == 1 exactly; p=1 states the constraint as "the
        # weights sum to a constant", which is what a resource budget literally means
        # for non-negative weights, and leaves diag(W^T W) varying with how
        # concentrated the receptive field is. Only the per-column scale differs
        # between the two - every neuron's preferred direction is identical.
        #
        # p=1 is an experimental configuration, not the production default. It is
        # measured end to end in
        # RESULT/Deep-DCNet-Init-Pipeline-Audit/03-normalization: it reaches 94.83%
        # against p=2's 96.14%, and it gets there with the variance turning point
        # firing on only 24% of batches - three quarters exit by exhausting
        # optimizer_max_iters rather than by satisfying the stop criterion, which the
        # cap is not meant to do. It is kept selectable so that run can be reproduced
        # and so the open questions in that document can be picked up later.
        if int(weight_norm_p) not in (1, 2):
            raise ValueError("weight_norm_p must be 1 or 2.")
        self.weight_norm_p = int(weight_norm_p)
        if optimizer_profile not in {"variance_dense", "l1_sparse"}:
            raise ValueError(
                "optimizer_profile must be 'variance_dense' or 'l1_sparse'."
            )
        self.optimizer_profile = str(optimizer_profile)
        if self.optimizer_profile == "variance_dense":
            if connection_ratio_schedule is not None:
                raise ValueError(
                    "connection_ratio_schedule is only supported by the "
                    "l1_sparse profile."
                )
            self.connection_ratio_schedule = None
            if optimizer_initial_support_threshold is None:
                optimizer_initial_support_threshold = 0.05
            if optimizer_initial_max_iters is None:
                optimizer_initial_max_iters = 20000
        else:
            schedule = (
                DEFAULT_CONNECTION_RATIO_SCHEDULE
                if connection_ratio_schedule is None
                else connection_ratio_schedule
            )
            self.connection_ratio_schedule = self._validate_connection_schedule(
                schedule
            )
            if optimizer_initial_support_threshold is None:
                optimizer_initial_support_threshold = optimizer_support_threshold
            if optimizer_initial_max_iters is None:
                optimizer_initial_max_iters = optimizer_max_iters

        # The optimizer's starting point is tied to the norm, not chosen beside it.
        # Under p=2 diag(W^T W) == 1 and the two starting points are the same run, so
        # the default stays y0 = a and archived p=2 results reproduce bit for bit.
        # Under p=1 they are not the same run: y0 = a starts about s^2 ~ 300x below
        # its own solution, the suppression step never crosses zero, relu never fires,
        # and the turning point was measured firing on 12 of 80 cases with 68 of them
        # exhausting the 20000-iteration budget. So p=1 does not get a choice here.
        if self.optimizer_profile == "l1_sparse":
            if optimizer_y0_divide_by_diagonal:
                raise ValueError(
                    "optimizer_y0_divide_by_diagonal is not used by the "
                    "l1_sparse profile."
                )
            optimizer_y0_divide_by_diagonal = False
        elif self.weight_norm_p == 1:
            optimizer_y0_divide_by_diagonal = True
        else:
            optimizer_y0_divide_by_diagonal = bool(optimizer_y0_divide_by_diagonal)
        self.strength_decay_start = int(strength_decay_start)
        self.strength_decay_power = float(strength_decay_power)
        self.strength_decay_scale = float(strength_decay_scale)
        self.lr_init = float(lr_init)
        self.lr_min = float(min_lr)
        self.lr_max = float(max_lr)

        if initializer is None:
            initializer = RandomInitializer(
                non_negative=self.non_negative,
                non_negative_strategy=self.non_negative_strategy,
            )
        elif hasattr(initializer, "non_negative_strategy"):
            # A supplied initializer owns the transform applied to its weights.
            self.non_negative_strategy = str(initializer.non_negative_strategy)
        self.initializer_config = (
            initializer.configuration()
            if hasattr(initializer, "configuration")
            else {"type": type(initializer).__name__}
        )
        weights = initializer.weights((self.in_dim, self.out_dim))
        weights = F.normalize(weights, p=self.weight_norm_p, dim=0)
        self.neuron_weights = nn.Parameter(weights)

        optimizer_type = (
            L1NormalizedActivityOptimizer
            if self.optimizer_profile == "l1_sparse"
            else IterativeActivityOptimizer
        )
        self.activity_optimizer = optimizer_type(
            max_iters=optimizer_max_iters,
            support_threshold=optimizer_support_threshold,
            initial_support_threshold=optimizer_initial_support_threshold,
            initial_max_iters=optimizer_initial_max_iters,
            variance_stop_window=optimizer_variance_stop_window,
            y0_divide_by_diagonal=optimizer_y0_divide_by_diagonal,
        )
        self.activation = nn.ReLU()
        self.organizer = DiscriminationOrganizer(
            in_dim=in_dim,
            out_dim=out_dim,
            default_lr=self.lr_init,
            beta=beta,
            use_cooldown=use_cooldown,
        )
        self.stats = NeuronStateTracker(
            out_dim=out_dim,
            half_count=half_count,
            strong_bonus=strong_bonus,
            std_scale=std_scale,
            strong_bonus_power=strong_bonus_power,
            strong_bonus_cap=strong_bonus_cap,
            min_active_for_std=min_active_for_std,
            std_eps=std_eps,
            activity_cache_capacity=state_cache_capacity,
        )
        self.lr_policy = ProtectWithRecoveryLR(
            lr_init=self.lr_init,
            min_lr=min_lr,
            max_lr=max_lr,
            recover_step=recover_step,
            recover_alpha=recover_alpha,
            strength_gate_k=strength_gate_k,
            lr_gate_n=lr_gate_n,
        )

        self.register_buffer("lr_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("last_raw_strength", torch.zeros(self.out_dim))
        self.register_buffer("last_effective_strength", torch.zeros(self.out_dim))
        self.register_buffer("last_lr_vec", torch.full((self.out_dim,), self.lr_init))
        self.register_buffer("last_inactive_mask", torch.zeros(self.out_dim, dtype=torch.bool))
        self.register_buffer("last_cycles_since_active", torch.zeros(self.out_dim, dtype=torch.long))
        self.register_buffer("last_cycle_hits", torch.zeros(self.out_dim, dtype=torch.bool))
        self.register_buffer("last_cycle_total_increment", torch.zeros(self.out_dim, dtype=torch.long))
        self.register_buffer("last_cycle_weighted_increment", torch.zeros(self.out_dim))
        self.register_buffer("last_cycle_sample_count", torch.tensor(0, dtype=torch.long))
        with torch.no_grad():
            corr0 = torch.matmul(self.neuron_weights.detach().transpose(0, 1), self.neuron_weights.detach())
        self.register_buffer("organize_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("neuron_correlation_matrix", corr0.clone())
        self.register_buffer(
            "lateral_connection_mask", torch.ones_like(corr0, dtype=torch.bool)
        )
        self.register_buffer("effective_lateral_matrix", corr0.clone())
        self.register_buffer(
            "current_connection_ratio", torch.tensor(1.0, dtype=corr0.dtype)
        )
        self._refresh_lateral_operator(corr0)

    @staticmethod
    def _validate_connection_schedule(
        schedule: Mapping[int, float],
    ) -> Dict[int, float]:
        if not schedule:
            raise ValueError("connection_ratio_schedule cannot be empty.")

        normalized = {int(start): float(ratio) for start, ratio in schedule.items()}
        if 0 not in normalized:
            raise ValueError("connection_ratio_schedule must start at organize 0.")
        if any(start < 0 for start in normalized):
            raise ValueError("connection schedule starts must be non-negative.")
        if any(not 0.0 < ratio <= 1.0 for ratio in normalized.values()):
            raise ValueError("connection ratios must be inside (0, 1].")
        return dict(sorted(normalized.items()))

    def connection_ratio_for_organize(self, organize_count: int) -> float:
        if self.connection_ratio_schedule is None:
            return 1.0
        active_start = max(
            start for start in self.connection_ratio_schedule if start <= organize_count
        )
        return self.connection_ratio_schedule[active_start]

    @torch.no_grad()
    def _build_lateral_mask(
        self,
        dense_correlation: torch.Tensor,
        connection_ratio: float,
    ) -> torch.Tensor:
        neuron_count = dense_correlation.shape[0]
        mask = torch.zeros_like(dense_correlation, dtype=torch.bool)
        if neuron_count > 1:
            top_k = min(
                neuron_count - 1,
                max(1, round(connection_ratio * neuron_count)),
            )
            similarity = dense_correlation.clone()
            similarity.fill_diagonal_(-torch.inf)
            neighbor_indices = torch.topk(similarity, k=top_k, dim=0).indices
            target_indices = torch.arange(
                neuron_count, device=dense_correlation.device
            ).expand(top_k, neuron_count)
            mask[neighbor_indices, target_indices] = True
        mask.fill_diagonal_(True)
        return mask

    @torch.no_grad()
    def _refresh_lateral_operator(
        self,
        dense_correlation: Optional[torch.Tensor] = None,
    ) -> None:
        if dense_correlation is None:
            dense_correlation = self.compute_neuron_correlation_matrix()
        self.neuron_correlation_matrix.copy_(dense_correlation)

        ratio = self.connection_ratio_for_organize(int(self.organize_count.item()))
        if self.optimizer_profile == "l1_sparse":
            mask = self._build_lateral_mask(dense_correlation, ratio)
        else:
            mask = torch.ones_like(dense_correlation, dtype=torch.bool)
        effective = dense_correlation * mask.to(dense_correlation.dtype)

        self.lateral_connection_mask.copy_(mask)
        self.effective_lateral_matrix.copy_(effective)
        self.current_connection_ratio.fill_(ratio)
        self.activity_optimizer.update_cached_gain(self.effective_lateral_matrix)

    def compute_neuron_correlation_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            weights = self.neuron_weights.detach()
            return torch.matmul(weights.transpose(0, 1), weights)

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        if x.dim() != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"input must be [batch, {self.in_dim}]")

        weights = self.neuron_weights
        corr = self.effective_lateral_matrix
        act_raw = torch.matmul(x, weights)
        self.activity_optimizer.set_initial_phase(
            not bool(self.lr_ready.item())
        )
        act_opt = self.activity_optimizer(act_raw, corr)
        act = self.activation(act_opt)

        if self.training:
            self.organizer.step(x, act)
            self.stats.cache_activity(act)

        if return_intermediate:
            return {
                "act_raw": act_raw,
                "act_opt": act_opt,
                "act": act,
                "correlation_matrix": corr,
                "dense_correlation_matrix": self.neuron_correlation_matrix,
                "lateral_connection_mask": self.lateral_connection_mask,
            }
        return act

    @torch.no_grad()
    def refresh_learning_rates(self) -> torch.Tensor:
        raw_strength = self.stats.get_raw_strength().to(self.neuron_weights.device, dtype=self.neuron_weights.dtype)
        inactive_mask, cycles = self.stats.finalize_cycle()
        inactive_mask = inactive_mask.to(self.neuron_weights.device)
        cycles = cycles.to(self.neuron_weights.device)

        current_lr = self.organizer.lr_vec.to(self.neuron_weights.device, dtype=self.neuron_weights.dtype)
        lr_span = max(self.lr_max - self.lr_min, 1e-12)
        lr_norm = ((current_lr - self.lr_min) / lr_span).clamp(0.0, 1.0)
        decay_from_lr = self.strength_decay_scale * torch.pow(lr_norm, self.strength_decay_power)
        decay_from_lr = decay_from_lr.clamp(0.0, 1.0)

        decay_enabled = cycles >= self.strength_decay_start
        effective_strength = torch.where(
            decay_enabled,
            raw_strength * (1.0 - decay_from_lr),
            raw_strength,
        )

        lr_vec = self.lr_policy.compute(
            strength_scores=effective_strength,
            inactive_mask=inactive_mask,
            cycles_since_active=cycles,
        ).to(self.neuron_weights.device, dtype=self.neuron_weights.dtype)

        self.organizer.set_learning_rates(lr_vec)
        self.lr_ready.fill_(True)
        self.last_raw_strength.copy_(raw_strength)
        self.last_effective_strength.copy_(effective_strength)
        self.last_lr_vec.copy_(lr_vec)
        self.last_inactive_mask.copy_(inactive_mask)
        self.last_cycles_since_active.copy_(cycles)
        self.last_cycle_hits.copy_(self.stats.last_cycle_hits)
        self.last_cycle_total_increment.copy_(self.stats.last_cycle_total_increment)
        self.last_cycle_weighted_increment.copy_(self.stats.last_cycle_weighted_increment)
        self.last_cycle_sample_count.copy_(self.stats.last_cycle_sample_count)
        return lr_vec

    @torch.no_grad()
    def organize(self, unit_norm: bool = True) -> torch.Tensor:
        self.refresh_learning_rates()
        updated_weights = self.organizer.organize(self.neuron_weights.data)

        if self.non_negative:
            updated_weights = updated_weights.clamp_min(0.0)
            zero_cols = updated_weights.sum(dim=0) == 0
            if zero_cols.any():
                reinit = torch.rand(
                    updated_weights.size(0),
                    int(zero_cols.sum().item()),
                    device=updated_weights.device,
                    dtype=updated_weights.dtype,
                )
                reinit = F.normalize(reinit, p=self.weight_norm_p, dim=0)
                updated_weights[:, zero_cols] = reinit

        if unit_norm:
            updated_weights = F.normalize(updated_weights, p=self.weight_norm_p, dim=0)

        self.neuron_weights.data.copy_(updated_weights)
        self.organize_count.add_(1)
        self._refresh_lateral_operator()
        return updated_weights

    @torch.no_grad()
    def reset_organizer_state(self) -> None:
        self.organizer.potential_hebb.zero_()
        self.organizer.potential_antihebb.zero_()
        self.organizer.cooldown_state.zero_()
        self.organizer.beta_accum.fill_(self.organizer.beta)
        self.organizer.set_learning_rates(torch.full_like(self.organizer.lr_vec, self.organizer.default_lr))
        self.last_lr_vec.copy_(self.organizer.lr_vec)
        self.lr_ready.fill_(False)
        self.organize_count.zero_()
        self._refresh_lateral_operator()

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, torch.Tensor]:
        return {
            "raw_strength": self.last_raw_strength.detach().clone(),
            "effective_strength": self.last_effective_strength.detach().clone(),
            "lr_vec": self.last_lr_vec.detach().clone(),
            "inactive_mask": self.last_inactive_mask.detach().clone(),
            "cycles_since_active": self.last_cycles_since_active.detach().clone(),
            "cycle_hits": self.last_cycle_hits.detach().clone(),
            "cycle_total_increment": self.last_cycle_total_increment.detach().clone(),
            "cycle_weighted_increment": self.last_cycle_weighted_increment.detach().clone(),
            "cycle_sample_count": self.last_cycle_sample_count.detach().clone(),
            "zero_norm_rows_total": self.organizer.zero_norm_rows_total.detach().clone(),
            "organize_count": self.organize_count.detach().clone(),
            "connection_ratio": self.current_connection_ratio.detach().clone(),
        }
