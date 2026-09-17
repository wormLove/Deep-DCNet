from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from core.initializers import RandomInitializer
from core.lr_policy import ProtectWithRecoveryLR
from core.organizer import DiscriminationOrganizer
from core.stats import NeuronStateTracker
from modules.activation import LayerThresholding
from modules.optimizer import IterativeActivityOptimizer


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
        threshold_factor: float = 1.0,
        sparsity: float = 0.05,
        optimizer_max_iters: int = 1000,
        optimizer_lambda: float = 0.1,
        optimizer_gain_factor: float = 10.0,
        optimizer_estimate_steps: int = 50,
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
        self.strength_decay_start = int(strength_decay_start)
        self.strength_decay_power = float(strength_decay_power)
        self.strength_decay_scale = float(strength_decay_scale)
        self.lr_init = float(lr_init)
        self.lr_min = float(min_lr)
        self.lr_max = float(max_lr)

        if initializer is None:
            initializer = RandomInitializer(non_negative=self.non_negative)
        weights = initializer.weights((self.in_dim, self.out_dim))
        weights = F.normalize(weights, p=2, dim=0)
        self.neuron_weights = nn.Parameter(weights)

        self.activity_optimizer = IterativeActivityOptimizer(
            # Phase-1 baseline values: intentionally lightweight for the first
            # GPU-friendly rebuild benchmark configuration.
            max_iters=optimizer_max_iters,
            lambda_=optimizer_lambda,
            gain_factor=optimizer_gain_factor,
            estimate_steps=optimizer_estimate_steps,
        )
        self.activation = LayerThresholding(
            threshold_factor=threshold_factor,
            sparsity=sparsity,
            soft=False,
        )
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
        self.register_buffer("neuron_correlation_matrix", corr0)
        self.activity_optimizer.update_cached_gain(self.neuron_correlation_matrix)

    def compute_neuron_correlation_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            weights = self.neuron_weights.detach()
            return torch.matmul(weights.transpose(0, 1), weights)

    def forward(self, x: torch.Tensor, return_intermediate: bool = False, accumulate: bool = True):
        """
        accumulate: when False, skip the Hebbian/anti-Hebbian potential
        update and the neuron-activity stats cache even in training mode -
        a pure feature-extraction pass. Defaults to True, so existing
        single-layer behavior is unchanged. Used by
        architectures/stacked.py for (a) layers that are currently gated
        off waiting for their upstream layer to stabilize, and (b) review
        replay through downstream layers.
        """
        if x.dim() != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"input must be [batch, {self.in_dim}]")

        weights = self.neuron_weights
        corr = self.neuron_correlation_matrix
        act_raw = torch.matmul(x, weights)
        act_opt = self.activity_optimizer(act_raw, corr)
        act = self.activation(act_opt)

        if self.training and accumulate:
            self.organizer.step(x, act)
            self.stats.cache_activity(act)

        if return_intermediate:
            return {
                "act_raw": act_raw,
                "act_opt": act_opt,
                "act": act,
                "correlation_matrix": corr,
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
                reinit = F.normalize(reinit, p=2, dim=0)
                updated_weights[:, zero_cols] = reinit

        if unit_norm:
            updated_weights = F.normalize(updated_weights, p=2, dim=0)

        self.neuron_weights.data.copy_(updated_weights)
        self.neuron_correlation_matrix.copy_(self.compute_neuron_correlation_matrix())
        self.activity_optimizer.update_cached_gain(self.neuron_correlation_matrix)
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
        }
