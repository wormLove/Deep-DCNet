import math

import torch
from torch import nn


class NeuronStateTracker(nn.Module):
    """
    Minimal GPU-friendly neuron statistics for memory-strength tracking.
    """

    def __init__(
        self,
        out_dim: int,
        half_count: float = 100.0,
        strong_bonus: float = 0.5,
        std_scale: float = 1.5,
        strong_bonus_power: float = 2.0,
        strong_bonus_cap: float = 3.0,
        min_active_for_std: int = 2,
        std_eps: float = 1e-8,
        activity_cache_capacity: int = 256,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.k_count = math.log(2.0) / max(float(half_count), 1e-6)
        self.strong_bonus = float(strong_bonus)
        self.std_scale = float(std_scale)
        self.strong_bonus_power = float(strong_bonus_power)
        self.strong_bonus_cap = float(strong_bonus_cap)
        self.min_active_for_std = int(min_active_for_std)
        self.std_eps = float(std_eps)

        self.register_buffer("total_count", torch.zeros(self.out_dim, dtype=torch.long))
        self.register_buffer("weighted_count", torch.zeros(self.out_dim, dtype=torch.float32))
        self.register_buffer("cycles_since_active", torch.zeros(self.out_dim, dtype=torch.long))
        self.register_buffer("last_cycle_hits", torch.zeros(self.out_dim, dtype=torch.bool))
        self.register_buffer("last_cycle_total_increment", torch.zeros(self.out_dim, dtype=torch.long))
        self.register_buffer("last_cycle_weighted_increment", torch.zeros(self.out_dim, dtype=torch.float32))
        self.register_buffer("last_cycle_sample_count", torch.tensor(0, dtype=torch.long))
        cache_capacity = max(int(activity_cache_capacity), 1)
        self.register_buffer(
            "activity_cache",
            torch.empty(cache_capacity, self.out_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("activity_cache_size", torch.tensor(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def _ensure_cache_capacity(self, required_size: int, device: torch.device) -> None:
        current_capacity = int(self.activity_cache.shape[0])
        if required_size <= current_capacity and self.activity_cache.device == device:
            return

        new_capacity = max(required_size, current_capacity * 2)
        new_cache = torch.empty(new_capacity, self.out_dim, device=device, dtype=torch.float32)
        current_size = int(self.activity_cache_size.item())
        if current_size > 0:
            new_cache[:current_size].copy_(self.activity_cache[:current_size].to(device=device, dtype=torch.float32))
        self.activity_cache = new_cache

    @torch.no_grad()
    def cache_activity(self, activity_values: torch.Tensor) -> None:
        if activity_values.dim() != 2:
            raise ValueError("activity_values must be [batch, out_dim].")
        a = activity_values.detach().to(dtype=torch.float32)
        batch_size = int(a.shape[0])
        start = int(self.activity_cache_size.item())
        end = start + batch_size
        self._ensure_cache_capacity(end, a.device)
        self.activity_cache[start:end].copy_(a)
        self.activity_cache_size.fill_(end)

    @torch.no_grad()
    def get_raw_strength(self) -> torch.Tensor:
        wc = self.weighted_count.clamp_min(0.0)
        strength = 1.0 - torch.exp(-self.k_count * wc)
        return strength.clamp(0.0, 1.0)

    @torch.no_grad()
    def finalize_cycle(self):
        sample_count = int(self.activity_cache_size.item())
        if sample_count > 0:
            a = self.activity_cache[:sample_count]
            active = a > 0

            total_increment = active.sum(dim=0).to(self.total_count.dtype)
            cycle_hits = active.any(dim=0)

            weighted_increment = total_increment.to(self.weighted_count.dtype)

            row_active_count = active.sum(dim=1)
            valid = row_active_count >= self.min_active_for_std
            if valid.any():
                masked_a = torch.where(active, a, torch.zeros_like(a))
                denom = row_active_count.clamp_min(1).to(a.dtype).unsqueeze(1)
                mean_val = masked_a.sum(dim=1, keepdim=True) / denom

                centered = torch.where(active, a - mean_val, torch.zeros_like(a))
                var_val = centered.square().sum(dim=1, keepdim=True) / denom
                std_val = torch.sqrt(var_val.clamp_min(self.std_eps))
                valid = valid & (std_val.squeeze(1) > self.std_eps)
                if valid.any():
                    z = (a - mean_val) / std_val
                    strong = active & valid.unsqueeze(1) & (z > self.std_scale)
                    if strong.any():
                        z_margin = (z - self.std_scale).clamp_min(0.0)
                        bonus = self.strong_bonus * torch.pow(1.0 + z_margin, self.strong_bonus_power)
                        bonus = bonus.clamp(max=self.strong_bonus_cap)
                        bonus = torch.where(strong, bonus, torch.zeros_like(bonus))
                        weighted_increment += bonus.sum(dim=0)
        else:
            sample_count = 0
            cycle_hits = torch.zeros_like(self.last_cycle_hits)
            total_increment = torch.zeros_like(self.total_count)
            weighted_increment = torch.zeros_like(self.weighted_count)

        self.total_count += total_increment
        self.weighted_count += weighted_increment

        inactive = ~cycle_hits
        self.cycles_since_active[inactive] += 1
        self.cycles_since_active[~inactive] = 0

        self.last_cycle_hits.copy_(cycle_hits)
        self.last_cycle_total_increment.copy_(total_increment)
        self.last_cycle_weighted_increment.copy_(weighted_increment)
        self.last_cycle_sample_count.fill_(sample_count)
        self.activity_cache_size.zero_()
        return inactive.clone(), self.cycles_since_active.clone()
