from typing import Callable, Dict, Optional

import torch
from torch import nn

from core.activation_cache import ActivationCache
from core.weight_monitor import WeightDriftMonitor
from models.discrimination import DiscriminationLayer
from models.readout import ReadoutHead


class BiologicalClassifier(nn.Module):
    """
    Minimal wrapper around a single DiscriminationLayer.

    Supports two modes:
    - discrimination-only mode (no readout)
    - simple classification mode with a linear readout head

    This stage intentionally excludes:
    - review / replay
    - task-specific multi-head logic
    - readout confidence gating
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        data_initializer=None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        discrimination_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.transform = transform if transform is not None else (lambda t: t)
        self.discrimination_config = discrimination_config or {}

        self.discrimination_layer = DiscriminationLayer(
            in_dim=input_dim,
            out_dim=hidden_dim,
            initializer=data_initializer,
            **self.discrimination_config,
        )
        self.output_dim = output_dim
        self.readout_head = ReadoutHead(hidden_dim, output_dim) if output_dim is not None else None

        for param in self.discrimination_layer.parameters():
            param.requires_grad = False

        self.monitor_last_dl = WeightDriftMonitor(name="DL_Last")
        self.activation_cache = ActivationCache()
        self._review_enabled = False
        self._p = 0.0
        self._p_max = 0.8
        self._p_delta = 0.2

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        record_cache: bool = False,
        return_intermediate: bool = False,
    ):
        x = self.transform(x)
        act = self.discrimination_layer(x, return_intermediate=return_intermediate)

        if self.readout_head is None:
            return act

        if return_intermediate:
            act_dict = act
            if record_cache and self.activation_cache is not None and y is not None:
                self.activation_cache.add(act_dict["act"].detach(), y.detach())
            logits = self.readout_head(act_dict["act"])
            act_dict["logits"] = logits
            return act_dict

        if record_cache and self.activation_cache is not None and y is not None:
            self.activation_cache.add(act.detach(), y.detach())
        return self.readout_head(act)

    def forward_with_activations(self, activations: torch.Tensor) -> torch.Tensor:
        if self.readout_head is None:
            raise RuntimeError("forward_with_activations requires a readout head.")
        return self.readout_head(activations)

    @torch.no_grad()
    def organize(self, unit_norm: bool = True) -> torch.Tensor:
        updated = self.discrimination_layer.organize(unit_norm=unit_norm)
        was_state = self.monitor_last_dl.state
        self.monitor_last_dl.step(self.discrimination_layer.neuron_weights)
        if was_state == "stable" and self.monitor_last_dl.state == "unstable":
            if self.activation_cache is not None:
                self.activation_cache.clear()
        return updated

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, torch.Tensor]:
        return self.discrimination_layer.diagnostics()

    @torch.no_grad()
    def reset_training_state(self) -> None:
        self.discrimination_layer.reset_organizer_state()
        self.monitor_last_dl.reset()
        if self.activation_cache is not None:
            self.activation_cache.clear()
        self._p = 0.0

    def is_stable(self) -> bool:
        return self.monitor_last_dl.state == "stable"

    def enable_review(self, enabled: bool = True, p0: float = 0.0, pmax: float = 0.8, delta: float = 0.2) -> None:
        self._review_enabled = bool(enabled)
        self._p = float(p0 if enabled else 0.0)
        self._p_max = float(pmax)
        self._p_delta = float(delta)

    def review_after_organize(self) -> None:
        if not self._review_enabled:
            return

        st = self.monitor_last_dl.state
        changed = self.monitor_last_dl.state_changed

        if st == "stable" and changed:
            self._p = 0.0
        elif st == "stable" and not changed:
            self._p = min(self._p + self._p_delta, self._p_max)
        elif st == "unstable" and changed:
            self._p = 0.0
            if self.activation_cache is not None:
                self.activation_cache.clear()

    def review_sample(self, batch_size: int, device, out_dtype=torch.float32):
        return self.activation_cache.sample_batch(batch_size, device=device, out_dtype=out_dtype)

    def save_model(self, path: str) -> None:
        state = {
            "discrimination_layer": self.discrimination_layer.state_dict(),
            "meta": {
                "version": "gpu_rebuild_stage1",
                "has_readout": self.readout_head is not None,
                "discrimination_config": self.discrimination_config,
            },
        }
        if self.readout_head is not None:
            state["readout_head"] = self.readout_head.state_dict()
        torch.save(state, path)

    def load_model(self, path: str, map_location=None) -> None:
        state = torch.load(path, map_location=map_location)
        self.discrimination_layer.load_state_dict(state["discrimination_layer"])
        if self.readout_head is not None and "readout_head" in state:
            self.readout_head.load_state_dict(state["readout_head"])
