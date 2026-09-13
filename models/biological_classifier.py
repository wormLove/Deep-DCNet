from typing import Callable, Dict, Optional

import torch
from torch import nn

from core.activation_cache import ActivationCache
from core.weight_monitor import WeightDriftMonitor
from models.discrimination import DiscriminationLayer
from models.integration import IntegrationLayer
from models.readout import ReadoutHead


class BiologicalClassifier(nn.Module):
    """
    Minimal wrapper around a single DiscriminationLayer.

    Supports two modes:
    - discrimination-only mode (no readout)
    - simple classification mode with a linear readout head

    Optionally inserts a dense IntegrationLayer between the discrimination
    layer and the readout.  When enabled, the integration layer receives the
    concatenation of the raw transformed input (Layer 0) and the sparse
    discrimination activations (Layer 1), and compresses them into a joint
    representation before classification.

    This stage intentionally excludes:
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
        integration_dim: Optional[int] = None,
        integration_activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.transform = transform if transform is not None else (lambda t: t)
        self.discrimination_config = discrimination_config or {}

        self.discrimination_layer = DiscriminationLayer(
            in_dim=input_dim,
            out_dim=hidden_dim,
            initializer=data_initializer,
            **self.discrimination_config,
        )

        for param in self.discrimination_layer.parameters():
            param.requires_grad = False

        # --- Integration layer (optional) ---
        # When present: cat([layer0, layer1]) → IntegrationLayer → ReadoutHead
        # When absent:  layer1 → ReadoutHead  (original behaviour)
        self.output_dim = output_dim
        if output_dim is not None:
            if integration_dim is not None:
                self.integration_layer = IntegrationLayer(
                    in_dim=input_dim + hidden_dim,
                    out_dim=integration_dim,
                    activation=integration_activation,
                )
                self.readout_head = ReadoutHead(integration_dim, output_dim)
            else:
                self.integration_layer = None
                self.readout_head = ReadoutHead(hidden_dim, output_dim)
        else:
            self.integration_layer = None
            self.readout_head = None

        self.monitor_last_dl = WeightDriftMonitor(name="DL_Last")
        self.activation_cache = ActivationCache()
        self._review_enabled = False
        self._p = 0.0
        self._p_max = 0.8
        self._p_delta = 0.2

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        record_cache: bool = False,
        return_intermediate: bool = False,
    ):
        x_t = self.transform(x)  # Layer 0 output — kept for integration
        act = self.discrimination_layer(x_t, return_intermediate=return_intermediate)

        if self.readout_head is None:
            return act

        if return_intermediate:
            act_dict = act  # dict: act_raw, act_opt, act, correlation_matrix
            act_sparse = act_dict["act"]

            if self.integration_layer is not None:
                # cat([layer0, layer1]) is cached for review replay
                integration_input = torch.cat([x_t, act_sparse], dim=1)
                integration_out = self.integration_layer(integration_input)
                act_dict["integration_input"] = integration_input
                act_dict["integration_out"] = integration_out
                if record_cache and self.activation_cache is not None and y is not None:
                    self.activation_cache.add(integration_input.detach(), y.detach())
                logits = self.readout_head(integration_out)
            else:
                if record_cache and self.activation_cache is not None and y is not None:
                    self.activation_cache.add(act_sparse.detach(), y.detach())
                logits = self.readout_head(act_sparse)

            act_dict["logits"] = logits
            return act_dict

        # Non-intermediate path
        if self.integration_layer is not None:
            integration_input = torch.cat([x_t, act], dim=1)
            integration_out = self.integration_layer(integration_input)
            if record_cache and self.activation_cache is not None and y is not None:
                self.activation_cache.add(integration_input.detach(), y.detach())
            return self.readout_head(integration_out)

        if record_cache and self.activation_cache is not None and y is not None:
            self.activation_cache.add(act.detach(), y.detach())
        return self.readout_head(act)

    def forward_with_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Run cached activations through the trainable head(s) only.

        When an integration layer is present, `activations` is expected to be
        the cached cat([layer0, layer1]) tensor (in_dim + hidden_dim wide).
        It is passed through the integration layer before the readout head.

        When no integration layer is present, `activations` is the cached
        Layer 1 sparse activation (hidden_dim wide).
        """
        if self.readout_head is None:
            raise RuntimeError("forward_with_activations requires a readout head.")
        if self.integration_layer is not None:
            return self.readout_head(self.integration_layer(activations))
        return self.readout_head(activations)

    # ------------------------------------------------------------------
    # Organiser / diagnostics
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        state = {
            "discrimination_layer": self.discrimination_layer.state_dict(),
            "meta": {
                "version": "gpu_rebuild_stage1",
                "has_readout": self.readout_head is not None,
                "has_integration": self.integration_layer is not None,
                "discrimination_config": self.discrimination_config,
            },
        }
        if self.readout_head is not None:
            state["readout_head"] = self.readout_head.state_dict()
        if self.integration_layer is not None:
            state["integration_layer"] = self.integration_layer.state_dict()
        torch.save(state, path)

    def load_model(self, path: str, map_location=None) -> None:
        state = torch.load(path, map_location=map_location)
        self.discrimination_layer.load_state_dict(state["discrimination_layer"])
        if self.readout_head is not None and "readout_head" in state:
            self.readout_head.load_state_dict(state["readout_head"])
        if self.integration_layer is not None and "integration_layer" in state:
            self.integration_layer.load_state_dict(state["integration_layer"])
