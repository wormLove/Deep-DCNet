import random
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from core.weight_monitor import WeightDriftMonitor
from modules.discrimination import DiscriminationModule
from modules.readout import ReadoutHead
from modules.activation_cache import ActivationCache


class BiologicalClassifier(nn.Module):
    """
    Single-layer biological classifier:
      Input -> DiscriminationModule (no gradients, Hebbian-style updates via organize())
            -> ReadoutHead (trainable with backprop)

    Optional: review mechanism that caches (activation, label) pairs during stable periods
    and re-trains the readout head using cached activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        data_initializer=None,
        transform=None,
        non_negative_strategy: str = "softplus",
        discrimination_config: Optional[Dict[str, Any]] = None,
        head_cls=ReadoutHead,
        head_kwargs: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        super().__init__()

        self.transform = transform if transform is not None else (lambda t: t)
        self.debug = debug

        if discrimination_config is None:
            discrimination_config = {}

        # Discrimination layer (Hebbian-style; no gradients)
        self.discrimination_layer = DiscriminationModule(
            input_dim,
            hidden_dim,
            initializer=data_initializer,
            non_negative_strategy=non_negative_strategy,
            **discrimination_config,
        )
        for p in self.discrimination_layer.parameters():
            p.requires_grad = False

        # Readout head. Defaults to the linear ReadoutHead (unchanged behavior);
        # pass head_cls to swap in a different head - e.g.
        # modules.classifier_heads.TraditionalMLPHead - for comparison
        # experiments (Prof. Yu's point #5). Any head_cls must accept
        # (in_dim, out_dim, **kwargs) and implement forward(x) -> logits,
        # matching ReadoutHead's interface, so it's a drop-in swap.
        self.readout_head = head_cls(hidden_dim, output_dim, **(head_kwargs or {}))

        # Monitoring + review cache
        self.monitor_last_dl = WeightDriftMonitor(name="DL_Last")
        self.activation_cache = ActivationCache()

        # Review schedule state
        self._review_enabled = False
        self._p = 0.0
        self._p_max = 0.8
        self._p_delta = 0.2

        # Local RNG (does not affect global RNG states)
        self._rng = random.Random(0)

    # ---------------------------------------------------------------------
    # Discrimination layer update / stability
    # ---------------------------------------------------------------------
    def organize(self) -> None:
        """
        Update discrimination-layer weights based on accumulated activity.
        Also updates drift monitor and clears cache if stability turns unstable.
        """
        self.discrimination_layer.organize()

        prev_state = self.monitor_last_dl.state
        self.monitor_last_dl.step(self.discrimination_layer.neuron_weights)

        # If we just became unstable, clear cached activations
        if prev_state == "stable" and self.monitor_last_dl.state == "unstable":
            self.activation_cache.clear()

    def is_stable(self) -> bool:
        """Return True if the discrimination layer is currently stable."""
        return self.monitor_last_dl.state == "stable"

    # ---------------------------------------------------------------------
    # Review mechanism controls
    # ---------------------------------------------------------------------
    def enable_review(self, enabled: bool = True, p0: float = 0.0, pmax: float = 0.8, delta: float = 0.2) -> None:
        """
        Enable/disable review.
        p is the review ratio used by the training loop to decide the number of review steps.
        """
        self._review_enabled = bool(enabled)
        self._p = float(p0 if enabled else 0.0)
        self._p_max = float(pmax)
        self._p_delta = float(delta)

    def review_after_organize(self) -> None:
        """
        Call after organize(). Adjusts review ratio p based on stability transitions.
          - first stable block: p = 0
          - continued stable blocks: p increases by delta up to p_max
          - unstable transition: p = 0 and cache cleared
        """
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
            self.activation_cache.clear()

    def review_sample(self, batch_size: int, device: torch.device, out_dtype: torch.dtype = torch.float32):
        """Sample a batch of (activations, labels) from cache."""
        return self.activation_cache.sample_batch(batch_size, device=device, out_dtype=out_dtype)

    # ---------------------------------------------------------------------
    # Save / load
    # ---------------------------------------------------------------------
    def save_model(self, path: str, include_training_state: bool = False) -> None:
        state = {
            "discrimination_layer": self.discrimination_layer.save_state(include_training_state),
            "readout_head": self.readout_head.state_dict(),
            "meta": {
                "version": "v-test-SingleLayer",
                "include_training_state": include_training_state,
            },
        }
        torch.save(state, path)

    def load_model(self, path: str, include_training_state: bool = False, map_location: Optional[str] = None) -> None:
        """
        Load model weights.
        For student (CPU-only) release, default map_location is current device.
        """
        if map_location is None:
            map_location = str(next(self.parameters()).device)

        state = torch.load(path, map_location=map_location)
        self.discrimination_layer.load_state(state["discrimination_layer"], include_training_state)
        self.readout_head.load_state_dict(state["readout_head"])

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, record_cache: bool = False) -> torch.Tensor:
        """
        Forward pass:
          - DiscriminationModule runs under no_grad
          - Optionally cache (activation, y) for review training
          - ReadoutHead is trainable and outputs logits
        """
        x = self.transform(x)

        with torch.no_grad():
            act = self.discrimination_layer(x)
            if record_cache and self._review_enabled and y is not None:
                self.activation_cache.add(act.detach(), y.detach())

        return self.readout_head(act)

    def forward_with_activations(self, dl_output: torch.Tensor) -> torch.Tensor:
        """Forward readout head only (used for review mini-steps)."""
        return self.readout_head(dl_output)
