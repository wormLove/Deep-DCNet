"""
Multi-layer ("stacked") biological classifier.

  Input -> DL_0 -> DL_1 -> ... -> DL_{N-1} -> [readout head]

This is a from-scratch build against Deep-DCNet's REAL components (the
batched DiscriminationLayer with its adaptive per-neuron learning rates via
ProtectWithRecoveryLR/DiscriminationOrganizer/NeuronStateTracker, the real
ActivationCache, the real WeightDriftMonitor) - not a copy of the old
DC_Net_Hebbian_Model/architectures/stacked.py module, which wraps a
different (single-sample, custom save_state/load_state) DiscriminationModule
implementation that doesn't exist in this repo.

It IS a faithful port of that module's *design*, since that's what was
asked for:
  - Stability-gated organize(): layer 0 always organizes on schedule; layer
    i (i > 0) only organizes once layer i-1 has stabilized (drift below a
    threshold for k consecutive checks - see core/weight_monitor.py).
  - Per-layer WeightDriftMonitor + ActivationCache, so each layer boundary
    can independently track stability and cache (activation, label) pairs
    for review once THAT layer (and everything upstream of it) is stable.
  - forward_with_activations(layer_idx, cached_acts): replay cached
    layer-i activations through layers i+1..N-1 and the head, for review
    mini-steps, without recomputing anything upstream of layer i.

Deliberate differences from architectures/single_layer.py (the
single-layer model):
  - No IntegrationLayer support yet. The old stacked model didn't have one
    either; if cat([layer0, layerN])-style fusion on top of a stack is
    wanted later, that's a separate, additive follow-up.
  - head_cls/head_kwargs work exactly the same way as the single-layer
    model's, so TraditionalMLPHead (Prof. Yu's point #5 comparison head)
    is a drop-in swap here too - the stacked model isn't limited to the
    plain linear ReadoutHead.

Caveat (same honesty note as everywhere else in this codebase so far):
this class has been syntax-checked and logic-reviewed against the real
DiscriminationLayer/ActivationCache/WeightDriftMonitor APIs, but has NOT
been run end-to-end with real torch (this sandbox can't load the CUDA
libs). Run a quick smoke test on your machine before trusting numbers
out of it.
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from core.activation_cache import ActivationCache
from core.weight_monitor import WeightDriftMonitor
from modules.discrimination import DiscriminationLayer
from modules.readout import ReadoutHead


class StackedBiologicalClassifier(nn.Module):
    def __init__(
        self,
        layer_dims: List[int],
        data_initializer=None,
        transform=None,
        discrimination_config: Optional[Dict[str, Any]] = None,
        head_cls=ReadoutHead,
        head_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            layer_dims: e.g. [784, 2000, 2000, 10]. First entry is the
                (flattened) input dim, last entry is the number of output
                classes, everything in between is one DiscriminationLayer's
                out_dim (== the next layer's in_dim). len(layer_dims) - 2
                DiscriminationLayers are built.
            data_initializer: passed to layer 0 only (PCA/dataset-based
                init), matching the original stacked design - deeper
                layers use DiscriminationLayer's own default (random,
                column-normalized) init.
            transform: input preprocessing applied once before DL_0
                (e.g. flatten_to_vector).
            discrimination_config: kwargs forwarded to every
                DiscriminationLayer (same config shared across all layers,
                matching the original design).
            head_cls/head_kwargs: classifier head on top of the last DL
                layer's output - defaults to the plain linear ReadoutHead;
                pass modules.classifier_heads.TraditionalMLPHead here for
                the point #5 comparison head.
        """
        super().__init__()
        if len(layer_dims) < 3:
            raise ValueError(
                "layer_dims must have at least 3 entries: [input, hidden, output]. "
                f"Got {layer_dims}."
            )
        self.layer_dims = list(int(d) for d in layer_dims)
        self.transform = transform if transform is not None else (lambda t: t)
        self.discrimination_config = discrimination_config or {}

        num_dl_layers = len(self.layer_dims) - 2
        dl_layers = []
        for i in range(num_dl_layers):
            in_d = self.layer_dims[i]
            out_d = self.layer_dims[i + 1]
            initializer = data_initializer if i == 0 else None
            dl = DiscriminationLayer(
                in_dim=in_d,
                out_dim=out_d,
                initializer=initializer,
                **self.discrimination_config,
            )
            for p in dl.parameters():
                p.requires_grad = False
            dl_layers.append(dl)
        self.discrimination_layers = nn.ModuleList(dl_layers)

        self.readout_head = head_cls(self.layer_dims[-2], self.layer_dims[-1], **(head_kwargs or {}))

        self.monitors: List[WeightDriftMonitor] = [
            WeightDriftMonitor(name=f"DL_{i}") for i in range(num_dl_layers)
        ]
        self.activation_caches: List[ActivationCache] = [
            ActivationCache() for _ in range(num_dl_layers)
        ]

        self._review_enabled = False
        self._p: List[float] = [0.0] * num_dl_layers
        self._p_max = 0.8
        self._p_delta = 0.2

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------
    @property
    def num_dl_layers(self) -> int:
        return len(self.discrimination_layers)

    def is_stable(self, layer_idx: int = -1) -> bool:
        """True if DL layer `layer_idx` is currently stable (default: the
        last layer, i.e. 'is the representation feeding the head stable')."""
        return self.monitors[layer_idx].state == "stable"

    # ------------------------------------------------------------------
    # Organize (stability-gated cascade)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def organize(self, unit_norm: bool = True) -> None:
        """
        Layer 0 always organizes. Layer i (i > 0) only organizes once
        layer i-1 is stable - same gating rule as the original stacked
        design, so a downstream layer doesn't chase a moving target.
        """
        for i, dl in enumerate(self.discrimination_layers):
            if i > 0 and not self.is_stable(i - 1):
                continue

            dl.organize(unit_norm=unit_norm)

            prev_state = self.monitors[i].state
            self.monitors[i].step(dl.neuron_weights)

            if prev_state == "stable" and self.monitors[i].state == "unstable":
                self.activation_caches[i].clear()

    @torch.no_grad()
    def reset_training_state(self) -> None:
        for dl in self.discrimination_layers:
            dl.reset_organizer_state()
        for m in self.monitors:
            m.reset()
        for c in self.activation_caches:
            c.clear()
        self._p = [0.0] * self.num_dl_layers

    def diagnostics(self, layer_idx: int = -1) -> Dict[str, torch.Tensor]:
        """Diagnostics for one layer (default: the last). For all layers,
        index model.discrimination_layers directly."""
        return self.discrimination_layers[layer_idx].diagnostics()

    # ------------------------------------------------------------------
    # Review mechanism (per layer)
    # ------------------------------------------------------------------
    def enable_review(self, enabled: bool = True, p0: float = 0.0, pmax: float = 0.8, delta: float = 0.2) -> None:
        self._review_enabled = bool(enabled)
        self._p = [float(p0 if enabled else 0.0)] * self.num_dl_layers
        self._p_max = float(pmax)
        self._p_delta = float(delta)

    def review_after_organize(self) -> None:
        """
        Call after organize(). Per layer:
          - first stable block:     p[i] = 0
          - continued stable block: p[i] += delta (capped at p_max)
          - transition to unstable: p[i] = 0, cache[i] cleared
        """
        if not self._review_enabled:
            return
        for i in range(self.num_dl_layers):
            st = self.monitors[i].state
            changed = self.monitors[i].state_changed

            if st == "stable" and changed:
                self._p[i] = 0.0
            elif st == "stable" and not changed:
                self._p[i] = min(self._p[i] + self._p_delta, self._p_max)
            elif st == "unstable" and changed:
                self._p[i] = 0.0
                self.activation_caches[i].clear()

    def review_sample(self, layer_idx: int, batch_size: int, device, out_dtype: torch.dtype = torch.float32):
        return self.activation_caches[layer_idx].sample_batch(batch_size, device=device, out_dtype=out_dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, record_cache: bool = False) -> torch.Tensor:
        """
        Transform once, run through all DL layers under no_grad (each
        DiscriminationLayer accumulates its own Hebbian/anti-Hebbian
        potential internally via its forward(), same as the single-layer
        model), optionally caching each stable layer's activations for
        review, then the trainable head.
        """
        x = self.transform(x)
        with torch.no_grad():
            for i, dl in enumerate(self.discrimination_layers):
                x = dl(x)
                if record_cache and self._review_enabled and y is not None and self.is_stable(i):
                    self.activation_caches[i].add(x.detach(), y.detach())
        return self.readout_head(x)

    def forward_with_activations(self, layer_idx: int, cached_acts: torch.Tensor) -> torch.Tensor:
        """
        Replay cached activations from `layer_idx` through layers
        layer_idx+1..N-1 (no_grad) and the head. Used for per-layer review
        mini-steps so upstream layers aren't recomputed.
        """
        x = cached_acts
        with torch.no_grad():
            for i in range(layer_idx + 1, self.num_dl_layers):
                x = self.discrimination_layers[i](x)
        return self.readout_head(x)

    # ------------------------------------------------------------------
    # Save / load (whole-model; for a reusable, named, cross-experiment
    # checkpoint of just the discrimination stack, see
    # core.checkpointing.save_stack/load_stack)
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        state = {
            "layer_dims": self.layer_dims,
            "discrimination_layers": [dl.state_dict() for dl in self.discrimination_layers],
            "readout_head": self.readout_head.state_dict(),
            "meta": {
                "version": "StackedBiologicalClassifier",
                "discrimination_config": self.discrimination_config,
            },
        }
        torch.save(state, path)

    def load_model(self, path: str, map_location=None) -> None:
        state = torch.load(path, map_location=map_location)
        for i, dl in enumerate(self.discrimination_layers):
            dl.load_state_dict(state["discrimination_layers"][i])
        self.readout_head.load_state_dict(state["readout_head"])
