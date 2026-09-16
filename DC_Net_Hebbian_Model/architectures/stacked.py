import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from core.weight_monitor import WeightDriftMonitor
from modules.discrimination import DiscriminationModule
from modules.readout import ReadoutHead
from modules.activation_cache import ActivationCache


class StackedBiologicalClassifier(nn.Module):
    """
    Multi-layer biological classifier:
      Input -> DL_0 -> DL_1 -> ... -> DL_{N-1} -> ReadoutHead

    Each DiscriminationModule (DL) uses local Hebbian-style updates via organize().
    No gradients flow through any DL layer; only the ReadoutHead is trained with backprop.

    Stability-gated organize:
      DL_0 always organizes every interval.
      DL_i (i > 0) only organizes once DL_{i-1} is stable.

    Per-layer review:
      Each DL layer has its own ActivationCache and WeightDriftMonitor.
      Caching is gated on the layer's own stability.
      Review replays layer-i cached activations through layers i+1..N-1 and readout.
    """

    def __init__(
        self,
        layer_dims: List[int],
        data_initializer=None,
        transform=None,
        non_negative_strategy: str = "shift",
        discrimination_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        """
        Args:
            layer_dims: List of dimensions, e.g. [784, 1000, 1000, 10].
                        The last entry is the number of output classes.
                        There will be len(layer_dims) - 2 DiscriminationModule layers
                        and 1 ReadoutHead.
            data_initializer: DatasetInitializer for the first DL layer (PCA-based init).
                              Deeper layers use random (column-normalised) init.
            transform: Input preprocessing callable (applied once before DL_0).
            non_negative_strategy: Weight non-negativity strategy passed to all DL layers.
            discrimination_config: Dict of kwargs forwarded to every DiscriminationModule.
            debug: If True, extra prints may be added in future.
        """
        super().__init__()

        if len(layer_dims) < 3:
            raise ValueError(
                "layer_dims must have at least 3 entries: [input, hidden, output]. "
                f"Got {layer_dims}."
            )

        self.layer_dims = layer_dims
        self.transform = transform if transform is not None else (lambda t: t)
        self.debug = debug

        if discrimination_config is None:
            discrimination_config = {}

        num_dl_layers = len(layer_dims) - 2  # number of DiscriminationModule layers

        # ------------------------------------------------------------------ #
        # Build DiscriminationModule layers
        # ------------------------------------------------------------------ #
        dl_layers = []
        for i in range(num_dl_layers):
            in_d = layer_dims[i]
            out_d = layer_dims[i + 1]
            initializer = data_initializer if i == 0 else None
            dl = DiscriminationModule(
                in_d,
                out_d,
                initializer=initializer,
                non_negative_strategy=non_negative_strategy,
                **discrimination_config,
            )
            # Disable gradients for all DL parameters
            for p in dl.parameters():
                p.requires_grad = False
            dl_layers.append(dl)

        self.discrimination_layers = nn.ModuleList(dl_layers)

        # ------------------------------------------------------------------ #
        # Readout head (trainable)
        # ------------------------------------------------------------------ #
        self.readout_head = ReadoutHead(layer_dims[-2], layer_dims[-1])

        # ------------------------------------------------------------------ #
        # Per-layer monitors and caches
        # ------------------------------------------------------------------ #
        self.monitors: List[WeightDriftMonitor] = [
            WeightDriftMonitor(name=f"DL_{i}") for i in range(num_dl_layers)
        ]
        self.activation_caches: List[ActivationCache] = [
            ActivationCache() for _ in range(num_dl_layers)
        ]

        # ------------------------------------------------------------------ #
        # Review schedule state (one entry per DL layer)
        # ------------------------------------------------------------------ #
        self._review_enabled = False
        self._p: List[float] = [0.0] * num_dl_layers
        self._p_max = 0.8
        self._p_delta = 0.2

        self._rng = random.Random(0)

    # ---------------------------------------------------------------------- #
    # Properties / helpers
    # ---------------------------------------------------------------------- #
    @property
    def num_dl_layers(self) -> int:
        return len(self.discrimination_layers)

    def is_stable(self, layer_idx: int) -> bool:
        """Return True if DL layer `layer_idx` is currently stable."""
        return self.monitors[layer_idx].state == "stable"

    # ---------------------------------------------------------------------- #
    # Organize (stability-gated)
    # ---------------------------------------------------------------------- #
    def organize(self) -> None:
        """
        Update DL weights based on accumulated activity.

        Layer 0 always organizes.
        Layer i (i > 0) only organizes when layer i-1 is stable.

        After organizing each layer, its WeightDriftMonitor is stepped and the
        cache is cleared if the layer just became unstable.
        """
        for i, dl in enumerate(self.discrimination_layers):
            # Gate: only organize layer i if layer i-1 is stable (or it is layer 0)
            if i > 0 and not self.is_stable(i - 1):
                continue

            dl.organize()

            prev_state = self.monitors[i].state
            self.monitors[i].step(dl.neuron_weights)

            if prev_state == "stable" and self.monitors[i].state == "unstable":
                self.activation_caches[i].clear()

    # ---------------------------------------------------------------------- #
    # Review mechanism
    # ---------------------------------------------------------------------- #
    def enable_review(
        self,
        enabled: bool = True,
        p0: float = 0.0,
        pmax: float = 0.8,
        delta: float = 0.2,
    ) -> None:
        """Enable/disable per-layer review."""
        self._review_enabled = bool(enabled)
        self._p = [float(p0 if enabled else 0.0)] * self.num_dl_layers
        self._p_max = float(pmax)
        self._p_delta = float(delta)

    def review_after_organize(self) -> None:
        """
        Call after organize(). Adjusts per-layer review ratio based on stability.
          - first stable block:     p[i] = 0
          - continued stable block: p[i] += delta (up to p_max)
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

    def review_sample(
        self,
        layer_idx: int,
        batch_size: int,
        device: torch.device,
        out_dtype: torch.dtype = torch.float32,
    ):
        """Sample a batch of (activations, labels) from layer `layer_idx`'s cache."""
        return self.activation_caches[layer_idx].sample_batch(
            batch_size, device=device, out_dtype=out_dtype
        )

    # ---------------------------------------------------------------------- #
    # Save / load
    # ---------------------------------------------------------------------- #
    def save_model(self, path: str, include_training_state: bool = False) -> None:
        state = {
            "layer_dims": self.layer_dims,
            "discrimination_layers": [
                dl.save_state(include_training_state)
                for dl in self.discrimination_layers
            ],
            "readout_head": self.readout_head.state_dict(),
            "meta": {
                "version": "v-StackedBiologicalClassifier",
                "include_training_state": include_training_state,
            },
        }
        torch.save(state, path)

    def load_model(
        self,
        path: str,
        include_training_state: bool = False,
        map_location: Optional[str] = None,
    ) -> None:
        if map_location is None:
            map_location = str(next(self.parameters()).device)

        state = torch.load(path, map_location=map_location)
        for i, dl in enumerate(self.discrimination_layers):
            dl.load_state(state["discrimination_layers"][i], include_training_state)
        self.readout_head.load_state_dict(state["readout_head"])

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        record_cache: bool = False,
    ) -> torch.Tensor:
        """
        Full forward pass.
          - Transform input once.
          - Pass through all DL layers (no_grad).
          - Optionally cache (activation, label) per layer when that layer is stable.
          - ReadoutHead is trainable and outputs logits.
        """
        x = self.transform(x)

        with torch.no_grad():
            for i, dl in enumerate(self.discrimination_layers):
                x = dl(x)
                if (
                    record_cache
                    and self._review_enabled
                    and y is not None
                    and self.is_stable(i)
                ):
                    self.activation_caches[i].add(x.detach(), y.detach())

        return self.readout_head(x)

    def forward_with_activations(
        self, layer_idx: int, cached_acts: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass starting from cached activations at `layer_idx`.

        Runs cached_acts through DL layers layer_idx+1 .. N-1 (no_grad),
        then through the ReadoutHead.  Used for per-layer review mini-steps.

        Args:
            layer_idx: The layer whose cache was sampled (0-indexed).
            cached_acts: Tensor of shape [B, layer_dims[layer_idx+1]].

        Returns:
            Logits of shape [B, output_dim].
        """
        x = cached_acts
        with torch.no_grad():
            for i in range(layer_idx + 1, self.num_dl_layers):
                x = self.discrimination_layers[i](x)
        return self.readout_head(x)
