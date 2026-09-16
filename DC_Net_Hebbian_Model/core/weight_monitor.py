import torch


class WeightDriftMonitor:
    """
    Track how much a weight matrix changes in direction over time.

    We compare the current weight matrix W to an exponential moving average (EMA) of W.
    For each column (neuron), drift is defined as:
        drift_j = 1 - cosine_similarity(w_j, ema_j)
    The reported drift is the mean over columns.

    A small drift for multiple consecutive steps indicates stability.
    """

    def __init__(
        self,
        name: str = "Monitor",
        alpha: float = 0.3,
        tau_enter: float = 0.3,
        tau_exit: float = 0.3,
        k_enter: int = 2,
        assume_unit_columns: bool = True,
    ):
        self.name = name
        self.alpha = float(alpha)
        self.tau_enter = float(tau_enter)
        self.tau_exit = float(tau_exit)
        self.k_enter = int(k_enter)
        self.assume_unit_columns = bool(assume_unit_columns)

        self.W_ema = None
        self.state = "cold"  # "cold" | "stable" | "unstable"
        self.stable_count = 0
        self.last_drift = -1.0  # <0 means "uninitialized"
        self.state_changed = False

    @torch.no_grad()
    def step(self, W: torch.Tensor):
        """
        Update monitor state using current weights W.

        Args:
            W: Tensor of shape [in_dim, out_dim].

        Returns:
            (is_stable: bool, drift: float)
        """
        self.state_changed = False
        W = W.detach()

        if self.W_ema is None:
            self.W_ema = W.clone()
            self.last_drift = -1.0
            return (False, self.last_drift)

        if self.assume_unit_columns:
            # If columns are L2-normalized, dot product equals cosine similarity.
            cos = (W * self.W_ema).sum(dim=0)
        else:
            # Safe cosine similarity per column.
            eps = 1e-12
            num = (W * self.W_ema).sum(dim=0)
            den = (W.norm(dim=0) * self.W_ema.norm(dim=0)).clamp_min(eps)
            cos = num / den

        cos = cos.clamp(-1.0, 1.0)
        drift_t = 1.0 - cos
        drift_t = torch.nan_to_num(drift_t, nan=1.0, posinf=1.0, neginf=1.0)

        drift = float(drift_t.mean().item())
        self.last_drift = drift

        # EMA update
        self.W_ema = (1.0 - self.alpha) * self.W_ema + self.alpha * W

        # Patience counter for entering stable state
        if drift < self.tau_enter:
            self.stable_count += 1
        else:
            self.stable_count = 0

        prev_state = self.state
        if self.state != "stable" and self.stable_count >= self.k_enter:
            self.state = "stable"
        elif self.state == "stable" and drift > self.tau_exit:
            self.state = "unstable"
            self.stable_count = 0

        self.state_changed = (self.state != prev_state)
        return (self.state == "stable", drift)

    def reset(self) -> None:
        """Reset internal EMA and state machine."""
        self.W_ema = None
        self.state = "cold"
        self.stable_count = 0
        self.last_drift = -1.0
        self.state_changed = False

    def __str__(self) -> str:
        if self.last_drift < 0.0:
            return f"{self.name}: state={self.state}, drift=uninitialized"
        return f"{self.name}: state={self.state}, drift={self.last_drift:.6f}"





