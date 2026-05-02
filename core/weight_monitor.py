import torch


class WeightDriftMonitor:
    """
    Monitor the drift of a weight matrix and gate review when the
    discrimination representation becomes stable.
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
        self.state = "cold"
        self.stable_count = 0
        self.last_drift = -1.0
        self.state_changed = False

    @torch.no_grad()
    def step(self, W: torch.Tensor):
        self.state_changed = False
        W = W.detach()

        if self.W_ema is None:
            self.W_ema = W.clone()
            self.last_drift = -1.0
            return False, self.last_drift

        if self.assume_unit_columns:
            cos = (W * self.W_ema).sum(dim=0)
        else:
            eps = 1e-12
            num = (W * self.W_ema).sum(dim=0)
            den = (W.norm(dim=0) * (self.W_ema.norm(dim=0) + eps) + eps)
            cos = num / den

        cos = cos.clamp(-1.0, 1.0)
        drift_t = 1.0 - cos
        drift_t = torch.nan_to_num(drift_t, nan=1.0, posinf=1.0, neginf=1.0)

        drift = drift_t.mean().item()
        self.last_drift = drift
        self.W_ema = (1.0 - self.alpha) * self.W_ema + self.alpha * W

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

        self.state_changed = self.state != prev_state
        return self.state == "stable", drift

    def reset(self) -> None:
        self.W_ema = None
        self.state = "cold"
        self.stable_count = 0
        self.last_drift = -1.0
        self.state_changed = False
