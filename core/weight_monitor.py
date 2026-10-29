import torch


class EffectiveOperatorDriftMonitor:
    """Gate review using consecutive effective lateral operators.

    Relative Frobenius drift observes both continuous correlation changes and
    discrete sparse-mask edge changes.  Hysteresis keeps review from toggling
    when the operator sits close to the stability boundary.
    """

    def __init__(
        self,
        name: str = "EffectiveOperator",
        tau_enter: float = 0.01,
        tau_exit: float = 0.02,
        k_enter: int = 2,
    ):
        if not 0.0 <= tau_enter <= tau_exit:
            raise ValueError("Expected 0 <= tau_enter <= tau_exit.")
        if k_enter <= 0:
            raise ValueError("k_enter must be positive.")
        self.name = name
        self.tau_enter = float(tau_enter)
        self.tau_exit = float(tau_exit)
        self.k_enter = int(k_enter)
        self.previous = None
        self.state = "cold"
        self.stable_count = 0
        self.last_drift = -1.0
        self.state_changed = False

    @torch.no_grad()
    def step(self, operator: torch.Tensor):
        self.state_changed = False
        current = operator.detach()
        if self.previous is None:
            self.previous = current.clone()
            self.last_drift = -1.0
            return False, self.last_drift

        denominator = self.previous.norm().clamp_min(1e-12)
        drift = float((current - self.previous).norm().div(denominator).item())
        self.previous = current.clone()
        self.last_drift = drift

        if drift < self.tau_enter:
            self.stable_count += 1
        else:
            self.stable_count = 0

        previous_state = self.state
        if self.state != "stable" and self.stable_count >= self.k_enter:
            self.state = "stable"
        elif self.state == "stable" and drift > self.tau_exit:
            self.state = "unstable"
            self.stable_count = 0

        self.state_changed = self.state != previous_state
        return self.state == "stable", drift

    def reset(self) -> None:
        self.previous = None
        self.state = "cold"
        self.stable_count = 0
        self.last_drift = -1.0
        self.state_changed = False
