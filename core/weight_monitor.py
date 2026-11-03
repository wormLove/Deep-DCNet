import statistics
from collections import deque

import torch


def relative_drift(previous: torch.Tensor, current: torch.Tensor) -> float:
    """Relative Frobenius change between two consecutive operators."""
    denominator = previous.norm().clamp_min(1e-12)
    return float((current - previous).norm().div(denominator).item())


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

        drift = relative_drift(self.previous, current)
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


class PlateauDriftMonitor:
    """Gate review once the drift has levelled off, whatever its level.

    The drift is EffectiveOperatorDriftMonitor's. Its level is set by the organizer's
    step size and the lateral mask's boundary swaps, not by how stale cached activity
    is, so only its trend is used. After each organize the plateau test passes if the
    median drift over the last ``window`` organizes is within ``allowed_decrease``
    (relative, either way) of the median over the ``window`` organizes before: a drift
    still falling or rising beyond that is not a plateau. Review becomes stable after
    ``k_enter`` consecutive passes; the median of the last window at that organize
    becomes the reference level, and a stable organize whose drift exceeds
    ``exit_ratio`` times the reference makes it unstable.
    """

    def __init__(
        self,
        name: str = "EffectiveOperator",
        window: int = 10,
        allowed_decrease: float = 0.1,
        k_enter: int = 2,
        exit_ratio: float = 2.0,
    ):
        if window <= 0 or k_enter <= 0:
            raise ValueError("window and k_enter must be positive.")
        if not 0.0 <= allowed_decrease < 1.0 or exit_ratio < 1.0:
            raise ValueError("Expected 0 <= allowed_decrease < 1 and exit_ratio >= 1.")
        self.name = name
        self.window = int(window)
        self.allowed_decrease = float(allowed_decrease)
        self.k_enter = int(k_enter)
        self.exit_ratio = float(exit_ratio)
        self.reset()

    @torch.no_grad()
    def step(self, operator: torch.Tensor):
        self.state_changed = False
        current = operator.detach()
        if self.previous is None:
            self.previous = current.clone()
            self.last_drift = -1.0
            return False, self.last_drift

        drift = relative_drift(self.previous, current)
        self.previous = current.clone()
        self.last_drift = drift
        self.drift_history.append(drift)

        recent = None
        passed = False
        if len(self.drift_history) == self.drift_history.maxlen:
            history = list(self.drift_history)
            recent = statistics.median(history[self.window :])
            before = statistics.median(history[: self.window])
            passed = (1.0 - self.allowed_decrease) * before <= recent <= (1.0 + self.allowed_decrease) * before
        self.stable_count = self.stable_count + 1 if passed else 0

        previous_state = self.state
        if self.state != "stable" and self.stable_count >= self.k_enter:
            self.state = "stable"
            self.reference_drift = recent
        elif self.state == "stable" and drift > self.exit_ratio * self.reference_drift:
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
        self.drift_history = deque(maxlen=2 * self.window)
        self.reference_drift = None
