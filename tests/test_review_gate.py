import tempfile
import unittest
from pathlib import Path

import torch

from core.weight_monitor import EffectiveOperatorDriftMonitor, PlateauDriftMonitor
from models.biological_classifier import BiologicalClassifier


# Falling, then flat: with window 3 the plateau test first passes at the 9th drift.
FALLING_THEN_FLAT = [0.08, 0.04, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]


def _operators(drifts, size: int = 4):
    """Operators whose consecutive relative drifts are the given values."""
    base = torch.rand(size, size, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    scale = 1.0
    operators = [base.clone()]
    for drift in drifts:
        scale *= 1.0 + drift
        operators.append(base * scale)
    return operators


def _states(monitor, operators):
    return [(monitor.step(operator), monitor.state)[1] for operator in operators]


class PlateauDriftMonitorTest(unittest.TestCase):
    def test_both_gates_measure_the_same_drift(self):
        generator = torch.Generator().manual_seed(1)
        plateau, absolute = PlateauDriftMonitor(), EffectiveOperatorDriftMonitor()

        for _ in range(6):
            operator = torch.rand(5, 5, generator=generator)
            plateau.step(operator)
            absolute.step(operator)
            self.assertEqual(plateau.last_drift, absolute.last_drift)

    def test_opens_after_k_enter_passes_once_the_drift_stops_falling(self):
        monitor = PlateauDriftMonitor(window=3, allowed_decrease=0.1, k_enter=2)

        states = _states(monitor, _operators(FALLING_THEN_FLAT))

        # states[i] follows the i-th drift (states[0]: first operator, no drift yet).
        self.assertEqual(states[:10], ["cold"] * 10)
        self.assertEqual(states[10], "stable")
        self.assertAlmostEqual(monitor.reference_drift, 0.005)

    def test_opening_does_not_depend_on_the_drift_level(self):
        for level in (1.0, 20.0):
            monitor = PlateauDriftMonitor(window=3, allowed_decrease=0.1, k_enter=2)

            states = _states(monitor, _operators([level * d for d in FALLING_THEN_FLAT]))

            self.assertEqual(states.index("stable"), 10, level)

    def test_closes_above_exit_ratio_times_the_reference(self):
        monitor = PlateauDriftMonitor(window=3, allowed_decrease=0.1, k_enter=2, exit_ratio=2.0)
        operators = _operators(FALLING_THEN_FLAT + [0.009, 0.011])

        states = _states(monitor, operators)

        self.assertEqual(states[-2], "stable")
        self.assertEqual(states[-1], "unstable")
        self.assertTrue(monitor.state_changed)

    def test_a_rising_drift_is_not_a_plateau(self):
        monitor = PlateauDriftMonitor(window=3, allowed_decrease=0.1, k_enter=2)

        # Falling, then doubling: a one-sided "not falling" test would open at the 8th drift.
        states = _states(monitor, _operators([0.08, 0.04, 0.02, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]))

        self.assertNotIn("stable", states)

    def test_does_not_reopen_while_the_drift_is_still_rising(self):
        monitor = PlateauDriftMonitor(window=3, allowed_decrease=0.1, k_enter=2, exit_ratio=2.0)

        states = _states(monitor, _operators(FALLING_THEN_FLAT + [0.011, 0.02, 0.04, 0.08]))

        self.assertEqual(states[10], "stable")
        self.assertEqual(states[11:], ["unstable"] * 4)

    def test_reset_clears_the_drift_history(self):
        monitor = PlateauDriftMonitor(window=3)
        _states(monitor, _operators(FALLING_THEN_FLAT))

        monitor.reset()

        self.assertEqual(monitor.state, "cold")
        self.assertEqual(len(monitor.drift_history), 0)
        self.assertIsNone(monitor.reference_drift)
        self.assertIsNone(monitor.previous)


class ClassifierReviewGateTest(unittest.TestCase):
    def test_plateau_gate_is_the_default(self):
        model = BiologicalClassifier(input_dim=6, hidden_dim=5, output_dim=3)

        self.assertIsInstance(model.monitor_last_dl, PlateauDriftMonitor)
        self.assertEqual(
            model.review_gate_config,
            {"review_gate": "plateau", "window": 10, "allowed_decrease": 0.1, "k_enter": 2, "exit_ratio": 2.0},
        )

    def test_absolute_gate_is_the_previous_monitor(self):
        model = BiologicalClassifier(input_dim=6, hidden_dim=5, output_dim=3, review_gate="absolute")

        self.assertIsInstance(model.monitor_last_dl, EffectiveOperatorDriftMonitor)
        self.assertEqual(
            model.review_gate_config,
            {"review_gate": "absolute", "tau_enter": 0.01, "tau_exit": 0.02, "k_enter": 2},
        )

    def test_gate_settings_are_passed_and_saved(self):
        model = BiologicalClassifier(
            input_dim=6, hidden_dim=5, output_dim=3, review_gate_config={"window": 25}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save_model(str(path))
            meta = torch.load(path)["meta"]

        self.assertEqual(model.monitor_last_dl.window, 25)
        self.assertEqual(meta["review_gate_config"], model.review_gate_config)

    def test_unknown_gate_is_rejected(self):
        with self.assertRaises(ValueError):
            BiologicalClassifier(input_dim=6, hidden_dim=5, review_gate="drift")


if __name__ == "__main__":
    unittest.main()
