import unittest

import torch
from torch import nn

from core.initializers import (
    DatasetInitializerWhole,
    RandomInitializer,
    apply_non_negative_transform,
)
from core.weight_monitor import EffectiveOperatorDriftMonitor
from models.biological_classifier import BiologicalClassifier
from models.discrimination import DiscriminationLayer
from models.optimizer import L1NormalizedActivityOptimizer, IterativeActivityOptimizer


class OptimizerRouteTest(unittest.TestCase):
    def test_default_connection_schedule_uses_verified_stepwise_profile(self):
        layer = DiscriminationLayer(in_dim=4, out_dim=10)

        expected = {0: 0.50, 4: 0.40, 5: 0.30, 6: 0.20}
        self.assertEqual(layer.connection_ratio_schedule, expected)
        self.assertAlmostEqual(layer.connection_ratio_for_organize(0), 0.50)
        self.assertAlmostEqual(layer.connection_ratio_for_organize(3), 0.50)
        self.assertAlmostEqual(layer.connection_ratio_for_organize(4), 0.40)
        self.assertAlmostEqual(layer.connection_ratio_for_organize(5), 0.30)
        self.assertAlmostEqual(layer.connection_ratio_for_organize(100), 0.20)

    def test_sparse_mask_keeps_directed_top_k_neighbors_and_self_diagonal(self):
        layer = DiscriminationLayer(
            in_dim=4,
            out_dim=10,
            connection_ratio_schedule={0: 0.20},
        )

        mask = layer.lateral_connection_mask
        torch.testing.assert_close(mask.diagonal(), torch.ones(10, dtype=torch.bool))
        self.assertTrue(torch.equal(mask.sum(dim=0), torch.full((10,), 3)))
        torch.testing.assert_close(
            layer.effective_lateral_matrix,
            layer.neuron_correlation_matrix * mask,
        )

    def test_variance_dense_rejects_sparse_connection_schedule(self):
        with self.assertRaisesRegex(ValueError, "only supported by the l1_sparse"):
            DiscriminationLayer(
                in_dim=4,
                out_dim=8,
                optimizer_profile="variance_dense",
                connection_ratio_schedule={0: 0.20},
            )

    def test_l1_sparse_rejects_legacy_diagonal_start_option(self):
        with self.assertRaisesRegex(ValueError, "not used by the l1_sparse"):
            DiscriminationLayer(
                in_dim=4,
                out_dim=8,
                optimizer_y0_divide_by_diagonal=True,
            )

    def test_effective_operator_monitor_uses_relative_frobenius_hysteresis(self):
        monitor = EffectiveOperatorDriftMonitor(
            tau_enter=0.01,
            tau_exit=0.02,
            k_enter=2,
        )
        baseline = torch.eye(4)

        self.assertFalse(monitor.step(baseline)[0])
        self.assertFalse(monitor.step(baseline * 1.005)[0])
        stable, drift = monitor.step(baseline * 1.009)
        self.assertTrue(stable)
        self.assertLess(drift, 0.01)

        stable, drift = monitor.step(baseline * 1.04)
        self.assertFalse(stable)
        self.assertGreater(drift, 0.02)

    def test_exact_eigenvalue_recovers_largest_eigenvalue(self):
        matrix = torch.diag(torch.tensor([1.0, 2.0, 4.0]))
        optimizer = IterativeActivityOptimizer()

        self.assertAlmostEqual(
            float(optimizer.update_cached_gain(matrix)), 4.0, places=6
        )

    def test_initial_and_later_support_gates_are_distinct(self):
        optimizer = IterativeActivityOptimizer(
            support_threshold=0.08,
            initial_support_threshold=0.05,
            initial_max_iters=20000,
            max_iters=1000,
        )

        optimizer.set_initial_phase(True)
        self.assertEqual(optimizer._iteration_limit(), 20000)
        self.assertAlmostEqual(optimizer._support_limit(), 0.05)
        optimizer.set_initial_phase(False)
        self.assertEqual(optimizer._iteration_limit(), 1000)
        self.assertAlmostEqual(optimizer._support_limit(), 0.08)

    def test_discrimination_defaults_to_final_profile(self):
        layer = DiscriminationLayer(in_dim=4, out_dim=8)
        optimizer = layer.activity_optimizer

        self.assertIsInstance(layer.activation, nn.ReLU)
        self.assertEqual(layer.optimizer_profile, "l1_sparse")
        self.assertIsInstance(optimizer, L1NormalizedActivityOptimizer)
        self.assertEqual(layer.non_negative_strategy, "abs")
        self.assertEqual(layer.weight_norm_p, 2)
        self.assertFalse(optimizer.y0_divide_by_diagonal)
        self.assertEqual(optimizer.initial_support_threshold, 0.08)
        self.assertEqual(optimizer.support_threshold, 0.08)
        self.assertEqual(optimizer.initial_max_iters, 1000)

    def test_non_negative_transforms_remain_selectable(self):
        weights = torch.tensor([[-2.0, 1.0], [3.0, -4.0]])

        torch.testing.assert_close(
            apply_non_negative_transform(weights, strategy="abs"),
            torch.tensor([[2.0, 1.0], [3.0, 4.0]]),
        )
        torch.testing.assert_close(
            apply_non_negative_transform(weights, strategy="shift"),
            torch.tensor([[2.0, 5.0], [7.0, 0.0]]),
        )
        torch.testing.assert_close(
            apply_non_negative_transform(weights, strategy="relu"),
            torch.tensor([[0.0, 1.0], [3.0, 0.0]]),
        )

    def test_dataset_initializer_uses_audited_defaults(self):
        initializer = DatasetInitializerWhole(dataset=[], transform=lambda value: value)

        self.assertEqual(initializer.non_negative_strategy, "abs")
        self.assertFalse(initializer.mid_matrix)
        self.assertEqual(
            initializer.configuration(),
            {
                "type": "DatasetInitializerWhole",
                "init_ratio": 0.25,
                "non_negative": True,
                "non_negative_strategy": "abs",
                "mid_matrix": False,
            },
        )

    def test_variance_dense_l1_weights_force_diagonal_scaled_start(self):
        layer = DiscriminationLayer(
            in_dim=4,
            out_dim=8,
            weight_norm_p=1,
            optimizer_profile="variance_dense",
            optimizer_y0_divide_by_diagonal=False,
        )

        self.assertTrue(layer.activity_optimizer.y0_divide_by_diagonal)

    def test_model_records_resolved_initialization_profile(self):
        initializer = RandomInitializer(non_negative_strategy="shift")
        model = BiologicalClassifier(
            input_dim=4,
            hidden_dim=8,
            data_initializer=initializer,
            discrimination_config={"non_negative_strategy": "relu"},
        )

        self.assertEqual(model.discrimination_config["non_negative_strategy"], "shift")
        self.assertEqual(model.discrimination_config["weight_norm_p"], 2)
        self.assertEqual(model.discrimination_config["optimizer_profile"], "l1_sparse")
        self.assertEqual(
            model.discrimination_config["connection_ratio_schedule"],
            {0: 0.50, 4: 0.40, 5: 0.30, 6: 0.20},
        )
        self.assertEqual(
            model.discrimination_config["optimizer_support_threshold"], 0.08
        )
        self.assertNotIn(
            "optimizer_y0_divide_by_diagonal", model.discrimination_config
        )
        self.assertNotIn("optimizer_variance_stop_window", model.discrimination_config)
        self.assertEqual(model.initializer_config, initializer.configuration())

    def test_clean_optimizer_matches_verified_numeric_behavior(self):
        correlation = torch.tensor(
            [
                [1.0, 0.82, 0.35, 0.10],
                [0.82, 1.0, 0.25, 0.08],
                [0.35, 0.25, 1.0, 0.55],
                [0.10, 0.08, 0.55, 1.0],
            ]
        )
        raw_activity = torch.tensor(
            [[3.2, 2.7, 1.1, 0.4], [1.0, 0.8, 2.9, 2.1]]
        )
        optimizer = IterativeActivityOptimizer(
            max_iters=1000,
        )
        optimizer.update_cached_gain(correlation)
        optimizer.set_initial_phase(False)

        actual = optimizer(raw_activity, correlation)
        expected = torch.tensor(
            [
                [3.0012323856, 0.2324874848, 0.0, 0.0812777653],
                [0.0, 0.1250384748, 2.4648630619, 0.7343221903],
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(optimizer.last_completed_iters, 1000)

    def test_l1_optimizer_restores_mean_raw_drive_and_stops_on_support(self):
        correlation = torch.tensor(
            [
                [1.0, 0.90, 0.80, 0.70],
                [0.90, 1.0, 0.60, 0.50],
                [0.80, 0.60, 1.0, 0.40],
                [0.70, 0.50, 0.40, 1.0],
            ]
        )
        raw_activity = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        optimizer = L1NormalizedActivityOptimizer(
            max_iters=100,
            support_threshold=0.50,
            initial_support_threshold=0.50,
            initial_max_iters=100,
        )
        optimizer.update_cached_gain(correlation)
        optimizer.set_initial_phase(False)

        actual = optimizer(raw_activity, correlation)

        torch.testing.assert_close(
            actual.sum(dim=1), raw_activity.abs().mean(dim=1)
        )
        self.assertLessEqual(float((actual > 0.0).float().mean()), 0.50)
        self.assertTrue(optimizer.last_stop_triggered)
        self.assertLess(optimizer.last_completed_iters, 100)

    def test_discrimination_can_select_legacy_variance_dense_profile(self):
        layer = DiscriminationLayer(
            in_dim=4,
            out_dim=8,
            optimizer_profile="variance_dense",
        )

        self.assertEqual(layer.optimizer_profile, "variance_dense")
        self.assertIsNone(layer.connection_ratio_schedule)
        self.assertIsInstance(layer.activity_optimizer, IterativeActivityOptimizer)
        self.assertNotIsInstance(
            layer.activity_optimizer, L1NormalizedActivityOptimizer
        )
        self.assertEqual(
            layer.activity_optimizer.initial_support_threshold, 0.05
        )
        self.assertEqual(layer.activity_optimizer.initial_max_iters, 20000)
        self.assertTrue(layer.lateral_connection_mask.all())


if __name__ == "__main__":
    unittest.main()
