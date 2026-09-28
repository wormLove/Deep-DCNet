import unittest

import torch
from torch import nn

from models.discrimination import DiscriminationLayer
from models.optimizer import IterativeActivityOptimizer


class OptimizerRouteTest(unittest.TestCase):
    def test_exact_eigenvalue_recovers_largest_eigenvalue(self):
        matrix = torch.diag(torch.tensor([1.0, 2.0, 4.0]))
        optimizer = IterativeActivityOptimizer()

        self.assertAlmostEqual(
            float(optimizer.update_cached_gain(matrix)), 4.0, places=6
        )

    def test_initial_and_later_variance_gates_are_distinct(self):
        optimizer = IterativeActivityOptimizer(
            variance_stop_nonzero_ratio=0.08,
            variance_stop_initial_nonzero_ratio=0.05,
            variance_stop_initial_max_iters=20000,
            max_iters=1000,
        )

        optimizer.set_variance_stop_initial_phase(True)
        self.assertEqual(optimizer._iteration_limit(), 20000)
        self.assertAlmostEqual(optimizer._variance_nonzero_ratio_limit(), 0.05)
        optimizer.set_variance_stop_initial_phase(False)
        self.assertEqual(optimizer._iteration_limit(), 1000)
        self.assertAlmostEqual(optimizer._variance_nonzero_ratio_limit(), 0.08)

    def test_discrimination_defaults_to_final_profile(self):
        layer = DiscriminationLayer(in_dim=4, out_dim=8)
        optimizer = layer.activity_optimizer

        self.assertIsInstance(layer.activation, nn.ReLU)
        self.assertEqual(optimizer.variance_stop_initial_nonzero_ratio, 0.05)
        self.assertEqual(optimizer.variance_stop_nonzero_ratio, 0.08)

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
        optimizer.set_variance_stop_initial_phase(False)

        actual = optimizer(raw_activity, correlation)
        expected = torch.tensor(
            [
                [3.0012323856, 0.2324874848, 0.0, 0.0812777653],
                [0.0, 0.1250384748, 2.4648630619, 0.7343221903],
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(optimizer.last_completed_iters, 1000)


if __name__ == "__main__":
    unittest.main()
