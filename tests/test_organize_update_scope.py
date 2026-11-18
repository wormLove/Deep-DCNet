import unittest

import torch

from models.biological_classifier import BiologicalClassifier
from models.discrimination import DiscriminationLayer


IN_DIM = 16
OUT_DIM = 50


def _layer(scope: str, seed: int = 0) -> DiscriminationLayer:
    torch.manual_seed(seed)
    return DiscriminationLayer(
        in_dim=IN_DIM,
        out_dim=OUT_DIM,
        organize_update_scope=scope,
    )


def _cycle_inputs(active, seed: int = 1, batch: int = 4):
    """Inputs plus an activity pattern in which only `active` neurons fire."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, IN_DIM, generator=generator)
    act = torch.zeros(batch, OUT_DIM)
    act[:, active] = torch.rand(batch, len(active), generator=generator) + 0.1
    return x, act


def _feed(layer: DiscriminationLayer, x: torch.Tensor, act: torch.Tensor) -> None:
    # The two calls DiscriminationLayer.forward makes in training mode, with the
    # activity fixed so the test controls exactly which neurons fire.
    layer.organizer.step(x, act)
    layer.stats.cache_activity(act)


class OrganizeUpdateScopeTest(unittest.TestCase):
    def test_default_scope_is_activated_and_recorded(self):
        layer = DiscriminationLayer(in_dim=4, out_dim=8)
        model = BiologicalClassifier(input_dim=4, hidden_dim=8)

        self.assertEqual(layer.organize_update_scope, "activated")
        self.assertEqual(
            model.discrimination_config["organize_update_scope"], "activated"
        )

    def test_rejects_unknown_scope(self):
        with self.assertRaisesRegex(ValueError, "organize_update_scope"):
            DiscriminationLayer(in_dim=4, out_dim=8, organize_update_scope="all")

    def test_update_mask_is_diagnostic_only(self):
        layer = _layer("activated")

        self.assertNotIn("last_update_mask", layer.state_dict())

    def test_activated_scope_leaves_silent_neurons_bit_identical(self):
        layer = _layer("activated")
        before = layer.neuron_weights.detach().clone()
        active = [0, 3, 7, 12, 21]
        _feed(layer, *_cycle_inputs(active))

        layer.organize()

        hits = layer.last_cycle_hits
        expected_hits = torch.zeros(OUT_DIM, dtype=torch.bool)
        expected_hits[active] = True
        self.assertTrue(torch.equal(hits, expected_hits))
        self.assertTrue(torch.equal(layer.last_update_mask, expected_hits))
        self.assertTrue(
            torch.equal(layer.diagnostics()["update_mask"], expected_hits)
        )
        weights = layer.neuron_weights.detach()
        self.assertTrue(torch.equal(weights[:, ~hits], before[:, ~hits]))
        self.assertTrue((weights[:, hits] != before[:, hits]).any(dim=0).all())

    def test_first_organize_matches_global_scope_on_fired_neurons(self):
        activated = _layer("activated")
        global_scope = _layer("global")
        before = activated.neuron_weights.detach().clone()
        self.assertTrue(torch.equal(before, global_scope.neuron_weights))
        cycle = _cycle_inputs([1, 4, 9, 30, 44])
        _feed(activated, *cycle)
        _feed(global_scope, *cycle)

        activated.organize()
        global_scope.organize()

        fired = activated.last_cycle_hits
        a_weights = activated.neuron_weights.detach()
        g_weights = global_scope.neuron_weights.detach()
        self.assertTrue(torch.equal(a_weights[:, fired], g_weights[:, fired]))
        # A neuron silent since initialization holds zero potential, so the global
        # rule only renormalizes it: equal up to rounding, not bit-identical.
        torch.testing.assert_close(g_weights[:, ~fired], before[:, ~fired])
        self.assertTrue(torch.equal(a_weights[:, ~fired], before[:, ~fired]))

    def test_gate_leaves_learning_rates_potentials_and_beta_accum_unchanged(self):
        activated = _layer("activated")
        global_scope = _layer("global")
        cycle = _cycle_inputs([2, 5, 8])
        _feed(activated, *cycle)
        _feed(global_scope, *cycle)

        activated.organize()
        global_scope.organize()

        for name in ("lr_vec", "potential_hebb", "potential_antihebb", "beta_accum"):
            self.assertTrue(
                torch.equal(
                    getattr(activated.organizer, name),
                    getattr(global_scope.organizer, name),
                ),
                name,
            )
        self.assertTrue(torch.equal(activated.last_lr_vec, global_scope.last_lr_vec))

    def test_organize_does_not_consume_potential(self):
        for scope in ("activated", "global"):
            layer = _layer(scope)
            _feed(layer, *_cycle_inputs([0, 1, 2]))
            hebb = layer.organizer.potential_hebb.clone()
            antihebb = layer.organizer.potential_antihebb.clone()

            layer.organize()

            self.assertTrue(torch.equal(layer.organizer.potential_hebb, hebb), scope)
            self.assertTrue(
                torch.equal(layer.organizer.potential_antihebb, antihebb), scope
            )

    def test_silent_neuron_with_stale_potential_changes_only_under_global(self):
        silent_tonight = 6
        results = {}
        for scope in ("activated", "global"):
            layer = _layer(scope)
            _feed(layer, *_cycle_inputs([silent_tonight, 10, 11], seed=2))
            layer.organize()
            after_first = layer.neuron_weights.detach().clone()
            _feed(layer, *_cycle_inputs([10, 11, 15], seed=3))
            layer.organize()
            results[scope] = (
                after_first[:, silent_tonight],
                layer.neuron_weights.detach()[:, silent_tonight],
            )

        before, after = results["activated"]
        self.assertTrue(torch.equal(after, before))
        before, after = results["global"]
        self.assertFalse(torch.allclose(after, before, atol=1e-4))

    def test_update_mask_covers_all_neurons_under_global(self):
        layer = _layer("global")
        _feed(layer, *_cycle_inputs([0, 1]))

        layer.organize()

        self.assertTrue(layer.last_update_mask.all())

    def test_forward_cycle_gates_on_observed_activity(self):
        layer = _layer("activated")
        layer.train()
        before = layer.neuron_weights.detach().clone()
        generator = torch.Generator().manual_seed(4)
        x = torch.rand(8, IN_DIM, generator=generator)
        for batch in x.split(4):
            layer(batch)

        layer.organize()

        hits = layer.last_cycle_hits
        self.assertTrue((~hits).any())
        self.assertTrue(torch.equal(layer.last_update_mask, hits))
        self.assertTrue(
            torch.equal(layer.neuron_weights.detach()[:, ~hits], before[:, ~hits])
        )

    def test_reset_clears_update_mask(self):
        layer = _layer("activated")
        _feed(layer, *_cycle_inputs([0, 1]))
        layer.organize()
        self.assertTrue(layer.last_update_mask.any())

        layer.reset_organizer_state()

        self.assertFalse(layer.last_update_mask.any())


if __name__ == "__main__":
    unittest.main()
