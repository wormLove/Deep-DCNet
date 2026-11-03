import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from core.organizer import DiscriminationOrganizer
from models.biological_classifier import BiologicalClassifier
from models.discrimination import DiscriminationLayer


IN_DIM = 12
OUT_DIM = 7


def _organizer(lr: float, seed: int = 0, weight_norm_p: int = 2, **kwargs):
    generator = torch.Generator().manual_seed(seed)
    organizer = DiscriminationOrganizer(IN_DIM, OUT_DIM, weight_norm_p=weight_norm_p, **kwargs)
    organizer.potential_hebb.copy_(torch.rand(IN_DIM, OUT_DIM, generator=generator))
    antihebb = torch.rand(OUT_DIM, OUT_DIM, generator=generator)
    organizer.potential_antihebb.copy_(0.1 * (antihebb + antihebb.T))
    organizer.set_learning_rates(torch.full((OUT_DIM,), lr))
    weights = F.normalize(torch.rand(IN_DIM, OUT_DIM, generator=generator), p=weight_norm_p, dim=0)
    return organizer, weights


def _residual(organizer: DiscriminationOrganizer, weights: torch.Tensor) -> torch.Tensor:
    return organizer.potential_hebb - weights @ organizer.potential_antihebb


class OrganizeDirectionTest(unittest.TestCase):
    def test_direction_is_the_default_step(self):
        self.assertEqual(DiscriminationOrganizer(IN_DIM, OUT_DIM).organize_step, "direction")
        layer = DiscriminationLayer(in_dim=IN_DIM, out_dim=OUT_DIM)
        self.assertEqual(layer.organizer.organize_step, "direction")

    def test_update_does_not_depend_on_potential_scale(self):
        organizer, weights = _organizer(lr=0.3)
        reference = organizer.organize(weights)

        for scale in (1e-6, 1e-2, 50.0):
            organizer.potential_hebb.mul_(scale)
            organizer.potential_antihebb.mul_(scale)
            torch.testing.assert_close(organizer.organize(weights), reference, msg=str(scale))
            organizer.potential_hebb.div_(scale)
            organizer.potential_antihebb.div_(scale)

    def test_full_rate_replaces_weights_with_the_residual_direction(self):
        organizer, weights = _organizer(lr=1.0)

        updated = organizer.organize(weights)

        expected = F.normalize(_residual(organizer, weights), dim=0)
        torch.testing.assert_close(updated, expected)

    def test_rate_mixes_weights_and_unit_direction(self):
        organizer, weights = _organizer(lr=0.25)

        updated = organizer.organize(weights)

        direction = F.normalize(_residual(organizer, weights), dim=0)
        torch.testing.assert_close(updated, 0.75 * weights + 0.25 * direction)
        torch.testing.assert_close(direction.norm(dim=0), torch.ones(OUT_DIM))

    def test_neuron_without_potential_keeps_its_direction(self):
        organizer, weights = _organizer(lr=0.6)
        organizer.potential_hebb[:, 2] = 0.0
        organizer.potential_antihebb[:, 2] = 0.0

        updated = organizer.organize(weights)

        self.assertTrue(torch.isfinite(updated).all())
        torch.testing.assert_close(updated[:, 2], 0.4 * weights[:, 2])

    def test_residual_parallel_to_weights_is_a_fixed_point(self):
        organizer, weights = _organizer(lr=0.7)
        organizer.potential_antihebb.zero_()
        organizer.potential_hebb.copy_(weights * torch.linspace(0.01, 3.0, OUT_DIM))

        updated = F.normalize(organizer.organize(weights), dim=0)

        torch.testing.assert_close(updated, weights)

    def test_direction_uses_the_weight_norm(self):
        organizer, weights = _organizer(lr=1.0, weight_norm_p=1)

        updated = organizer.organize(weights)

        torch.testing.assert_close(updated.abs().sum(dim=0), torch.ones(OUT_DIM))
        layer = DiscriminationLayer(in_dim=IN_DIM, out_dim=OUT_DIM, weight_norm_p=1)
        self.assertEqual(layer.organizer.weight_norm_p, 1)

    def test_direction_step_leaves_correction_state_alone(self):
        organizer, weights = _organizer(lr=0.3, correction_beta=0.9)

        organizer.organize(weights)
        organizer.organize(weights)

        self.assertTrue(torch.equal(organizer.beta_accum, torch.tensor(0.9)))

    def test_correction_factor_step_is_the_previous_rule(self):
        organizer, weights = _organizer(lr=0.3, correction_beta=0.9, organize_step="correction_factor")
        lr_col = organizer.lr_vec.unsqueeze(0)

        for accum in (0.9, 0.81):
            residual = _residual(organizer, weights)
            expected = (1.0 - lr_col) * weights + ((1.0 / (1.0 - torch.tensor(accum))) * lr_col) * residual

            updated = organizer.organize(weights)

            torch.testing.assert_close(updated, expected)
        torch.testing.assert_close(organizer.beta_accum, torch.tensor(0.9**3))

    def test_unknown_organize_step_is_rejected(self):
        with self.assertRaises(ValueError):
            DiscriminationOrganizer(IN_DIM, OUT_DIM, organize_step="raw")

    def test_checkpoint_without_correction_state_loads(self):
        model = BiologicalClassifier(input_dim=IN_DIM, hidden_dim=OUT_DIM, output_dim=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save_model(str(path))
            state = torch.load(path)
            del state["discrimination_layer"]["organizer.beta_accum"]
            torch.save(state, path)
            restored = BiologicalClassifier(input_dim=IN_DIM, hidden_dim=OUT_DIM, output_dim=3)

            restored.load_model(str(path))

        self.assertTrue(
            torch.equal(restored.discrimination_layer.neuron_weights, model.discrimination_layer.neuron_weights)
        )

    def test_classifier_records_organize_step(self):
        default = BiologicalClassifier(input_dim=IN_DIM, hidden_dim=OUT_DIM)
        previous = BiologicalClassifier(
            input_dim=IN_DIM,
            hidden_dim=OUT_DIM,
            discrimination_config={"organize_step": "correction_factor"},
        )

        self.assertEqual(default.discrimination_config["organize_step"], "direction")
        self.assertEqual(previous.discrimination_config["organize_step"], "correction_factor")


if __name__ == "__main__":
    unittest.main()
