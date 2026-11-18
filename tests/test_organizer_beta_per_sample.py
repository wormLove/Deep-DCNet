import unittest

import torch

from core.organizer import DiscriminationOrganizer
from models.biological_classifier import BiologicalClassifier
from models.discrimination import DiscriminationLayer


IN_DIM = 6
OUT_DIM = 5
PER_BATCH_BETA = 199 / 200
PER_SAMPLE_BETA = PER_BATCH_BETA ** (1 / 4)


def _samples(count: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(count, IN_DIM, generator=generator)
    act = torch.rand(count, OUT_DIM, generator=generator)
    return x, act


def _feed(organizer: DiscriminationOrganizer, x, act, batch_size: int) -> None:
    for x_batch, act_batch in zip(x.split(batch_size), act.split(batch_size)):
        organizer.step(x_batch, act_batch)


class OrganizerBetaPerSampleTest(unittest.TestCase):
    def test_batch_of_one_is_the_per_sample_ema(self):
        organizer = DiscriminationOrganizer(IN_DIM, OUT_DIM, beta=PER_SAMPLE_BETA)
        x, act = _samples(12)
        hebb = torch.zeros(IN_DIM, OUT_DIM)
        antihebb = torch.zeros(OUT_DIM, OUT_DIM)
        for x_i, act_i in zip(x, act):
            a = act_i / act_i.norm()
            hebb = PER_SAMPLE_BETA * hebb + (1 - PER_SAMPLE_BETA) * torch.outer(x_i, a)
            antihebb = PER_SAMPLE_BETA * antihebb + (1 - PER_SAMPLE_BETA) * torch.outer(a, a)

        _feed(organizer, x, act, batch_size=1)

        torch.testing.assert_close(organizer.potential_hebb, hebb)
        torch.testing.assert_close(organizer.potential_antihebb, antihebb)

    def test_retention_is_independent_of_batch_size(self):
        x, act = _samples(1)
        x, act = x.repeat(8, 1), act.repeat(8, 1)
        fresh = DiscriminationOrganizer(IN_DIM, OUT_DIM, beta=PER_SAMPLE_BETA)
        fresh.step(x[:1], act[:1])
        one_sample = fresh.potential_hebb / (1 - PER_SAMPLE_BETA)

        for batch_size in (1, 2, 4, 8):
            organizer = DiscriminationOrganizer(IN_DIM, OUT_DIM, beta=PER_SAMPLE_BETA)
            organizer.potential_hebb.fill_(1.0)
            _feed(organizer, x, act, batch_size)
            # Old content keeps beta**8 after 8 samples; 8 identical samples add
            # (1 - beta**8) of their content, however they were batched.
            expected = PER_SAMPLE_BETA**8 + (1 - PER_SAMPLE_BETA**8) * one_sample
            torch.testing.assert_close(organizer.potential_hebb, expected, msg=str(batch_size))

    def test_batch_of_four_reproduces_the_previous_per_batch_beta_bit_for_bit(self):
        organizer = DiscriminationOrganizer(IN_DIM, OUT_DIM, beta=PER_SAMPLE_BETA)
        x, act = _samples(200, seed=3)
        hebb = torch.zeros(IN_DIM, OUT_DIM)
        antihebb = torch.zeros(OUT_DIM, OUT_DIM)
        for x_batch, act_batch in zip(x.split(4), act.split(4)):
            # DiscriminationOrganizer.step as of v0.3.0, with beta = 199/200 per batch.
            a = act_batch / act_batch.norm(dim=1, keepdim=True).clamp_min(1e-12)
            hebb.mul_(PER_BATCH_BETA).add_((1.0 - PER_BATCH_BETA) * (x_batch.T @ a / 4))
            antihebb.mul_(PER_BATCH_BETA).add_((1.0 - PER_BATCH_BETA) * (a.T @ a / 4))

        _feed(organizer, x, act, batch_size=4)

        self.assertTrue(torch.equal(organizer.potential_hebb, hebb))
        self.assertTrue(torch.equal(organizer.potential_antihebb, antihebb))

    def test_correction_schedule_follows_correction_beta_not_beta(self):
        split = DiscriminationOrganizer(
            IN_DIM,
            OUT_DIM,
            beta=PER_SAMPLE_BETA,
            correction_beta=PER_BATCH_BETA,
            organize_step="correction_factor",
        )
        shared = DiscriminationOrganizer(
            IN_DIM, OUT_DIM, beta=PER_BATCH_BETA, organize_step="correction_factor"
        )
        weights = torch.rand(IN_DIM, OUT_DIM)

        for _ in range(5):
            split.organize(weights)
            shared.organize(weights)
            self.assertTrue(torch.equal(split.beta_accum, shared.beta_accum))

    def test_correction_beta_defaults_to_beta(self):
        organizer = DiscriminationOrganizer(IN_DIM, OUT_DIM, beta=0.9)

        self.assertEqual(organizer.correction_beta, 0.9)
        self.assertTrue(torch.equal(organizer.beta_accum, torch.tensor(0.9)))

    def test_layer_reset_restores_correction_beta(self):
        layer = DiscriminationLayer(
            in_dim=IN_DIM,
            out_dim=OUT_DIM,
            beta=PER_SAMPLE_BETA,
            correction_beta=0.9,
            organize_step="correction_factor",
        )
        x, act = _samples(4)
        layer.organizer.step(x, act)
        layer.stats.cache_activity(act)
        layer.organize()
        layer.organize()

        layer.reset_organizer_state()

        self.assertTrue(torch.equal(layer.organizer.beta_accum, torch.tensor(0.9)))

    def test_classifier_records_resolved_betas(self):
        model = BiologicalClassifier(
            input_dim=IN_DIM,
            hidden_dim=OUT_DIM,
            discrimination_config={"beta": PER_SAMPLE_BETA},
        )

        self.assertEqual(model.discrimination_config["beta"], PER_SAMPLE_BETA)
        self.assertEqual(model.discrimination_config["correction_beta"], PER_SAMPLE_BETA)


if __name__ == "__main__":
    unittest.main()
