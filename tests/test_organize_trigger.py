import functools
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

import training.training_engines as engines
from core.organize_trigger import FixedIntervalTrigger, ScheduledIntervalTrigger
from models.biological_classifier import BiologicalClassifier


BATCH = 4


class _OrganizeTimes:
    """Stands in for the engine's analysis hook; records the samples seen at each organize."""

    def __init__(self):
        self.samples = []

    def save_step_state(self, model, step):
        pass

    def save_organize_state(self, model, step):
        self.samples.append(step * BATCH)


def _train(samples: int, **engine_kwargs):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    x = torch.rand(samples, 6, generator=generator)
    y = torch.randint(0, 3, (samples,), generator=generator)
    model = BiologicalClassifier(input_dim=6, hidden_dim=5, output_dim=3)
    model.enable_review(enabled=True)
    times = _OrganizeTimes()
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=False)
    engines.train_classifier_with_review(
        model=model,
        train_loader=loader,
        test_loader=loader,
        device=torch.device("cpu"),
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.readout_head.parameters(), lr=1e-3),
        analysis=times,
        enable_checkpoints=False,
        quiet_train=True,
        **engine_kwargs,
    )
    return model, times.samples


class OrganizeTriggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tqdm = engines.tqdm
        engines.tqdm = functools.partial(engines.tqdm, disable=True)

    @classmethod
    def tearDownClass(cls):
        engines.tqdm = cls._tqdm

    def test_fixed_trigger_is_due_after_its_interval(self):
        trigger = FixedIntervalTrigger(200)

        self.assertFalse(trigger.due(196))
        self.assertTrue(trigger.due(200))
        self.assertTrue(trigger.due(204))

    def test_scheduled_trigger_follows_its_intervals_in_turn(self):
        trigger = ScheduledIntervalTrigger([8, 12])

        self.assertTrue(trigger.due(8))
        trigger.organized()
        self.assertFalse(trigger.due(8))
        self.assertTrue(trigger.due(12))
        trigger.organized()
        with self.assertRaises(RuntimeError):
            trigger.due(100)

    def test_invalid_intervals_are_rejected(self):
        for intervals in ([], [8, 0]):
            with self.assertRaises(ValueError):
                ScheduledIntervalTrigger(intervals)
        with self.assertRaises(ValueError):
            FixedIntervalTrigger(0)

    def test_fixed_trigger_reproduces_the_interval_argument(self):
        by_interval, times_interval = _train(48, organize_interval_samples=8)
        by_trigger, times_trigger = _train(48, organize_trigger=FixedIntervalTrigger(8))
        by_schedule, times_schedule = _train(48, organize_trigger=ScheduledIntervalTrigger([8] * 7))

        self.assertEqual(times_interval, [8, 16, 24, 32, 40, 48])
        self.assertEqual(times_trigger, times_interval)
        self.assertEqual(times_schedule, times_interval)
        for other in (by_trigger, by_schedule):
            self.assertTrue(
                torch.equal(other.discrimination_layer.neuron_weights, by_interval.discrimination_layer.neuron_weights)
            )

    def test_scheduled_intervals_set_the_organize_times(self):
        _, times = _train(40, organize_trigger=ScheduledIntervalTrigger([8, 12, 4, 100]))

        # The last interval is cut short by the end of the data: the engine's final organize.
        self.assertEqual(times, [8, 20, 24, 40])


if __name__ == "__main__":
    unittest.main()
