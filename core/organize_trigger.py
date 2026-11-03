from typing import Sequence


class FixedIntervalTrigger:
    """Organize every ``interval_samples`` training samples: the fixed sleep rhythm."""

    def __init__(self, interval_samples: int = 200):
        if interval_samples <= 0:
            raise ValueError("interval_samples must be positive.")
        self.interval_samples = int(interval_samples)

    def due(self, samples_since_organize: int, model=None) -> bool:
        return samples_since_organize >= self.interval_samples

    def organized(self) -> None:
        pass


class ScheduledIntervalTrigger:
    """Organize after each given interval in turn, fixed in advance.

    A study control: intervals that vary but do not depend on the model's state.
    """

    def __init__(self, intervals_samples: Sequence[int]):
        intervals = [int(interval) for interval in intervals_samples]
        if not intervals or min(intervals) <= 0:
            raise ValueError("intervals_samples must be a non-empty list of positive counts.")
        self.intervals_samples = intervals
        self._index = 0

    def due(self, samples_since_organize: int, model=None) -> bool:
        if self._index >= len(self.intervals_samples):
            raise RuntimeError("ScheduledIntervalTrigger ran out of intervals.")
        return samples_since_organize >= self.intervals_samples[self._index]

    def organized(self) -> None:
        self._index += 1
