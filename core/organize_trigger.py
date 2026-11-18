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
