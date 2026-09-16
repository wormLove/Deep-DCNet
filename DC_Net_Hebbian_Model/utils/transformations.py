import torch
from abc import ABC, abstractmethod
from typing import Sequence


class Transform(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class Scale(Transform):
    """Min-max normalize to [0, 1]. If constant input, returns zeros."""
    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_min = x.min()
        x_max = x.max()
        if (x_max - x_min).abs() < 1e-6:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)


class ToVector(Transform):
    """Flatten and make it a row vector: [*, ...] -> [1, D]."""
    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x.flatten().unsqueeze(0)


class Compose(Transform):
    """Apply transforms in order."""
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for t in self.transforms:
            x = t(x, **kwargs)
        return x
