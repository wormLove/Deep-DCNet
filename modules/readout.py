import torch
from torch import nn


class ReadoutHead(nn.Module):
    """
    Minimal linear readout head.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
