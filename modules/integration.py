import torch
from torch import nn


class IntegrationLayer(nn.Module):
    """
    Dense integration layer that merges Layer 0 (raw input) and
    Layer 1 (discrimination activations) into a joint representation.

    Expected input: cat([layer0_output, layer1_output])  →  [batch, in_dim]
    Output: dense representation                          →  [batch, out_dim]

    This layer is trained with standard backprop (Adam) alongside the
    readout head. The discrimination layer itself remains frozen.
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {activation!r}. "
                f"Choose from {list(self.ACTIVATIONS)}."
            )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.act = self.ACTIVATIONS[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
