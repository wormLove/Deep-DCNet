import torch
from torch import nn

class ActivityOptimizerLeastSquares(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, neuron_correlation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Solve: W_lateral * y ~= x   (least squares)
        where:
          - neuron_correlation_matrix is W_lateral (out_dim x out_dim)
          - input is x with shape (1, out_dim)
        Returns:
          y with shape (1, out_dim)
        """
        A = neuron_correlation_matrix
        b = input.T  # (out_dim, 1)

        try:
            y = torch.linalg.lstsq(A, b).solution  # (out_dim, 1)
        except (RuntimeError, AttributeError):
            # Robust fallback (works across versions)
            y = torch.linalg.pinv(A) @ b

        return y.T
