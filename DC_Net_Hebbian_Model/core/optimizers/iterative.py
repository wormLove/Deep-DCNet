import torch
import torch.nn as nn
import torch.nn.functional as F

class IterativeActivityOptimizer(nn.Module):
    def __init__(self, max_iters=1000, lambda_=0.1, gain_factor=10.0, 
                 patience=5, return_best_y=True):
        super().__init__()
        self.max_iters = max_iters
        self.lambda_ = lambda_
        self.gain_factor = gain_factor
        self.patience = patience
        self.return_best_y = return_best_y

    def _calculate_gain(self, matrix: torch.Tensor) -> float:
        """
        Approximates the maximum eigenvalue using power iteration.
        """
        b = torch.rand(matrix.shape[1], device=matrix.device)
        b = b / torch.norm(b)
        for _ in range(50):
            b = torch.matmul(matrix, b)
            b = b / torch.norm(b)
        return torch.dot(b, torch.matmul(matrix, b)).item()

    def _threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """One-sided (non-negative) soft thresholding."""
        return F.relu(x - threshold)

    def forward(self, input: torch.Tensor, W_lateral: torch.Tensor):
        assert input.dim() == 2 and input.shape[0] == 1

        x = input.squeeze(0)
        gain = self.gain_factor * self._calculate_gain(W_lateral)
        gain_inv = 1.0 / (gain + 1e-12)

        y_prev = x.clone()
        y = y_prev.clone()

        best_y = y.clone()
        best_update_norm = float('inf')
        patience_counter = 0

        # Schedule-based dynamic step size
        decay_schedule = {300, 600, 900}

        for i in range(self.max_iters):
            l = 0.0 if i < 2 else self.lambda_

            if i in decay_schedule:
                gain_inv *= 0.5  # decay step size

            update = x - torch.matmul(W_lateral, y_prev)
            update_norm = float(torch.norm(update))
            y = y_prev + gain_inv * update
            y = self._threshold(y, l * gain_inv)

            # Early stopping if update doesn't improve
            if update_norm + 1e-8 < best_update_norm:
                best_update_norm = update_norm
                best_y = y.clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

            y_prev = y.clone()

        return (best_y if self.return_best_y else y).unsqueeze(0)
