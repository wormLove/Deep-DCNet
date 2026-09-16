import torch
import torch.nn as nn
from torch.linalg import matrix_rank

class ActivityOptimizerInverseMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, neuron_correlation_matrix: torch.Tensor) -> torch.Tensor:
        approx_inverse = self._construct_invertible_matrix(neuron_correlation_matrix, input)
        return torch.mm(input, approx_inverse)
    
    def _construct_invertible_matrix(self, correlation_matrix: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        dim = input.shape[1]

        selection_diag = torch.zeros(dim, device=correlation_matrix.device, dtype=correlation_matrix.dtype)
        selection_indx = torch.argsort(input.flatten(), descending=True)

        left, right = 0, dim
        while left < right:
            mid = (left + right) // 2
            selection_diag.zero_()
            selection_diag[selection_indx[: mid + 1]] = 1

            sys_mat = self._system_matrix(selection_diag, correlation_matrix)
            if matrix_rank(sys_mat) < dim:
                right = mid
            else:
                left = mid + 1

        selection_diag.zero_()
        selection_diag[selection_indx[:left]] = 1
        return torch.linalg.inv(self._system_matrix(selection_diag, correlation_matrix))

    
    def _system_matrix(self, selection_diag: torch.Tensor, correlation_matrix: torch.Tensor) -> torch.Tensor:
        dim = len(selection_diag)
        identity_matrix = torch.eye(dim, device=correlation_matrix.device, dtype=correlation_matrix.dtype)

        selected_rows = torch.mm(torch.diag(selection_diag), correlation_matrix)
        system_matrix = identity_matrix + selected_rows - torch.diag(selection_diag)
        return system_matrix
