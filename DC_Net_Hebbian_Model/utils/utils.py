import os
import random
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

class Standarize(nn.Module):
    """
        This function performs whitening on the input data along the specified dimension, 
        transforming it to have a mean of 0 and a standard deviation of 1.，
        
        Note:
        There is an issue with the current implementation: if the standard deviation of any sub-tensor is 0, 
        the entire input tensor will be set to 0. It's unclear if this was the intended design. 
        A more typical approach would be to set the sub-tensor to 0 if its standard deviation is 0, 
        and otherwise, normalize the sub-tensor.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: torch.Tensor):
        if all(input.std(dim=self.dim, keepdim=True) > 0):
            return (input - input.mean(dim=self.dim, keepdim=True))/input.std(dim=self.dim, keepdim=True)
        return torch.zeros_like(input)

class Standarize_REF(nn.Module):
    """
        This function performs whitening on the input data along the specified dimension, 
        transforming it to have a mean of 0 and a standard deviation of 1.
        
        Note:
        Updated version, set the sub-tensor to 0 if its standard deviation is 0, and otherwise, 
        normalize the sub-tensor.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: torch.Tensor):
        std = input.std(dim=self.dim, keepdim=True)
        mean = input.mean(dim=self.dim, keepdim=True)
        standardized_input = torch.where(std > 0, (input - mean) / std, torch.zeros_like(input))
        return standardized_input

def get_timestamp():
    ct = datetime.datetime.now()
    return ct.strftime("%Y-%m-%d_%H-%M-%S")

def set_random_seed(seed: int = 42, deterministic: bool = False):
    """
    Set the random seed for reproducibility across numpy, torch, and python.

    Args:
        seed (int): The seed value to set. Defaults to 42.
        deterministic (bool): Whether to enforce deterministic behavior in PyTorch. Defaults to False.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def non_negative_transform(matrix: torch.Tensor, strategy: str = 'shift') -> torch.Tensor:
    """
    Applies a non-negative transformation to the matrix.

    Args:
        matrix (torch.Tensor): The input weight matrix.
        strategy (str): The strategy for non-negative transformation. Options:
            - 'abs': Absolute value transformation.
            - 'min-max': Min-Max normalization.
            - 'shift': Min-value shift to make all values non-negative.
            - 'softplus': Apply softplus activation to ensure non-negativity.

    Returns:
        torch.Tensor: The transformed non-negative matrix.

    Raises:
        ValueError: If an unknown strategy is provided.
    """
    if strategy == 'abs':
        return matrix.abs()
    elif strategy == 'min-max':
        min_val, max_val = matrix.min(), matrix.max()
        if max_val == min_val:
            raise ValueError("Min-Max scaling cannot be applied to a constant matrix.")
        return (matrix - min_val) / (max_val - min_val)
    elif strategy == 'shift':
        return matrix - matrix.min()
    elif strategy == 'softplus':
        return torch.nn.functional.softplus(matrix)
    elif strategy == 'no-change':
        return matrix
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def write_log(file_path: str, message: str):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
