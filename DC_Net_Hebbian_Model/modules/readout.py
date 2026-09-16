import torch
import torch.nn as nn

class ReadoutHead(nn.Module):
    """
    General-purpose readout head.
    Can be a simple linear layer or a multi-layer classifier.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.head(x)