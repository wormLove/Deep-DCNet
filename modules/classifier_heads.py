"""
Swappable classifier heads for BiologicalClassifier.

Both classes below are drop-in replacements for modules/readout.py's
ReadoutHead: same interface (__init__(in_dim, out_dim, **kwargs),
forward(x) -> logits), so BiologicalClassifier's head_cls/head_kwargs
params can swap between them without touching the discrimination layer
or (when present) the integration layer.
"""
import torch.nn as nn


class TraditionalMLPHead(nn.Module):
    """
    A conventional, multi-layer feedforward classifier trained end-to-end
    with backprop - the standard deep-learning approach to turning a
    feature vector into class logits. Exists specifically as a comparison
    baseline against ReadoutHead's single linear layer (Prof. Yu's point #5):
    same discrimination-layer (or integration-layer) features in, does a
    deeper/nonlinear head do better than a single linear layer at reading
    them out?

    Deliberately NOT biologically-inspired - no Hebbian update, no sparsity
    constraint, just Linear -> ReLU -> Dropout stacked normally.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dims=(256,), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
