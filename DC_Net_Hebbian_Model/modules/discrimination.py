import torch
import warnings
from torch import nn
from torch.nn.functional import normalize

from core.abstract import Organizer
from core.optimizers.registry import ACTIVITY_OPTIMIZERS
from core.activations import LayerThresholding
import utils.utils as util

class DiscriminationModule(nn.Module):
    """
    Pipeline:
      input (1 x in_dim)
        -> raw activity:  act_raw = input @ W
        -> optimized:     act_opt = ActivityOptimizer(act_raw, Corr)
        -> thresholding:  act = LayerThresholding(act_opt)
        -> organizer step (training only)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        initializer=None,
        activity_optimizer: str = "least_squares",
        non_negative_strategy: str = "min-max",
        **kwargs,
    ):
        super().__init__()

        # Initialize weights
        self.neuron_weights = self.initialize_weights(
            in_dim=in_dim,
            out_dim=out_dim,
            initializer=initializer,
            non_negative_strategy=non_negative_strategy,
        )

        # Activity optimizer
        self.activity_optimizer = self._build_activity_optimizer(activity_optimizer)

        # Correlation matrix used by activity optimizer
        self.neuron_correlation_matrix = self.compute_neuron_correlation_matrix()

        # Thresholding / activation
        self.activation = LayerThresholding(kwargs.get("threshold_factor", 1.0))

        # Organizer (Hebbian / Anti-Hebbian potentials + weight updates)
        self.organizer = DiscriminationOrganizer(in_dim, out_dim, **kwargs)
    
    def initialize_weights(
        self,
        in_dim: int,
        out_dim: int,
        initializer=None,
        non_negative_strategy: str = "min-max",
    ) -> torch.Tensor:
        """
        Initialize neuron weights with optional dataset-aware initializer.
        The returned matrix is column-wise L2-normalized.
        """
        if initializer is not None:
            weights = initializer.weights(
                (in_dim, out_dim),
                non_negative_strategy=non_negative_strategy,
            )
        else:
            weights = torch.randn(in_dim, out_dim)
            weights = util.non_negative_transform(weights, strategy=non_negative_strategy)

        return normalize(weights, p=2, dim=0)

    @staticmethod
    def _build_activity_optimizer(name: str):
        name = str(name).lower()

        if name not in ACTIVITY_OPTIMIZERS:
            raise ValueError(
                f"Unknown activity_optimizer='{name}'. "
                f"Choose from: {sorted(ACTIVITY_OPTIMIZERS)}."
            )

        return ACTIVITY_OPTIMIZERS[name]()

    def compute_neuron_correlation_matrix(self) -> torch.Tensor:
        """
        Correlation matrix between neurons (out_dim x out_dim).
        If neuron_weights columns are unit-norm, this is cosine similarity matrix.
        """
        return torch.mm(self.neuron_weights.T, self.neuron_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (1 x in_dim) row vector.

        Returns:
            act: (1 x out_dim) activity vector after optimization + thresholding.
        """
        assert input.dim() == 2 and input.shape[0] == 1, "Input must be a row vector [1, in_dim]."

        act_raw = torch.mm(input, self.neuron_weights)  # (1 x out_dim)
        act_opt = self.activity_optimizer(act_raw, self.neuron_correlation_matrix)
        act = self.activation(act_opt)

        if self.training:
            self.organizer.step(input, act)

        return act
    
    def organize(self, unit_norm: bool = True) -> None:
        """
        Apply organizer to update neuron weights, then refresh correlation matrix.
        """
        updated_weights = self.organizer.organize(self.neuron_weights)

        if unit_norm:
            self.neuron_weights = normalize(updated_weights, p=2, dim=0)
        else:
            self.neuron_weights = updated_weights

        self.neuron_correlation_matrix = self.compute_neuron_correlation_matrix()
    
    def save_state(self, include_training_state=False):
        state = {
            'neuron_weights': self.neuron_weights,
            'correlation_matrix': self.neuron_correlation_matrix,
        }
        if include_training_state:
            state['organizer_state'] = {
                'potential_hebb': self.organizer.potential_hebb,
                'potential_antihebb': self.organizer.potential_antihebb,
                'cooldown_state': self.organizer.cooldown_state,
                'lr': self.organizer.lr,
                'lr_decay': self.organizer.lr_decay,
                'beta': self.organizer.beta,
                'beta_accum': self.organizer.beta_accum,
            }
        return state

    def load_state(self, state_dict, include_training_state=False):
        self.neuron_weights = state_dict['neuron_weights']
        self.neuron_correlation_matrix = state_dict['correlation_matrix']

        if include_training_state and 'organizer_state' in state_dict:
            organizer_state = state_dict['organizer_state']
            self.organizer.potential_hebb = organizer_state['potential_hebb']
            self.organizer.potential_antihebb = organizer_state['potential_antihebb']
            self.organizer.cooldown_state = organizer_state['cooldown_state']
            self.organizer.lr = organizer_state['lr']
            self.organizer.lr_decay = organizer_state['lr_decay']
            self.organizer.beta = organizer_state['beta']
            self.organizer.beta_accum = organizer_state['beta_accum']
    
    
class DiscriminationOrganizer(Organizer):
    """
    Update potentials (Hebbian & Anti-Hebbian) and organize weights for DiscriminationModule.

    - potential_hebb:   input^T @ activity
    - potential_antihebb: activity^T @ activity
    """
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.potential_hebb = torch.zeros(in_dim, out_dim)
        self.potential_antihebb = torch.zeros(out_dim, out_dim)
        self.cooldown_state = torch.zeros(out_dim, dtype=torch.bool)
        
        self.lr = kwargs.get('lr', 0.99)
        self.lr_decay = kwargs.get('lr_decay', 0.99)
        self.beta = kwargs.get('beta', 0.99)
        self.beta_accum = self.beta
    
    def step(
        self,
        input_signal: torch.Tensor,
        activity_signal: torch.Tensor,
    ) -> None:
        """
        Update potentials using the current input and activity.
        """
        activity_signal = self._apply_cooldown(activity_signal)
        activity_signal = self._normalize(activity_signal)

        hebb_update = (1.0 - self.beta) * self._potential(input_signal, activity_signal)
        antihebb_update = (1.0 - self.beta) * self._potential(activity_signal, activity_signal)

        self.potential_hebb = self.beta * self.potential_hebb + hebb_update
        self.potential_antihebb = self.beta * self.potential_antihebb + antihebb_update
    
    def organize(self, weights: torch.Tensor):
        """
        Updates the weight matrix based on the current potentials.
        """
        correction_factor = 1 / (1 - self.beta_accum)
        potential_diff = self.potential_hebb - torch.mm(weights, self.potential_antihebb)
        updated_weights = (1 - self.lr) * weights + correction_factor * self.lr * potential_diff     
        
        self.lr *= self.lr_decay
        self.beta_accum *= self.beta

        return updated_weights
    
    @staticmethod
    def _potential(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Potential matrix = input^T @ target.
        """
        return torch.mm(input_tensor.T, target_tensor)
    
    def _apply_cooldown(self, activity_signal: torch.Tensor) -> torch.Tensor:
        """
        Prevent consecutive activations of the same neurons.
        """
        activity_signal = self.cooldown_state.logical_not() * activity_signal
        self.cooldown_state = activity_signal > 0
        return activity_signal

    @staticmethod
    def _normalize(activity_signal: torch.Tensor) -> torch.Tensor:
        """
        Normalize activity to unit Frobenius norm.
        """
        activity_norm = torch.norm(activity_signal, p="fro").item()
        if activity_norm == 0:
            warnings.warn("Activity signal norm is zero.")
            return activity_signal
        return activity_signal / activity_norm
