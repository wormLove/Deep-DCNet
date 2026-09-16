import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.linalg import multi_dot

# From Workspace
from core.abstract import Initializer
from utils.transformations import Transform
import utils.utils as util


class DatasetInitializer(Initializer):
    """
    DatasetInitializer initializes weights by performing PCA on a sample of the dataset.

    NOTE (Student release):
      - Channel/multi-channel interface is removed for consistency with the no-channel DiscriminationModule.
      - PCA balancing logic is preserved exactly as your current tested version.
    """
    def __init__(self, dataset: Dataset, transforms: Transform, init_ratio: float = 0.25):
        self.dataset = dataset
        self.transforms = transforms
        self.init_ratio = init_ratio

    def weights(self, dims: tuple[int, int], non_negative_strategy: str = "min-max") -> torch.Tensor:
        """
        Computes the initial weight matrix based on PCA of a sample of the dataset.

        Args:
            dims: (in_dim, out_dim)
        Returns:
            torch.Tensor: initialized weight matrix (in_dim x out_dim)
        """
        in_dim, out_dim = dims
        sample_size = int(self.init_ratio * len(self.dataset))
        sample_data = self._sample_data(sample_size)

        V_n, Sn_inv, n = self._extract_pca_components(sample_data)

        randomizer = self._randomizer(out_dim, n)
        neuron_weights = multi_dot((V_n, Sn_inv, randomizer.T))

        neuron_weights = util.non_negative_transform(neuron_weights, strategy=non_negative_strategy)
        assert neuron_weights.dim() == 2, "Weight matrix must be 2D"

        return neuron_weights

    def _sample_data(self, sample_size: int) -> torch.Tensor:
        """
        Samples a subset of the dataset and applies the specified transformations.

        Returns:
            torch.Tensor: shape (sample_size, in_dim) after transforms.
        """
        batch = next(
            iter(DataLoader(self.dataset, sampler=RandomSampler(self.dataset), batch_size=sample_size))
        )
        data_batch = batch[0]
        sample_data = torch.stack([self.transforms(data).squeeze() for data in data_batch])
        return sample_data

    def _extract_pca_components(self, sample_data: torch.Tensor):
        """
        Performs PCA on the sampled data and extracts key components for weight initialization.

        Returns:
            V_n (torch.Tensor): Top principal components matrix.
            Sn_inv (torch.Tensor): Inverse scaling matrix (PRESERVED logic).
            n (int): Effective number of components to retain.
        """
        if sample_data.numel() == 0:
            raise ValueError("Sample data is empty. Ensure dataset is not empty.")

        q = min(sample_data.shape[0], sample_data.shape[1])
        U, S, V = torch.pca_lowrank(sample_data, q=q, center=True)

        Sn, n = self._effective_dims(S)
        V_n = V[:, :n]
        Sn_inv = torch.diag(Sn[:n] ** (-1))

        return V_n, Sn_inv, n

    @staticmethod
    def _effective_dims(S: torch.Tensor, explained_variance: float = 0.95):
        """
        Determines the number of components needed to explain a certain percentage of variance.

        Returns:
            Sn (torch.Tensor): normalized explained-variance ratios.
            n (int): number of components to retain.
        """
        assert S.dim() == 1, "singular values must be in 1D tensor"
        Sn = S**2 / torch.sum(S**2)
        n = (Sn.cumsum(0) < explained_variance).sum().item() + 1
        return Sn, n

    @staticmethod
    def _randomizer(m: int, n: int) -> torch.Tensor:
        """
        Generates a random orthogonal-like matrix for adding randomness to the weights.

        Returns:
            torch.Tensor: shape (m, n)
        """
        assert m >= n, f"{m=} should be greater than or equal to {n=}"

        _, _, V = torch.pca_lowrank(torch.rand(n, n), q=n)

        if m > n:
            random_matrix = normalize(torch.randn(m - n, n), p=2, dim=1, eps=1e-12)
            Vc = torch.mm(random_matrix, V)
            return torch.vstack((V, Vc))
        else:
            return V





