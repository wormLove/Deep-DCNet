from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler


def apply_non_negative_transform(weights: Tensor, enabled: bool = True) -> Tensor:
    if not enabled:
        return weights
    return weights - weights.min()


class RandomInitializer:
    def __init__(self, non_negative: bool = True):
        self.non_negative = bool(non_negative)

    def weights(self, dims: tuple[int, int]) -> Tensor:
        in_dim, out_dim = dims
        weights = torch.randn(in_dim, out_dim)
        weights = apply_non_negative_transform(weights, enabled=self.non_negative)
        return weights


class DatasetInitializerWhole:
    """
    PCA-based whole-image initializer aligned with the CPU research version's
    DatasetInitializer_Whole, adapted to the current flat 784-dim pipeline.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Callable[[Tensor], Tensor],
        init_ratio: float = 0.25,
        non_negative: bool = True,
    ):
        self.dataset = dataset
        self.transform = transform
        self.init_ratio = float(init_ratio)
        self.non_negative = bool(non_negative)

    def weights(self, dims: tuple[int, int]) -> Tensor:
        in_dim, out_dim = dims
        sample_size = max(1, int(self.init_ratio * len(self.dataset)))
        sample_data = self._sample_data(sample_size)
        if sample_data.shape[1] != in_dim:
            raise ValueError(
                f"Dataset initializer feature dim mismatch: expected {in_dim}, got {sample_data.shape[1]}"
            )

        v_n, sn_inv, n = self._extract_pca_components(sample_data)
        randomizer = self._randomizer(out_dim, n)
        weights = torch.linalg.multi_dot((v_n, sn_inv, randomizer.T))
        weights = apply_non_negative_transform(weights, enabled=self.non_negative)
        return weights

    def _sample_data(self, sample_size: int) -> Tensor:
        batch = next(
            iter(
                DataLoader(
                    self.dataset,
                    sampler=RandomSampler(self.dataset),
                    batch_size=sample_size,
                    drop_last=False,
                )
            )
        )
        data_batch = batch[0]
        sample_data = torch.stack([self.transform(data).squeeze() for data in data_batch])
        return sample_data

    def _extract_pca_components(self, sample_data: Tensor) -> tuple[Tensor, Tensor, int]:
        if sample_data.numel() == 0:
            raise ValueError("Sample data is empty.")

        q = min(sample_data.shape[0], sample_data.shape[1])
        _, s, v = torch.pca_lowrank(sample_data, q=q, center=True)
        sn, n = self._effective_dims(s)
        v_n = v[:, :n]
        sn_inv = torch.diag(sn[:n].pow(-1))
        return v_n, sn_inv, n

    @staticmethod
    def _effective_dims(s: Tensor, explained_variance: float = 0.95) -> tuple[Tensor, int]:
        if s.dim() != 1:
            raise ValueError("singular values must be 1D.")
        sn = s.pow(2) / torch.sum(s.pow(2))
        n = int((sn.cumsum(0) < explained_variance).sum().item() + 1)
        return sn, n

    @staticmethod
    def _randomizer(m: int, n: int) -> Tensor:
        if m < n:
            raise ValueError(f"{m=} should be >= {n=}")
        _, _, v = torch.pca_lowrank(torch.rand(n, n), q=n)
        if m > n:
            random_matrix = F.normalize(torch.randn(m - n, n), p=2, dim=1, eps=1e-12)
            vc = torch.mm(random_matrix, v)
            return torch.vstack((v, vc))
        return v
