from typing import Optional, Tuple

import torch


class ActivationCache:
    """
    CPU ring-buffer cache for review activations.

    Activations are stored on CPU to avoid holding long-lived replay memory on
    GPU. Sampling moves a minibatch back to the requested device.
    """

    def __init__(self, max_size: int = 20000, dtype: torch.dtype = torch.float16):
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        self.max_size = int(max_size)
        self.dtype = dtype

        self.acts: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.head = 0
        self.size = 0
        self.dim: Optional[int] = None

    def _lazy_alloc(self, dim: int) -> None:
        self.dim = int(dim)
        self.acts = torch.empty((self.max_size, self.dim), dtype=self.dtype, device="cpu")
        self.labels = torch.empty((self.max_size,), dtype=torch.int16, device="cpu")

    @torch.no_grad()
    def add(self, activation: torch.Tensor, label: torch.Tensor) -> None:
        a = activation.detach().to(dtype=self.dtype, device="cpu")
        y = label.detach().to(device="cpu")

        if a.dim() == 1:
            a = a.unsqueeze(0)
        elif a.dim() != 2:
            raise ValueError(f"activation must be 1D or 2D, got shape {tuple(a.shape)}")

        if y.dim() == 0:
            y = y.unsqueeze(0)
        else:
            y = y.view(-1)

        if a.size(0) != y.numel():
            raise ValueError(
                f"activation batch and label batch must match, got {a.size(0)} and {y.numel()}"
            )

        if self.acts is None:
            self._lazy_alloc(a.size(1))

        if a.size(1) != self.dim:
            raise ValueError(
                f"activation feature dim mismatch: cache expects {self.dim}, got {a.size(1)}"
            )

        for i in range(a.size(0)):
            self.acts[self.head].copy_(a[i])
            self.labels[self.head] = int(y[i].item())
            self.head = (self.head + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        out_dtype: Optional[torch.dtype] = None,
        strict: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0 or (strict and self.size < batch_size):
            raise ValueError(f"ActivationCache not ready: have {self.size}, need {batch_size}.")

        k = min(batch_size, self.size)
        idx = torch.randint(low=0, high=self.size, size=(k,), device="cpu")
        acts_cpu = self.acts.index_select(0, idx)
        labels_cpu = self.labels.index_select(0, idx).to(torch.long)

        target_device = device if device is not None else acts_cpu.device
        target_dtype = out_dtype if out_dtype is not None else acts_cpu.dtype

        acts = acts_cpu.to(device=target_device, dtype=target_dtype, non_blocking=True)
        labels = labels_cpu.to(device=target_device, non_blocking=True)
        return acts, labels

    def clear(self) -> None:
        self.head = 0
        self.size = 0

    def ready(self, min_size: int = 64) -> bool:
        return self.size >= min_size

    def __len__(self) -> int:
        return self.size
