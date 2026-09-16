import torch
from typing import Optional

class ActivationCache:
    """
    Ring-buffer activation cache.
    - Activations stored on CPU in fp16: [max_size, D]
    - Labels stored on CPU in int16: [max_size]
    - Lazy allocation on first add()
    - O(1) insert, O(B) sampling (with replacement)
    """
    def __init__(self, max_size: int = 20000, dtype: torch.dtype = torch.float16):
        assert max_size > 0
        self.max_size = int(max_size)
        self.dtype = dtype

        self.acts: Optional[torch.Tensor] = None     # [max_size, D] on CPU
        self.labels: Optional[torch.Tensor] = None   # [max_size] on CPU
        self.head = 0
        self.size = 0
        self.dim: Optional[int] = None

    def _lazy_alloc(self, dim: int) -> None:
        self.dim = int(dim)
        self.acts = torch.empty((self.max_size, self.dim), dtype=self.dtype, device="cpu")
        self.labels = torch.empty((self.max_size,), dtype=torch.int16, device="cpu")

    def add(self, activation: torch.Tensor, label: torch.Tensor) -> None:
        a = activation.detach().to(dtype=self.dtype, device="cpu")
        if a.dim() == 2 and a.size(0) == 1:
            a = a.squeeze(0)
        elif a.dim() != 1:
            a = a.view(-1)

        if self.acts is None:
            self._lazy_alloc(a.numel())

        self.acts[self.head].copy_(a)

        y = label.detach().to("cpu")
        if y.dim() > 0:
            y = y.view(-1)[0]
        self.labels[self.head] = int(y.item())

        self.head = (self.head + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        out_dtype: Optional[torch.dtype] = None,
        strict: bool = False,
    ):
        """Uniform sampling with replacement. Returns [B, D], [B]."""
        if self.acts is None or self.labels is None:
            raise ValueError("ActivationCache not initialized. Call add() first.")
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


