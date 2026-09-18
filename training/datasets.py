"""
Dataset registry - one place that knows how to download, store and load
every dataset the training scripts can use (Prof. Yu's point #6).

Adding a dataset = adding one DatasetSpec to DATASETS below. Everything
downstream (train_classifier.py, train_classifier_stacked.py, the dated
experiment scripts, the point #4 robustness sweep) picks input_dim /
num_classes / image shape up from the spec, so nothing else needs to change.

Storage: every dataset lives under <data_root>/<torchvision folder>, i.e.
the project-local DATA/ by default (gitignored). torchvision skips files
that are already present, so downloading is idempotent.

    # list what is available and what is already on disk
    python training/datasets.py list

    # download (run this on the HPC LOGIN node - compute nodes have no internet)
    python training/datasets.py download mnist fashion_mnist cifar10
    python training/datasets.py download all

    # in code
    from training.datasets import get_dataset
    spec, train_ds, test_ds = get_dataset("fashion_mnist", data_root="DATA")
    spec.input_dim, spec.num_classes, spec.image_chw   # 784, 10, (1, 28, 28)

Notes on the two non-MNIST starters:
- fashion_mnist: exact drop-in for MNIST (1x28x28, 10 classes, 60k/10k).
- cifar10: 3x32x32 RGB flattened to 3072-d, 10 classes, 50k/10k. Natural
  images are a much harder target for a single sparse Hebbian layer than
  digits; treat the first numbers as a starting point, not a verdict.
- cifar10_gray: same images converted to 1x32x32 (1024-d). Cheaper and
  keeps the "one input channel" assumption the MNIST pipeline was built
  around; use it to separate "more pixels" from "colour" when comparing.
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from torchvision import datasets as tvd
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "DATA"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tv_class: Callable                      # torchvision dataset class
    image_chw: Tuple[int, int, int]         # what one sample looks like AFTER `transform`
    num_classes: int
    transform: Callable                     # PIL -> tensor in [0, 1], shape image_chw
    description: str = ""
    tv_kwargs: Dict = field(default_factory=dict)   # extra ctor kwargs (none of ours need any)
    folder: str = ""                        # subfolder torchvision uses under data_root (for `list`)

    @property
    def input_dim(self) -> int:
        c, h, w = self.image_chw
        return c * h * w

    @property
    def patch_hw(self) -> Tuple[int, int]:
        """
        (rows, cols) used by AnalysisEngine to draw one discrimination-layer
        neuron's incoming weights as an image. For 1-channel data this is the
        image itself; for RGB the three channel planes are stacked vertically
        (3*H rows) so the same greyscale patch-grid code keeps working.
        """
        c, h, w = self.image_chw
        return (c * h, w)


_TO_TENSOR = transforms.ToTensor()
_GRAY_TO_TENSOR = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

DATASETS: Dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist", tv_class=tvd.MNIST, image_chw=(1, 28, 28), num_classes=10,
        transform=_TO_TENSOR, folder="MNIST",
        description="Handwritten digits, 60k/10k. The original DCNet benchmark.",
    ),
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist", tv_class=tvd.FashionMNIST, image_chw=(1, 28, 28), num_classes=10,
        transform=_TO_TENSOR, folder="FashionMNIST",
        description="Zalando clothing items, 60k/10k. Drop-in replacement for MNIST, harder.",
    ),
    "kmnist": DatasetSpec(
        name="kmnist", tv_class=tvd.KMNIST, image_chw=(1, 28, 28), num_classes=10,
        transform=_TO_TENSOR, folder="KMNIST",
        description="Kuzushiji (cursive Japanese) characters, 60k/10k. Another MNIST drop-in.",
    ),
    "cifar10": DatasetSpec(
        name="cifar10", tv_class=tvd.CIFAR10, image_chw=(3, 32, 32), num_classes=10,
        transform=_TO_TENSOR, folder="cifar-10-batches-py",
        description="Natural colour images, 50k/10k, flattened to 3072-d.",
    ),
    "cifar10_gray": DatasetSpec(
        name="cifar10_gray", tv_class=tvd.CIFAR10, image_chw=(1, 32, 32), num_classes=10,
        transform=_GRAY_TO_TENSOR, folder="cifar-10-batches-py",
        description="CIFAR-10 converted to greyscale, 1024-d. Same files on disk as cifar10.",
    ),
}

DATASET_CHOICES = tuple(DATASETS.keys())


def get_spec(name: str) -> DatasetSpec:
    key = name.lower().replace("-", "_")
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(DATASET_CHOICES)}")
    return DATASETS[key]


def is_downloaded(name: str, data_root=DEFAULT_DATA_ROOT) -> bool:
    """Cheap on-disk check (does torchvision's folder exist and is non-empty)."""
    spec = get_spec(name)
    folder = Path(data_root) / spec.folder
    return folder.is_dir() and any(folder.iterdir())


def get_dataset(name: str, data_root=DEFAULT_DATA_ROOT, download: bool = True):
    """
    Returns (spec, train_dataset, test_dataset). Both datasets yield
    (tensor of shape spec.image_chw in [0, 1], int label).

    download=True only fetches files that are missing. If the machine has
    no internet (HPC compute nodes), the torchvision error is re-raised
    with the exact command to run on the login node instead.
    """
    spec = get_spec(name)
    root = str(data_root)
    os.makedirs(root, exist_ok=True)
    try:
        train_ds = spec.tv_class(root=root, train=True, transform=spec.transform, download=download, **spec.tv_kwargs)
        test_ds = spec.tv_class(root=root, train=False, transform=spec.transform, download=download, **spec.tv_kwargs)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load dataset '{spec.name}' from {root} "
            f"(download={'on' if download else 'off'}). If this machine has no internet "
            f"(e.g. an HPC compute node), download it first from a login node with:\n"
            f"    python training/datasets.py download {spec.name}\n"
            f"Original error: {exc}"
        ) from exc
    return spec, train_ds, test_ds


def download(names: List[str], data_root=DEFAULT_DATA_ROOT) -> None:
    if len(names) == 1 and names[0].lower() == "all":
        names = list(DATASET_CHOICES)
    for n in names:
        spec = get_spec(n)
        print(f"[{spec.name}] downloading to {Path(data_root).resolve()} (skips files already present) ...")
        get_dataset(spec.name, data_root=data_root, download=True)
        print(f"[{spec.name}] ok")


def _cli():
    parser = argparse.ArgumentParser(description="Dataset registry: list / download datasets into DATA/.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show registered datasets and whether they are on disk")
    p_dl = sub.add_parser("download", help="download one or more datasets (or 'all')")
    p_dl.add_argument("names", nargs="+", help=f"any of: {', '.join(DATASET_CHOICES)}, or 'all'")
    args = parser.parse_args()

    if args.cmd == "list":
        print(f"data_root: {Path(args.data_root).resolve()}\n")
        print(f"{'name':<14}{'input_dim':>10}{'classes':>9}  {'on disk':<8} description")
        for spec in DATASETS.values():
            on_disk = "yes" if is_downloaded(spec.name, args.data_root) else "no"
            print(f"{spec.name:<14}{spec.input_dim:>10}{spec.num_classes:>9}  {on_disk:<8} {spec.description}")
    elif args.cmd == "download":
        download(args.names, data_root=args.data_root)


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    _cli()
