import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import utils.utils as util
import utils.transformations as transformations
from architectures.registry import ARCHITECTURES
from core.initializers import DatasetInitializer
from core.checkpointing import save_layer
from training.loop import train_with_review

util.set_random_seed(33, deterministic=True)

# Safe to change - affects only how often accuracy is checked, not the result.
EVAL_INTERVAL = 2000  # original default; raise (e.g. 2000) for a much faster wall-clock run


def main(data_dir, res_dir, checkpoints_dir):
    device = torch.device("cpu")
    os.makedirs(res_dir, exist_ok=True)
    log_path = os.path.join(res_dir, "log.txt")

    mnist_tr = MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=False)
    mnist_te = MNIST(root=data_dir, train=False, transform=transforms.ToTensor(), download=False)

    d_transform = transformations.Compose([transformations.Scale(), transformations.ToVector()])

    train_loader = DataLoader(mnist_tr, batch_size=1, shuffle=True)
    test_loader = DataLoader(mnist_te, batch_size=1, shuffle=False)

    discr_init = DatasetInitializer(dataset=mnist_tr, transforms=d_transform, init_ratio=0.25)

    model = ARCHITECTURES["single_layer"](
        input_dim=784,
        hidden_dim=1000,
        output_dim=10,
        data_initializer=discr_init,
        transform=d_transform,
        non_negative_strategy="shift",
        discrimination_config={"beta": 199 / 200},
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.enable_review()

    train_with_review(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        output_dir=res_dir,
        organize_interval=200,
        eval_interval=EVAL_INTERVAL,
        max_steps=None,  # full epoch, matching the original baseline
        log_path=log_path,
        r_cap=12,
        save_best=True,
    )

    save_layer(
        model.discrimination_layer,
        name="single_layer_mnist_h1000_full",
        checkpoints_dir=checkpoints_dir,
        meta={
            "architecture": "single_layer",
            "activity_optimizer": "least_squares",
            "source_experiment": "experiments/2026-09-10_single_layer_full_baseline.py",
            "dataset": "mnist",
            "hidden_dim": 1000,
            "notes": "full-epoch baseline reproduction",
        },
        overwrite=True,
    )
    print(f"Saved reusable checkpoint: single_layer_mnist_h1000_full ({checkpoints_dir}/manifest.json)")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "DATA")
    RES_DIR = os.path.join(BASE_DIR, "RESULT", "single_layer_full_baseline", util.get_timestamp())
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    main(DATA_DIR, RES_DIR, CKPT_DIR)