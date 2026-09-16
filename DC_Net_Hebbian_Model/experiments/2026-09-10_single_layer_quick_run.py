"""
Real MNIST, real code, deliberately short: a "small experiment" to validate
the reorganized single_layer architecture end to end, not a full baseline
reproduction (that's max_steps=None over the full 60k-step epoch, like the
original train.py). Uses hidden_dim=300 (vs. 1000 in the original baseline)
and max_steps=3000 (vs. a full epoch) specifically to finish in a few
minutes. Result: 10.28% (untrained) -> 60.36% after 3000/60000 steps.
"""
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
from core.checkpointing import save_layer, load_layer
from training.loop import train_with_review

util.set_random_seed(33, deterministic=True)


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
        hidden_dim=300,
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
        eval_interval=3000,  # keep to start + final eval only - full-test-set eval is ~70s each,
                              # frequent mid-training eval is what makes a "quick" run not quick.
        max_steps=3000,
        log_path=log_path,
        r_cap=12,
        save_best=True,
    )

    # promote the trained discrimination layer as a reusable checkpoint
    save_layer(
        model.discrimination_layer,
        name="single_layer_mnist_h300_quick",
        checkpoints_dir=checkpoints_dir,
        meta={
            "architecture": "single_layer",
            "activity_optimizer": "least_squares",
            "source_experiment": "experiments/2026-09-10_single_layer_quick_run.py",
            "dataset": "mnist",
            "hidden_dim": 300,
            "steps_trained": 3000,
            "notes": "quick real-MNIST run (3000/60000 steps), not the full baseline",
        },
        overwrite=True,
    )
    print(f"Saved reusable checkpoint: single_layer_mnist_h300_quick "
          f"({checkpoints_dir}/manifest.json)")

    # round-trip sanity check
    reloaded = load_layer("single_layer_mnist_h300_quick", checkpoints_dir=checkpoints_dir)
    probe = torch.rand(1, 784)
    with torch.no_grad():
        same = torch.allclose(model.discrimination_layer(probe), reloaded(probe))
    print(f"Checkpoint round-trip identical: {same}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "DATA")
    RES_DIR = os.path.join(BASE_DIR, "RESULT", "single_layer_quick_run", util.get_timestamp())
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    main(DATA_DIR, RES_DIR, CKPT_DIR)
