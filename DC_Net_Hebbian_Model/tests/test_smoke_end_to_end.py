"""
Fast, dependency-light integration check: trains a tiny single-layer model
for a few hundred steps on sklearn's bundled `digits` dataset (no MNIST
download needed), then verifies a checkpoint save -> reload round-trip is
exact. Run this any time you touch core/, modules/, or architectures/ to
catch an import or shape break in seconds, without needing MNIST or a
17-hour run.

This is NOT a substitute for eval/reproduce_baseline.py (not yet built) -
that one should check against the real, validated MNIST numbers
(95.23% single-layer / 92.59% 2-layer stack). This test only proves the
pipeline runs and checkpoints round-trip correctly.

Requires scikit-learn (pip install scikit-learn) in addition to
requirements.txt, just for the bundled digits dataset.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset

import utils.utils as util
import utils.transformations as transformations
from architectures.registry import ARCHITECTURES
from core.initializers import DatasetInitializer
from core.checkpointing import save_layer, load_layer


def run_smoke_test(max_steps: int = 600, organize_interval: int = 50):
    util.set_random_seed(33, deterministic=True)

    digits = load_digits()
    X = torch.tensor(digits.data, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)
    n_train = int(len(X) * 0.8)
    train_loader = DataLoader(TensorDataset(X[:n_train], y[:n_train]), batch_size=1, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[n_train:], y[n_train:]), batch_size=1, shuffle=False)

    d_transform = transformations.Compose([transformations.Scale(), transformations.ToVector()])
    discr_init = DatasetInitializer(dataset=train_loader.dataset, transforms=d_transform, init_ratio=0.5)

    model = ARCHITECTURES["single_layer"](
        input_dim=64, hidden_dim=120, output_dim=10,
        data_initializer=discr_init, transform=d_transform,
        non_negative_strategy="shift", discrimination_config={"beta": 0.98},
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.enable_review()
    model.train()

    step = 0
    for x, y_ in train_loader:
        if step > 0 and step % organize_interval == 0:
            model.organize()
            model.review_after_organize()

        record_cache = model._review_enabled and model.is_stable()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x, y=y_, record_cache=record_cache), y_)
        loss.backward()
        optimizer.step()

        p = model._p if model._review_enabled else 0.0
        if p > 0.0 and model.activation_cache.ready():
            r = int(math.ceil(p * 12))
            acts_buf, y_buf = model.activation_cache.sample_batch(
                r, device=next(model.parameters()).device, out_dtype=torch.float32
            )
            for i in range(r):
                optimizer.zero_grad(set_to_none=True)
                loss_rev = criterion(model.forward_with_activations(acts_buf[i].unsqueeze(0)), y_buf[i].unsqueeze(0))
                loss_rev.backward()
                optimizer.step()

        step += 1
        if step >= max_steps:
            break

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y_ in test_loader:
            correct += (model(x).argmax(dim=1) == y_).sum().item()
            total += y_.size(0)
    acc = 100.0 * correct / total

    return model, acc


def test_pipeline_runs_and_checkpoints_roundtrip(tmp_path=None):
    model, acc = run_smoke_test()
    assert acc > 20.0, f"sanity floor: expected clearly-better-than-random accuracy, got {acc:.2f}%"

    ckpt_dir = str(tmp_path) if tmp_path is not None else os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "_test_scratch"
    )
    save_layer(
        model.discrimination_layer, name="pytest_scratch", checkpoints_dir=ckpt_dir,
        meta={"architecture": "single_layer", "notes": "test-only, safe to delete"},
        overwrite=True,
    )
    reloaded = load_layer("pytest_scratch", checkpoints_dir=ckpt_dir)

    probe = torch.rand(1, 64)
    with torch.no_grad():
        out_a = model.discrimination_layer(probe)
        out_b = reloaded(probe)
    assert torch.allclose(out_a, out_b), "checkpoint round-trip produced different outputs"

    print(f"[smoke test] acc={acc:.2f}% (sklearn digits, not MNIST), checkpoint round-trip OK")


if __name__ == "__main__":
    test_pipeline_runs_and_checkpoints_roundtrip()
