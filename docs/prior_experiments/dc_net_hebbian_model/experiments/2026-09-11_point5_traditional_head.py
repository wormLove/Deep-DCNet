"""
Point #5 (traditional classification layer) comparison experiment.

Reuses the already-trained, checkpointed discrimination layer from the full
baseline reproduction (checkpoints/single_layer_mnist_h1000_full.pt - 94.88%
final / 95.15% best, see RESULT/single_layer_full_baseline/2026-09-10_17-31-42/)
as a FROZEN feature extractor, instead of re-running the multi-hour Hebbian
training pipeline. Each image's discrimination-layer activation is computed
once (forward-only, no backward, no organize() - the frozen layer never
changes), then ONLY the new TraditionalMLPHead (modules/classifier_heads.py)
is trained on those precomputed activations via standard batched backprop.

This is a fair, controlled comparison: same frozen features the ReadoutHead
baseline used, only the head architecture changes (single linear layer vs.
Linear -> ReLU -> Dropout -> Linear). Expected runtime: a few minutes total
(precompute activations for all 70k images once, then a handful of epochs
of ordinary mini-batch training on 1000-dim vectors) - nowhere near the
hours a full retrain would take, since organize() and the per-sample review
mechanism are skipped entirely.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms

import utils.utils as util
import utils.transformations as transformations
from core.checkpointing import load_layer
from modules.classifier_heads import TraditionalMLPHead

util.set_random_seed(33, deterministic=True)

# --- experiment knobs -------------------------------------------------
BASELINE_CHECKPOINT = "single_layer_mnist_h1000_full"  # frozen features from the 94.88%/95.15% baseline
HIDDEN_DIMS = (256,)   # TraditionalMLPHead's hidden layer(s)
DROPOUT = 0.2
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
# ------------------------------------------------------------------------


def precompute_activations(discrimination_layer, dataset, d_transform, device, log, log_every=10000):
    """
    Run every image through the frozen discrimination layer once (forward
    only - no grad, no organize) and stack the resulting activation vectors.
    DiscriminationModule.forward only accepts single-row [1, in_dim] input
    (see modules/discrimination.py's assert), so this loops image-by-image,
    same as every other experiment file in this repo - it's just much
    cheaper here since there's no backward pass or Hebbian update per step.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    acts, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = d_transform(x).to(device)
            act = discrimination_layer(x)
            acts.append(act.squeeze(0).cpu())
            labels.append(y.item())
            if (i + 1) % log_every == 0:
                log(f"    ...{i + 1}/{len(dataset)} ({time.time() - t0:.1f}s elapsed)")
    return torch.stack(acts), torch.tensor(labels, dtype=torch.long)


def main(data_dir, res_dir, checkpoints_dir, log_path):
    device = torch.device("cpu")
    os.makedirs(res_dir, exist_ok=True)

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    mnist_tr = MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=False)
    mnist_te = MNIST(root=data_dir, train=False, transform=transforms.ToTensor(), download=False)
    d_transform = transformations.Compose([transformations.Scale(), transformations.ToVector()])

    log(f"Loading frozen discrimination layer checkpoint: {BASELINE_CHECKPOINT}")
    discrimination_layer = load_layer(BASELINE_CHECKPOINT, checkpoints_dir=checkpoints_dir).to(device)
    discrimination_layer.eval()  # important: eval() (not just requires_grad=False) so forward()
                                  # skips organizer.step() entirely - the layer must stay frozen
    hidden_dim = discrimination_layer.neuron_weights.shape[1]
    log(f"  hidden_dim={hidden_dim}, frozen (all params requires_grad=False)")

    log("Precomputing train-set activations (forward-only, no organize)...")
    t0 = time.time()
    train_acts, train_labels = precompute_activations(discrimination_layer, mnist_tr, d_transform, device, log)
    log(f"  done in {time.time() - t0:.1f}s -> {tuple(train_acts.shape)}")

    log("Precomputing test-set activations...")
    t0 = time.time()
    test_acts, test_labels = precompute_activations(discrimination_layer, mnist_te, d_transform, device, log)
    log(f"  done in {time.time() - t0:.1f}s -> {tuple(test_acts.shape)}")

    train_loader = DataLoader(TensorDataset(train_acts, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_acts, test_labels), batch_size=1000, shuffle=False)

    head = TraditionalMLPHead(hidden_dim, 10, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head.parameters(), lr=LR)

    def evaluate():
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                correct += (head(xb).argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        head.train()
        return 100.0 * correct / total

    log(f"\nTraining TraditionalMLPHead (hidden_dims={HIDDEN_DIMS}, dropout={DROPOUT}, "
        f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR})")

    best_acc = evaluate()
    log(f"[Epoch 0] (untrained head) TestAcc: {best_acc:.2f}%")

    t0 = time.time()
    final_acc = best_acc
    for epoch in range(1, EPOCHS + 1):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
        final_acc = evaluate()
        tag = ""
        if final_acc > best_acc:
            best_acc = final_acc
            tag = " [NEW BEST]"
            torch.save(head.state_dict(), os.path.join(res_dir, "best_head.pth"))
        log(f"[Epoch {epoch}] TestAcc: {final_acc:.2f}%{tag}")

    torch.save(head.state_dict(), os.path.join(res_dir, "final_head.pth"))
    log(f"\nTraining done in {time.time() - t0:.1f}s")
    log(f"[FINAL] epoch={EPOCHS}, acc={final_acc:.2f}%")
    log(f"[BEST]  acc={best_acc:.2f}%")
    log(
        "\nComparison, same frozen discrimination-layer features "
        f"({BASELINE_CHECKPOINT}):\n"
        "  ReadoutHead      (single linear layer):        94.88% final / 95.15% best\n"
        f"  TraditionalMLPHead (Linear->ReLU->Dropout->Linear): {final_acc:.2f}% final / {best_acc:.2f}% best"
    )


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "DATA")
    RES_DIR = os.path.join(BASE_DIR, "RESULT", "point5_traditional_head", util.get_timestamp())
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(RES_DIR, exist_ok=True)
    LOG_PATH = os.path.join(RES_DIR, "log.txt")
    main(DATA_DIR, RES_DIR, CKPT_DIR, LOG_PATH)
