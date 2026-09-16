"""
Shared training loop for any DCNet architecture that exposes:
  - model(x, y=None, record_cache=False) -> logits
  - model.organize()
  - model.review_after_organize()   (no-op if review isn't enabled)
  - model.save_model(path, include_training_state=...)
  - model.is_stable() / model._review_enabled / model._p / model.activation_cache

This is train.py's train_with_review() and evaluate(), unchanged in logic,
pulled out so single-layer and stacked experiments (and future
architectures) share one implementation instead of each hand-editing a
copy. See experiments/ for the thin per-run wiring that calls this.
"""
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate classification accuracy (%) on a dataloader."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            y_pred = model(x)
            pred = y_pred.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return 100.0 * correct / max(total, 1)


def train_with_review(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    output_dir: str,
    organize_interval: int = 200,
    eval_interval: Optional[int] = None,
    log_path: Optional[str] = None,
    r_cap: int = 12,
    max_steps: Optional[int] = None,
    save_best: bool = True,
) -> None:
    """
    Training strategy:
      1) Normal step: forward through (DL + readout). If stable and review is enabled, cache activations.
      2) Review steps: train readout only using cached activations.
         R = ceil(p * r_cap), where p is a stability-dependent review ratio maintained by the model.
      3) Every 'organize_interval' steps: call model.organize() and model.review_after_organize().
      4) Evaluation runs every 'eval_interval' steps. If eval_interval is None, it defaults to organize_interval.
      5) Always run one final evaluation after training ends.
    """

    os.makedirs(output_dir, exist_ok=True)

    if eval_interval is None:
        eval_interval = organize_interval

    model.train()
    step = 0
    last_loss = float("nan")

    best_acc = -1.0
    best_step = -1
    best_path = None

    t_block_start = time.time()

    for x, y in tqdm(train_loader, desc="Train"):
        if step > 0 and step % organize_interval == 0:
            model.organize()
            model.review_after_organize()

            t_now = time.time()
            if log_path:
                with open(log_path, "a") as f:
                    f.write(f"[Step {step}] BlockTime: {t_now - t_block_start:.2f}s\n")
                    f.write(f"[Step {step}] LastLoss: {last_loss:.6f}\n")
                    if hasattr(model, "monitor_last_dl"):
                        f.write(f"[Step {step}] WeightDrift: {model.monitor_last_dl.last_drift}\n")
                    if hasattr(model, "_p") and hasattr(model, "activation_cache"):
                        f.write(f"[Step {step}] ReviewP: {model._p:.3f}, CacheSize: {len(model.activation_cache)}\n")

            t_block_start = time.time()

        if eval_interval > 0 and step % eval_interval == 0:
            acc = evaluate(model, test_loader)
            model.train()

            if save_best and acc > best_acc:
                best_acc = acc
                best_step = step
                best_path = os.path.join(output_dir, f"best_step_{step}.pth")
                model.save_model(best_path)

            if log_path:
                t_now = time.time()
                with open(log_path, "a") as f:
                    f.write(f"[Step {step}] TestAcc: {acc:.2f}%\n")
                    f.write(f"[Step {step}] EvalTime: {t_now - t_block_start:.2f}s\n")
                    if save_best and best_step == step:
                        f.write(f"[NEW BEST] acc={best_acc:.2f}% at step={best_step}\n")
                t_block_start = time.time()

        record_cache = getattr(model, "_review_enabled", False) and model.is_stable()

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x, y=y, record_cache=record_cache)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())

        p = getattr(model, "_p", 0.0) if getattr(model, "_review_enabled", False) else 0.0
        if p > 0.0 and model.activation_cache.ready():
            r = int(math.ceil(p * r_cap))
            acts_buf, y_buf = model.activation_cache.sample_batch(
                r, device=next(model.parameters()).device, out_dtype=torch.float32
            )

            for i in range(r):
                act_b = acts_buf[i].unsqueeze(0)
                y_b = y_buf[i].unsqueeze(0)

                optimizer.zero_grad(set_to_none=True)
                y_pred_rev = model.forward_with_activations(act_b)
                loss_rev = criterion(y_pred_rev, y_b)
                loss_rev.backward()
                optimizer.step()

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    final_acc = evaluate(model, test_loader)
    model.train()

    if log_path:
        with open(log_path, "a") as f:
            f.write(f"[FINAL] step={step}, acc={final_acc:.2f}%\n")

    final_path = os.path.join(output_dir, "final.pth")
    model.save_model(final_path)

    if log_path:
        with open(log_path, "a") as f:
            f.write(f"[BEST] acc={best_acc:.2f}% at step={best_step}; path={best_path}\n")
