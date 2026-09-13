import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def evaluate_classifier(model, loader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    progress = tqdm(loader, total=len(loader), desc="Classifier Eval", leave=False)
    for x, y in progress:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
        progress.set_postfix(samples=total, acc=f"{(100.0 * correct / max(total, 1)):.2f}%")
    model.train()
    return 100.0 * correct / max(total, 1)


def train_classifier_plain(
    model,
    train_loader,
    test_loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer,
    organize_interval_samples: int = 200,
    eval_interval_samples: int = 0,
    analysis=None,
    enable_checkpoints: bool = True,
    debug_stats: bool = False,
    summarize_cycle_fn=None,
    quiet_train: bool = False,
    eval_logger: Optional[Callable[[str, int, int, float], None]] = None,
    train_logger: Optional[Callable[[str], None]] = None,
):
    total_steps = 0
    total_samples = 0
    last_organize_samples = 0
    last_eval_samples = 0
    organize_count = 0
    last_loss = 0.0

    progress = tqdm(train_loader, total=len(train_loader), desc="Classifier Train", leave=True)
    for x, y in progress:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        total_steps += 1
        total_samples += x.shape[0]

        if analysis is not None:
            analysis.save_step_state(model, total_steps)
        progress.set_postfix(step=total_steps, samples=total_samples, loss=f"{last_loss:.4f}")

        if total_samples - last_organize_samples >= organize_interval_samples:
            weights_before = model.discrimination_layer.neuron_weights.detach().clone()
            model.organize()
            last_organize_samples = total_samples
            organize_count += 1
            diag = model.diagnostics()
            if analysis is not None:
                analysis.save_organize_state(model, total_steps)
                if enable_checkpoints:
                    analysis.save_checkpoint(model, total_steps)
            weights_after = model.discrimination_layer.neuron_weights.detach()
            weight_shift = torch.norm(weights_after - weights_before).item()
            if eval_interval_samples > 0 and total_samples - last_eval_samples >= eval_interval_samples:
                acc = evaluate_classifier(model, test_loader, device)
                last_eval_samples = total_samples
                msg = (
                    f"[organize {organize_count}] "
                    f"step={total_steps}, samples={total_samples}, loss={last_loss:.6f}, "
                    f"weight_shift={weight_shift:.6f}, acc={acc:.2f}%"
                )
                if not quiet_train:
                    print(msg)
                if train_logger is not None:
                    train_logger(msg)
                if eval_logger is not None:
                    eval_logger("organize", total_steps, total_samples, acc)
            else:
                msg = (
                    f"[organize {organize_count}] "
                    f"step={total_steps}, samples={total_samples}, loss={last_loss:.6f}, "
                    f"weight_shift={weight_shift:.6f}"
                )
                if not quiet_train:
                    print(msg)
                if train_logger is not None:
                    train_logger(msg)
            if debug_stats and summarize_cycle_fn is not None:
                detail = "  " + summarize_cycle_fn(diag)
                if not quiet_train:
                    print(detail)
                if train_logger is not None:
                    train_logger(detail)

    if total_samples > last_organize_samples:
        weights_before = model.discrimination_layer.neuron_weights.detach().clone()
        model.organize()
        organize_count += 1
        diag = model.diagnostics()
        if analysis is not None:
            analysis.save_organize_state(model, total_steps)
            if enable_checkpoints:
                analysis.save_checkpoint(model, total_steps)
        weights_after = model.discrimination_layer.neuron_weights.detach()
        weight_shift = torch.norm(weights_after - weights_before).item()
        acc = evaluate_classifier(model, test_loader, device)
        msg = (
            f"[final organize] step={total_steps}, samples={total_samples}, loss={last_loss:.6f}, "
            f"weight_shift={weight_shift:.6f}, acc={acc:.2f}%"
        )
        if not quiet_train:
            print(msg)
        if train_logger is not None:
            train_logger(msg)
        if debug_stats and summarize_cycle_fn is not None:
            detail = "  " + summarize_cycle_fn(diag)
            if not quiet_train:
                print(detail)
            if train_logger is not None:
                train_logger(detail)
        if eval_logger is not None:
            eval_logger("final_organize", total_steps, total_samples, acc)

    final_acc = evaluate_classifier(model, test_loader, device)
    if analysis is not None and enable_checkpoints:
        analysis.save_final_checkpoint(model)
    msg = f"[final eval] acc={final_acc:.2f}%"
    if not quiet_train:
        print(msg)
    if train_logger is not None:
        train_logger(msg)
    if eval_logger is not None:
        eval_logger("final_eval", total_steps, total_samples, final_acc)
    return final_acc


def train_classifier_with_review(
    model,
    train_loader,
    test_loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer,
    organize_interval_samples: int = 200,
    eval_interval_samples: int = 0,
    analysis=None,
    enable_checkpoints: bool = True,
    debug_stats: bool = False,
    summarize_cycle_fn=None,
    review_per_sample_max: float = 10.0,
    review_error_repeat: int = 1,
    quiet_train: bool = False,
    eval_logger: Optional[Callable[[str, int, int, float], None]] = None,
    train_logger: Optional[Callable[[str], None]] = None,
):
    total_steps = 0
    total_samples = 0
    last_organize_samples = 0
    last_eval_samples = 0
    organize_count = 0
    last_loss = 0.0

    progress = tqdm(train_loader, total=len(train_loader), desc="Classifier Train", leave=True)
    for x, y in progress:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, return_intermediate=True)
        logits = out["logits"]
        # When an integration layer is present, cache cat([layer0, layer1]) so
        # that review replay can pass through the full integration → readout path.
        # Without an integration layer this falls back to the sparse Layer 1 act.
        act = out.get("integration_input", out["act"]).detach()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        record_cache = model._review_enabled and model.is_stable()
        if record_cache:
            pred = logits.detach().argmax(dim=1)
            model.activation_cache.add(act, y)
            if review_error_repeat > 0:
                incorrect_mask = pred.ne(y)
                if incorrect_mask.any():
                    wrong_act = act[incorrect_mask]
                    wrong_y = y[incorrect_mask]
                    for _ in range(review_error_repeat):
                        model.activation_cache.add(wrong_act, wrong_y)

        p = model._p if model._review_enabled else 0.0
        if p > 0.0 and model.activation_cache.ready():
            review_batch_size = int(math.ceil(x.shape[0] * p * review_per_sample_max))
            if review_batch_size > 0:
                acts_big, y_big = model.review_sample(review_batch_size, device=device, out_dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                y_pred_rev = model.forward_with_activations(acts_big)
                loss_rev = criterion(y_pred_rev, y_big)
                loss_rev.backward()
                optimizer.step()

        last_loss = float(loss.item())
        total_steps += 1
        total_samples += x.shape[0]

        if analysis is not None:
            analysis.save_step_state(model, total_steps)
        progress.set_postfix(step=total_steps, samples=total_samples, loss=f"{last_loss:.4f}")

        if total_samples - last_organize_samples >= organize_interval_samples:
            weights_before = model.discrimination_layer.neuron_weights.detach().clone()
            model.organize()
            model.review_after_organize()
            last_organize_samples = total_samples
            organize_count += 1
            diag = model.diagnostics()
            if analysis is not None:
                analysis.save_organize_state(model, total_steps)
                if enable_checkpoints:
                    analysis.save_checkpoint(model, total_steps)
            weights_after = model.discrimination_layer.neuron_weights.detach()
            weight_shift = torch.norm(weights_after - weights_before).item()
            if eval_interval_samples > 0 and total_samples - last_eval_samples >= eval_interval_samples:
                acc = evaluate_classifier(model, test_loader, device)
                last_eval_samples = total_samples
                msg = (
                    f"[organize {organize_count}] "
                    f"step={total_steps}, samples={total_samples}, loss={last_loss:.6f}, "
                    f"weight_shift={weight_shift:.6f}, acc={acc:.2f}%, review_p={model._p:.3f}, "
                    f"cache_size={len(model.activation_cache)}"
                )
                if not quiet_train:
                    print(msg)
                if train_logger is not None:
                    train_logger(msg)
                if eval_logger is not None:
                    eval_logger("organize", total_steps, total_samples, acc)
            else:
                msg = (
                    f"[organize {organize_count}] "
                    f"step={total_steps}, samples={total_samples}, loss={last_loss:.6f}, "
                    f"weight_shift={weight_shift:.6f}, review_p={model._p:.3f}, "
                    f"cache_size={len(model.activation_cache)}"
                )
                if not quiet_train:
                    print(msg)
                if train_logger is not None:
                    train_logger(msg)
            if debug_stats and summarize_cycle_fn is not None:
                detail = "  " + summarize_cycle_fn(diag)
                if not quiet_train:
                    print(detail)
                if train_logger is not None:
                    train_logger(detail)

    if total_samples > last_organize_samples:
        weights_before = model.discrimination_layer.neuron_weights.detach().clone()
        model.organize()
        model.review_after_organize()
        organize_count += 1
        diag = model.diagnostics()
        if analysis is not None:
            analysis.save_organize_state(model, total_steps)
            if enable_checkpoints:
                analysis.save_checkpoint(model, total_steps)
        weights_after = model.discrimination_layer.neuron_weights.detach()
        weight_shift = torch.norm(weights_after - weights_before).item()
        acc = evaluate_classifier(model, test_loader, device)
        msg = (
            f"[final organize] step={total_steps}, samples={total_samples}, loss={last_loss:.6f}, "
            f"weight_shift={weight_shift:.6f}, acc={acc:.2f}%, review_p={model._p:.3f}, "
            f"cache_size={len(model.activation_cache)}"
        )
        if not quiet_train:
            print(msg)
        if train_logger is not None:
            train_logger(msg)
        if debug_stats and summarize_cycle_fn is not None:
            detail = "  " + summarize_cycle_fn(diag)
            if not quiet_train:
                print(detail)
            if train_logger is not None:
                train_logger(detail)
        if eval_logger is not None:
            eval_logger("final_organize", total_steps, total_samples, acc)

    final_acc = evaluate_classifier(model, test_loader, device)
    if analysis is not None and enable_checkpoints:
        analysis.save_final_checkpoint(model)
    msg = f"[final eval] acc={final_acc:.2f}%"
    if not quiet_train:
        print(msg)
    if train_logger is not None:
        train_logger(msg)
    if eval_logger is not None:
        eval_logger("final_eval", total_steps, total_samples, final_acc)
    return final_acc
