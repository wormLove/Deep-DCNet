"""
Freeze-then-train protocol (Plan 03): training_mode / protocol = "freeze".

The one-pass protocols (plain, review) train the Hebbian layer(s) and the
backprop head at the same time, so the head reads features that are still
changing and "MLP head vs linear readout" mixes feature quality with head
training speed. "freeze" separates the two:

  1. Validation split    the last `val_size` images of a seeded permutation
                         of the training subset are held out as validation
                         (val_size=0: no split, the training subset is used
                         as-is). The test set is never used for any choice.
  2. Hebbian phase       no labels, no head: `hebbian_epochs` passes over the
                         fit set, organize() every N images plus a final
                         organize - the plain loop minus the head forward
                         and the backprop step. Stacked models learn
                         layer-wise (not the stability-gated cascade): L0
                         alone, freeze; then L1 on frozen L0's outputs,
                         freeze; and so on.
  3. Freeze              every discrimination layer -> eval(),
                         requires_grad=False. From here on nothing calls
                         organize() or accumulates potentials (run_freeze
                         checks at the end that the layers did not change).
  4. Feature extraction  one no-grad pass of fit / val / test through the
                         frozen layers; features kept in memory on CPU
                         (float32), never written to disk.
  5. Head training       Adam on the cached fit features, shuffled with the
                         run seed, up to head_epochs, early stopping on val
                         accuracy; the head from the best-val epoch is kept.
                         Test accuracy per epoch goes to head_metrics.csv
                         for curves only.

The head is trained here, not by training/training_engines.py's loops:
those always end with organize(), which would change a frozen layer.

Equivalence with plain: the entry points (training/train_classifier.py,
training/train_classifier_stacked.py) build the model exactly as the plain
path does and hand it to run_freeze. Between model construction and the
first fit batch the global RNG sees the same draws as in plain (the val
split uses its own generator; run_freeze replays the one draw of plain's
step-0 test-loader iterator, see there). So with val_size=0 the fit loader
shuffles the same images in the same order as plain, and a single-layer
freeze run learns the identical discrimination layer as the plain run with
the readout head (tools/compare_layers.py checks this). Caveats:
  - not with the MLP head: its dropout draws from the same GPU RNG that
    organize() uses (zero-column re-init, eigenvalue estimate), so plain+MLP
    is not bit-comparable;
  - on CUDA only: on a CPU device organize() draws from the CPU RNG, which
    plain's periodic test evaluations (eval_interval_samples) also advance.

experiments/head_on_frozen.py reuses phases 4-5 (run_head_on_frozen) to
train a new head on an already-finished freeze run's frozen layers.
"""
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from analysis.analysis_engine import AnalysisEngine
from analysis.metrics_log import MetricsLog
from core.checkpointing import save_layer, save_stack
from training.provenance import run_provenance, set_seed
from training.run_config import (
    write_failed_eval,
    write_final_eval,
    write_run_config,
)
from training.training_engines import _diag_metrics, _log_organize

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Plan 03 defaults (FULL size). experiments/_configs.py derives its SMOKE
# variant from this dict, so there is one source of truth.
FREEZE_DEFAULTS: Dict[str, Any] = {
    "val_size": 5000,
    "hebbian_epochs": 1,
    "head_epochs": 30,
    "head_batch_size": 128,
    "head_lr": 1e-3,
    "patience": 5,
}

HEAD_METRICS_FILENAME = "head_metrics.csv"
EXTRACT_BATCH_SIZE = 512   # features are per-sample (per-sample std threshold + top-k), so batch size only affects speed
EVAL_BATCH_SIZE = 4096     # head evaluation on cached features

Features = Tuple[torch.Tensor, torch.Tensor]   # (features [N, D] float32 CPU, labels [N] long CPU)
Logger = Callable[[str], None]

# provenance keys that describe the run that wrote a run_config.json, not the model
_PROVENANCE_KEYS = ("git_commit", "git_dirty", "env", "argv", "created_at", "_run_dir")


def _noop(_msg: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------
def discrimination_layers_of(model) -> List[nn.Module]:
    """[DL] for BiologicalClassifier, [DL_0, DL_1, ...] for StackedBiologicalClassifier."""
    if hasattr(model, "discrimination_layers"):
        return list(model.discrimination_layers)
    return [model.discrimination_layer]


def split_train_val(dataset, val_size: int, seed: Optional[int]):
    """
    (fit, val): val = the last `val_size` items of a permutation of
    `dataset` drawn from a private generator seeded with `seed` (0 when the
    run is unseeded), fit = the rest. The global RNG is not touched, so the
    model init and the fit loader's shuffle stay exactly as in plain.

    val_size <= 0 returns (dataset, None) - the very same object, in the
    original order, which is what makes the plain-equivalence check work.
    """
    n = len(dataset)
    val_size = int(val_size or 0)
    if val_size <= 0:
        return dataset, None
    if val_size >= n:
        raise ValueError(f"val_size={val_size} leaves no training images (training subset has {n})")
    gen = torch.Generator()
    gen.manual_seed(int(seed) if seed is not None else 0)
    perm = torch.randperm(n, generator=gen).tolist()
    return Subset(dataset, perm[: n - val_size]), Subset(dataset, perm[n - val_size:])


def freeze_layers(layers) -> None:
    """eval() (forward then skips all Hebbian accumulation) + requires_grad=False."""
    for dl in layers:
        dl.eval()
        for p in dl.parameters():
            p.requires_grad_(False)


def _snapshot(layers) -> List[Dict[str, torch.Tensor]]:
    return [{k: v.detach().clone() for k, v in dl.state_dict().items()} for dl in layers]


def _check_unchanged(layers, snapshot) -> None:
    for i, (dl, snap) in enumerate(zip(layers, snapshot)):
        for key, value in dl.state_dict().items():
            if not torch.equal(value, snap[key]):
                raise RuntimeError(
                    f"frozen discrimination layer {i} changed after freezing ({key}) - "
                    "something called organize() or accumulated potentials"
                )


def train_hebbian_single(
    model,
    fit_loader,
    device: torch.device,
    organize_interval_samples: int,
    hebbian_epochs: int = 1,
    analysis=None,
    enable_checkpoints: bool = True,
    metrics_log=None,
    log: Logger = _noop,
) -> Dict[str, int]:
    """
    Phase 1 for a BiologicalClassifier: train_classifier_plain's loop minus
    the head forward and the backprop step. Same per-batch forward of the
    discrimination layer (training mode, accumulate=True), same organize()
    schedule (every `organize_interval_samples` images, counted across
    epochs, plus one final organize for the remainder), same model.organize()
    call. metrics.csv rows: today's columns plus phase/layer/epoch; loss and
    test_acc stay blank (no head, no evaluation).
    """
    if int(hebbian_epochs) < 1:
        raise ValueError("hebbian_epochs must be >= 1")
    dl = model.discrimination_layer
    model.train()
    total_steps = 0
    total_samples = 0
    last_organize_samples = 0
    organize_count = 0
    t0 = time.time()

    def organize(epoch: int) -> None:
        nonlocal organize_count
        weights_before = dl.neuron_weights.detach().clone()
        model.organize()
        organize_count += 1
        diag = model.diagnostics()
        if analysis is not None:
            analysis.save_organize_state(model, total_steps)
            if enable_checkpoints:
                analysis.save_checkpoint(model, total_steps)
        weight_shift = torch.norm(dl.neuron_weights.detach() - weights_before).item()
        _log_organize(metrics_log, organize_count, total_steps, total_samples, None, None, t0,
                      weight_shift=weight_shift, **_diag_metrics(diag), phase="hebbian", layer=0, epoch=epoch)
        log(f"[hebbian L0 organize {organize_count}] epoch={epoch} step={total_steps} "
            f"samples={total_samples} weight_shift={weight_shift:.6f}")

    epoch = 0
    for epoch in range(1, int(hebbian_epochs) + 1):
        progress = tqdm(fit_loader, total=len(fit_loader), desc=f"Hebbian L0 (epoch {epoch}/{hebbian_epochs})", leave=True)
        for x, _ in progress:
            x = x.to(device)
            with torch.no_grad():
                dl(model.transform(x))
            total_steps += 1
            total_samples += x.shape[0]
            if analysis is not None:
                analysis.save_step_state(model, total_steps)
            progress.set_postfix(step=total_steps, samples=total_samples)
            if total_samples - last_organize_samples >= organize_interval_samples:
                organize(epoch)
                last_organize_samples = total_samples
    if total_samples > last_organize_samples:
        organize(epoch)
    return {"n_organizes": organize_count, "steps": total_steps, "samples": total_samples}


def train_hebbian_stacked(
    model,
    fit_loader,
    device: torch.device,
    organize_interval_samples: int,
    hebbian_epochs: int = 1,
    analysis=None,
    enable_checkpoints: bool = True,
    metrics_log=None,
    log: Logger = _noop,
) -> Dict[str, int]:
    """
    Phase 1 for a StackedBiologicalClassifier, layer-wise: for i = 0..N-1,
    layer i learns for `hebbian_epochs` passes over the fit set on the
    outputs of the already-frozen layers 0..i-1 (eval mode,
    accumulate=False), with the single-layer organize schedule, then is
    frozen. Layers after i are not run. The stability-gated cascade
    (model.organize()) is not used; each layer's own WeightDriftMonitor is
    still stepped after its organize() so drift/stable are recorded.

    metrics.csv: one row per organize with an unsuffixed column set plus
    layer=<i> (tidy format); organize counts across all layers, step/samples
    restart at 0 for each layer.
    """
    if int(hebbian_epochs) < 1:
        raise ValueError("hebbian_epochs must be >= 1")
    layers = list(model.discrimination_layers)
    organize_count = 0
    global_steps = 0   # only for full-level analysis filenames, so layers don't overwrite each other's files
    t0 = time.time()

    for i, dl in enumerate(layers):
        upstream = layers[:i]
        freeze_layers(upstream)
        dl.train()
        monitor = model.monitors[i]
        steps = 0
        samples = 0
        last_organize_samples = 0
        layer_organizes = 0

        def organize(epoch: int) -> None:
            nonlocal organize_count, layer_organizes
            weights_before = dl.neuron_weights.detach().clone()
            dl.organize()
            monitor.step(dl.neuron_weights)
            organize_count += 1
            layer_organizes += 1
            if analysis is not None:
                analysis.save_organize_state(model, global_steps)
                if enable_checkpoints:
                    analysis.save_checkpoint(model, global_steps)
            weight_shift = torch.norm(dl.neuron_weights.detach() - weights_before).item()
            _log_organize(metrics_log, organize_count, steps, samples, None, None, t0,
                          weight_shift=weight_shift, **_diag_metrics(dl.diagnostics()),
                          drift=monitor.last_drift, stable=monitor.state == "stable",
                          phase="hebbian", layer=i, epoch=epoch)
            log(f"[hebbian L{i} organize {layer_organizes}] epoch={epoch} step={steps} samples={samples} "
                f"weight_shift={weight_shift:.6f} drift={monitor.last_drift:.4f} stable={monitor.state == 'stable'}")

        epoch = 0
        for epoch in range(1, int(hebbian_epochs) + 1):
            progress = tqdm(fit_loader, total=len(fit_loader),
                            desc=f"Hebbian L{i} (epoch {epoch}/{hebbian_epochs})", leave=True)
            for x, _ in progress:
                x = x.to(device)
                with torch.no_grad():
                    h = model.transform(x)
                    for up in upstream:
                        h = up(h, accumulate=False)
                    dl(h)
                steps += 1
                samples += x.shape[0]
                global_steps += 1
                if analysis is not None:
                    analysis.save_step_state(model, global_steps)
                progress.set_postfix(step=steps, samples=samples)
                if samples - last_organize_samples >= organize_interval_samples:
                    organize(epoch)
                    last_organize_samples = samples
        if samples > last_organize_samples:
            organize(epoch)
        freeze_layers([dl])
        log(f"[hebbian L{i}] done: {layer_organizes} organizes over {samples} samples; layer frozen")

    return {"n_organizes": organize_count, "steps": global_steps}


@torch.no_grad()
def extract_features(layers, transform, dataset, device: torch.device,
                     batch_size: int = EXTRACT_BATCH_SIZE, desc: str = "Extract features") -> Features:
    """
    One pass of `dataset` (in order) through the frozen layers. Returns
    (features float32 on CPU, labels long on CPU). Refuses layers that are
    still in training mode, since their forward() would accumulate.
    """
    for i, dl in enumerate(layers):
        if dl.training:
            raise RuntimeError(f"extract_features: discrimination layer {i} is in training mode - freeze it first")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    feats, labels = [], []
    for x, y in tqdm(loader, total=len(loader), desc=desc, leave=False):
        h = transform(x.to(device))
        for dl in layers:
            h = dl(h, accumulate=False)
        feats.append(h.to(torch.float32).cpu())
        labels.append(torch.as_tensor(y).cpu())
    return torch.cat(feats), torch.cat(labels).long()


@torch.no_grad()
def _accuracy(head: nn.Module, data: Features, device: torch.device, batch_size: int = EVAL_BATCH_SIZE) -> float:
    x_all, y_all = data
    was_training = head.training
    head.eval()
    correct = 0
    for start in range(0, x_all.shape[0], batch_size):
        xb = x_all[start:start + batch_size].to(device)
        yb = y_all[start:start + batch_size].to(device)
        correct += (head(xb).argmax(dim=1) == yb).sum().item()
    head.train(was_training)
    return 100.0 * correct / max(x_all.shape[0], 1)


def train_head_on_features(
    head: nn.Module,
    fit: Features,
    val: Optional[Features],
    test: Features,
    device: torch.device,
    head_epochs: int = FREEZE_DEFAULTS["head_epochs"],
    head_batch_size: int = FREEZE_DEFAULTS["head_batch_size"],
    head_lr: float = FREEZE_DEFAULTS["head_lr"],
    patience: int = FREEZE_DEFAULTS["patience"],
    seed: Optional[int] = None,
    head_log=None,
    log: Logger = _noop,
) -> Dict[str, Any]:
    """
    Phase 5: Adam(lr=head_lr) + cross-entropy on the cached fit features,
    mini-batches of head_batch_size from a per-epoch permutation drawn from
    a private generator seeded with `seed`. After every epoch: val and test
    accuracy (head in eval mode); one head_log row
    (epoch, train_loss, train_acc, val_acc, test_acc, elapsed_sec);
    train_loss/train_acc are averaged over the epoch's batches as they were
    trained (dropout on for the MLP head).

    Early stopping on val accuracy: keep the weights of the epoch with the
    highest val accuracy (ties keep the earlier epoch); stop once `patience`
    epochs in a row did not beat it (patience <= 0: never stop early). The
    head ends with the best-val weights loaded.

    Without a validation set (val=None, i.e. val_size=0) there is nothing to
    select on: all head_epochs run, the last epoch is kept, best_val_acc is
    None.

    Returns final_acc (test accuracy of the kept head), best_val_acc,
    best_epoch, epochs_run, early_stopped.
    """
    head_epochs = int(head_epochs)
    if head_epochs < 1:
        raise ValueError("head_epochs must be >= 1")
    head.to(device)
    for p in head.parameters():
        p.requires_grad_(True)
    optimizer = optim.Adam(head.parameters(), lr=head_lr)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    else:
        gen.seed()

    fit_x, fit_y = fit
    n = fit_x.shape[0]
    best_val, best_epoch, best_state = None, None, None
    since_best = 0
    epochs_run = 0
    early_stopped = False
    t0 = time.time()

    for epoch in range(1, head_epochs + 1):
        head.train()
        perm = torch.randperm(n, generator=gen)
        loss_sum = 0.0
        correct = 0
        for start in range(0, n, head_batch_size):
            idx = perm[start:start + head_batch_size]
            xb = fit_x[idx].to(device)
            yb = fit_y[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * idx.numel()
            correct += (logits.detach().argmax(dim=1) == yb).sum().item()
        epochs_run = epoch
        train_loss = loss_sum / max(n, 1)
        train_acc = 100.0 * correct / max(n, 1)
        val_acc = _accuracy(head, val, device) if val is not None else None
        test_acc = _accuracy(head, test, device)
        if head_log is not None:
            head_log.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "elapsed_sec": round(time.time() - t0, 2),
            })

        if val_acc is None or best_val is None or val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            since_best = 0
            mark = " *"
        else:
            since_best += 1
            mark = ""
        val_str = f"{val_acc:.2f}%" if val_acc is not None else "n/a"
        log(f"[head epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_acc={val_str} test_acc={test_acc:.2f}%{mark}")

        if val is not None and patience > 0 and since_best >= patience:
            early_stopped = True
            log(f"[head] early stop after epoch {epoch}: val accuracy did not improve for {patience} epochs")
            break

    head.load_state_dict(best_state)
    head.eval()
    final_acc = _accuracy(head, test, device)
    return {
        "final_acc": final_acc,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
    }


def _make_logger(train_log_path: Optional[Path], quiet: bool) -> Callable[..., None]:
    def log(message: str, echo: bool = False) -> None:
        if echo or not quiet:
            print(message)
        if train_log_path is not None:
            with open(train_log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
    return log


def _round(v, digits):
    return None if v is None else round(float(v), digits)


# ---------------------------------------------------------------------------
# Full freeze run (phases 1-5), called by the two training entry points
# ---------------------------------------------------------------------------
def run_freeze(
    *,
    model,
    arch: str,
    spec,
    device: torch.device,
    train_subset,
    test_subset,
    result_root: Path,
    run_name: str,
    batch_size: int,
    organize_interval_samples: int,
    record: Dict[str, Any],
    enable_analysis: bool = True,
    enable_checkpoints: bool = True,
    quiet_train: bool = True,
    save_train_log: bool = True,
    checkpoint_name: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    analysis_level: str = "lean",
    run_start: Optional[float] = None,
    val_size: int = FREEZE_DEFAULTS["val_size"],
    hebbian_epochs: int = FREEZE_DEFAULTS["hebbian_epochs"],
    head_epochs: int = FREEZE_DEFAULTS["head_epochs"],
    head_batch_size: int = FREEZE_DEFAULTS["head_batch_size"],
    head_lr: float = FREEZE_DEFAULTS["head_lr"],
    patience: int = FREEZE_DEFAULTS["patience"],
) -> float:
    """
    One freeze-protocol run on a freshly built (untrained) model. `record`
    holds the architecture-specific run_config.json fields from the entry
    point (arch, dataset, sizes, head, discrimination_config, setup, ...);
    this adds the protocol fields and provenance.

    Writes to RESULT/<run_name>/<timestamp>/: run_config.json, metrics.csv
    (phase 1), head_metrics.csv (phase 5), train_log.txt, final_eval.json,
    checkpoints/final.pth (frozen layers + best head, via the model's
    save_model, so load_trained_run and the robustness script work
    unchanged) and the final weight grid(s). If `checkpoint_name` is given,
    the frozen layer(s) are also saved right after freezing as a named
    reusable checkpoint (save_layer / save_stack).

    Returns final_acc: test accuracy of the best-val-epoch head.
    """
    run_start = time.time() if run_start is None else run_start
    if arch not in ("single_layer", "stacked"):
        raise ValueError(f"unknown arch {arch!r}")
    layers = discrimination_layers_of(model)

    fit_ds, val_ds = split_train_val(train_subset, val_size, seed)
    fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # RNG alignment with plain. Plain evaluates the untrained model on the
    # test set before training ("eval step 0"). Creating that DataLoader
    # iterator draws one value from the global CPU RNG (torch's per-iterator
    # base seed, even with shuffle=False), and the training loader's shuffle
    # seed is drawn after it. Creating - not consuming - the same iterator
    # here makes the fit loader's shuffle identical to plain's. Nothing is
    # loaded or evaluated; the test set is not used for anything.
    iter(DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=False))

    analysis = AnalysisEngine(base_dir=result_root, run_name=run_name, image_hw=spec.patch_hw,
                              level=analysis_level) if enable_analysis else None
    run_dir = analysis.run_dir if analysis is not None else None
    train_log_path = run_dir / "train_log.txt" if (run_dir is not None and save_train_log) else None
    log = _make_logger(train_log_path, quiet_train)

    freeze_cfg = {
        "val_size": int(val_size),
        "hebbian_epochs": int(hebbian_epochs),
        "head_epochs": int(head_epochs),
        "head_batch_size": int(head_batch_size),
        "head_lr": float(head_lr),
        "patience": int(patience),
    }
    if analysis is not None:
        write_run_config(run_dir, {
            **record,
            "training_mode": "freeze",
            "protocol": "freeze",
            **freeze_cfg,
            "fit_size": len(fit_ds),
            "num_train_samples": len(train_subset),
            "num_test_samples": len(test_subset),
            "batch_size": batch_size,
            "organize_interval_samples": organize_interval_samples,
            "run_name": run_name,
            "seed": seed,
            "analysis_level": analysis_level,
            **run_provenance(device),
        })
    metrics_log = MetricsLog(run_dir) if run_dir is not None else None
    head_log = MetricsLog(run_dir, filename=HEAD_METRICS_FILENAME) if run_dir is not None else None

    for line in [
        f"[run_name] {run_name}",
        f"[protocol] freeze ({arch}, head={record.get('head')})",
        f"[device] {device}",
        f"[dataset] {spec.name} (input_dim={spec.input_dim}, num_classes={spec.num_classes})",
        f"[train_subset] {len(train_subset)} -> fit {len(fit_ds)} + val {0 if val_ds is None else len(val_ds)}",
        f"[test_subset] {len(test_subset)}",
        f"[batch_size] {batch_size}  [organize_interval_samples] {organize_interval_samples}",
        "[freeze] " + " ".join(f"{k}={v}" for k, v in freeze_cfg.items()),
        f"[result_dir] {run_dir if run_dir is not None else 'disabled (analysis off)'}",
    ]:
        log(line)

    try:
        # Phase 1 - Hebbian learning (no labels, no head)
        t = time.time()
        trainer = train_hebbian_single if arch == "single_layer" else train_hebbian_stacked
        hebb = trainer(model, fit_loader, device, organize_interval_samples, hebbian_epochs=hebbian_epochs,
                       analysis=analysis, enable_checkpoints=enable_checkpoints, metrics_log=metrics_log, log=log)
        hebbian_wall_sec = time.time() - t
        log(f"[phase 1] Hebbian learning done: {hebb['n_organizes']} organizes, {hebbian_wall_sec:.1f} s", echo=True)

        # Freeze
        freeze_layers(layers)
        frozen = _snapshot(layers)
        if checkpoint_name is not None:
            ckpt_dir = str(checkpoints_dir) if checkpoints_dir is not None else str(PROJECT_ROOT / "checkpoints")
            saver = save_layer if arch == "single_layer" else save_stack
            path = saver(model, name=checkpoint_name, checkpoints_dir=ckpt_dir,
                         meta={"run_name": run_name, "protocol": "freeze", "source": "training/freeze_protocol.py"},
                         overwrite=True)
            log(f"[checkpoint] frozen layer(s) saved as '{checkpoint_name}' -> {path}")

        # Phase 2 - feature extraction
        t = time.time()
        fit_feats = extract_features(layers, model.transform, fit_ds, device, desc="Extract fit")
        val_feats = extract_features(layers, model.transform, val_ds, device, desc="Extract val") if val_ds is not None else None
        test_feats = extract_features(layers, model.transform, test_subset, device, desc="Extract test")
        extract_wall_sec = time.time() - t
        log(f"[phase 2] features: fit {tuple(fit_feats[0].shape)}, "
            f"val {None if val_feats is None else tuple(val_feats[0].shape)}, test {tuple(test_feats[0].shape)}; "
            f"{extract_wall_sec:.1f} s", echo=True)

        # Phase 3 - head training on cached features
        t = time.time()
        res = train_head_on_features(model.readout_head, fit_feats, val_feats, test_feats, device,
                                     head_epochs=head_epochs, head_batch_size=head_batch_size, head_lr=head_lr,
                                     patience=patience, seed=seed, head_log=head_log, log=log)
        head_wall_sec = time.time() - t
        _check_unchanged(layers, frozen)
        final_acc = res["final_acc"]
        best_val_str = "n/a" if res["best_val_acc"] is None else f"{res['best_val_acc']:.2f}%"
        log(f"[phase 3] head: best epoch {res['best_epoch']}/{res['epochs_run']} run, best val {best_val_str}, "
            f"test {final_acc:.2f}%; {head_wall_sec:.1f} s", echo=True)

        model.eval()
        if analysis is not None and enable_checkpoints:
            analysis.save_final_checkpoint(model)
        if analysis is not None:
            write_final_eval(
                run_dir,
                final_acc=final_acc,
                best_acc=None,
                best_acc_samples=None,
                n_organizes=hebb["n_organizes"],
                train_wall_sec=hebbian_wall_sec + extract_wall_sec + head_wall_sec,
                total_wall_sec=time.time() - run_start,
                device=device,
                extra={
                    "best_val_acc": _round(res["best_val_acc"], 4),
                    "best_epoch": res["best_epoch"],
                    "epochs_run": res["epochs_run"],
                    "early_stopped": res["early_stopped"],
                    "hebbian_wall_sec": _round(hebbian_wall_sec, 1),
                    "extract_wall_sec": _round(extract_wall_sec, 1),
                    "head_wall_sec": _round(head_wall_sec, 1),
                    "fit_size": len(fit_ds),
                    "val_size": 0 if val_ds is None else len(val_ds),
                },
            )
    except Exception as exc:
        if analysis is not None:
            write_failed_eval(run_dir, exc)
        raise

    return final_acc


# ---------------------------------------------------------------------------
# New head on an already-finished freeze run (phases 4-5 only)
# ---------------------------------------------------------------------------
def run_head_on_frozen(
    *,
    source_run: str,
    device: torch.device,
    data_root: Path,
    result_root: Path,
    run_name: str,
    head: str,
    head_hidden_dims: Tuple[int, ...] = (256,),
    head_dropout: float = 0.2,
    source_timestamp: Optional[str] = None,
    head_epochs: int = FREEZE_DEFAULTS["head_epochs"],
    head_batch_size: int = FREEZE_DEFAULTS["head_batch_size"],
    head_lr: float = FREEZE_DEFAULTS["head_lr"],
    patience: int = FREEZE_DEFAULTS["patience"],
    seed: Optional[int] = None,
    analysis_level: str = "lean",
    quiet_train: bool = True,
) -> float:
    """
    Load a finished freeze run's frozen layer(s) from its checkpoints/final.pth
    (load_trained_run, newest finished folder unless source_timestamp is
    given), rebuild its exact fit / val / test split (its dataset,
    num_train/test_samples, val_size and seed from its run_config.json),
    extract features and train only a new `head` (seeded with `seed`). No
    Hebbian retraining.

    The new run folder RESULT/<run_name>/<timestamp>/ has the same layout as
    a freeze run (minus metrics.csv): its run_config.json is the source
    run's model config with the new head, so load_trained_run and the
    robustness script load it like any other run.
    """
    # local imports: training.train_classifier imports this module at load time
    from training.datasets import get_dataset
    from training.run_config import load_trained_run
    from training.train_classifier import make_subset, resolve_head

    run_start = time.time()
    model, src_cfg, spec = load_trained_run(source_run, result_root, device, timestamp=source_timestamp)
    src_dir = Path(src_cfg["_run_dir"])
    if src_cfg.get("protocol") != "freeze":
        raise SystemExit(f"{src_dir} is a '{src_cfg.get('protocol')}' run, not a freeze run; "
                         "head_on_frozen needs a run trained with the freeze protocol")
    layers = discrimination_layers_of(model)
    freeze_layers(layers)

    # rebuild the source run's split exactly
    _, train_ds, test_ds = get_dataset(src_cfg["dataset"], data_root=data_root)
    train_subset = make_subset(train_ds, src_cfg["num_train_samples"])
    test_subset = make_subset(test_ds, src_cfg["num_test_samples"])
    fit_ds, val_ds = split_train_val(train_subset, src_cfg["val_size"], src_cfg.get("seed"))
    if "fit_size" in src_cfg and len(fit_ds) != src_cfg["fit_size"]:
        raise RuntimeError(f"re-split gave {len(fit_ds)} fit images, source run had {src_cfg['fit_size']}")

    # new head, seeded independently of the source run
    if seed is not None:
        set_seed(seed)
    head_cls, head_kwargs = resolve_head(head, tuple(head_hidden_dims), head_dropout)
    in_dim = src_cfg["hidden_dim"] if src_cfg["arch"] == "single_layer" else src_cfg["layer_dims"][-2]
    model.readout_head = head_cls(in_dim, spec.num_classes, **head_kwargs).to(device)

    analysis = AnalysisEngine(base_dir=result_root, run_name=run_name, image_hw=spec.patch_hw, level=analysis_level)
    run_dir = analysis.run_dir
    log = _make_logger(run_dir / "train_log.txt", quiet_train)

    cfg = {k: v for k, v in src_cfg.items() if k not in _PROVENANCE_KEYS}
    cfg.update({
        "head": head,
        "head_hidden_dims": list(head_hidden_dims),
        "head_dropout": head_dropout,
        "head_epochs": int(head_epochs),
        "head_batch_size": int(head_batch_size),
        "head_lr": float(head_lr),
        "patience": int(patience),
        "run_name": run_name,
        "seed": seed,
        "hebbian_seed": src_cfg.get("seed"),
        "source_run": source_run,
        "source_timestamp": src_dir.name,
        "source_git_commit": src_cfg.get("git_commit"),
        "head_only": True,
        "analysis_level": analysis_level,
        **run_provenance(device),
    })
    write_run_config(run_dir, cfg)
    head_log = MetricsLog(run_dir, filename=HEAD_METRICS_FILENAME)
    for line in [
        f"[run_name] {run_name}",
        f"[source] {source_run} <- {src_dir}",
        f"[head] {head}  [seed] {seed}",
        f"[split] fit {len(fit_ds)} + val {0 if val_ds is None else len(val_ds)}; test {len(test_subset)}",
        f"[result_dir] {run_dir}",
    ]:
        log(line)

    try:
        frozen = _snapshot(layers)
        t = time.time()
        fit_feats = extract_features(layers, model.transform, fit_ds, device, desc="Extract fit")
        val_feats = extract_features(layers, model.transform, val_ds, device, desc="Extract val") if val_ds is not None else None
        test_feats = extract_features(layers, model.transform, test_subset, device, desc="Extract test")
        extract_wall_sec = time.time() - t
        log(f"[phase 2] features extracted in {extract_wall_sec:.1f} s", echo=True)

        t = time.time()
        res = train_head_on_features(model.readout_head, fit_feats, val_feats, test_feats, device,
                                     head_epochs=head_epochs, head_batch_size=head_batch_size, head_lr=head_lr,
                                     patience=patience, seed=seed, head_log=head_log, log=log)
        head_wall_sec = time.time() - t
        _check_unchanged(layers, frozen)
        final_acc = res["final_acc"]
        best_val_str = "n/a" if res["best_val_acc"] is None else f"{res['best_val_acc']:.2f}%"
        log(f"[phase 3] head: best epoch {res['best_epoch']}/{res['epochs_run']} run, best val {best_val_str}, "
            f"test {final_acc:.2f}%; {head_wall_sec:.1f} s", echo=True)

        model.eval()
        analysis.save_final_checkpoint(model)
        write_final_eval(
            run_dir,
            final_acc=final_acc,
            best_acc=None,
            best_acc_samples=None,
            n_organizes=None,
            train_wall_sec=extract_wall_sec + head_wall_sec,
            total_wall_sec=time.time() - run_start,
            device=device,
            extra={
                "best_val_acc": _round(res["best_val_acc"], 4),
                "best_epoch": res["best_epoch"],
                "epochs_run": res["epochs_run"],
                "early_stopped": res["early_stopped"],
                "hebbian_wall_sec": None,
                "extract_wall_sec": _round(extract_wall_sec, 1),
                "head_wall_sec": _round(head_wall_sec, 1),
                "fit_size": len(fit_ds),
                "val_size": 0 if val_ds is None else len(val_ds),
                "source_run": source_run,
                "source_timestamp": src_dir.name,
            },
        )
    except Exception as exc:
        write_failed_eval(run_dir, exc)
        raise
    return final_acc
