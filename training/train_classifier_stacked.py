"""
Training entry point for the multi-layer (stacked) biological classifier -
the multi-layer counterpart to training/train_classifier.py.

Reuses train_classifier.py's device/head/logging helpers rather than
duplicating them, and drives architectures.stacked.StackedBiologicalClassifier
via training.training_engines.train_classifier_stacked_plain /
train_classifier_stacked_with_review (see that file for why the stacked
model gets its own engines instead of reusing the single-layer ones).

Caveat (same as everywhere else new in this repo so far): syntax-checked
and logic-reviewed only, not yet run with real torch on this machine.
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analysis_engine import AnalysisEngine
from core.checkpointing import save_stack
from core.initializers import DatasetInitializerWhole
from architectures.stacked import StackedBiologicalClassifier
from training.train_classifier import (
    HEAD_CHOICES,
    build_eval_log_path,
    build_train_log_path,
    flatten_to_vector,
    resolve_head,
    select_device,
    summarize_cycle,
)
from training.training_engines import (
    evaluate_classifier,
    train_classifier_stacked_plain,
    train_classifier_stacked_with_review,
)

BASE_DIR = str(PROJECT_ROOT)

# Same discrimination_config as train_classifier.py's single-layer baseline,
# shared across every DL layer in the stack (matching the original stacked
# design, which also used one shared config for all layers).
DEFAULT_DISCRIMINATION_CONFIG = {
    "non_negative": True,
    "beta": 199 / 200,
    "lr_init": 0.99,
    "min_lr": 1e-3,
    "max_lr": 0.99,
    "threshold_factor": 1.0,
    "sparsity": 0.05,
    "optimizer_max_iters": 1000,
    "optimizer_lambda": 0.1,
    "optimizer_gain_factor": 10.0,
    "optimizer_estimate_steps": 50,
    "recover_step": 0.05,
    "recover_alpha": 0.2,
    "strength_gate_k": 1.0,
    "lr_gate_n": 0.5,
    "half_count": 100.0,
    "strong_bonus": 0.5,
    "std_scale": 1.5,
    "strong_bonus_power": 2.0,
    "strong_bonus_cap": 3.0,
    "state_cache_capacity": 200,
    "strength_decay_start": 5,
    "strength_decay_power": 2.0,
    "strength_decay_scale": 1.0,
    "use_cooldown": False,
}


def run_classifier_train_stacked(
    device: torch.device,
    data_root: Path,
    result_root: Path,
    run_name: str = "gpu_rebuild_stacked_classifier",
    layer_dims: Tuple[int, ...] = (784, 2000, 2000, 10),
    num_train_samples: int = 1000,
    num_test_samples: int = 1000,
    batch_size: int = 4,
    organize_interval_samples: int = 200,
    eval_interval_samples: int = 0,
    enable_analysis: bool = True,
    enable_checkpoints: bool = True,
    training_mode: str = "review",
    init_mode: str = "random",
    init_ratio: float = 0.25,
    init_dataset_scope: str = "full",
    review_per_sample_max: float = 10.0,
    review_pmax: float = 0.8,
    review_delta: float = 0.2,
    review_cache_size: int = 20000,
    quiet_train: bool = True,
    save_eval_log: bool = True,
    save_train_log: bool = True,
    head: str = "readout",
    head_hidden_dims: Tuple[int, ...] = (256,),
    head_dropout: float = 0.2,
    checkpoint_name: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
) -> float:
    """
    Runs one stacked-classifier training experiment. `layer_dims` is
    [input, hidden_1, ..., hidden_N, output] - e.g. (784, 2000, 2000, 10)
    for a 2-discrimination-layer stack on MNIST. Defaults to
    training_mode="review" (not "plain" like the single-layer entry point)
    because the original stacked baseline this is modeled on reached its
    92.59%/92.99% result WITH review; organize-only is expected to
    underperform substantially, same as it would for the single-layer case.

    Returns the final test accuracy (%).
    """
    layer_dims = tuple(int(d) for d in layer_dims)

    train_dataset = MNIST(root=str(data_root), train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(root=str(data_root), train=False, transform=transforms.ToTensor(), download=True)

    train_subset = Subset(train_dataset, list(range(num_train_samples)))
    test_subset = Subset(test_dataset, list(range(num_test_samples)))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=False)

    data_initializer = None
    if init_mode == "dataset":
        init_dataset = train_dataset if init_dataset_scope == "full" else train_subset
        data_initializer = DatasetInitializerWhole(
            dataset=init_dataset,
            transform=flatten_to_vector,
            init_ratio=init_ratio,
            non_negative=True,
        )

    head_cls, head_kwargs = resolve_head(head, head_hidden_dims, head_dropout)

    model = StackedBiologicalClassifier(
        layer_dims=list(layer_dims),
        data_initializer=data_initializer,
        transform=flatten_to_vector,
        discrimination_config=dict(DEFAULT_DISCRIMINATION_CONFIG),
        head_cls=head_cls,
        head_kwargs=head_kwargs,
    ).to(device)
    model.train()
    for cache in model.activation_caches:
        cache.max_size = int(review_cache_size)
    if training_mode == "review":
        model.enable_review(enabled=True, p0=0.0, pmax=review_pmax, delta=review_delta)
    else:
        model.enable_review(enabled=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.readout_head.parameters(), lr=1e-3)

    analysis = AnalysisEngine(base_dir=result_root, run_name=run_name) if enable_analysis else None
    eval_log_path = build_eval_log_path(result_root, analysis) if save_eval_log else None
    train_log_path = build_train_log_path(result_root, analysis) if save_train_log else None

    def eval_logger(tag: str, step: int, samples: int, acc: float) -> None:
        if eval_log_path is None:
            return
        with open(eval_log_path, "a", encoding="utf-8") as f:
            f.write(f"{tag}\tstep={step}\tsamples={samples}\tacc={acc:.4f}\n")

    def train_logger(message: str) -> None:
        if train_log_path is None:
            return
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    setup_lines = [
        f"[run_name] {run_name}",
        f"[device] {device}",
        f"[layer_dims] {layer_dims}",
        f"[num_dl_layers] {model.num_dl_layers}",
        f"[train_subset] {num_train_samples}",
        f"[test_subset] {num_test_samples}",
        f"[batch_size] {batch_size}",
        f"[organize_interval_samples] {organize_interval_samples}",
        f"[eval_interval_samples] {eval_interval_samples}",
        f"[training_mode] {training_mode}",
        f"[init_mode] {init_mode}",
        f"[head] {head}",
    ]
    if head == "traditional_mlp":
        setup_lines.append(f"[head_hidden_dims] {tuple(head_hidden_dims)}")
        setup_lines.append(f"[head_dropout] {head_dropout}")
    if init_mode == "dataset":
        setup_lines.append(f"[init_ratio] {init_ratio}")
        setup_lines.append(f"[init_dataset_scope] {init_dataset_scope}")
    if training_mode == "review":
        setup_lines.extend([
            f"[review_per_sample_max] {review_per_sample_max}",
            f"[review_pmax] {review_pmax}",
            f"[review_delta] {review_delta}",
            f"[review_cache_size] {review_cache_size}",
        ])
    if analysis is not None:
        setup_lines.append(f"[result_dir] {analysis.run_dir}")
    else:
        setup_lines.append("[result_dir] disabled (analysis off)")
    if checkpoint_name is not None:
        setup_lines.append(f"[checkpoint_name] {checkpoint_name}")
    for line in setup_lines:
        train_logger(line)

    base_acc = evaluate_classifier(model, test_loader, device)
    train_logger(f"[eval step 0] acc={base_acc:.2f}%")
    eval_logger("eval_step_0", 0, 0, base_acc)

    engine = train_classifier_stacked_with_review if training_mode == "review" else train_classifier_stacked_plain
    engine_kwargs = dict(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        organize_interval_samples=organize_interval_samples,
        eval_interval_samples=eval_interval_samples,
        analysis=analysis,
        enable_checkpoints=enable_checkpoints,
        quiet_train=quiet_train,
        eval_logger=eval_logger,
        train_logger=train_logger,
    )
    if training_mode == "review":
        engine_kwargs["review_per_sample_max"] = review_per_sample_max
    final_acc = engine(**engine_kwargs)

    if checkpoint_name is not None:
        ckpt_dir = str(checkpoints_dir) if checkpoints_dir is not None else str(Path(BASE_DIR) / "checkpoints")
        path = save_stack(
            model,
            name=checkpoint_name,
            checkpoints_dir=ckpt_dir,
            meta={
                "run_name": run_name,
                "head": head,
                "final_acc": final_acc,
                "source": "training/train_classifier_stacked.py",
            },
            overwrite=True,
        )
        msg = f"[checkpoint] saved reusable stack checkpoint '{checkpoint_name}' -> {path}"
        if not quiet_train:
            print(msg)
        train_logger(msg)

    return final_acc


def _parse_layer_dims(s: str) -> Tuple[int, ...]:
    return tuple(int(v.strip()) for v in s.split(",") if v.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-layer (stacked) discrimination-layer classifier training.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", type=str, default=str(Path(BASE_DIR) / "DATA"))
    parser.add_argument("--result-root", type=str, default=str(Path(BASE_DIR) / "RESULT"))
    parser.add_argument("--run-name", type=str, default="gpu_rebuild_stacked_classifier")
    parser.add_argument(
        "--layer-dims", type=str, default="784,2000,2000,10",
        help="Comma-separated layer sizes: input,hidden_1,...,hidden_N,output. "
             "Default is a 2-discrimination-layer stack matching the single-layer "
             "baseline's hidden width (2000) at each level.",
    )
    parser.add_argument("--num-train-samples", type=int, default=1000)
    parser.add_argument("--num-test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--organize-interval-samples", type=int, default=200)
    parser.add_argument("--eval-interval-samples", type=int, default=0)
    parser.add_argument("--disable-analysis", action="store_true")
    parser.add_argument("--disable-checkpoints", action="store_true")
    parser.add_argument("--training-mode", type=str, default="review", choices=["plain", "review"])
    parser.add_argument("--init-mode", type=str, default="random", choices=["random", "dataset"])
    parser.add_argument("--init-ratio", type=float, default=0.25)
    parser.add_argument("--init-dataset-scope", type=str, default="full", choices=["full", "subset"])
    parser.add_argument("--review-per-sample-max", type=float, default=10.0)
    parser.add_argument("--review-pmax", type=float, default=0.8)
    parser.add_argument("--review-delta", type=float, default=0.2)
    parser.add_argument("--review-cache-size", type=int, default=20000)
    parser.add_argument("--show-train-log", action="store_true")
    parser.add_argument("--disable-eval-log", action="store_true")
    parser.add_argument("--disable-train-log", action="store_true")
    parser.add_argument("--head", type=str, default="readout", choices=list(HEAD_CHOICES))
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = select_device(args.device)
    run_classifier_train_stacked(
        device=device,
        data_root=Path(args.data_root),
        result_root=Path(args.result_root),
        run_name=args.run_name,
        layer_dims=_parse_layer_dims(args.layer_dims),
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        batch_size=args.batch_size,
        organize_interval_samples=args.organize_interval_samples,
        eval_interval_samples=args.eval_interval_samples,
        enable_analysis=not args.disable_analysis,
        enable_checkpoints=not args.disable_checkpoints,
        training_mode=args.training_mode,
        init_mode=args.init_mode,
        init_ratio=args.init_ratio,
        init_dataset_scope=args.init_dataset_scope,
        review_per_sample_max=args.review_per_sample_max,
        review_pmax=args.review_pmax,
        review_delta=args.review_delta,
        review_cache_size=args.review_cache_size,
        quiet_train=not args.show_train_log,
        save_eval_log=not args.disable_eval_log,
        save_train_log=not args.disable_train_log,
        head=args.head,
        head_hidden_dims=(args.head_hidden_dim,),
        head_dropout=args.head_dropout,
        checkpoint_name=args.checkpoint_name,
    )
