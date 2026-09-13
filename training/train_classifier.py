import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
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
from core.initializers import DatasetInitializerWhole
from models.biological_classifier import BiologicalClassifier
from training.training_engines import evaluate_classifier, train_classifier_plain, train_classifier_with_review

BASE_DIR = str(PROJECT_ROOT)


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flatten_to_vector(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1)


@torch.no_grad()
def summarize_cycle(diag: dict) -> str:
    cycle_hits_ratio = diag["cycle_hits"].to(torch.float32).mean().item()
    total_inc = diag["cycle_total_increment"].to(torch.float32)
    weighted_inc = diag["cycle_weighted_increment"].to(torch.float32)
    return (
        f"cycle_samples={int(diag['cycle_sample_count'].item())}, "
        f"cycle_hits_ratio={cycle_hits_ratio:.6f}, "
        f"cycle_total_inc_mean={total_inc.mean().item():.6f}, "
        f"cycle_total_inc_max={total_inc.max().item():.6f}, "
        f"cycle_weighted_inc_mean={weighted_inc.mean().item():.6f}, "
        f"cycle_weighted_inc_max={weighted_inc.max().item():.6f}, "
        f"zero_norm_rows_total={int(diag['zero_norm_rows_total'].item())}"
    )


def build_eval_log_path(result_root: Path, analysis) -> Path:
    if analysis is not None:
        return analysis.run_dir / "eval_log.txt"
    run_dir = result_root / "gpu_rebuild_classifier_logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "eval_log.txt"


def build_train_log_path(result_root: Path, analysis) -> Path:
    if analysis is not None:
        return analysis.run_dir / "train_log.txt"
    run_dir = result_root / "gpu_rebuild_classifier_logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "train_log.txt"


def run_classifier_train(
    device: torch.device,
    data_root: Path,
    result_root: Path,
    num_train_samples: int = 1000,
    num_test_samples: int = 1000,
    batch_size: int = 4,
    organize_interval_samples: int = 200,
    eval_interval_samples: int = 0,
    enable_analysis: bool = True,
    enable_checkpoints: bool = True,
    debug_stats: bool = False,
    training_mode: str = "plain",
    init_mode: str = "random",
    init_ratio: float = 0.25,
    init_dataset_scope: str = "full",
    review_per_sample_max: float = 10.0,
    review_pmax: float = 0.8,
    review_delta: float = 0.2,
    review_cache_size: int = 20000,
    review_error_repeat: int = 1,
    quiet_train: bool = True,
    save_eval_log: bool = True,
    save_train_log: bool = True,
    integration_dim: Optional[int] = None,
    integration_activation: str = "relu",
) -> None:
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

    model = BiologicalClassifier(
        input_dim=784,
        hidden_dim=2000,
        output_dim=10,
        data_initializer=data_initializer,
        transform=flatten_to_vector,
        integration_dim=integration_dim,
        integration_activation=integration_activation,
        discrimination_config={
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
        },
    ).to(device)
    model.train()
    model.activation_cache.max_size = int(review_cache_size)
    if training_mode == "review":
        model.enable_review(enabled=True, p0=0.0, pmax=review_pmax, delta=review_delta)
    else:
        model.enable_review(enabled=False)

    criterion = nn.CrossEntropyLoss()
    # Collect all backprop-trained parameters: integration layer (when present) + readout head
    trainable_params = list(model.readout_head.parameters())
    if model.integration_layer is not None:
        trainable_params += list(model.integration_layer.parameters())
    optimizer = optim.Adam(trainable_params, lr=1e-3)

    analysis = AnalysisEngine(base_dir=result_root, run_name="gpu_rebuild_classifier") if enable_analysis else None
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
        f"[device] {device}",
        f"[train_subset] {num_train_samples}",
        f"[test_subset] {num_test_samples}",
        f"[batch_size] {batch_size}",
        f"[organize_interval_samples] {organize_interval_samples}",
        f"[eval_interval_samples] {eval_interval_samples}",
        f"[training_mode] {training_mode}",
        f"[init_mode] {init_mode}",
    ]
    if integration_dim is not None:
        setup_lines.append(f"[integration_dim] {integration_dim}")
        setup_lines.append(f"[integration_activation] {integration_activation}")
    else:
        setup_lines.append("[integration_layer] disabled")
    if init_mode == "dataset":
        setup_lines.append(f"[init_ratio] {init_ratio}")
        setup_lines.append(f"[init_dataset_scope] {init_dataset_scope}")
    if training_mode == "review":
        setup_lines.extend([
            f"[review_per_sample_max] {review_per_sample_max}",
            f"[review_pmax] {review_pmax}",
            f"[review_delta] {review_delta}",
            f"[review_cache_size] {review_cache_size}",
            f"[review_error_repeat] {review_error_repeat}",
        ])
    if analysis is not None:
        setup_lines.append(f"[result_dir] {analysis.run_dir}")
    else:
        setup_lines.append("[result_dir] disabled (analysis off)")
    if eval_log_path is not None:
        setup_lines.append(f"[eval_log] {eval_log_path}")
    if train_log_path is not None:
        setup_lines.append(f"[train_log] {train_log_path}")
    for line in setup_lines:
        train_logger(line)

    base_acc = evaluate_classifier(model, test_loader, device)
    train_logger(f"[eval step 0] acc={base_acc:.2f}%")
    eval_logger("eval_step_0", 0, 0, base_acc)

    engine = train_classifier_with_review if training_mode == "review" else train_classifier_plain
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
        debug_stats=debug_stats,
        summarize_cycle_fn=summarize_cycle,
        quiet_train=quiet_train,
        eval_logger=eval_logger,
        train_logger=train_logger,
    )
    if training_mode == "review":
        engine_kwargs["review_per_sample_max"] = review_per_sample_max
        engine_kwargs["review_error_repeat"] = review_error_repeat
    engine(**engine_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal classifier training with one discrimination layer + linear readout.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", type=str, default=str(Path(BASE_DIR) / "DATA"))
    parser.add_argument("--result-root", type=str, default=str(Path(BASE_DIR) / "RESULT"))
    parser.add_argument("--num-train-samples", type=int, default=1000)
    parser.add_argument("--num-test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--organize-interval-samples", type=int, default=200)
    parser.add_argument("--eval-interval-samples", type=int, default=0)
    parser.add_argument("--disable-analysis", action="store_true")
    parser.add_argument("--disable-checkpoints", action="store_true")
    parser.add_argument("--debug-stats", action="store_true")
    parser.add_argument("--training-mode", type=str, default="plain", choices=["plain", "review"])
    parser.add_argument("--init-mode", type=str, default="random", choices=["random", "dataset"])
    parser.add_argument("--init-ratio", type=float, default=0.25)
    parser.add_argument("--init-dataset-scope", type=str, default="full", choices=["full", "subset"])
    parser.add_argument("--review-per-sample-max", type=float, default=10.0)
    parser.add_argument("--review-pmax", type=float, default=0.8)
    parser.add_argument("--review-delta", type=float, default=0.2)
    parser.add_argument("--review-cache-size", type=int, default=20000)
    parser.add_argument("--review-error-repeat", type=int, default=1)
    parser.add_argument("--show-train-log", action="store_true")
    parser.add_argument("--disable-eval-log", action="store_true")
    parser.add_argument("--disable-train-log", action="store_true")
    parser.add_argument(
        "--integration-dim",
        type=int,
        default=None,
        help="Hidden size of the dense integration layer (cat([L0,L1]) → integration_dim → readout). "
             "Omit to disable the integration layer and use the original L1 → readout path.",
    )
    parser.add_argument(
        "--integration-activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "gelu"],
        help="Activation function for the integration layer (default: relu).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = select_device(args.device)
    run_classifier_train(
        device=device,
        data_root=Path(args.data_root),
        result_root=Path(args.result_root),
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        batch_size=args.batch_size,
        organize_interval_samples=args.organize_interval_samples,
        eval_interval_samples=args.eval_interval_samples,
        enable_analysis=not args.disable_analysis,
        enable_checkpoints=not args.disable_checkpoints,
        debug_stats=args.debug_stats,
        training_mode=args.training_mode,
        init_mode=args.init_mode,
        init_ratio=args.init_ratio,
        init_dataset_scope=args.init_dataset_scope,
        review_per_sample_max=args.review_per_sample_max,
        review_pmax=args.review_pmax,
        review_delta=args.review_delta,
        review_cache_size=args.review_cache_size,
        review_error_repeat=args.review_error_repeat,
        quiet_train=not args.show_train_log,
        save_eval_log=not args.disable_eval_log,
        save_train_log=not args.disable_train_log,
        integration_dim=args.integration_dim,
        integration_activation=args.integration_activation,
    )
