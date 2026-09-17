import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analysis_engine import AnalysisEngine
from core.initializers import DatasetInitializerWhole
from architectures.single_layer import BiologicalClassifier

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
def summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    t = tensor.detach().to(torch.float32)
    return (
        f"{name}: mean={t.mean().item():.6f}, std={t.std(unbiased=False).item():.6f}, "
        f"min={t.min().item():.6f}, max={t.max().item():.6f}"
    )


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


def run_smoke_train(
    device: torch.device,
    data_root: Path,
    result_root: Path,
    num_samples: int = 1000,
    batch_size: int = 4,
    organize_interval_samples: int = 200,
    enable_analysis: bool = True,
    enable_checkpoints: bool = True,
    debug_stats: bool = False,
    init_mode: str = "random",
    init_ratio: float = 0.25,
    init_dataset_scope: str = "full",
) -> None:
    dataset = MNIST(root=str(data_root), train=True, transform=transforms.ToTensor(), download=True)
    subset = Subset(dataset, list(range(num_samples)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)

    data_initializer = None
    if init_mode == "dataset":
        init_dataset = dataset if init_dataset_scope == "full" else subset
        data_initializer = DatasetInitializerWhole(
            dataset=init_dataset,
            transform=flatten_to_vector,
            init_ratio=init_ratio,
            non_negative=True,
        )

    model = BiologicalClassifier(
        input_dim=784,
        hidden_dim=2000,
        data_initializer=data_initializer,
        transform=flatten_to_vector,
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
    analysis = AnalysisEngine(base_dir=result_root, run_name="gpu_rebuild_discrimination") if enable_analysis else None

    total_steps = 0
    total_samples = 0
    last_organize_samples = 0
    organize_count = 0

    print(f"[device] {device}")
    print(f"[dataset] MNIST train subset = {num_samples} samples")
    print(f"[batch_size] {batch_size}")
    print(f"[organize_interval_samples] {organize_interval_samples}")
    print(f"[init_mode] {init_mode}")
    if init_mode == "dataset":
        print(f"[init_ratio] {init_ratio}")
        print(f"[init_dataset_scope] {init_dataset_scope}")
    print(f"[expected_steps] {len(loader)}")
    print(f"[expected_organize_calls] {num_samples // organize_interval_samples}")
    if analysis is not None:
        print(f"[result_dir] {analysis.run_dir}")
    else:
        print("[result_dir] disabled (analysis off)")

    progress = tqdm(loader, total=len(loader), desc="Smoke Train", leave=True)
    for x, _ in progress:
        x = x.to(device)
        _ = model(x)
        total_steps += 1
        total_samples += x.shape[0]
        if analysis is not None:
            analysis.save_step_state(model, total_steps)
        progress.set_postfix(step=total_steps, samples=total_samples)

        if total_samples - last_organize_samples >= organize_interval_samples:
            weights_before = model.discrimination_layer.neuron_weights.detach().clone()
            model.organize()
            last_organize_samples = total_samples
            organize_count += 1
            if analysis is not None:
                analysis.save_organize_state(model, total_steps)
                if enable_checkpoints:
                    analysis.save_checkpoint(model, total_steps)
            weights_after = model.discrimination_layer.neuron_weights.detach()
            weight_shift = torch.norm(weights_after - weights_before).item()
            diag = model.diagnostics()

            print(
                f"[organize {organize_count}] "
                f"step={total_steps}, samples={total_samples}, weight_shift={weight_shift:.6f}"
            )
            if debug_stats:
                print("  " + summarize_tensor("raw_strength", diag["raw_strength"]))
                print("  " + summarize_tensor("effective_strength", diag["effective_strength"]))
                print("  " + summarize_tensor("lr_vec", diag["lr_vec"]))
                print(
                    "  "
                    + f"inactive_ratio={diag['inactive_mask'].to(torch.float32).mean().item():.6f}, "
                    + f"cycles_mean={diag['cycles_since_active'].to(torch.float32).mean().item():.6f}"
                )
                print("  " + summarize_cycle(diag))
                print(
                    "  "
                    + summarize_tensor("potential_hebb", model.discrimination_layer.organizer.potential_hebb)
                )
                print(
                    "  "
                    + summarize_tensor("potential_antihebb", model.discrimination_layer.organizer.potential_antihebb)
                )

    if total_samples > last_organize_samples:
        weights_before = model.discrimination_layer.neuron_weights.detach().clone()
        model.organize()
        if analysis is not None:
            analysis.save_organize_state(model, total_steps)
            if enable_checkpoints:
                analysis.save_checkpoint(model, total_steps)
        weights_after = model.discrimination_layer.neuron_weights.detach()
        weight_shift = torch.norm(weights_after - weights_before).item()
        diag = model.diagnostics()
        print(
            f"[final organize] step={total_steps}, samples={total_samples}, weight_shift={weight_shift:.6f}"
        )
        if debug_stats:
            print("  " + summarize_tensor("raw_strength", diag["raw_strength"]))
            print("  " + summarize_tensor("effective_strength", diag["effective_strength"]))
            print("  " + summarize_tensor("lr_vec", diag["lr_vec"]))
            print("  " + summarize_cycle(diag))

    if analysis is not None and enable_checkpoints:
        analysis.save_final_checkpoint(model)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal discrimination-layer smoke training on MNIST.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data-root", type=str, default=str(Path(BASE_DIR) / "DATA"))
    parser.add_argument("--result-root", type=str, default=str(Path(BASE_DIR) / "RESULT"))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--organize-interval-samples", type=int, default=200)
    parser.add_argument("--disable-analysis", action="store_true")
    parser.add_argument("--disable-checkpoints", action="store_true")
    parser.add_argument("--debug-stats", action="store_true")
    parser.add_argument("--init-mode", type=str, default="random", choices=["random", "dataset"])
    parser.add_argument("--init-ratio", type=float, default=0.25)
    parser.add_argument("--init-dataset-scope", type=str, default="full", choices=["full", "subset"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = select_device(args.device)
    run_smoke_train(
        device=device,
        data_root=Path(args.data_root),
        result_root=Path(args.result_root),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        organize_interval_samples=args.organize_interval_samples,
        enable_analysis=not args.disable_analysis,
        enable_checkpoints=not args.disable_checkpoints,
        debug_stats=args.debug_stats,
        init_mode=args.init_mode,
        init_ratio=args.init_ratio,
        init_dataset_scope=args.init_dataset_scope,
    )
