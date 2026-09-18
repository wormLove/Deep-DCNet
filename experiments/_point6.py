"""
Shared driver for the point #6 dated scripts (one script per dataset).

Each dataset script trains one of the same three configurations the MNIST
baselines use, chosen by flags, so results line up 1:1 with the MNIST runs:

    --arch single_layer --head readout          (== 2026-09-17_single_layer_baseline)
    --arch single_layer --head traditional_mlp  (== 2026-09-17_point5_traditional_head)
    --arch stacked      --head readout          (== 2026-09-17_stacked_2layer_baseline)

run_name = point6_<dataset>_<arch>_<head>, so the point #4 sweep can later
be pointed at any subset of them with --runs.
"""
import argparse
from pathlib import Path

import torch

from _configs import FULL, SEED, SMOKE
from training.train_classifier import run_classifier_train, select_device
from training.train_classifier_stacked import run_classifier_train_stacked

BASE_DIR = Path(__file__).resolve().parent.parent


def parse_point6_args(dataset: str):
    parser = argparse.ArgumentParser(description=f"Point #6: {dataset} with the MNIST baseline configurations.")
    parser.add_argument("--arch", default="single_layer", choices=["single_layer", "stacked"])
    parser.add_argument("--head", default="readout", choices=["readout", "traditional_mlp"])
    parser.add_argument("--hidden-dim", type=int, default=2000, help="discrimination-layer width (per layer for stacked)")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run that only checks the code paths work")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def run_point6(dataset: str, args) -> float:
    size = dict(SMOKE if args.smoke else FULL)
    suffix = "_smoke" if args.smoke else ""
    run_name = f"point6_{dataset}_{args.arch}_{args.head}{suffix}"
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    common = dict(
        device=device,
        data_root=BASE_DIR / "DATA",
        result_root=BASE_DIR / "RESULT",
        run_name=run_name,
        dataset=dataset,
        head=args.head,
        head_hidden_dims=(256,),
        head_dropout=0.2,
        checkpoint_name=run_name,
        **size,
    )
    if args.arch == "single_layer":
        acc = run_classifier_train(training_mode="plain", integration_dim=None, hidden_dim=args.hidden_dim, **common)
    else:
        acc = run_classifier_train_stacked(training_mode="review", review_pmax=0.8,
                                           hidden_dims=(args.hidden_dim, args.hidden_dim), **common)
    print(f"\n[{run_name}] final acc={acc:.2f}%")
    return acc
