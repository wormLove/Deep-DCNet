"""
Prof. Yu's point #4: robustness / noise testing.

Evaluation-only sweep over already-trained runs. For each run name given,
the newest finished RESULT/<run_name>/<timestamp>/ (run_config.json +
checkpoints/final.pth) is rebuilt and scored on the test split under every
(perturbation, level) in core.perturbations.DEFAULT_GRID:

    gaussian     sigma 0.1 / 0.2 / 0.3 / 0.5
    salt_pepper  p     0.05 / 0.1 / 0.2
    occlusion    k x k square, k = 6 / 10 / 14
    shift        max +-2 / +-4 px

Every model sees the identical corrupted images (same seed per grid
cell). All runs in one sweep must have been trained on the same dataset -
the dataset is read from the first run's config and the test split comes
from training/datasets.py, so this works unchanged for MNIST,
Fashion-MNIST, CIFAR-10, ... (point #6).

Research question: do sparse Hebbian features degrade more gracefully than
a backprop MLP head on the same features, and does a second discrimination
layer help or hurt under corruption?

    # after the three MNIST full runs have finished:
    python experiments/2026-09-18_point4_robustness.py
    # any set of runs that share a dataset:
    python experiments/2026-09-18_point4_robustness.py --runs point6_fashion_mnist_readout point6_fashion_mnist_mlp
    # code-path check against the *_smoke runs, 256 test images:
    python experiments/2026-09-18_point4_robustness.py --smoke

Output: RESULT/point4_robustness_<dataset>/<timestamp>/robustness.tsv (tidy,
one row per model x cell) and table.txt (pivot, models as columns).
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from analysis.robustness import format_table, run_robustness_sweep
from core.perturbations import DEFAULT_GRID
from training.datasets import get_dataset
from training.run_config import load_trained_run
from training.train_classifier import make_subset, select_device

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = ["single_layer_baseline", "point5_traditional_head", "stacked_2layer_baseline"]


def main():
    parser = argparse.ArgumentParser(description="Point #4: robustness sweep over trained runs.")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                        help="run_names under RESULT/ to compare (newest finished timestamp of each is used)")
    parser.add_argument("--smoke", action="store_true",
                        help="use the <run>_smoke runs and 256 test images; code-path check only")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-test-samples", type=int, default=0, help="0 = whole test split")
    parser.add_argument("--result-root", default=str(BASE_DIR / "RESULT"))
    parser.add_argument("--data-root", default=str(BASE_DIR / "DATA"))
    args = parser.parse_args()

    device = select_device(args.device)
    run_names = [r + "_smoke" if args.smoke and not r.endswith("_smoke") else r for r in args.runs]
    num_test = 256 if args.smoke and args.num_test_samples == 0 else args.num_test_samples

    models, dataset_name = {}, None
    for rn in run_names:
        model, cfg, spec = load_trained_run(rn, args.result_root, device)
        if dataset_name is None:
            dataset_name = spec.name
        elif spec.name != dataset_name:
            raise SystemExit(f"Run '{rn}' was trained on {spec.name}, but the sweep is on {dataset_name}; "
                             "compare only runs that share a dataset.")
        models[rn] = model
        print(f"[loaded] {rn} ({cfg['arch']} / head={cfg['head']}) <- {cfg['_run_dir']}")

    spec, _, test_ds = get_dataset(dataset_name, data_root=args.data_root)
    test_subset = make_subset(test_ds, num_test)
    loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    tag = "_smoke" if args.smoke else ""
    out_dir = Path(args.result_root) / f"point4_robustness_{spec.name}{tag}" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[dataset] {spec.name}  [test images] {len(test_subset)}  [device] {device}\n[out] {out_dir}\n")

    rows = run_robustness_sweep(models, loader, device, DEFAULT_GRID, out_dir, seed=args.seed, dataset_name=spec.name)
    table = format_table(rows)
    with open(out_dir / "table.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset: {spec.name}   test images: {len(test_subset)}   seed: {args.seed}\n\n{table}\n")
    print("\n" + table)
    if args.smoke:
        print("\n[smoke] numbers above come from smoke-trained models on 256 images - code-path check only.")


if __name__ == "__main__":
    main()
