"""
Freeze-then-train baselines (Plan 03): the main protocol from now on.

Learn the Hebbian discrimination layer(s) without labels, freeze them, cache
their features, then train the head on the cached features with early
stopping on a 5,000-image validation split (training/freeze_protocol.py).
The one-pass protocols (plain, review) stay available in the older scripts.

    --dataset  mnist | fashion_mnist | kmnist | cifar10 | cifar10_gray   (default mnist)
    --arch     single_layer | stacked (2 x 2000, learned layer-wise)     (default single_layer)
    --head     readout | traditional_mlp                                (default readout)

Run size comes from experiments/_configs.py (FULL / SMOKE); the freeze knobs
from FREEZE_FULL / FREEZE_SMOKE there (val_size 5000, hebbian_epochs 1,
head_epochs 30, head_batch_size 128, head_lr 1e-3, patience 5; SMOKE:
val_size 64, head_epochs 3). --val-size / --hebbian-epochs / --head-epochs /
--head-batch-size / --head-lr / --patience override them; a non-default
value is added to the run name.

run_name = freeze_<dataset>_<arch>_<head>[_<overrides>]_s<seed>[_smoke]
The frozen layer(s) are also saved as checkpoint '<run_name>_frozen'.

    python experiments/2026-11-15_freeze_baselines.py --dataset mnist --arch single_layer --head readout
    python experiments/2026-11-15_freeze_baselines.py --dataset mnist --arch stacked --head readout
    python experiments/2026-11-15_freeze_baselines.py --smoke                    # code-path check only
    python experiments/2026-11-15_freeze_baselines.py --smoke --val-size 0       # plain-equivalence check

To compare heads on the same frozen features without Hebbian retraining,
use experiments/head_on_frozen.py on a finished run of this script.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _configs import ANALYSIS_LEVELS, FULL, SEED, SMOKE, add_freeze_args, freeze_params, seed_tag
from training.datasets import DATASET_CHOICES
from training.train_classifier import run_classifier_train, select_device
from training.train_classifier_stacked import run_classifier_train_stacked

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HIDDEN_DIM = 2000


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze-then-train baselines (Plan 03).")
    parser.add_argument("--dataset", default="mnist", choices=list(DATASET_CHOICES))
    parser.add_argument("--arch", default="single_layer", choices=["single_layer", "stacked"])
    parser.add_argument("--head", default="readout", choices=["readout", "traditional_mlp"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM,
                        help="discrimination-layer width (per layer for stacked); a non-default width "
                             "is added to the run name as _h<width>")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run that only checks the code paths work")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--analysis-level", default="lean", choices=list(ANALYSIS_LEVELS),
                        help="lean (default): final checkpoint + weight grid + metrics.csv; "
                             "full: every-step/every-organize PNGs + checkpoints (GBs, debugging only)")
    add_freeze_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    size = dict(SMOKE if args.smoke else FULL)
    fp, override_tag = freeze_params(args)
    width_tag = "" if args.hidden_dim == DEFAULT_HIDDEN_DIM else f"_h{args.hidden_dim}"
    run_name = (f"freeze_{args.dataset}_{args.arch}_{args.head}{width_tag}{override_tag}"
                f"{seed_tag(args)}{'_smoke' if args.smoke else ''}")
    device = select_device(args.device)
    common = dict(
        device=device,
        data_root=BASE_DIR / "DATA",
        result_root=BASE_DIR / "RESULT",
        run_name=run_name,
        dataset=args.dataset,
        training_mode="freeze",
        head=args.head,
        head_hidden_dims=(256,),
        head_dropout=0.2,
        checkpoint_name=run_name + "_frozen",
        seed=args.seed,
        analysis_level=args.analysis_level,
        **size,
        **fp,
    )
    if args.arch == "single_layer":
        acc = run_classifier_train(integration_dim=None, hidden_dim=args.hidden_dim, **common)
    else:
        acc = run_classifier_train_stacked(hidden_dims=(args.hidden_dim, args.hidden_dim), **common)
    print(f"\n[{run_name}] final acc={acc:.2f}% (test accuracy of the best-val-epoch head)")


if __name__ == "__main__":
    main()
