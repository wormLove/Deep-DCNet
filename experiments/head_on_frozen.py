"""
Train a new head on an already-finished freeze run's frozen features
(Plan 03) - no Hebbian retraining, so a head comparison takes minutes.

Loads RESULT/<from-run>/<newest finished timestamp, or --timestamp>/
checkpoints/final.pth (load_trained_run), rebuilds that run's exact fit /
val / test split (its dataset, sample counts, val_size and seed from its
run_config.json), extracts features through the frozen layer(s) and trains
only the new head (phases 2-3 of training/freeze_protocol.py). The head's
init and batch order are seeded with --seed.

run_name = <from-run without _s<seed>[_smoke]>_head-<head>[_<overrides>]_s<seed>[_smoke]
e.g. --from-run freeze_mnist_single_layer_readout_s33 --head traditional_mlp
  -> freeze_mnist_single_layer_readout_head-traditional_mlp_s33

    python experiments/head_on_frozen.py --from-run freeze_mnist_single_layer_readout_s33 --head traditional_mlp
    python experiments/head_on_frozen.py --smoke --from-run freeze_mnist_single_layer_readout_s33_smoke --head traditional_mlp

--smoke only sets head_epochs 3 (unless given) and adds _smoke; the data
sizes always come from the source run. Head knobs default to FREEZE_FULL in
experiments/_configs.py (head_epochs 30, head_batch_size 128, head_lr 1e-3,
patience 5); a non-default value is added to the run name.
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _configs import ANALYSIS_LEVELS, SEED, add_freeze_args, freeze_params, seed_tag
from training.freeze_protocol import run_head_on_frozen
from training.train_classifier import HEAD_CHOICES, select_device

BASE_DIR = Path(__file__).resolve().parent.parent
HEAD_KEYS = ("head_epochs", "head_batch_size", "head_lr", "patience")
RUN_SUFFIX = re.compile(r"_s\d+(_smoke)?$")


def parse_args():
    parser = argparse.ArgumentParser(description="New head on a finished freeze run's frozen features (Plan 03).")
    parser.add_argument("--from-run", required=True, help="run_name of a finished freeze run under RESULT/")
    parser.add_argument("--timestamp", default=None, help="that run's timestamp folder (default: newest finished)")
    parser.add_argument("--head", required=True, choices=list(HEAD_CHOICES))
    parser.add_argument("--smoke", action="store_true", help="head_epochs 3 and a _smoke run name; code-path check only")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=SEED, help="seeds the new head's init and batch order")
    parser.add_argument("--analysis-level", default="lean", choices=list(ANALYSIS_LEVELS))
    add_freeze_args(parser, keys=HEAD_KEYS)
    return parser.parse_args()


def main():
    args = parse_args()
    # a head on a smoke run is a smoke run too (smoke defaults, _smoke name, left out of summaries)
    args.smoke = args.smoke or args.from_run.endswith("_smoke")
    smoke = args.smoke
    fp, override_tag = freeze_params(args, keys=HEAD_KEYS)
    run_name = (f"{RUN_SUFFIX.sub('', args.from_run)}_head-{args.head}{override_tag}"
                f"{seed_tag(args)}{'_smoke' if smoke else ''}")
    device = select_device(args.device)
    acc = run_head_on_frozen(
        source_run=args.from_run,
        source_timestamp=args.timestamp,
        device=device,
        data_root=BASE_DIR / "DATA",
        result_root=BASE_DIR / "RESULT",
        run_name=run_name,
        head=args.head,
        head_hidden_dims=(256,),
        head_dropout=0.2,
        seed=args.seed,
        analysis_level=args.analysis_level,
        **fp,
    )
    print(f"\n[{run_name}] final acc={acc:.2f}% (test accuracy of the best-val-epoch head)")


if __name__ == "__main__":
    main()
