"""
Shared run-size configs for the dated experiment scripts in this folder.

FULL mirrors the one real full-scale run that exists for this codebase so
far (RESULT/gpu_rebuild_classifier/2026-05-25_00-37-27 on the HPC: full
MNIST, batch 32, organize every 1000 samples, eval every 5000), so numbers
from these scripts are comparable with it and with each other.

SMOKE is a minutes-on-a-laptop-CPU run whose only job is to exercise every
code path (train, organize, stability gating, review, eval, RESULT logging,
named checkpoint save). Its accuracy is meaningless - never report it.

Every script takes:   python experiments/<script>.py            # FULL
                      python experiments/<script>.py --smoke    # SMOKE
"""
import argparse

FULL = dict(
    num_train_samples=60000,
    num_test_samples=10000,
    batch_size=32,
    organize_interval_samples=1000,
    eval_interval_samples=5000,
)

SMOKE = dict(
    num_train_samples=640,
    num_test_samples=128,
    batch_size=32,
    organize_interval_samples=64,
    eval_interval_samples=320,
)

# Same seed in every script so head-vs-head / depth-vs-depth comparisons
# start from the same random discrimination-layer init and data order.
SEED = 33


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny CPU-friendly run that only checks the code paths work")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def run_size(args):
    """Returns (size_kwargs, run_name_suffix, checkpoint_name_suffix)."""
    if args.smoke:
        return dict(SMOKE), "_smoke", "_smoke"
    return dict(FULL), "", ""
