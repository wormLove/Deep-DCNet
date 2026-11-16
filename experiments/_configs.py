"""
Shared run-size configs for the dated experiment scripts in this folder.

FULL mirrors the one real full-scale run that exists for this codebase so
far (RESULT/gpu_rebuild_classifier/2026-05-25_00-37-27 on the HPC: full
MNIST, batch 32, organize every 1000 samples, eval every 5000), so numbers
from these scripts are comparable with it and with each other. It is
dataset-agnostic: "full split" rather than a fixed sample count.

SMOKE is a minutes-on-a-laptop-CPU run whose only job is to exercise every
code path (train, organize, stability gating, review, eval, RESULT logging,
named checkpoint save). Its accuracy is meaningless - never report it.

Every script takes:   python experiments/<script>.py            # FULL
                      python experiments/<script>.py --smoke    # SMOKE
plus --seed N (default SEED) and --analysis-level lean|full (default lean).

Run names carry the seed: <name>_s<seed>[_smoke], e.g.
single_layer_baseline_s33, so the 3 seeds of one configuration get
separate RESULT/<run_name>/ folders (and can run in parallel) and
tools/build_results_index.py can group them by stripping _s<seed>.
"""
import argparse

from training.freeze_protocol import FREEZE_DEFAULTS

FULL = dict(
    num_train_samples=None,   # None = the whole training split (60k MNIST/Fashion, 50k CIFAR-10)
    num_test_samples=None,    # None = the whole test split (10k)
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

ANALYSIS_LEVELS = ("lean", "full")  # mirrors analysis.analysis_engine.ANALYSIS_LEVELS


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny CPU-friendly run that only checks the code paths work")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--analysis-level", default="lean", choices=list(ANALYSIS_LEVELS),
                        help="lean (default): final checkpoint + weight grid + metrics.csv; "
                             "full: every-step/every-organize PNGs + checkpoints (GBs, debugging only)")
    return parser.parse_args()


def seed_tag(args) -> str:
    """'_s<seed>' - appended to every run_name and named checkpoint."""
    return f"_s{args.seed}"


def run_size(args):
    """Returns (size_kwargs, run_name_suffix, checkpoint_name_suffix); suffixes are '_s<seed>[_smoke]'."""
    tag = seed_tag(args)
    if args.smoke:
        return dict(SMOKE), tag + "_smoke", tag + "_smoke"
    return dict(FULL), tag, tag


# ---------------------------------------------------------------------------
# Freeze-then-train protocol (Plan 03; training/freeze_protocol.py).
# FREEZE_FULL is the protocol's own default; SMOKE only shrinks the val split
# and the number of head epochs. Used by 2026-11-15_freeze_baselines.py and
# head_on_frozen.py.
# ---------------------------------------------------------------------------
FREEZE_FULL = dict(FREEZE_DEFAULTS)
FREEZE_SMOKE = dict(FREEZE_DEFAULTS, val_size=64, head_epochs=3)

# A value that differs from the size's default is written into the run name
# (before _s<seed>), e.g. freeze_mnist_single_layer_readout_val0_s33_smoke, so
# such a run never shares a RESULT/ folder with - or is averaged in the
# results index together with - the default configuration.
_FREEZE_FLAGS = {
    # key: (CLI flag, type, run-name tag)
    "val_size": ("--val-size", int, "val"),
    "hebbian_epochs": ("--hebbian-epochs", int, "heb"),
    "head_epochs": ("--head-epochs", int, "hep"),
    "head_batch_size": ("--head-batch-size", int, "hbs"),
    "head_lr": ("--head-lr", float, "hlr"),
    "patience": ("--patience", int, "pat"),
}


def add_freeze_args(parser, keys=tuple(_FREEZE_FLAGS)) -> None:
    """Optional overrides of the freeze defaults (default None = use FREEZE_FULL / FREEZE_SMOKE)."""
    for key in keys:
        flag, typ, _ = _FREEZE_FLAGS[key]
        parser.add_argument(flag, dest=key, type=typ, default=None,
                            help=f"override {key} (FULL default {FREEZE_FULL[key]}, SMOKE {FREEZE_SMOKE[key]}); "
                                 "a non-default value is added to the run name")


def freeze_params(args, keys=tuple(_FREEZE_FLAGS)):
    """Returns (freeze kwargs for this run size, run-name tag like '_val0' or '')."""
    defaults = FREEZE_SMOKE if args.smoke else FREEZE_FULL
    params, tag = {}, ""
    for key in keys:
        value = getattr(args, key, None)
        if value is None:
            value = defaults[key]
        elif value != defaults[key]:
            tag += f"_{_FREEZE_FLAGS[key][2]}{value:g}" if isinstance(value, float) else f"_{_FREEZE_FLAGS[key][2]}{value}"
        params[key] = value
    return params, tag
