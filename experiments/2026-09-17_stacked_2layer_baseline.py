"""
Multi-layer (stacked) baseline, matching the spirit of the 2-layer stacked
baseline from DC_Net_Hebbian_Model / DCNet-V-1.0 (92.59% final / 92.99% best
on MNIST), reproduced natively against this repo's real architecture
(architectures.stacked.StackedBiologicalClassifier) rather than by
importing that old codebase.

Not a bit-for-bit reproduction: the underlying DiscriminationLayer here is
a different (batched, GPU-ready, adaptive-per-neuron-LR) implementation
than the old single-sample DiscriminationModule the original run used, and
the exact original hyperparameters/layer widths were never recorded in a
runnable script (only the RESULT log survived - see
docs/prior_experiments/dc_net_hebbian_model/README.md). This uses 2
discrimination layers of width 2000 each (matching this repo's own
single-layer baseline's hidden width, for a fair same-repo comparison)
with the review mechanism enabled (review_pmax=0.8), since the original
stacked baseline's ~92% result relied on review - organize-only training
is expected to underperform substantially, same as it would single-layer.

Run size comes from experiments/_configs.py (default FULL: 60k/10k, batch
32, organize every 1000 samples - same as the single-layer baseline so the
two are directly comparable). Depth needs the long run: layer 1 only starts
learning after layer 0 has been stable for a few organize cycles.

    python experiments/2026-09-17_stacked_2layer_baseline.py            # full run (GPU/HPC)
    python experiments/2026-09-17_stacked_2layer_baseline.py --smoke    # code-path check only
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from _configs import parse_args, run_size
from training.train_classifier import select_device
from training.train_classifier_stacked import run_classifier_train_stacked

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_NAME = "stacked_2layer_h2000_readout"


def main():
    args = parse_args("Stacked 2-layer DCNet baseline (review mode, linear readout head).")
    size, run_suffix, ckpt_suffix = run_size(args)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    final_acc = run_classifier_train_stacked(
        device=device,
        data_root=BASE_DIR / "DATA",
        result_root=BASE_DIR / "RESULT",
        run_name="stacked_2layer_baseline" + run_suffix,
        dataset="mnist",
        hidden_dims=(2000, 2000),
        training_mode="review",
        review_pmax=0.8,
        head="readout",
        checkpoint_name=CHECKPOINT_NAME + ckpt_suffix,
        **size,
    )
    print(f"\n[stacked_2layer_baseline{run_suffix}] final acc={final_acc:.2f}%")


if __name__ == "__main__":
    main()
