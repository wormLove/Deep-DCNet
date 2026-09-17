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
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from training.train_classifier import select_device
from training.train_classifier_stacked import run_classifier_train_stacked

if __name__ == "__main__":
    BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    device = select_device("auto")
    run_classifier_train_stacked(
        device=device,
        data_root=BASE_DIR / "DATA",
        result_root=BASE_DIR / "RESULT",
        run_name="stacked_2layer_baseline",
        layer_dims=(784, 2000, 2000, 10),
        num_train_samples=1000,
        num_test_samples=1000,
        batch_size=4,
        organize_interval_samples=200,
        eval_interval_samples=200,
        training_mode="review",
        review_pmax=0.8,
        head="readout",
        checkpoint_name="stacked_2layer_h2000_readout",
    )
