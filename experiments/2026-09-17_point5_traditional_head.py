"""
Prof. Yu's point #5: traditional classifier head, native to this codebase.

Compares BiologicalClassifier with --head traditional_mlp (a conventional
Linear->ReLU->Dropout->Linear head, trained end to end with ordinary
backprop) against 2026-09-17_single_layer_baseline.py's --head readout
(a single linear layer) - same discrimination-layer hyperparameters and
training setup in both, only the head architecture differs.

Note this is an independent training run, not a frozen-feature reuse of
the baseline script's saved checkpoint: training_engines.py's train loop
always calls model.organize() at least once (a final catch-up call after
the last batch, regardless of organize_interval_samples), which would
mutate a loaded "frozen" discrimination layer's weights in place and
defeat the point of comparing against a literally-fixed feature set. Doing
that safely would need a real "never organize" mode added to
training_engines.py, which hasn't been built/verified here - so for now,
this trains its own discrimination layer from scratch with the same
config and a fixed seed, which is still a fair, like-for-like comparison
of head architecture, just not a bit-for-bit-identical-features one.

Compare against docs/prior_experiments/dc_net_hebbian_model/'s point5
result (96.43% vs 94.88%/95.15%) for historical context - that used a
different, simpler, single-sample model, so the numbers aren't directly
comparable, only the comparison logic mirrors it.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from training.train_classifier import run_classifier_train, select_device

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_NAME = "point5_h2000_traditional_mlp"
SEED = 33


def main():
    torch.manual_seed(SEED)
    device = select_device("auto")
    final_acc = run_classifier_train(
        device=device,
        data_root=BASE_DIR / "DATA",
        result_root=BASE_DIR / "RESULT",
        run_name="point5_traditional_head",
        num_train_samples=1000,
        num_test_samples=1000,
        batch_size=4,
        organize_interval_samples=200,
        eval_interval_samples=200,
        head="traditional_mlp",
        head_hidden_dim=256,
        head_dropout=0.2,
        checkpoint_name=CHECKPOINT_NAME,
    )
    print(f"\n[point5_traditional_head] final acc={final_acc:.2f}%")
    print("[point5_traditional_head] compare against a --head readout run with the same "
          "num_train_samples/batch_size/seed (see 2026-09-17_single_layer_baseline.py) for "
          "the actual head-vs-head comparison in this codebase.")


if __name__ == "__main__":
    main()
