"""
Single-layer baseline: DiscriminationLayer + the original linear ReadoutHead,
trained end to end (Hebbian organize() on the discrimination layer, ordinary
backprop on the readout head). This is the reference point everything else
in this experiments/ folder gets compared against - in particular
2026-09-17_point5_traditional_head.py, which reuses the checkpoint this
script saves.

This is a thin, named, dated wrapper around training.train_classifier's
run_classifier_train() - the actual training logic lives there and in
training/training_engines.py; this file just fixes a specific, documented
configuration so the experiment is reproducible and discoverable by name,
matching the convention: one dated file per named experiment, config baked
in rather than left to remember as CLI flags.

Defaults deliberately match train_classifier.py's own CLI defaults
(1000 train / 1000 test samples, batch_size=4) rather than a full 60k-image
epoch - this codebase's discrimination layer is considerably more complex
per-step (iterative activity optimization, adaptive per-neuron learning
rates) than the simpler CPU-only model used in earlier baseline work, and
its real per-step cost on this machine hasn't been measured yet. Scale
num_train_samples up once you've confirmed this runs correctly and have a
sense of the wall-clock cost.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_classifier import run_classifier_train, select_device

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_NAME = "single_layer_h2000_readout"


def main():
    device = select_device("auto")
    final_acc = run_classifier_train(
        device=device,
        data_root=BASE_DIR / "DATA",
        result_root=BASE_DIR / "RESULT",
        run_name="single_layer_baseline",
        num_train_samples=1000,
        num_test_samples=1000,
        batch_size=4,
        organize_interval_samples=200,
        eval_interval_samples=200,
        head="readout",
        checkpoint_name=CHECKPOINT_NAME,
    )
    print(f"\n[single_layer_baseline] final acc={final_acc:.2f}%")
    print(f"[single_layer_baseline] discrimination layer saved as checkpoint '{CHECKPOINT_NAME}' "
          f"- see checkpoints/manifest.json")


if __name__ == "__main__":
    main()
