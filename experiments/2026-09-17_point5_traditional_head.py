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

Framing note: the default ReadoutHead is ALSO trained with backprop (Adam +
cross-entropy, same as on main) - but it is a single linear layer, so no
error is ever propagated through a hidden layer. TraditionalMLPHead is the
first place gradients flow through a hidden layer. The comparison is
"linear readout vs. multi-layer backprop head on identical Hebbian
features", not "no-backprop vs. backprop".

    python experiments/2026-09-17_point5_traditional_head.py            # full run (GPU/HPC)
    python experiments/2026-09-17_point5_traditional_head.py --smoke    # code-path check only
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from _configs import parse_args, run_size
from training.train_classifier import run_classifier_train, select_device

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_NAME = "point5_h2000_traditional_mlp"


def main():
    args = parse_args("Point #5: traditional MLP head on the same Hebbian features.")
    size, run_suffix, ckpt_suffix = run_size(args)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    final_acc = run_classifier_train(
        device=device,
        data_root=BASE_DIR / "DATA",
        result_root=BASE_DIR / "RESULT",
        run_name="point5_traditional_head" + run_suffix,
        training_mode="plain",
        integration_dim=None,
        head="traditional_mlp",
        head_hidden_dims=(256,),
        head_dropout=0.2,
        checkpoint_name=CHECKPOINT_NAME + ckpt_suffix,
        **size,
    )
    print(f"\n[point5_traditional_head{run_suffix}] final acc={final_acc:.2f}%")
    print("[point5_traditional_head] compare against 2026-09-17_single_layer_baseline.py run at the "
          "same size/seed - identical in everything except the head.")


if __name__ == "__main__":
    main()
