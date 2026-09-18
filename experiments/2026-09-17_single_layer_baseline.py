"""
Single-layer baseline: DiscriminationLayer + the original linear ReadoutHead,
trained end to end (Hebbian organize() on the discrimination layer, ordinary
backprop on the readout head). This is the reference point everything else
in this experiments/ folder gets compared against - in particular
2026-09-17_point5_traditional_head.py (same config and seed, different
head).

This is a thin, named, dated wrapper around training.train_classifier's
run_classifier_train() - the actual training logic lives there and in
training/training_engines.py; this file just fixes a specific, documented
configuration so the experiment is reproducible and discoverable by name,
matching the convention: one dated file per named experiment, config baked
in rather than left to remember as CLI flags.

Run size comes from experiments/_configs.py: the default is the FULL config
(full MNIST 60k/10k, batch 32, organize every 1000 samples), mirroring the
2026-05-25 HPC run (RESULT/gpu_rebuild_classifier/2026-05-25_00-37-27,
96.52%). Note that HPC run had the IntegrationLayer ON
(--integration-dim 256, which also feeds raw pixels to the head); this
baseline keeps it OFF so the head sees ONLY the Hebbian features and the
point #5 head comparison stays clean. Expect a somewhat different number
from 96.52% for that reason.

    python experiments/2026-09-17_single_layer_baseline.py            # full run (GPU/HPC)
    python experiments/2026-09-17_single_layer_baseline.py --smoke    # code-path check only
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from _configs import parse_args, run_size
from training.train_classifier import run_classifier_train, select_device

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_NAME = "single_layer_h2000_readout"


def main():
    args = parse_args("Single-layer DCNet baseline (linear readout head, no integration layer).")
    size, run_suffix, ckpt_suffix = run_size(args)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    final_acc = run_classifier_train(
        device=device,
        data_root=BASE_DIR / "DATA",
        result_root=BASE_DIR / "RESULT",
        run_name="single_layer_baseline" + run_suffix,
        dataset="mnist",
        training_mode="plain",
        integration_dim=None,
        head="readout",
        checkpoint_name=CHECKPOINT_NAME + ckpt_suffix,
        **size,
    )
    print(f"\n[single_layer_baseline{run_suffix}] final acc={final_acc:.2f}%")
    print(f"[single_layer_baseline{run_suffix}] discrimination layer saved as checkpoint "
          f"'{CHECKPOINT_NAME + ckpt_suffix}' - see checkpoints/manifest.json")


if __name__ == "__main__":
    main()
