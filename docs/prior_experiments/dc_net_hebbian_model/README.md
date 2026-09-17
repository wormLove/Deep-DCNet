# Prior experiments (DC_Net_Hebbian_Model)

This folder is a **record of completed work, not active code**. It holds the
experiment scripts and results Isha produced in a separate repo
(`DC_Net_Hebbian_Model`) while your shared `Deep-DCNet` repo wasn't yet
accessible to her, plus two original reference runs from even earlier
(`DCNet-V-1.0`).

**These scripts will not run as-is in this repo.** They import from a
separate, no-longer-maintained implementation (their own `core/`,
`modules/discrimination.py`, `architectures/`) that is architecturally
different from this repo's actual model - single-sample forward passes
instead of batched, different hyperparameters, different save/load
conventions. That implementation was deliberately **not** brought into
Deep-DCNet (see git history on this branch), since running two parallel
implementations of the same model side by side isn't sustainable. Only the
*results* are kept here, as evidence of the work and a numeric baseline to
compare future Deep-DCNet-native runs against.

The full, runnable version of this code still exists in the separate
`DC_Net_Hebbian_Model` repo, unchanged, if anyone wants to inspect or rerun
it directly.

## What's here

- `experiments/` - the 3 experiment scripts, kept for provenance (exact
  hyperparameters, exact logic used to produce each result below).
- `RESULT/` - full logs for every run:
  - `single_layer_baseline/2026-02-22_16-18-59/` and
    `stacked_2layer_baseline/2026-03-17_14-35-38/` - the original reference
    runs from `DCNet-V-1.0`, before any reorganization.
  - `single_layer_quick_run/2026-09-10_03-33-30/` - first real-MNIST smoke
    test after reorganizing into `DC_Net_Hebbian_Model`.
  - `single_layer_full_baseline/2026-09-10_17-31-42/` - full reproduction of
    the original single-layer baseline in the reorganized code.
  - `point5_traditional_head/2026-09-11_13-51-50/` - Prof. Yu's point #5
    (traditional classifier head) comparison.
- `checkpoints/` - the two trained discrimination-layer checkpoints those
  runs produced (`manifest.json` + weights), for anyone who wants to inspect
  them without rerunning training.

## Results summary

| Run | Final acc | Best acc |
|---|---|---|
| Single-layer baseline (original, `DCNet-V-1.0`) | 95.23% | 95.46% |
| Stacked 2-layer baseline (original, `DCNet-V-1.0`) | 92.59% | 92.99% |
| Single-layer quick run (reorg smoke test, 3000/60000 steps) | 60.36% | - |
| Single-layer full baseline (reorg reproduction) | 94.88% | 95.15% |
| Point #5: ReadoutHead (linear, same run) | 94.88% | 95.15% |
| Point #5: TraditionalMLPHead (Linear->ReLU->Dropout->Linear) | 96.43% | 96.43% |

The quick run and full baseline are both single-layer/single-sample runs
and match closely (94.88% vs the original 95.23%), confirming the reorg
didn't change the model's behavior. The Point #5 comparison shows a
traditional MLP head reading out the same frozen features does noticeably
better than a single linear layer - see `training/train_classifier.py
--head traditional_mlp` in this repo's own codebase for the Deep-DCNet-
native version of that same comparison, which is the one to trust going
forward.
