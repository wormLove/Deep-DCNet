# DC_Net_Hebbian_Model

Biologically-inspired classifier modeled on the olfactory system (Maximal
Dependence Capturing: dimensional expansion + sparse non-negative coding +
local Hebbian/anti-Hebbian learning, no backprop in the discrimination
layer), developed in Prof. Ron Yu's lab at CWRU.

## Layout

- `core/` - stable learning-rule primitives: the organizer (Hebbian /
  anti-Hebbian), the three activity optimizers (`core/optimizers/`),
  thresholding, and the weight-drift monitor. Changes here need strong
  justification - this is what produces the validated baseline numbers.
- `modules/` - building blocks composed from `core/` primitives:
  `DiscriminationModule`, `ReadoutHead`, `ActivationCache`.
- `architectures/` - how modules are composed into full models
  (`single_layer.py`, `stacked.py`, and future variants). Pick one by name
  via `architectures/registry.py` rather than importing a class directly.
- `utils/` - shared helpers: transforms, seeding, logging.
- `data/`, `eval/`, `training/`, `experiments/`, `RESULT/`, `tests/` -
  being filled in as part of the ongoing reorganization; see
  `docs/DECISIONS.md` and `docs/RESEARCH_LOG.md` for the running history.
- `docs/literature_notes/` - the two foundational papers (MDC, Hebbian
  classification module) this project is built on.

## Status

Actively being reorganized from the original DCNet-V-1.0 prototype so it
can hold Prof. Yu's newer research directions (parallel branches, tiled
receptive fields, robustness testing, more datasets, etc.) without one-off
edits to shared code. Validated baselines so far: 95.23% (single layer) /
92.59% (2-layer stack) on MNIST - see `RESULT/`.
