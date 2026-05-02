# DCNet-GPU-Rebuild

This folder is the standalone GPU-oriented rebuild of DCNet.

## Scope

- Local-learning discrimination layer
- Iterative activity optimizer
- Organize-cycle neuron statistics and per-neuron learning-rate protection
- Optional review-based readout training
- Optional dataset-based initialization

## Folder Layout

- `models/`: high-level model components
- `core/`: learning rules, memory protection, initialization, and monitoring
- `training/`: training engines and runnable entry scripts
- `analysis/`: analysis and visualization helpers
- `experiments/`: reserved for experiment-specific scripts
- `configs/`: reserved for configuration files
- `docs/`: reserved for project documents
- `DATA/`: local dataset root used by the training scripts
- `RESULT/`: local output root for logs, checkpoints, and visualizations

## Quick Start

Classifier with review:

```bash
python training/train_classifier.py --device cuda --training-mode review
```

Classifier without review:

```bash
python training/train_classifier.py --device cuda --training-mode plain
```

Discrimination-only smoke run:

```bash
python training/train_discrimination.py --device cuda
```

## Notes

- Training scripts now default to the local `DATA/` and `RESULT/` directories inside this project.
- This folder is prepared to be separated into its own GitHub repository.
- Generated data, logs, checkpoints, and visualizations are ignored by default through `.gitignore`.
