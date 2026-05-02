# Deep-DCNet

Deep-DCNet is the standalone GPU-oriented rebuild of DCNet.

## Project Overview

Deep-DCNet is a GPU-oriented reconstruction of DCNet, a biologically inspired continual-learning framework built around local representation learning and memory-protected adaptation.

Instead of relying on standard end-to-end backpropagation through the full network, DCNet separates learning into two interacting parts:

- a **Discrimination Layer**, which learns sparse internal representations through local iterative activity optimization with lateral competition, Hebbian / Anti-Hebbian organization, and neuron-level plasticity control
- a lightweight **Readout Layer**, which maps stabilized representations to task outputs and can be further reinforced through replay-style **review**

The current GPU rebuild preserves the core ideas of the original CPU research prototype while reformulating the training process for mini-batch computation. In particular, it includes:

- iterative activity optimization with lateral competition
- Hebbian / Anti-Hebbian potential accumulation
- neuron-level memory statistics and nonlinear learning-rate recovery
- review-based readout reinforcement
- optional dataset-based initialization

This repository focuses on the single-target GPU baseline of DCNet and serves as the main codebase for scalable training, controlled experiments, and future continual-learning extensions.

## Development Note

- This repository currently focuses on the single-target GPU baseline of DCNet.
- Core mechanisms including local discrimination learning, review-based readout reinforcement, and dataset-based initialization are already implemented.
- Documentation, experiment packaging, and additional continual-learning extensions are still being refined.

## Key Features

- Biologically inspired representation learning for continual adaptation
- Local learning in the discrimination layer, without end-to-end backpropagation through the full model
- Iterative activity optimization with lateral competition and sparse activation selection
- Hebbian / Anti-Hebbian organization with organize-cycle updates
- Per-neuron learning-rate protection and nonlinear recovery based on memory statistics
- Optional review-based readout training for replay-style reinforcement
- Optional dataset-based initialization for structure-aware early representation shaping
- GPU-oriented mini-batch training and evaluation pipeline

## Installation

The current codebase mainly depends on PyTorch, TorchVision, `tqdm`, and `matplotlib`.

### Option 1: Conda

Create and activate a new environment:

```bash
conda create -n dcnet-gpu python=3.10
conda activate dcnet-gpu
```

Install the core packages:

```bash
conda install pytorch torchvision matplotlib tqdm -c pytorch
```

If you want CUDA support, install the matching PyTorch build for your CUDA version following the official PyTorch instructions.

### Option 2: Pip

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the core packages:

```bash
pip install torch torchvision matplotlib tqdm
```

If you want GPU acceleration, make sure the installed PyTorch build matches your local CUDA setup.

## Dataset Preparation

The current training scripts use `torchvision.datasets.MNIST`.

- If MNIST is not found locally, it will be downloaded automatically at runtime.
- By default, datasets are stored under the project-local `DATA/` directory.
- You can override the dataset location with the `--data-root` argument.

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

## Usage

### Classifier Training

Classifier training with review:

```bash
python training/train_classifier.py --device cuda --training-mode review
```

Classifier training without review:

```bash
python training/train_classifier.py --device cuda --training-mode plain
```

CPU classifier run:

```bash
python training/train_classifier.py --device cpu --training-mode plain
```

### Discrimination-only Training

Discrimination-only run:

```bash
python training/train_discrimination.py --device cuda
```

### Common Options

- `--device`: choose the execution device (`cpu`, `cuda`, or `mps`)
- `--training-mode`: choose classifier training with or without review (`plain` or `review`)
- `--init-mode`: choose random or dataset-based initialization (`random` or `dataset`)
- `--batch-size`: set the mini-batch size
- `--organize-interval-samples`: set how many samples are processed between organize updates
- `--eval-interval-samples`: set how many samples are processed between evaluation passes
- `--review-per-sample-max`: set the maximum replay budget per real sample in review mode
- `--data-root`: override the default dataset directory

For the full CLI configuration, see:

```bash
python training/train_classifier.py --help
python training/train_discrimination.py --help
```

## Notes

- Training scripts now default to the local `DATA/` and `RESULT/` directories inside this project.
- This folder is prepared to be separated into its own GitHub repository.
- Generated data, logs, checkpoints, and visualizations are ignored by default through `.gitignore`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
