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

Datasets are registered in `training/datasets.py` (currently `mnist`, `fashion_mnist`, `kmnist`, `cifar10`, `cifar10_gray`) and stored under the project-local `DATA/` directory (override with `--data-root`). Input size and class count follow the dataset automatically in every training script.

```bash
python training/datasets.py list                       # what is registered / already on disk
python training/datasets.py download mnist cifar10     # fetch (skips files already present)
python training/datasets.py download all
```

Training scripts download a missing dataset on the fly when the machine has internet. On the HPC, compute nodes do not, so run `bash hpc/download_data.sh` from a login node first. To add a dataset, add one `DatasetSpec` entry to `DATASETS`.

## Folder Layout

- `architectures/`: full classifier architectures assembled from `modules/` (`single_layer.py`, `stacked.py`, `registry.py`)
- `modules/`: layer-level components (discrimination layer, readout/classifier heads, integration layer)
- `core/`: learning rules, memory protection, initialization, monitoring, named checkpoints, and input perturbations (`perturbations.py`)
- `training/`: training engines and runnable entry scripts
- `analysis/`: analysis and visualization helpers, incl. the robustness sweep (`robustness.py`, point #4)
- `experiments/`: named, dated experiment scripts (`<date>_<name>.py`). Each runs the full-scale config by default and takes `--smoke` for a minutes-long CPU code-path check; run sizes live in `experiments/_configs.py`. Every run writes `run_config.json` next to its logs so it can be rebuilt later (`training/run_config.py`)
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

### Stacked (Multi-layer) Classifier Training

Multi-layer stack (2 discrimination layers by default), with review:

```bash
python training/train_classifier_stacked.py --device cuda --training-mode review
```

Custom layer sizes (input/output follow `--dataset`):

```bash
python training/train_classifier_stacked.py --hidden-dims 2000,2000,2000
```

### Other Datasets

Any registered dataset works with both entry points:

```bash
python training/train_classifier.py --dataset fashion_mnist --num-train-samples 0 --num-test-samples 0
python training/train_classifier_stacked.py --dataset cifar10 --hidden-dims 2000,2000
```

### Experiments

Named, dated scripts under `experiments/` fix one configuration each (full-scale by default, `--smoke` for a quick code-path check):

```bash
python experiments/2026-09-17_single_layer_baseline.py
python experiments/2026-09-17_point5_traditional_head.py
python experiments/2026-09-17_stacked_2layer_baseline.py
python experiments/2026-09-18_point6_fashion_mnist.py --arch stacked --head readout
python experiments/2026-09-18_point6_cifar10.py --head traditional_mlp
python experiments/2026-09-18_point4_robustness.py --runs single_layer_baseline point5_traditional_head stacked_2layer_baseline
```

On the HPC: `bash hpc/download_data.sh` (login node), then `bash hpc/submit_all.sh` or `sbatch hpc/run_experiment.slurm <script> [flags]`.

### Common Options

- `--device`: choose the execution device (`cpu`, `cuda`, or `mps`)
- `--training-mode`: choose classifier training with or without review (`plain` or `review`)
- `--init-mode`: choose random or dataset-based initialization (`random` or `dataset`)
- `--batch-size`: set the mini-batch size
- `--organize-interval-samples`: set how many samples are processed between organize updates
- `--eval-interval-samples`: set how many samples are processed between evaluation passes
- `--review-per-sample-max`: set the maximum replay budget per real sample in review mode
- `--dataset`: any dataset registered in `training/datasets.py`
- `--num-train-samples` / `--num-test-samples`: subset sizes (`0` = the whole split)
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
