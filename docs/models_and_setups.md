# Models and setups

Every experiment = **setup × architecture × head × dataset × protocol × seed**. This page defines each axis.

## 1. Math setups
Lin's math changes on `main` are ported by hand into `branch_Aishani` as named setups; `main` and `branch_Aishani` are never merged. Every setup stays runnable.

| Setup | Source (Lin's `main`) | Activity optimizer | Initialization | Status |
|---|---|---|---|---|
| **S0** legacy | v0.1.0 (aac90f6), 2026-05-02 | L1 shrinkage (λ 0.1), step 1/(10·λ_max) with λ_max estimated, fixed 1000 iterations, then std threshold + top-5% cut | non-negative by `shift`; PCA init includes Mid Matrix | In use (branch default) |
| **S1** | b1cabf4, 2026-09-28 | No L1, no threshold, no top-k; step 1/λ_max (exact); stops at variance turning point with active fraction ≤ 5% (first phase) / 8% (later) | same as S0 | Planned (Plan 04) |
| **S2** | 37d799f, 2026-10-06 | same as S1 | non-negative by `abs`; PCA init without Mid Matrix | Planned (Plan 04) |

Variants of S0 requested by Prof. Yu will be added as S0a, S0b, … with a row here and an entry in the experiment log.

## 2. Training protocols
| Protocol | What happens | Status |
|---|---|---|
| `plain` | One pass over the training set. Each batch: Hebbian layer learns, head takes one backprop step; `organize()` every N images. | In use |
| `review` | `plain` + replay: once a layer is stable, the head also trains on up to 256 cached past activations per step (misclassified ones repeated). Lin's main mode. | In use |
| `freeze` | Hebbian layer(s) first without labels, freeze, then train the head on the frozen features with early stopping. | In use (main protocol from Plan 03) |

`freeze` in detail (`training/freeze_protocol.py`, script `experiments/2026-11-15_freeze_baselines.py`):
1. **Validation split:** the last 5,000 images of a seeded permutation of the training set (`val_size`; 0 = no split). The test set is never used for any choice.
2. **Hebbian phase:** one pass over the rest (`hebbian_epochs` 1), `organize()` every N images plus a final one, as in `plain` but without the head. Stacked: layer by layer (L0, freeze, then L1 on frozen L0's output), not the stability-gated cascade.
3. **Freeze + features:** layers to eval mode, no gradients, nothing organizes again; fit / val / test pass through them once and the features stay in memory.
4. **Head:** Adam (lr 1e-3), batch 128, up to 30 epochs, early stopping after 5 epochs without a better val accuracy; the best-val epoch's head is kept. Reported accuracy = that head's test accuracy. Per-epoch curves in `head_metrics.csv`.
5. `experiments/head_on_frozen.py` trains another head on a finished freeze run's frozen layers in minutes (no Hebbian retraining).

With `--val-size 0` the Hebbian phase sees the same images in the same order as `plain`, so a single-layer freeze run learns the identical layer as the plain run with the readout head (`tools/compare_layers.py`).

Only the head (and the optional integration layer) is trained with backprop; Hebbian layers never get gradients.

## 3. Architectures
| Name | Shape | Status |
|---|---|---|
| `single_layer` | input → sparse discrimination layer → head | In use |
| `stacked` | input → sparse → sparse → head (layer *i* learns only once layer *i−1* is stable) | In use |
| sparse → dense(input ⊕ sparse) → sparse | dense layer sees raw input + sparse code, then re-expands (meeting action item; related to point #3) | Planned |
| Parallel discrimination layers (point #1) | several sparse layers side by side → dense | Planned |
| Class-specific classification module (point #2) | Raj & Yu Hebbian classification layer instead of backprop readout | Planned |
| Local receptive fields (point #8) | overlapping tiles / strided kernels | Planned |

```text
single_layer:  input (784) ──► sparse DL (2000) ──► head ──► 10 classes
stacked:       input (784) ──► sparse DL0 (2000) ──► sparse DL1 (2000) ──► head ──► 10 classes
```
The "dense layer" in the September stacked runs was a second **sparse** discrimination layer, not a dense one.

## 4. Heads
| Head | Definition | Trained by |
|---|---|---|
| `readout` | one linear layer | backprop (Adam, lr 1e-3) |
| `traditional_mlp` | Linear → ReLU → Dropout(0.2) → Linear, hidden 256 | backprop (Adam, lr 1e-3) |

## 5. Default run sizes (`experiments/_configs.py`)
| | FULL | SMOKE (code-path check only) |
|---|---|---|
| Train / test images | full split (60k/10k; CIFAR 50k/10k) | 640 / 128 |
| Batch size | 32 | 32 |
| `organize()` every | 1000 images | 64 images |
| Eval every | 5000 images | 320 images |
| Width | 2000 per layer | 2000 |
| Init | random | random |
| Seed | 33 (34, 35 for repeats) | 33 |
| Output level | lean | lean |

Lin's validation runs use a different size: width 1000, batch 4, organize every 200. Numbers from the two sizes are not directly comparable.

## 6. Datasets (`training/datasets.py`)
| Name | Input | Classes | Status |
|---|---|---|---|
| `mnist` | 1×28×28 → 784 | 10 | Run |
| `fashion_mnist` | 1×28×28 → 784 | 10 | Run |
| `cifar10` | 3×32×32 → 3072 | 10 | Run |
| `kmnist` | 1×28×28 → 784 | 10 | Registered, not run |
| `cifar10_gray` | 1×32×32 → 1024 | 10 | Registered, not run |

## 7. Where runs happen
| Cluster | Repo path | Environment | GPU |
|---|---|---|---|
| Pioneer (primary) | `/home/jxw1862/DCNet/Aishani/Deep-DCNet` | pip torch 2.1.2 + cu121 venv (`DCNET_TORCH=pip`, default); module fallback — see `hpc/pioneer/README.md` | NVIDIA L40S, pinned with `--constraint=gpul40s` |
| Markov (fallback) | `/mnt/vstor/courses/wids/axp1343/DCNet_Updated/Deep-DCNet-git` | Python 3.13 venv, torch 2.14 | varies |

All arms of a comparison run on the same cluster and GPU type.
