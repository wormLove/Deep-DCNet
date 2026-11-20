# Experiment log

Newest first. One entry per batch of runs or work session.

**Template**
### YYYY-MM-DD — short title
- **What:** what changed or was run
- **Runs:** run names · cluster · GPU · commit
- **Result:** numbers, with units
- **Takeaway:** one or two lines
- **Next:** what this leads to

---

### 2026-11-15 — Freeze-then-train protocol: first MNIST numbers
- **What:** Plan 03. New protocol `freeze` (`training/freeze_protocol.py`): 5,000-image validation split, Hebbian layer(s) learned without labels (stacked: layer by layer), frozen, features cached, head trained up to 30 epochs with early stopping on val accuracy. `--training-mode freeze` in both training entry points; new scripts `experiments/2026-11-15_freeze_baselines.py`, `experiments/head_on_frozen.py` (new head on a finished run's frozen layers) and `tools/compare_layers.py`. `plain` and `review` unchanged.
- **Runs:** `freeze_mnist_single_layer_readout_s33`, `freeze_mnist_stacked_readout_s33`, `freeze_mnist_single_layer_readout_head-traditional_mlp_s33` (head only, on the first run's frozen layer) + smoke checks · Pioneer · NVIDIA L40S · eb45afc
- **Result:** equivalence check passed: with `--val-size 0` the frozen layer equals the plain run's exactly (every tensor Δ = 0). Full MNIST, test accuracy of the best-val head: single + readout **96.59 %** (plain 94.01), single + MLP head **96.90 %** on the same frozen features (plain 96.17), stacked + readout **95.30 %** (one-pass 93.93). Readout head ran all 30 epochs (best 28), stacked all 30 (best 30), MLP head early-stopped at 17 (best 12). Wall time 1 min 29 s single, 3 min 37 s stacked, 28 s head-only.
- **Takeaway:** freezing adds ~2.6 points to the linear readout; the readout vs MLP gap shrinks from 2.16 to 0.31 points, so most of the September head gap was head training, not feature quality. Stacking is still below single (−1.3), but its head was still improving at epoch 30. One seed only.
- **Next:** check whether 30 head epochs cap the readout/stacked numbers; then 3 seeds and the other datasets under `freeze`.

### 2026-11-15 — Pioneer on L40S with pip PyTorch
- **What:** Plan 02c. pip-installed torch 2.1.2 + cu121 in its own venv (same torch version as the module, all current GPU types); GPU type pinned to L40S; failed runs now marked `failed` in `final_eval.json` and the results index.
- **Runs:** smoke ×3 and `single_layer_baseline_s33` (full) · Pioneer · NVIDIA L40S (gput064) · a17ce60
- **Result:** smoke 78.91 / 75.78 / 75.00 %; full MNIST **94.01 %**, 3 min 44 s (vs 10 min 39 s on the RTX 2080 Ti), 44 MB. Queue wait 1 s – 2 min (was ~1 day).
- **Takeaway:** identical numbers on L40S, RTX 2080 Ti and Markov. Pioneer + L40S is now the default for all runs.
- **Next:** 3-seed runs and the freeze-then-train protocol (Plan 03).

### 2026-11-14 — Pioneer verified; switching to pip PyTorch on L40S
- **What:** Plan 02b. Pioneer setup + smoke runs + one full calibration run.
- **Runs:** smoke ×4 and `single_layer_baseline_s33` (full) · Pioneer · RTX 2080 Ti (gput048) · c284501
- **Result:** smoke 78.91 / 75.78 / 75.00 % (identical to Markov); full MNIST **94.01 %** (identical to Sept), 10 min 39 s, 44 MB.
- **Takeaway:** results don't depend on cluster, torch version (2.1.2 vs 2.14) or GPU for this config. The Pioneer PyTorch module only runs on RTX 2080 GPUs (sm_75), so jobs queued ~1 day.
- **Next:** Plan 02c, pip-installed torch 2.1.2 and pinned L40S GPUs; repeat calibration there.

### 2026-11-02 — Lean results and complete run records
- **What:** Plan 02. No more per-step images or intermediate checkpoints; each run writes `metrics.csv`, `final_eval.json` and a full `run_config.json` (seed, setup, commit, cluster, GPU, versions). Seed added to run names.
- **Runs:** smoke tests · Markov · RTX 2080 Ti · ef245c4, 7db6ae5, c284501
- **Result:** old vs new code, same GPU and seed: 78.91 / 75.78 / 75.00 % both, so math unchanged. Run folder 550 MB → 45 MB (single), 108 MB (stacked).
- **Takeaway:** runs are now small and fully traceable.
- **Next:** verify on Pioneer.

### 2026-11-01 — Git-based cluster workflow
- **What:** Plan 01. Pioneer scripts (`hpc/pioneer/`); Markov re-cloned as a git repo; Markov jobs use QOS `l1`.
- **Runs:** Markov smoke job 58880 · 78.91 % · f7df4cc / f861a63
- **Takeaway:** code reaches the clusters through `git pull` instead of manual uploads.

### 2026-10-30 — Math setups named
- **What:** went through Lin's history on `main` and named S0 (v0.1.0), S1 (variance-stop optimizer), S2 (init audit defaults). Confirmed branch_Aishani's math = S0.
- **Takeaway:** old and new math can be compared side by side; Prof. Yu's variants of S0 will be S0a, S0b, …

### 2026-10-29 — September results re-checked against raw logs
- **What:** re-read every log of the nine September runs.
- **Result:** all numbers confirmed (see results.md, archived section). Stacked 93.93 % vs single 94.01 % on MNIST: no difference (not the ~2 % drop reported at the meeting; the ~92 % figure came from the older repo).
- **Takeaway:** results are preliminary. Single seed; stacked used `review`, single used `plain`; nothing was frozen; setup differs from Lin's.
- **Next:** 3 seeds, freeze-then-train protocol, lean outputs (the nine runs used ~93 GB).

### Before 2026-10-20 — Meeting with Prof. Yu and Lin
- **Decisions:** test clean data before noise; dense layer should take input + sparse output (input → sparse → dense → sparse); class-specific classification layer needed before CIFAR/ImageNet; aim for orientation independence.
- **Action items:** build the input + sparse dense layer; organize results; integrate Lin's new discrimination-layer rule.

### 2026-09-18/19 — September runs (Markov, S0, seed 33)
- **What:** 3 configurations (single + readout, single + MLP head, stacked + readout) × MNIST, Fashion-MNIST, CIFAR-10, plus noise/occlusion/shift robustness sweeps.
- **Result:** see results.md, archived section.

### September 2026 — Code built on branch_Aishani
- Swappable heads (point #5), stacked architecture with stability gating, dated experiment scripts, reusable checkpoints, dataset registry (point #6), robustness sweeps (point #4), Markov Slurm scripts.
