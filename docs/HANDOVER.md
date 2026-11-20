# Handover guide — Deep-DCNet, `branch_Aishani`

Aishani Patil · M.S. CS (AI), CWRU · advisor Prof. Ron Yu · state as of 2026-11-19 (code at commit `eb45afc`)

This guide gets you from zero to running and tracking experiments on this branch. Read sections 1–5 once, then use the rest as reference.

---

## 1. The project in one paragraph

DCNet learns image features with **local Hebbian / anti-Hebbian rules** (no backpropagation) in sparse "discrimination layers". Only a small classification **head** on top is trained with backprop. This branch builds an experiment framework around Lin's DCNet code (`main`): swappable heads, stacked layers, more datasets, noise/occlusion/shift robustness, a freeze-then-train protocol, lean and fully traceable result folders, and a results index. Background papers are in `docs/literature_notes/`; text logs of the September 2026 runs are in `docs/prior_experiments/deep_dcnet_sept_2026/`. Prof. Yu's research agenda (8 points) and its status are in section 9.

## 2. Where everything lives

| What | Where | In git? |
|---|---|---|
| Code | `github.com/wormLove/Deep-DCNet`, branch **`branch_Aishani`** | yes |
| Lin's code (the math) | same repo, branch `main` (**never merged** into this branch, see section 8.6) | yes |
| Pioneer clone (primary cluster) | `/home/jxw1862/DCNet/Aishani/Deep-DCNet` (inside Lin's home; keep outputs small) | clone |
| Markov clone (fallback) | `/mnt/vstor/courses/wids/axp1343/DCNet_Updated/Deep-DCNet-git` | clone |
| September 2026 raw results | Markov: `.../DCNet_Updated/Deep-DCNet_old/RESULT` (~33 GB, superseded) | no |
| Datasets | `DATA/` in each clone (downloaded on the login node) | no |
| Run results | `RESULT/<run name>/<timestamp>/` in each clone | no |
| Reusable frozen layers | `checkpoints/*.pt` + `checkpoints/manifest.json` in each clone | allowed, but keep large `.pt` files out of commits |
| Tracking docs | `docs/` (this folder) | yes |

## 3. Repo map

```text
architectures/   single_layer.py (BiologicalClassifier), stacked.py (StackedBiologicalClassifier), registry.py
modules/         discrimination.py (the Hebbian layer), optimizer.py, activation.py, readout.py,
                 classifier_heads.py (MLP head), integration.py
core/            organizer.py, lr_policy.py, stats.py, initializers.py   ← learning rules (the "math")
                 checkpointing.py (named reusable layers), perturbations.py (robustness corruptions),
                 activation_cache.py, weight_monitor.py
training/        train_classifier.py, train_classifier_stacked.py   ← entry points (plain / review / freeze)
                 training_engines.py (one-pass loops), freeze_protocol.py (freeze-then-train),
                 datasets.py (dataset registry), run_config.py, provenance.py (seed, git, env)
analysis/        analysis_engine.py (what a run saves), metrics_log.py, robustness.py
experiments/     dated, named experiment scripts (section 6) + _configs.py (run sizes, seeds)
tools/           build_results_index.py (results index + docs/results.md), compare_layers.py
hpc/pioneer/     Pioneer setup + job wrapper (primary)     hpc/ (top level): Markov versions
docs/            this guide + results.md, experiment_log.md, models_and_setups.md
```

## 4. Quick start on Pioneer (first time, about 30 min)

1. **Access:** a Pioneer account with a Slurm association (Aishani used Lin's account `cxy567`; ask Lin). Write access to `/home/jxw1862/DCNet/Aishani`, or clone into your own space and change the paths below.
2. **Clone** (login node):
   ```
   cd /home/jxw1862/DCNet/Aishani && git clone https://github.com/wormLove/Deep-DCNet.git && cd Deep-DCNet && git checkout branch_Aishani
   ```
3. **Environment** (pip torch 2.1.2 + CUDA 12.1 in `~/dcnet_pioneer_pip_venv`):
   ```
   cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && bash hpc/pioneer/setup_env_pip.sh
   ```
4. **Data** (login node only; compute nodes may have no internet; CIFAR-10 takes about an hour):
   ```
   cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && bash hpc/pioneer/download_data.sh
   ```
5. **Smoke test** and check against the reference numbers:
   ```
   cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && sbatch hpc/pioneer/run_experiment.slurm 2026-09-17_single_layer_baseline.py --smoke
   ```
   Expect `final acc 78.91%` in `hpc/logs/slurm-<id>.out`. Point-5 smoke gives 75.78 and stacked smoke 75.00. These match on every cluster, GPU and torch version tried so far.
6. **Full calibration (4 min on L40S):** run the same script without `--smoke`; expect **94.01%**.
7. **Clean up smoke outputs** (cannot be undone; removes only `_smoke` runs):
   ```
   cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && ls -d RESULT/*_smoke* checkpoints/*_smoke* 2>/dev/null
   cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && rm -rf RESULT/*_smoke* && rm -f checkpoints/*_smoke* && source hpc/pioneer/activate.sh && python -c "import json;p='checkpoints/manifest.json';m=json.load(open(p));m={k:v for k,v in m.items() if not k.endswith('_smoke')};json.dump(m,open(p,'w'),indent=2,sort_keys=True);print(sorted(m))"
   ```

More detail (GPU pinning, the module fallback, overrides): `hpc/pioneer/README.md`.

## 5. Everyday workflow

```text
edit code on your laptop ─► commit + push branch_Aishani ─► Pioneer: git pull ─► sbatch ─►
check job log ─► build_results_index.py --update-doc docs/results.md ─► add an experiment_log entry ─► commit docs
```

- Jobs: `squeue --me` · live log `tail -f hpc/logs/slurm-<id>.out` · after: `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,NodeList`.
- Every job header prints host, git commit, `git status`, torch version, GPU name and the full command.
- GPU type is pinned to **L40S** (`#SBATCH --constraint=gpul40s`) so all arms of a comparison run on the same hardware. Queue waits are usually seconds.
- Always chain `cd … && command` so nothing runs in the wrong folder.
- Getting `docs/results.md` from Pioneer into git: Aishani couldn't use scp off campus. She pasted the printed auto block (`sed -n '/AUTO-RESULTS:BEGIN/,/AUTO-RESULTS:END/p' docs/results.md`) into the laptop copy, committed there, then ran `git checkout -- docs/results.md && git pull` on Pioneer. With VPN access, scp or a GitHub token on Pioneer is simpler.

## 6. Running experiments

All scripts take `--smoke` (tiny run; numbers meaningless), `--device`, `--seed` (default 33; use 34 and 35 for repeats) and `--analysis-level lean|full`. Submit any of them with `sbatch hpc/pioneer/run_experiment.slurm <script> [flags]`.

| Script | What it runs | Example flags |
|---|---|---|
| `2026-11-15_freeze_baselines.py` | **Main protocol.** Freeze-then-train for any dataset / architecture / head | `--dataset mnist --arch single_layer --head readout` |
| `head_on_frozen.py` | Train a new head on a finished freeze run's frozen layer(s) (~30 s) | `--from-run freeze_mnist_single_layer_readout_s33 --head traditional_mlp` |
| `2026-09-17_single_layer_baseline.py` | One-pass `plain`: single layer + linear readout (MNIST) | — |
| `2026-09-17_point5_traditional_head.py` | One-pass `plain`: single layer + MLP head (MNIST) | — |
| `2026-09-17_stacked_2layer_baseline.py` | One-pass `review`: two stacked layers + readout (MNIST) | — |
| `2026-09-18_point6_fashion_mnist.py`, `…_point6_cifar10.py` | Same three one-pass configs on other datasets | `--arch single_layer\|stacked --head readout\|traditional_mlp` |
| `2026-09-18_point4_robustness.py` | Evaluates finished runs on corrupted test images (13 corruption levels) | `--runs <run names that share a dataset>` (CPU job is fine: `-p batch --device cpu`) |

Freeze-protocol options (defaults in `training/freeze_protocol.py`): `--val-size 5000 --hebbian-epochs 1 --head-epochs 30 --head-batch-size 128 --head-lr 1e-3 --patience 5`. **A non-default value is written into the run name** (e.g. `_hep100`), so variants never overwrite or get averaged with the default.

Tools:
- `python tools/build_results_index.py [--result-root RESULT] [--update-doc docs/results.md]`: rebuilds the index (section 8.3).
- `python tools/compare_layers.py <run_dir_a> <run_dir_b>`: max difference per tensor between two runs' discrimination layers. Use it to prove a code change didn't alter the learned layer.

## 7. What a run produces (~45 MB single layer, ~110 MB stacked)

`RESULT/<run name>/<timestamp>/`

| File | Contents |
|---|---|
| `run_config.json` | Everything needed to rebuild the run: architecture, dataset, head, sizes, all hyperparameters, **seed, setup, protocol, git commit + dirty flag, cluster, Slurm job, GPU, Python/torch versions, full command line** |
| `final_eval.json` | `status` (ok/failed), final and best accuracy, wall times, GPU memory; freeze runs add best_val_acc, best_epoch, epochs_run, phase times |
| `metrics.csv` | One row per `organize()`: loss, test accuracy (when evaluated), weight change, learning-rate stats, inactive neurons, stability (per layer for stacked) |
| `head_metrics.csv` | Freeze runs only: per head epoch, train loss/acc, val acc, test acc |
| `eval_log.txt`, `train_log.txt` | Human-readable logs (one-pass runs have both) |
| `weights_final[_L0/_L1]_grid.png` | What each neuron learned (receptive fields) |
| `checkpoints/final.pth` | Full model (layers + head); `training/run_config.py: load_trained_run()` rebuilds it |

`--analysis-level full` restores the old per-step images and per-organize checkpoints for debugging only. They made each run 4–14 GB.

## 8. How results and experiments are tracked

The system has five layers. Each one answers a different question, so nothing has to be reconstructed from memory.

**8.1 Each run describes itself.** A run folder contains its full configuration, provenance (commit, seed, GPU, versions) and results (section 7). Any number can be traced to the exact code and settings that produced it. A crashed run says `failed` with the error instead of looking incomplete.

**8.2 Names carry meaning.** Run name = `<experiment>[_<non-default options>]_s<seed>`, e.g. `freeze_mnist_stacked_readout_s34`. A timestamp sub-folder keeps reruns apart. Smoke runs end in `_smoke` and are excluded from every summary. Parallel seeds never collide.

**8.3 One command builds the overview.** `tools/build_results_index.py` scans `RESULT/` and writes:
- `RESULT/INDEX.md`, `index_runs.csv` (one row per run), `index_summary.csv` (one row per configuration: n seeds, mean, std, min, max) and robustness CSVs;
- with `--update-doc docs/results.md`, the **auto-generated table** in `docs/results.md` (dataset · architecture · head · setup · protocol · seeds · mean ± std · runs), plus a robustness summary when sweeps exist. Only the text between the `AUTO-RESULTS` markers is rewritten.

**8.4 Three docs carry the story** (`docs/README.md` is the index):

| Doc | Question it answers | How it's updated |
|---|---|---|
| `results.md` | What are the numbers now? | Auto block via the index tool; calibration and archived sections by hand |
| `experiment_log.md` | What was done when, and what did it show? | One dated entry per batch of runs, newest first: what · runs (names, cluster, GPU, commit) · result · takeaway · next |
| `models_and_setups.md` | What do "S0", "freeze", "stacked" mean? | When a setup, protocol, architecture, head or dataset is added |

**8.5 Every experiment sits on the same axes:** setup × architecture × head × dataset × protocol × seed. These are the columns of the results table and fields in every `run_config.json`.

**8.6 Math versions have names.** Lin develops the learning rules on `main`. This branch never merges `main`; each of Lin's changes is ported by hand as a named **setup** (S0 = his May 2026 release, current default; S1 = variance-stop optimizer; S2 = S1 + new init defaults; S1/S2 not ported yet). Every setup stays runnable, so old vs new math can run on identical experiments. See `models_and_setups.md` §1.

**8.7 Built-in checks.**
- Smoke references 78.91 / 75.78 / 75.00 and full calibration 94.01 confirm an environment.
- `compare_layers.py` confirms a refactor didn't change learning (the freeze protocol was verified this way: Δ = 0 on every tensor).
- Each comparison runs on one cluster and one GPU type.

## 9. Current state

**Results (MNIST, setup S0, seed 33 only; see `results.md`):**

| Configuration | One-pass | Freeze-then-train |
|---|---|---|
| Single layer + linear readout | 94.01 | **96.59** |
| Single layer + MLP head | 96.17 | **96.90** (same frozen layer) |
| Two stacked sparse layers + readout | 93.93 | **95.30** |

Freezing closes most of the readout–MLP gap (2.16 → 0.31 points), so the September "MLP head is better" result was mostly about head training. Stacking still doesn't help.

**Prof. Yu's agenda:**

| # | Point | Status |
|---|---|---|
| 1 | Parallel discrimination layers → dense layer | Not started |
| 2 | Class-specific classification module (Raj & Yu) | Not started; confirm interpretation with Prof. Yu |
| 3 | Combine sparse + dense, then expand again | Not started; overlaps the "input + sparse → dense → sparse" layer agreed at the October meeting |
| 4 | Noise / masking robustness | Final-accuracy sweeps done (September, one-pass); per-layer representation analysis not done |
| 5 | Traditional classification head | Done |
| 6 | More datasets | Done (registry: MNIST, Fashion-MNIST, KMNIST, CIFAR-10, CIFAR-10 gray) |
| 7 | Fourier-domain input | Deferred by Prof. Yu |
| 8 | Overlapping tiles / strided kernels | Not started; motivated by shift robustness (±4 px → ~35 %) |

## 10. Next steps and known issues

**Next, in order:**
1. Check whether 30 head epochs cap the readout numbers: `head_on_frozen.py --from-run <run> --head readout --head-epochs 100` for the single and stacked freeze runs (seconds each).
2. Run freeze baselines with **3 seeds** (33, 34, 35) on MNIST, Fashion-MNIST, CIFAR-10 (and KMNIST, CIFAR-10 gray).
3. Robustness sweeps on freeze runs (the script loading a freeze run was submitted but its output never checked).
4. Port Lin's S1 and S2 setups; rerun the same suite on them.
5. Architectures: input + sparse → dense → sparse, then agenda points #1, #2, #3, #8.

**Known issues:**
- `final.pth` stores the correlation matrix and Hebbian potentials (~35–40 MB single layer). Fine for now.
- Stacked `review` mode replays layer-0's cache through layer 1 every step, which is slow.
- Restoring a checkpoint to *resume* Hebbian training is unverified (frozen reuse is fine).
- The one-pass training loops call a final `organize()`; never run a frozen layer through them (the freeze protocol avoids them on purpose).
- `2026-09-18_point4_robustness.py --smoke` needs `--runs <name>_s33_smoke` passed explicitly.
- Pioneer's PyTorch *module* only runs on RTX 2080 GPUs; that's why the pip environment is the default.
- The September results on Markov (~33 GB) can be deleted once nobody needs them; their numbers are archived in `docs/results.md` and their text logs in `docs/prior_experiments/deep_dcnet_sept_2026/`.

## 11. Conventions

- Lin owns `main`; never merge it into this branch (section 8.6).
- Seeds 33 / 34 / 35; report mean ± std; never report smoke numbers.
- Same cluster and GPU type for every arm of a comparison.
- New experiment → a dated script in `experiments/` (`YYYY-MM-DD_<name>.py`) with a fixed run name.
- After every batch of runs: update `results.md` (index tool) and add an `experiment_log.md` entry.

## 12. People

- **Prof. Ron Yu** (CWRU): advisor; sets the research agenda.
- **Lin (Junlin)**: author of Deep-DCNet and owner of `main` and the Pioneer folder; ask him about the math and Pioneer access.
- **Aishani Patil**: author of `branch_Aishani`.
