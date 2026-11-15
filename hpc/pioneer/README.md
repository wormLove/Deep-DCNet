# Running experiments on Pioneer

Experiments run from a git clone of `branch_Aishani` in
`/home/jxw1862/DCNet/Aishani/Deep-DCNet`. Code changes are made on the Mac,
pushed, and pulled on Pioneer - nothing is copied by hand.

The Markov scripts in `hpc/` (one level up) are kept for reference; the
files in this folder are the Pioneer versions.

| File | What it does |
|---|---|
| `setup_env_pip.sh` | One-time setup (login node) of the **default pip environment**: Python module + venv `~/dcnet_pioneer_pip_venv` with pip torch 2.1.2 + cu121, prints versions and GPU archs. |
| `setup_env.sh` | One-time setup (login node) of the **module fallback**: loads the PyTorch module, adds tqdm/matplotlib via a small venv only if needed. |
| `activate.sh` | `source` it to get the same environment the jobs use; `DCNET_TORCH` picks which (default `pip`). |
| `download_data.sh` | Downloads datasets into `DATA/` (login node only). |
| `run_experiment.slurm` | Job wrapper: runs one script from `experiments/` on 1 GPU, pinned to L40S by default. |

The shared filesystem is inside Lin's home - keep outputs small.

## Environments and GPU type

Two PyTorch environments, selected with the `DCNET_TORCH` variable
(read by `activate.sh`, so it applies to jobs too):

| `DCNET_TORCH` | What | Runs on |
|---|---|---|
| `pip` (default) | `Python/3.11.3-GCCcore-12.3.0` module + venv `~/dcnet_pioneer_pip_venv` with **torch 2.1.2 + cu121** from pip (set up by `setup_env_pip.sh`) | L40S, V100, H100, 4090, 2080 Ti, ... |
| `module` | `PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1` (+ `~/dcnet_pioneer_venv` if present) | **RTX 2080 Ti only** (`gpu2080`) - the module is built for `sm_75` only |

Both are torch 2.1.2, so only the GPU support differs. Results did not
depend on torch version or GPU in Plan 02b (smoke 78.91 / 75.78 / 75.00,
full baseline 94.01 on Markov and Pioneer alike).

**GPU type is pinned** in `run_experiment.slurm` with
`#SBATCH --constraint=gpul40s` (9 nodes x 4 L40S), so every arm of a
comparison runs on the same GPU. Override for one job on the command line:
```
cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && sbatch --constraint=gpu4v100 hpc/pioneer/run_experiment.slurm <script> [flags]
```
GPU features in partition `gpu`: `gpul40s`, `gpu2080`, `gpu4v100`,
`gpu2v100`, `gpu2h100`, `gpu4090`, `gpup100`.

To use the module fallback, pass both the variable and the 2080 constraint:
```
cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && sbatch --constraint=gpu2080 --export=ALL,DCNET_TORCH=module hpc/pioneer/run_experiment.slurm <script> [flags]
```
The job header in `hpc/logs/slurm-<id>.out` prints `DCNET_TORCH`, the
torch version, `torch.cuda.get_arch_list()` and the GPU name, so you can
check which environment and GPU a run used.

If a run raises during training (e.g. `no kernel image is available` on an
unsupported GPU), its `final_eval.json` says `"status": "failed"` with the
error, and `tools/build_results_index.py` lists it as `failed`.

## First-time setup (login node)

1. Clone the repo and check out the branch:
   ```
   cd /home/jxw1862/DCNet/Aishani && git clone https://github.com/wormLove/Deep-DCNet.git && cd Deep-DCNet && git checkout branch_Aishani && git log --oneline -3
   ```
   If git asks for a password, the repo is private: use a GitHub personal
   access token as the password, or set up an SSH key on Pioneer.

2. Set up the pip environment (prints Python / torch / torchvision / CUDA versions and GPU archs):
   ```
   cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && bash hpc/pioneer/setup_env_pip.sh
   ```
   If the Python module name is wrong, `module spider Python/3.11` lists the
   available ones; re-run with `PY_MODULE=<name> bash hpc/pioneer/setup_env_pip.sh`
   and change the `PY_MODULE` default in `activate.sh` to match.
   (Only for the module fallback: `bash hpc/pioneer/setup_env.sh`.)

3. Download datasets (default: mnist fashion_mnist kmnist cifar10; pass names or `all` for others):
   ```
   cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && bash hpc/pioneer/download_data.sh
   ```

4. Submit a smoke test:
   ```
   cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && sbatch hpc/pioneer/run_experiment.slurm 2026-09-17_single_layer_baseline.py --smoke
   ```
   If sbatch is rejected for an account, find yours with
   ```
   sacctmgr -nP show assoc user=$USER format=account,partition
   ```
   and fill in the commented `--account=` line in `run_experiment.slurm`
   (remove the extra leading `#`).

## Everyday loop

- **Mac:** edit -> commit -> push `branch_Aishani`.
- **Pioneer:**
  ```
  cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && git pull && sbatch hpc/pioneer/run_experiment.slurm <script> [flags]
  ```
  `<script>` is a file name in `experiments/`; `--device cuda` is added
  automatically and any extra flags are passed through to the script.

Each job log starts with the host, commit hash and `git status`, so a run
from a dirty tree is visible in the log.

## Checking jobs

```
squeue --me                                             # your queued / running jobs
tail -f hpc/logs/slurm-<id>.out                         # follow a job's output (errors in .err)
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS       # state, run time, peak memory
```

## Transient GPU errors

Occasionally a job fails at start-up with a CUDA / GPU error (e.g. no
device found, CUDA init failure) that has nothing to do with the code -
usually a bad node. Just resubmit the same command. (`no kernel image is
available for execution on the device` is different: the torch build does
not support that GPU - e.g. the module env on a non-2080 node.) If it keeps failing
the same way, check the `.err` file and `nvidia-smi -L` line in the `.out`
file.
