# Running experiments on Pioneer

Experiments run from a git clone of `branch_Aishani` in
`/home/jxw1862/DCNet/Aishani/Deep-DCNet`. Code changes are made on the Mac,
pushed, and pulled on Pioneer - nothing is copied by hand.

The Markov scripts in `hpc/` (one level up) are kept for reference; the
files in this folder are the Pioneer versions.

| File | What it does |
|---|---|
| `setup_env.sh` | One-time setup (login node): loads the PyTorch module, adds tqdm/matplotlib via a small venv only if needed, prints versions. |
| `activate.sh` | `source` it to get the same environment the jobs use (module + venv if present). |
| `download_data.sh` | Downloads datasets into `DATA/` (login node only). |
| `run_experiment.slurm` | Job wrapper: runs one script from `experiments/` on 1 GPU. |

The shared filesystem is inside Lin's home - keep outputs small.

## First-time setup (login node)

1. Clone the repo and check out the branch:
   ```
   cd /home/jxw1862/DCNet/Aishani && git clone https://github.com/wormLove/Deep-DCNet.git && cd Deep-DCNet && git checkout branch_Aishani && git log --oneline -3
   ```
   If git asks for a password, the repo is private: use a GitHub personal
   access token as the password, or set up an SSH key on Pioneer.

2. Set up the environment (prints Python / torch / torchvision / CUDA versions):
   ```
   cd /home/jxw1862/DCNet/Aishani/Deep-DCNet && bash hpc/pioneer/setup_env.sh
   ```

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
usually a bad node. Just resubmit the same command. If it keeps failing
the same way, check the `.err` file and `nvidia-smi -L` line in the `.out`
file.
