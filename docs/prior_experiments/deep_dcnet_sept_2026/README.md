# September 2026 runs (archived, superseded)

Text records of the nine full training runs and three robustness sweeps from 2026-09-18/19 (Markov, setup S0, one pass, seed 33, random init, batch 32, width 2000, `full` analysis level). Only the logs and configs are kept; checkpoints and images stayed on Markov (`/mnt/vstor/courses/wids/axp1343/DCNet_Updated/Deep-DCNet_old/RESULT`).

Single-layer runs used `plain`, stacked runs used `review`, and nothing was frozen, so these numbers are superseded by the freeze-then-train runs. Summary tables: [`../../results.md`](../../results.md), archived section.

Each folder: `run_config.json`, `eval_log.txt` (test accuracy every 5,000 images; last line `final_eval`), `train_log.txt`; robustness folders: `robustness.tsv`, `table.txt`.
