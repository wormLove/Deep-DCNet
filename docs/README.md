# Project docs — start here

Aishani Patil · branch_Aishani of Deep-DCNet · advisor Prof. Ron Yu (CWRU)

| File | Read it for | Updated |
|---|---|---|
| [results.md](results.md) | The latest numbers (mean ± std over seeds) | After each batch of runs (auto-generated block) |
| [experiment_log.md](experiment_log.md) | What was done when, and what it showed | After every batch of runs or work session |
| [models_and_setups.md](models_and_setups.md) | Math setups, training protocols, architectures, default configs, datasets | When something new is added |

Also here: `literature_notes/` (papers), `prior_experiments/` (records from the older DC_Net_Hebbian_Model repo).

## Naming conventions
- **Run name:** `<experiment>_s<seed>` (e.g. `single_layer_baseline_s33`); results live in `RESULT/<run name>/<timestamp>/` on the cluster (not in git).
- **Setup:** math version, `S0`, `S1`, `S2`, … (see models_and_setups.md).
- **Protocol:** `plain`, `review` or `freeze`.
- **Seeds:** 33 by default; 34 and 35 for repeats.
