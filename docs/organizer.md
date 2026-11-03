# Organizer: Per-Neuron Consolidation and Review Gating

## Purpose

This document records the organizer changes made after `v0.3.0`, motivated by
treating the organize interval as a night of sleep: consolidation happens at
sleep, and it should apply only to the neurons that took in new information
during the preceding wake period. Four mechanisms changed; each previous
mechanism remains available as an option for controlled comparisons.

The organize interval itself is unchanged (200 training samples). Its value is
justified empirically: about 200 samples let almost every neuron fire at least
once before the first organize, so the first consolidation turns nearly every
neuron into a data-driven template.

## Organize Only Activated Neurons

At every organize, `DiscriminationLayer` updates only the neurons that were
active at least once since the previous organize. Silent neurons keep their
weights bit for bit. Without this gate a silent neuron was still moved by the
anti-Hebbian term and by learning rates that had recovered while it was
inactive, that is, it learned without a stimulus.

| Setting | Default | Previous behaviour |
|---|---|---|
| `organize_update_scope` | `"activated"` | `"global"` |

## Potential Memory Per Sample

The Hebbian and anti-Hebbian potentials are exponential moving averages. They
now decay once per training sample: a batch of `B` samples decays them by
`beta ** B` and adds the batch's total weight `1 - beta ** B`, shared equally
by its samples. The memory is therefore counted in samples, like the organize
interval, and does not depend on the batch size.

```text
P <- beta**B * P + (1 - beta**B) * (batch potential / B)
```

The training configurations use `beta = 199/200` per sample: after one
interval of 200 samples a sample's contribution has decayed to
`0.995**200 = 0.37`, as in the original single-sample design, where
`199/200` was set by the organize interval. `v0.3.0` decayed once per batch
of 4, a memory of 800 samples (4 intervals); `beta = (199/200) ** (1/4)`
reproduces it bit for bit.

The schedule used by the previous organize rule moved to its own
`correction_beta` (`199/200`, decayed once per organize).

## Organize Step

The organize step moves each neuron's weight column along the normalized
residual of its potentials:

```text
potential_diff = P_hebb - W P_antihebb
W_new = (1 - lr) W + lr * normalize(potential_diff)
```

followed, as before, by clamping to non-negative values, re-initializing empty
columns and normalizing columns. The step size is set only by the neuron's
learning rate. The previous rule scaled the raw residual by a global
`correction_factor = 1 / (1 - beta_accum)`, which made the effective step
depend on the absolute size of the potentials and on the organize count, and
shrank late plasticity for all neurons regardless of their learning rate.

| Setting | Default | Previous behaviour |
|---|---|---|
| `organize_step` | `"direction"` | `"correction_factor"` (with `correction_beta`) |

## Review Gate

Review replays cached activity to the readout once the lateral operator seen
by the optimizer is stable. Both gates measure the relative Frobenius drift of
the effective operator between consecutive organizes.

Under the direction step the drift settles on a floor of about 0.035, set by
top-k mask boundary swaps that continue at `min_lr`, so the previous absolute
threshold of 1% never opens review. The default gate therefore tests whether
the drift has levelled off, whatever its level:

```text
recent = median drift of the last 10 organizes
before = median drift of the 10 organizes before
pass   = (1 - 0.1) * before <= recent <= (1 + 0.1) * before
```

Review opens after 2 consecutive passes; the median at that organize becomes
the reference, and a later drift above 2 times the reference closes review.
The test is two-sided: a drift still falling, or rising, by more than 10% is
not a plateau.

| Setting | Default | Previous behaviour |
|---|---|---|
| `review_gate` | `"plateau"` (`window` 10, `allowed_decrease` 0.1, `k_enter` 2, `exit_ratio` 2.0) | `"absolute"` (`tau_enter` 0.01, `tau_exit` 0.02, `k_enter` 2) |

`allowed_decrease` is the relative tolerance in both directions.

## Organize Trigger

The training engines ask an organize trigger when to organize.
`FixedIntervalTrigger(200)` is the default and reproduces the previous
behaviour bit for bit; `ScheduledIntervalTrigger` takes an explicit list of
intervals for studies.

## Validated Default

One MNIST epoch (60,000 training cases, 10,000 test cases), 1,000 hidden
neurons, three seeds:

| Configuration | 10k test accuracy |
|---|---|
| `v0.3.0` | 95.78 / 95.91 / 95.75 |
| Activated-neuron gate, previous organize rule and gate | 95.59 / 95.64 / 95.70 |
| Direction step, absolute gate (review never opens) | 94.49 / 94.61 / 94.46 |
| Direction step, plateau gate, 800-sample memory | 95.92 / 95.84 / 95.73 |
| Direction step, plateau gate, one-interval memory (default) | 95.92 / 95.82 / 95.89 |

The activated-neuron gate has no resolvable accuracy effect on stationary
MNIST; its purpose is that silent neurons do not drift, which matters for
non-stationary streams. With review opened at about the same organize and with
the same replay volume as the previous mechanisms, the direction step with the
plateau gate is 0.03 to 0.33 above them in every seed. The one-interval memory
learns faster early (+0.16 to +0.35 at 10,000 samples) and ends at the same
accuracy; its late weight jitter is higher.

These results establish the default for single-target training. Behaviour
under non-stationary data (target switches, pool expansion) has not been
evaluated.

## Implementation Map

- `core/organizer.py`: per-sample potential decay, `organize_step`,
  `correction_beta`.
- `models/discrimination.py`: `organize_update_scope`, `organize_step`.
- `core/weight_monitor.py`: `relative_drift`, `EffectiveOperatorDriftMonitor`,
  `PlateauDriftMonitor`.
- `models/biological_classifier.py`: `review_gate`, `review_gate_config`,
  checkpoint metadata (`organize_direction_v1`); `load_model` accepts
  checkpoints without `organizer.beta_accum`.
- `core/organize_trigger.py`: `FixedIntervalTrigger`, `ScheduledIntervalTrigger`.
- `training/training_engines.py`: organize timing through the trigger.
- `training/train_classifier.py`, `training/train_discrimination.py`: `beta`,
  `correction_beta`.

## Verification

- Unit tests: `python -m unittest discover -s tests -t .`.
- Each previous behaviour is reproduced bit for bit by its option: the
  per-sample decay at `beta = (199/200) ** (1/4)` and batch 4, the
  `"correction_factor"` step with the absolute gate, and the default trigger.
  Full-scale runs must use the same thread count; a different count changes
  the dataset-based initial weights by about 1e-5.
- Study records, scripts and per-run data are stored outside the source
  repository under `RESULT/Deep-DCNet-Organizer-Study/`.
