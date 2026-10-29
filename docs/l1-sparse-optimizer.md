# L1-Normalized Sparse Lateral Optimizer

## Purpose

This document records the optimizer route selected after the `v0.2.0`
variance-based baseline. The new route keeps non-negative sparse activity and
local lateral competition while reducing the number of optimizer iterations by
more than twentyfold in the validated MNIST configuration.

The implementation exposes two tested profiles:

- `l1_sparse`: the current default described here;
- `variance_dense`: the dense `v0.2.0` baseline retained for controlled
  comparisons.

The profiles are complete mechanism bundles. Untested cross-combinations such
as the variance optimizer with a sparse connection schedule are rejected.

## Activity Update

Let `a` be the raw activity, `C_eff` the effective masked lateral operator, and
`y_t` the optimizer state. The raw activity and each projected state are
L1-normalized inside the optimizer:

```text
a_hat = a / ||a||_1
y_0   = a_hat

e_t     = a_hat - y_t C_eff
z_t     = ReLU(y_t + alpha e_t)
y_(t+1) = z_t / ||z_t||_1
```

The step size uses the largest singular value because directed top-k masking
makes `C_eff` asymmetric:

```text
alpha = 1 / sigma_max(C_eff)
```

L1 normalization keeps the optimization focused on the activity pattern rather
than its global mass. Before returning to the readout interface, the pattern is
scaled by the mean raw activity:

```text
y_out = y_final * mean(abs(a))
```

This scale was selected to keep the readout input in a trainable range without
restoring the excessive total mass of the raw activity.

## Sparse Lateral Operator

The dense neuron correlation matrix remains

```text
C_dense = W^T W
```

For each target neuron, the model retains incoming connections from its most
similar neurons. Selection is directed, so two neurons do not need to select
each other. The self diagonal is always retained and no row-sum compensation is
applied:

```text
C_eff = C_dense * M_directed_top_k
```

The mask is recomputed after every organize operation because neuron contexts
continue to change during training.

## Default Connection Schedule

Connection density is represented as a mapping from the first organize using a
stage to its retained-neighbor ratio:

```python
{
    0: 0.50,
    4: 0.40,
    5: 0.30,
    6: 0.20,
}
```

This means 50% through organize 3, followed by 40%, 30%, and then 20% from
organize 6 onward. The high initial density preserves early representation
formation; the final 20% density reduces winner concentration while retaining
classification performance.

## Stop Rule

The optimizer stops each input case when its active-neuron ratio reaches 8%.
Variance turning is not used in this profile. Under fixed-L1 normalization,
removing weak neurons redistributes the same mass over fewer active neurons and
can increase variance immediately, so the old variance interpretation no longer
identifies the transition between suppression and redistribution.

Complete numerical convergence is also undesirable: it is slower and drives
the representation toward excessive sparsity. The 8% support gate preserves a
multi-winner representation while keeping optimization inexpensive.

## Review Stability

Review is gated by the actual operator seen by the optimizer rather than by raw
neuron-context drift. For consecutive effective operators, the monitor computes

```text
drift_t = ||C_eff,t - C_eff,t-1||_F / max(||C_eff,t-1||_F, eps)
```

This captures both continuous correlation changes and discrete mask turnover.
The default hysteresis uses an enter threshold of 1%, an exit threshold of 2%,
and two consecutive stable organize operations before enabling review.

## Validated Default

The selected configuration is:

| Setting | Value |
|---|---:|
| Hidden neurons | 1,000 |
| Training cases | 60,000 |
| Test cases | 10,000 |
| Support threshold | 8% |
| Mask | Directed top-k plus self diagonal |
| Connection schedule | 50% -> 40% -> 30% -> 20% |
| Final connection ratio | 20% |
| Output scale | Mean raw activity |
| Row-sum compensation | Disabled |

The final scheduled run reached 95.97% accuracy on the complete 10,000-case
MNIST test set. The paired dense baseline was approximately 96.10%. Mean
optimizer cost was 10.39 iterations, with p95 of 12, maximum of 16, and no
unresolved cases, compared with the former fixed 300-iteration route. Final
mean active support was approximately 7.61%.

These results establish the selected implementation for the current DCNet
definition. They do not replace future multi-seed or cross-dataset evaluation.

## Implementation Map

- `models/optimizer.py`: fixed-L1 projection, support stop, singular-value step
  scale, and readout-interface rescaling.
- `models/discrimination.py`: profile selection, dense correlation, directed
  top-k mask, connection schedule, and effective-operator refresh.
- `models/biological_classifier.py`: resolved profile metadata and review
  monitoring of the effective operator.
- `core/weight_monitor.py`: relative-Frobenius stability monitor.
- `analysis/analysis_engine.py`: dense correlation, mask, and effective-operator
  visual diagnostics.
- `training/`: validated 1,000-neuron defaults and configurable `--hidden-dim`.
- `tests/test_optimizer_routes.py`: profile, schedule, mask, scale, stop, and
  monitor regression tests.

## Verification

Run the regression suite with:

```bash
python -m unittest discover -s tests -v
```

Historical development scripts, analysis utilities, and integrity hashes are
stored outside the source repository under:

```text
DCNet-Iterative-Optimizer-Sparsity-Study/
  reproduction-scripts/l1-sparse-optimizer-study-snapshot/
```
