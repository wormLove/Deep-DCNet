# Iterative Activity Optimizer Development

## Purpose

This document records why the DCNet activity optimizer was changed, how the
final mechanism works, and which experiments support the change. It describes
the implementation merged in commits `b1cabf4` and `c711bf2`.

The main goal was to make sparse activity emerge from lateral interaction,
residual correction, and the non-negative activity constraint. The final design
does not use an L1 penalty, an output threshold, or a hard top-k sparsity cut.

## Starting Point

For one input batch, let:

- `x` be the raw neuron activity with shape `[batch, neurons]`;
- `C` be the neuron lateral-correlation matrix;
- `y_t` be the optimized activity at iteration `t`.

The core residual is

```text
e_t = x - y_t C
```

and the original iterative update combined this residual with a small L1
shrinkage term. After optimization, an additional standard-deviation threshold
and a hard top-5% cut selected the output activity.

This produced sparse outputs, but it mixed three different mechanisms:

1. lateral competition inside the iterative optimizer;
2. uniform per-iteration shrinkage;
3. a final externally imposed sparsity rule.

The hard cut guaranteed a requested support size, but the support was selected
outside the activity dynamics. The L1 term also suppressed every coordinate by
the same absolute amount, which could collectively remove weak activities
rather than strengthen competition between neuron contexts.

## Research Path

### 1. Sparsity feasibility without an added penalty

Long optimizer trajectories were first run without the L1 term or the final
threshold/cut. Both 300-neuron mechanism probes and 1,000-neuron validation
probes eventually reached strong sparsity using only the residual update and
ReLU projection.

This established the key feasibility result: the original lateral dynamics can
generate sparse activity naturally. The practical problem was convergence
speed, not the absence of a sparse solution.

### 2. Step-size scale and the two optimizer stages

The optimizer trajectory consistently showed two time scales:

- **Global suppression:** the total activity falls rapidly while neuron
  contexts are still broadly correlated.
- **Selective redistribution:** slower neuron-specific directions determine
  which neurons remain active, reactivate, or become winners.

For an approximate equal-correlation matrix,

```text
C = (1 - c) I + c 11^T
```

the global direction has eigenvalue

```text
lambda_global = 1 - c + cN
```

while the remaining directions have eigenvalue `1 - c`. This large condition
number explains why the global activity scale can settle quickly while
selective redistribution continues much longer.

Before ReLU changes the active support, the error relative to a linear solution
evolves as

```text
z_(t+1) = z_t (I - alpha C)
```

and a component along eigenvalue `lambda_k` is multiplied by

```text
r_k = 1 - alpha lambda_k
```

The historical step used

```text
alpha = 1 / (g lambda_max),  g = 10
```

which was deliberately conservative. Sweeps showed that `g=1` is the practical
safe point for the projected system and gives a tenfold larger step than the
historical setting. Values below one can make the first update overshoot the
non-negative region, causing widespread deactivation followed by broad
reactivation. Although the corresponding unprojected linear system has the
weaker theoretical stability condition `g > 0.5`, that condition does not
protect the ReLU support dynamics.

The maximum eigenvalue is now computed exactly with `torch.linalg.eigvalsh`
after each context organization. The previous power-iteration estimate was
removed because this computation is infrequent and estimation error directly
changes the effective step size.

### 3. Fixed iterations versus convergence-based soft stop

A fixed 300-iteration route was compared with a convergence-based soft stop.
The soft stop could produce lower nonzero activity and slightly stronger frozen
representations, but it frequently spent hundreds of extra iterations removing
one or two weak neurons after the effective activity pattern had stabilized.

This showed that a generic magnitude/drift tolerance was not a satisfactory
endpoint. It added several numerical thresholds and did not distinguish useful
redistribution from late over-optimization.

### 4. Dynamic step size

An activity-dependent controller varied `g` between one and two. It was
mechanistically interpretable, but most late updates used a gain close to two,
which reduced the step to roughly half of the fixed `g=1` step. It added
per-iteration control overhead and did not provide a consistent performance or
runtime advantage.

The dynamic controller was therefore not retained. The final optimizer uses a
fixed `g=1` step and makes only the stopping decision state-dependent.

### 5. Variance-dynamics stopping

Direct activity visualizations showed that late optimization often continued
after the active pattern was already stable. Weak activities kept disappearing
while stronger winners absorbed the redistributed activity. This motivated a
stop based on a change in the population dynamics rather than another absolute
drift tolerance.

For each input case, the optimizer tracks population variance

```text
v_t = Var_i(y_(t,i)).
```

The recent `2w` values are split into two windows of length `w=20`. A turning
point is recorded when

```text
slope(v_(t-2w+1) ... v_(t-w)) < 0
slope(v_(t-w+1)  ... v_t)     > 0.
```

The negative-to-positive transition indicates that broad suppression has given
way to increasing winner separation. A case stops only when this turning point
has been observed and its nonzero activity ratio is below the phase-specific
support limit.

### 6. Initial organization as a special phase

Before the first context organization, neuron contexts are highly similar
because they come from shared dataset-level initialization. Natural sparsity is
therefore much slower to form than it is after learned patterns appear.

Early-training interventions also showed that the first support distribution
changes potential accumulation and the first context update. A natural 5%
support target improved the first-cycle result without restoring a hard cut,
although it required more optimizer iterations.

The final design therefore uses a separate initial-phase budget and support
limit. This is a phase distinction, not a different update rule.

## Final Algorithm

The production update is

```text
y_0       = x
alpha     = 1 / lambda_max(C)
e_t       = x - y_t C
y_(t+1)   = ReLU(y_t + alpha e_t)
```

There is no penalty term after `e_t`, and there is no thresholding stage after
the iterative optimizer.

For each case in a batch:

1. update activity with the exact `g=1` step;
2. update the two-window variance history;
3. remember whether a negative-to-positive variance turn has occurred;
4. stop the case when the turn has occurred and its support satisfies the
   current phase limit;
5. freeze completed cases while unresolved cases continue;
6. stop the batch when all cases finish or the phase cap is reached.

### Production configuration

| Setting | Initial phase | Later phases |
|---|---:|---:|
| Step size | `1 / lambda_max(C)` | `1 / lambda_max(C)` |
| Variance window | 20 | 20 |
| Nonzero support limit | 5% | 8% |
| Iteration cap | 20,000 | 1,000 |
| L1 or competitive penalty | Disabled | Disabled |
| Output threshold | Disabled | Disabled |
| Hard top-k cut | Disabled | Disabled |

The discrimination layer selects the initial phase while `lr_ready` is false.
After the first organization and learning-rate refresh, later-phase settings are
used. The exact maximum eigenvalue is refreshed after context organization and
after loading a model checkpoint.

## Final Paired Validation

The final mechanism was compared with the legacy route in a native end-to-end
MNIST run. Both routes used:

- 1,000 discrimination neurons;
- 60,000 training cases in the same order;
- batch size 4;
- one organization every 200 samples;
- the same initial discrimination and readout states;
- review-enabled readout training;
- 10,000 test cases for the final evaluation.

The legacy route used Old L1, `g=10`, 1,000 fixed iterations, the output
standard-deviation threshold, and the hard top-5% cut.

| Metric | Legacy | Final | Final minus legacy |
|---|---:|---:|---:|
| Final test accuracy | 95.35% | 96.03% | +0.68 pp |
| Final nonzero activity | 2.17% | 4.58% | +2.40 pp |
| Mean training optimizer iterations | 1,000.00 | 460.60 | -53.94% |
| Training forward time | 3277.95 s | 2381.04 s | -27.36% |
| Total training wall time | 3363.64 s | 2479.30 s | -26.29% |
| Final evaluation time | 503.23 s | 360.72 s | -28.32% |
| Training variance-stop trigger rate | 0% | 97.06% | +97.06 pp |

The final route is less extremely sparse than the legacy hard-cut route, but it
remains sparse, improves final accuracy in this paired run, and substantially
reduces optimizer work. The result supports replacing the old mechanism; it
does not establish that the same numerical settings are optimal for every
dataset or neuron count.

## Removed Mechanisms

The following are intentionally absent from the production implementation:

- per-iteration Old L1 shrinkage;
- the experimental competitive-inhibition/error2 route;
- gain-factor and power-iteration configuration;
- the original multi-threshold soft stop;
- the dynamic `g` controller;
- output standard-deviation thresholding;
- hard top-k sparsity selection.

The competitive-inhibition experiments are not treated as evidence for the
final mechanism because an early implementation used changing activity where
the proposed design expected fixed raw activity. Correctly archived Old L1
results are retained only as historical controls.

## Implementation Map

- `models/optimizer.py`: projected residual update, exact eigenvalue scale,
  per-case variance turning detection, support gate, and fallback caps.
- `models/discrimination.py`: initial/later phase selection and direct ReLU
  output.
- `models/biological_classifier.py`: cached eigenvalue refresh after checkpoint
  loading.
- `training/train_classifier.py`: final classifier defaults.
- `training/train_discrimination.py`: final discrimination-only defaults.
- `tests/test_optimizer_routes.py`: exact-eigenvalue, phase-configuration, final
  profile, and numerical-regression tests.

## Interpretation Boundaries

1. Linear eigenmode analysis explains the trajectory only while the active
   support is unchanged. ReLU makes the full optimizer piecewise linear.
2. The 5% initial support target and 8% later target are biological constraints,
   not mathematical convergence proofs.
3. A variance turn is not guaranteed before the cap. The cap remains a required
   fallback, and diagnostic fields report unresolved cases.
4. Exact eigendecomposition increases organization cost, but in the paired run
   this was much smaller than the time saved by reducing iterative updates.
5. The final accuracy comparison is one controlled MNIST configuration. Further
   datasets and random seeds are needed before making a general performance
   claim.

## Verification

Run the optimizer regression tests with:

```bash
python -m unittest discover -s tests -v
```

The formal research artifacts are stored separately from the source repository
under `DCNet-Iterative-Optimizer-Sparsity-Study`. That archive contains the raw
traces, checkpoints, comparison summaries, figures, and audit notes for each
development stage.
