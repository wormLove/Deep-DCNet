# Initialization Pipeline Audit

The discrimination layer's weights are set once, before any learning, by three
operations in sequence. This document records what each one was audited for, what
was measured, and what the code now does as a result.

```
DatasetInitializerWhole                        DiscriminationLayer.__init__
├─ PCA basis        V_n                        └─ F.normalize(p=weight_norm_p, dim=0)
├─ Mid Matrix       diag(Sn^-1)      <- study 02   <- study 03
└─ non-negative transform            <- study 01
```

Each study held everything else fixed and varied one operation. All runs share
hidden_dim 1000, 60000 MNIST training images, batch 4, one organize every 200
images, 300 organizes, and a 10000-image final evaluation. Full measurements,
including what the data does *not* support, are in
`RESULT/Deep-DCNet-Init-Pipeline-Audit/0{1,2,3}-*/RESULTS.md`.

---

## The conclusion, and where it lives in the code

| operation | audited alternatives | chosen | where |
|---|---|---|---|
| non-negative transform | `shift`, `relu`, `abs` | **`abs`** | `core/initializers.py`, `non_negative_strategy="abs"` |
| Mid Matrix | applied, omitted | **omitted** | `core/initializers.py`, `mid_matrix=False` |
| weight normalization | L1, L2 | **L2** | `models/discrimination.py`, `weight_norm_p=2` |

All three alternatives remain selectable. Two of the three choices are preferences
among configurations that all work; the third is not, and the code says so.

---

## Study 01 — the non-negative transform

The PCA initializer produces signed weights, but the layer requires non-negative
ones. Three ways to get there were trained end to end.

```
                 final accuracy   wall seconds   organize-0 median iterations
shift                   96.03%           2425                           6916
relu                    96.17%           2544                          20000
abs                     96.05%           2339                            416
```

`relu` hit the 20000-iteration ceiling on 29 of the 50 organize-0 batches. Accuracy
separates nothing — the three sit inside 0.14 points, below the noise of a
10000-image test — so the choice was made on cost and on structure.

What the three do to the lateral matrix differs sharply. Mean off-diagonal cosine
between neurons at initialization:

```
shift   0.9747      relu   0.3163      abs   0.6367
```

`shift` subtracts the global minimum, which preserves every pairwise difference
exactly — `center(shift(W))` equals `center(W)` to 5.9e-15 — but introduces a shared
constant that then carries **97.69%** of each column's energy. The individual
component is 2.31% of what the neuron is. `relu` discards the negative half
outright, leaving 53% of entries at zero. `abs` reflects negatives onto their
magnitude, which is the least principled of the three on paper: it maps a strong
negative correlation to a strong positive one. It was chosen anyway, because it was
measured fastest to organize 0 and best in the first epoch, and because the
structural objection to it did not show up in any measured quantity.

## Study 02 — the Mid Matrix

`diag(Sn^-1)`, where `Sn` is the explained-variance ratios, sits between the PCA
basis and the randomizer. The design intent recorded in the project's slides is
that it equalises each component's contribution. It does not: contribution goes as
`1/Sn`, so the weakest components dominate; uniform contribution would need
`Sn^-1/2`. Measured as the centre of mass over PC index, out of 153 components:

```
none   23.4        sqrt   77.3 (uniform would be 77)        inv   111.9
```

Removing it changed accuracy for none of the three transforms, and changed
organize-0 cost for two of them:

```
            accuracy          organize-0 median iterations
         inv      none            inv            none
shift  96.03%   96.07%           6916            1220      5.7x faster
relu   96.17%   96.10%          20000            1166       17x faster
abs    96.05%   96.14%            416             503      1.2x slower
```

`abs` is the only one that does not benefit — it was never slowed by the Mid Matrix
in the first place, which is why the knob looks inert from inside the chosen
configuration. This is the reason `mid_matrix` is still a parameter: reproducing
study 01's `shift` and `relu` runs requires turning it back on.

## Study 03 — L1 against L2 weight normalization

L2 fixes `||w||_2 = 1` per column. L1 fixes `||w||_1 = 1`, which is what a fixed
synaptic resource budget literally means for non-negative weights. The two differ
by a positive per-column scalar `s_j = ||w_j||_1` evaluated on the unit-L2 column —
about 17.4 for every neuron, because `s_j^2` is exactly the participation ratio and
every neuron effectively reads about 300 of its 784 pixels. Every receptive-field
pattern is identical under the two; every pairwise cosine is identical to 1.3e-06.

What differs is `W_lateral = W^T W`. L2 pins every diagonal entry at 1. L1 leaves
`W_lateral_jj = 1/s_j^2` free, and it spreads from 1.26x at initialization to 3.12x
by organize 300.

### A latent defect the audit surfaced

The activity optimizer started at `y0 = a`. The iteration

```
y <- relu( y + lr * (a - y @ W_lateral) )        lr = 1 / lambda_max
```

is exactly equivariant under `y -> s*y, W_lateral -> W_lateral/s^2, a -> a/s,
lr -> lr*s^2`, which requires the starting point to scale as `s*y0`. `y0 = a`
supplies `a/s` instead, missing by `s^2`. L2 hides this by fixing `s = 1`; under L1
the iteration starts about 300x below its own solution, the first step moves only
0.53 of `y0` instead of 1.00, it never crosses zero, and relu — the only thing in
the loop that creates sparsity — never fires at all.

`y0_j = a_j / W_lateral_jj` is the per-neuron least-squares solution ignoring
lateral interaction, and it restores the equivariance exactly. Under L2 it is the
identity. Measured on 80 cases at organize 0, with the production turn detector:

```
                     turn fired   median iterations   at the 20000 cap
L2   y0 = a               80/80                 115               0/80
L1   y0 = a               12/80               18902              68/80
L1   y0 = a / diag        80/80                 367               0/80
```

This is why `optimizer_y0_divide_by_diagonal` exists, and why it is not an
independent choice: `DiscriminationLayer` forces it on whenever `weight_norm_p == 1`,
because the other combination is a configuration that cannot stop.

### Why L1 remains experimental

With the starting point fixed, L1 trains to completion without numerical trouble —
no dead columns, no divergence, loss 1.78 to 0.166 — and reaches **94.83%** against
L2's 96.14%, at 2.17x the wall time. It is also *ahead* of L2 for the first 4000
images, by up to 3.4 points.

The disqualifying measurement is not the accuracy. It is this:

```
                          L2       L1
variance_trigger_rate   0.9532   0.2396
optimizer_iteration_mean  441.7    958.7
```

Three quarters of L1's batches end by exhausting `optimizer_max_iters`, not by
satisfying the stop criterion. The cap exists to stop a pathological case running
forever; a configuration in which it becomes the ordinary way the loop ends is not
appropriate as the production default, whatever the accuracy says. The failures
over-suppress rather than under-suppress — the 33 of 200 cases that never fire the
turn at organize 10 are all already **below** the sparsity target when the budget
runs out, median 3.4% against a target of 8%, one case down to 1.0%.

The mechanism is visible in what L1 learns. `W_lateral_jj` is the coefficient on a
neuron's suppression of itself, and under L1 it is set almost entirely by how
concentrated the receptive field is — correlation with the participation ratio
between -0.96 and -0.999, against -0.05 under L2. A concentrated column also wins
more activation, because a fixed `||w||_1` spent on few bright pixels beats the same
budget spread thin. Strong activation and strong self-suppression therefore land on
the same neurons, a coupling that L2 structurally cannot have. Classifying all 1000
learned columns against MNIST:

```
digit      0    1    2    3    4    5    6    7    8    9
L2       154   48  134  133   71   75  100   68  126   91
L1        81  123   32  115   63   67  110   89  133  187
```

Digits 1/7/9 — the most concentrated shapes — go from 20.7% of the population to
39.9%. Digit 2 falls to 24% of its L2 count. L1 also learns purer templates: 79.8%
of its columns match a single digit unanimously against L2's 69.3%, and ambiguous
columns that serve several digits are less than half as common. L1 selects against
the shared, blurred templates that L2 keeps.

L1 remains selectable as an experimental route so that run can be reproduced and so
the open questions can be picked up. Its raw `y0 = a` variant is excluded because it
does not recover the required two-stage dynamics; L1 therefore always uses
`y0 = a / diag(W_lateral)`. Two questions were left on the table: replacing the
variance turning point with the sign of the error on the active set, which is what
the turn is a proxy for and is scale free; and a per-neuron step size, which would
remove the remaining asymmetry between a varying `W_lateral_jj` and a single global
`lr`.

---

## What the audit did not cover

- The per-sample activity normalization in `core/organizer.py:_normalize`. Scope B
  of study 03, not started.
- `RandomInitializer`. It takes the same `non_negative_strategy` parameter and the
  same new default, but no study exercised that path.
- A clean accuracy comparison for L1. The 94.83% was produced by a run whose stop
  criterion was failing throughout, so normalization and a broken criterion are
  confounded in that number.

## Reproducing

Results, figures and the scripts that produced them are archived outside the
repository, under `RESULT/Deep-DCNet-Init-Pipeline-Audit/`:

```
00-static-grid/            the 3 Mid x 5 transform grid, no training
01-non-negative-transform/ RESULTS.md, 3 runs, 37 figures
02-mid-matrix/             RESULTS.md, 3 runs
03-normalization/          RESULTS.md, 1 run, 15 figures and 3808 frames
baseline-shift-reference/  provenance for the archived baseline
reproduction-scripts/      the audit's own experiment and analysis scripts
```

Each run directory holds `training_complete.json` (config, timings, stop criterion,
and the final weights' sha256), the per-batch and per-organize CSVs, a 100-point
accuracy curve, the final 10000-image evaluation, and 101 checkpoints — every
organize from 1 to 50, then every fifth to 300.

Note on reproducibility: `torch.pca_lowrank` is a randomized SVD and is bit-exact
only within a process. Two processes with the same seed differ by about 3.4e-4
relative, so a reproduction claim must cite the archived package and its sha256,
never the seed.
