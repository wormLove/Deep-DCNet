#!/usr/bin/env python3
"""
analysis/plot_training_curve.py

Parse a train_log.txt produced by train_classifier.py and produce a
two-panel figure:

  Top:    Test accuracy vs. training samples — marks the plateau
  Bottom: Discrimination-layer weight shift vs. samples (log scale)
          — marks where reductions become incremental

Usage:
    # auto-finds the most recently modified train_log.txt
    python analysis/plot_training_curve.py

    # explicit path
    python analysis/plot_training_curve.py --log path/to/train_log.txt

    # tune the detection thresholds
    python analysis/plot_training_curve.py --plateau-gap 1.0 --ws-threshold 0.10
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches both [organize N] and [final organize] lines
_ORG_RE = re.compile(
    r"\[(?:organize \d+|final organize)\] "
    r"step=\d+, samples=(\d+), loss=[\d.]+, "
    r"weight_shift=([\d.]+)"
    r"(?:, acc=([\d.]+)%)?"
)
_EVAL0_RE = re.compile(r"\[eval step 0\] acc=([\d.]+)%")


def parse_log(path: Path):
    """
    Returns four arrays:
        ws_samples   — sample count at each organize cycle
        ws_values    — weight shift at each organize cycle
        acc_samples  — sample count at each eval point (including step 0)
        acc_values   — test accuracy (%) at each eval point
    """
    text = path.read_text()

    ws_samples, ws_values = [], []
    acc_samples, acc_values = [], []

    m0 = _EVAL0_RE.search(text)
    if m0:
        acc_samples.append(0)
        acc_values.append(float(m0.group(1)))

    for m in _ORG_RE.finditer(text):
        s = int(m.group(1))
        ws_samples.append(s)
        ws_values.append(float(m.group(2)))
        if m.group(3) is not None:
            acc_samples.append(s)
            acc_values.append(float(m.group(3)))

    return (
        np.array(ws_samples, dtype=float),
        np.array(ws_values, dtype=float),
        np.array(acc_samples, dtype=float),
        np.array(acc_values, dtype=float),
    )


# ---------------------------------------------------------------------------
# Plateau / convergence detection
# ---------------------------------------------------------------------------

def find_accuracy_plateau(acc_samples, acc_values, gap_pp: float = 1.0):
    """
    Returns the first sample count where the model is within `gap_pp`
    percentage points of its final accuracy.

    This is more robust than a derivative-based approach given the small
    number of evaluation points and the oscillations late in training.
    """
    final = acc_values[-1]
    threshold = final - gap_pp
    for s, a in zip(acc_samples[1:], acc_values[1:]):  # skip random-init point
        if a >= threshold:
            return float(s)
    return float(acc_samples[-1])


def find_ws_incremental(ws_samples, ws_values, rel_threshold: float = 0.10):
    """
    Returns the first sample count where the relative weight-shift reduction
    between consecutive organize cycles drops below `rel_threshold`.

    Relative reduction = (ws[i-1] - ws[i]) / ws[i-1]
    """
    for i in range(1, len(ws_values)):
        rel = (ws_values[i - 1] - ws_values[i]) / (ws_values[i - 1] + 1e-12)
        if rel < rel_threshold:
            return float(ws_samples[i])
    return float(ws_samples[-1])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    ws_samples, ws_values,
    acc_samples, acc_values,
    out_path: Path,
    gap_pp: float = 1.0,
    ws_threshold: float = 0.10,
):
    plateau_s = find_accuracy_plateau(acc_samples, acc_values, gap_pp)
    ws_incr_s = find_ws_incremental(ws_samples, ws_values, ws_threshold)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"hspace": 0.38},
    )
    fig.suptitle("Deep-DCNet Training Dynamics", fontsize=14, fontweight="bold")

    # ── Panel 1: Accuracy ────────────────────────────────────────────────────
    ax1.plot(
        acc_samples / 1_000, acc_values,
        "o-", color="#1565C0", linewidth=2, markersize=6,
        label="Test accuracy",
    )
    ax1.axvline(
        plateau_s / 1_000, color="#C62828", linestyle="--", linewidth=1.8,
        label=f"Plateau  ≈ {int(plateau_s):,} samples\n"
              f"(first within {gap_pp:.1f} pp of final)",
    )
    ax1.axhline(acc_values[-1], color="grey", linestyle=":", linewidth=1, alpha=0.5)

    # annotate final accuracy
    ax1.annotate(
        f"Final: {acc_values[-1]:.2f}%",
        xy=(acc_samples[-1] / 1_000, acc_values[-1]),
        xytext=(-70, -18), textcoords="offset points",
        fontsize=9, color="#1565C0",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2),
    )

    ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax1.set_xlabel("Training samples seen (×1,000)", fontsize=11)
    ax1.set_ylim(0, 104)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Weight Shift ────────────────────────────────────────────────
    ax2.semilogy(
        ws_samples / 1_000, ws_values,
        "o-", color="#2E7D32", linewidth=2, markersize=3,
        label="Weight shift (log scale)",
    )
    ax2.axvline(
        ws_incr_s / 1_000, color="#E65100", linestyle="--", linewidth=1.8,
        label=f"Incremental drop ≈ {int(ws_incr_s):,} samples\n"
              f"(relative reduction < {ws_threshold*100:.0f}% per cycle)",
    )

    ax2.set_ylabel("Weight Shift (log scale)", fontsize=11)
    ax2.set_xlabel("Training samples seen (×1,000)", fontsize=11)
    ax2.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved → {out_path}")
    print(f"  Accuracy plateau        ≈ {int(plateau_s):,} samples "
          f"(first within {gap_pp} pp of {acc_values[-1]:.2f}%)")
    print(f"  Weight-shift incremental ≈ {int(ws_incr_s):,} samples "
          f"(rel. reduction < {ws_threshold*100:.0f}% per cycle)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def find_latest_log() -> Path:
    candidates = []
    for base in [Path("RESULT"), Path("gpu_rebuild_classifier")]:
        if base.exists():
            candidates.extend(base.rglob("train_log.txt"))
    if not candidates:
        raise FileNotFoundError(
            "No train_log.txt found automatically. Use --log to specify one."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy and weight-shift curves from train_log.txt"
    )
    parser.add_argument(
        "--log", type=Path, default=None,
        help="Path to train_log.txt (default: most recently modified one).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path (default: training_curve.png next to the log).",
    )
    parser.add_argument(
        "--plateau-gap", type=float, default=1.0,
        help="Accuracy gap (pp) below final used to mark the plateau (default: 1.0).",
    )
    parser.add_argument(
        "--ws-threshold", type=float, default=0.10,
        help="Relative weight-shift reduction below which drops are 'incremental' "
             "(default: 0.10 = 10%%).",
    )
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    out_path = args.out or (log_path.parent / "training_curve.png")

    print(f"Parsing: {log_path}")
    ws_s, ws_v, acc_s, acc_v = parse_log(log_path)
    plot(ws_s, ws_v, acc_s, acc_v, out_path,
         gap_pp=args.plateau_gap, ws_threshold=args.ws_threshold)


if __name__ == "__main__":
    main()
