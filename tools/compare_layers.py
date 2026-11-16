"""
Compare the discrimination layer(s) of two trained runs.

Loads each run's checkpoints/final.pth and prints, per discrimination layer
and per tensor in its state_dict (neuron_weights, correlation matrix, the
organizer's potentials and learning rates, the stats buffers, ...), the
largest absolute difference (for boolean buffers: the number of entries
that differ). Heads are ignored.

Main use - Plan 03 check 1: a freeze run with --val-size 0 must learn
exactly the same discrimination layer as the plain run with the readout
head (same seed, same images in the same order), so neuron_weights should
differ by 0 (at most ~1e-6).

    python tools/compare_layers.py RESULT/single_layer_baseline_s33_smoke \\
        RESULT/freeze_mnist_single_layer_readout_val0_s33_smoke

Each argument may be a run folder (RESULT/<run>/<timestamp>), a run-name
folder (RESULT/<run>; its newest timestamp with a final.pth is used) or a
.pth file. Exit code 0 when neuron_weights of every layer match within
--tol, 1 otherwise. CPU only; run it on the login node.
"""
import argparse
import sys
from pathlib import Path

import torch


def resolve_checkpoint(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    if (p / "checkpoints" / "final.pth").is_file():
        return p / "checkpoints" / "final.pth"
    if p.is_dir():
        candidates = sorted(d for d in p.iterdir() if (d / "checkpoints" / "final.pth").is_file())
        if candidates:
            return candidates[-1] / "checkpoints" / "final.pth"
    raise SystemExit(f"no checkpoints/final.pth found for {arg}")


def discrimination_states(path: Path):
    state = torch.load(str(path), map_location="cpu")
    if "discrimination_layers" in state:      # StackedBiologicalClassifier.save_model
        return list(state["discrimination_layers"])
    if "discrimination_layer" in state:       # BiologicalClassifier.save_model
        return [state["discrimination_layer"]]
    raise SystemExit(f"{path}: not a DCNet model checkpoint (keys: {sorted(state)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--tol", type=float, default=1e-6, help="max |delta| of neuron_weights counted as identical")
    args = parser.parse_args()

    path_a, path_b = resolve_checkpoint(args.run_a), resolve_checkpoint(args.run_b)
    layers_a, layers_b = discrimination_states(path_a), discrimination_states(path_b)
    print(f"A: {path_a}\nB: {path_b}\n")
    if len(layers_a) != len(layers_b):
        print(f"DIFFERENT: A has {len(layers_a)} discrimination layer(s), B has {len(layers_b)}")
        sys.exit(1)

    ok = True
    for i, (sa, sb) in enumerate(zip(layers_a, layers_b)):
        print(f"Layer {i}")
        print(f"  {'tensor':<40} {'shape':<16} {'max |delta|':>14}")
        for key in sorted(set(sa) | set(sb)):
            if key not in sa or key not in sb:
                print(f"  {key:<40} {'only in ' + ('A' if key in sa else 'B'):<16}")
                continue
            ta, tb = sa[key], sb[key]
            if ta.shape != tb.shape:
                print(f"  {key:<40} {str(tuple(ta.shape)) + ' vs ' + str(tuple(tb.shape)):<16} shape differs")
                if key == "neuron_weights":
                    ok = False
                continue
            if ta.numel() == 0:
                delta = 0.0
            elif ta.dtype == torch.bool:
                delta = float((ta != tb).sum().item())   # number of differing entries
            else:
                delta = (ta.to(torch.float64) - tb.to(torch.float64)).abs().max().item()
            marker = "  <-- weights" if key == "neuron_weights" else ""
            print(f"  {key:<40} {str(tuple(ta.shape)):<16} {delta:>14.3e}{marker}")
            if key == "neuron_weights" and not delta <= args.tol:
                ok = False
        print()

    if ok:
        print(f"IDENTICAL: neuron_weights of every layer match within {args.tol:g}")
        sys.exit(0)
    print(f"DIFFERENT: neuron_weights differ by more than {args.tol:g} - stop and report")
    sys.exit(1)


if __name__ == "__main__":
    main()
