"""
Prof. Yu's point #6 (more datasets): Fashion-MNIST: same shape as MNIST (1x28x28, 10 classes, 60k/10k), harder
content. Exact drop-in - the only thing that changes vs the MNIST baselines
is the data, which makes it the cleanest first test of whether the DCNet
numbers generalize beyond digits.

    python experiments/2026-09-18_point6_fashion_mnist.py                                   # single_layer + readout, full
    python experiments/2026-09-18_point6_fashion_mnist.py --head traditional_mlp            # point #5 head on this dataset
    python experiments/2026-09-18_point6_fashion_mnist.py --arch stacked                    # 2-layer stack, review mode
    python experiments/2026-09-18_point6_fashion_mnist.py --smoke                           # code-path check only

Download the data first on a machine with internet (HPC login node):
    python training/datasets.py download fashion_mnist
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _point6 import parse_point6_args, run_point6

DATASET = "fashion_mnist"

if __name__ == "__main__":
    run_point6(DATASET, parse_point6_args(DATASET))
