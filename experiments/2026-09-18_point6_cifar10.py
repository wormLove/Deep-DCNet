"""
Prof. Yu's point #6 (more datasets): CIFAR-10 (RGB, 3x32x32 flattened to 3072-d, 10 classes, 50k/10k). Natural
images are a much harder target for one sparse Hebbian layer on raw pixels
than digits are; expect a large gap to MNIST. For the greyscale 1024-d
variant pass --dataset cifar10_gray to training/train_classifier.py
directly, or copy this script and change DATASET.

    python experiments/2026-09-18_point6_cifar10.py                                   # single_layer + readout, full
    python experiments/2026-09-18_point6_cifar10.py --head traditional_mlp            # point #5 head on this dataset
    python experiments/2026-09-18_point6_cifar10.py --arch stacked                    # 2-layer stack, review mode
    python experiments/2026-09-18_point6_cifar10.py --smoke                           # code-path check only

Download the data first on a machine with internet (HPC login node):
    python training/datasets.py download cifar10
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _point6 import parse_point6_args, run_point6

DATASET = "cifar10"

if __name__ == "__main__":
    run_point6(DATASET, parse_point6_args(DATASET))
