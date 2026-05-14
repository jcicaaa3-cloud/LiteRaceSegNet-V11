"""Small no-dataset smoke check for GitHub reviewers.

It verifies that LiteRaceSegNet can be imported, instantiated, and run on a tiny
CPU tensor. This does not require a dataset or checkpoint.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from seg.core.lightweight_race import LiteRaceSegNet, count_trainable_params


def main() -> None:
    torch.set_num_threads(1)
    model = LiteRaceSegNet(num_classes=2).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    shapes = {k: None if v is None else tuple(v.shape) for k, v in y.items()}
    params = count_trainable_params(model)
    print("LiteRaceSegNet smoke check OK")
    print(f"trainable_params={params:,}")
    print(f"outputs={shapes}")


if __name__ == "__main__":
    main()
