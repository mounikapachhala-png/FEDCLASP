"""Shared paths, seeds and helpers. Must be IDENTICAL on both PCs."""
import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PART = ROOT / "data" / "partitions"
RES = ROOT / "results"
FIG = RES / "figures"
for _p in (RAW, PROC, PART, RES, FIG):
    _p.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = 42              # data split + client partition. NEVER change after Step 3.
TRAIN_SEEDS = [0, 1, 2, 3, 4]  # model-training seeds (Step 6 onwards)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def imbalance_ratio(counts: dict) -> float:
    vals = [v for v in counts.values() if v > 0]
    return round(max(vals) / min(vals), 2)
