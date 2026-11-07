"""Shared paths and a dataset-only adapter for the untouched author repository."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "not-double-descent"
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "fashion-mnist"
FIGURES = ROOT / "output" / "fashion-mnist"
COMMIT = "76c6ec5d6181380baa4554378bf87d32abaf6606"


def register_upstream():
    # The authors committed .pyc files; do not mutate the reference checkout.
    sys.dont_write_bytecode = True
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


def load_datasets():
    from torchvision.datasets import FashionMNIST
    return (FashionMNIST(str(DATA), train=True, download=False),
            FashionMNIST(str(DATA), train=False, download=False))


def load_data(pred_idx="all", center=False):
    """Use authors' exact selection, division, flattening, and OHE operations.

    The ONLY data-source change is MNIST -> FashionMNIST. In particular this
    preserves their deterministic first-1000-per-class selection and ordering.
    """
    register_upstream()
    from src import data_utils
    train, test = load_datasets()
    original = data_utils.get_raw_data

    def get_raw_data(dataset, split):
        if dataset == "Fashion-MNIST":
            return train if split == "train" else test
        return original(dataset, split)

    data_utils.get_raw_data = get_raw_data
    try:
        return data_utils.get_data("Fashion-MNIST", n_samples=10000,
                                   pred_idx=pred_idx, center=center)
    finally:
        data_utils.get_raw_data = original


def load_raw_tensors(device="cpu"):
    """Authors' process_data output, retaining integer class identities."""
    import torch
    train, test = load_datasets()
    indices = torch.cat([torch.where(train.targets == i)[0][:1000]
                         for i in range(10)])
    xtr = (train.data[indices] / 255).reshape(10000, -1).to(device)
    xte = (test.data / 255).reshape(10000, -1).to(device)
    ytr = train.targets[indices].to(torch.float64).to(device)
    yte = test.targets.to(torch.float64).to(device)
    return xtr, ytr, xte, yte


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def write_manifest():
    import numpy as np
    import torch
    import platform
    train, test = load_datasets()
    idx = torch.cat([torch.where(train.targets == i)[0][:1000]
                     for i in range(10)]).numpy()
    versions = {}
    for package in ["numpy", "torch", "torchvision", "scikit-learn", "scipy",
                    "matplotlib", "pandas"]:
        versions[package] = importlib.metadata.version(package)
    info = {
        "upstream_url": "https://github.com/alanjeffares/not-double-descent",
        "upstream_commit": COMMIT,
        "dataset": "Fashion-MNIST",
        "dataset_url": "https://github.com/zalandoresearch/fashion-mnist",
        "n_train": len(idx), "n_test": len(test.targets), "n_features": 784,
        "sampling": "First 1000 original training indices per class; concatenate classes 0..9, identical to authors' balance_data_set",
        "train_class_counts": np.bincount(train.targets[idx].numpy()).tolist(),
        "test_class_counts": np.bincount(test.targets.numpy()).tolist(),
        "train_indices_sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
        "preprocessing": "uint8 / 255 in torch float32, flatten 28x28 -> 784; no resizing, augmentation or additional scaling",
        "tree_boosting_seed": 42, "rff_seed": 0,
        "versions": versions, "python": platform.python_version(),
        "platform": platform.platform(),
        "version_limit": "Authors pin NumPy/PyTorch/torchvision; sklearn/scipy/matplotlib/pandas are not pinned in their requirements. These supplemental versions are recorded, not claimed identical to an unpublished environment.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS / "training_indices.npy", idx)
    atomic_json(RESULTS / "manifest.json", info)
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    write_manifest()
