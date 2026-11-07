"""
re-not-double-descent | run_fig2.py | ishan kunam
-------------------------------------------------
re-run figure 2 of curth et al. (2023): random forests on mnist
fit one forest per leaf cap & score the first k trees for each ensemble size
test error is the squared error summed over the 10 classes
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Forest implementation behind the experiment
from sklearn.ensemble import RandomForestRegressor

# -------------------
# input configuration
# -------------------

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA_DIR = ROOT / "data" / "mnist"  # decompressed MNIST IDX files

LEAVES = [2, 5, 10, 20, 50, 75, 100, 500, 1000, 2000, None]  # P^leaf (None == no cap)
ENS = [1, 2, 5, 10, 20, 50, 100]  # P^ens

N_CLASSES = 10  # digits 0-9
N_TRAIN = 10000  # training images used per run
N_PER_CLASS = N_TRAIN // N_CLASSES  # images per class in the balanced subset
N_TREES = max(ENS)  # trees fitted per forest

# ---------------
# load mnist data
# ---------------


def read_idx(path: Path):
    # Read the raw bytes of the IDX file
    data = path.read_bytes()

    # Header: magic number (last byte == number of dimensions), then one size per dimension
    magic = struct.unpack(">I", data[:4])[0]
    ndim = magic & 0xFF
    dims = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])

    # Remaining bytes are the array itself (unsigned 8-bit)
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(data_dir: Path, subset: str = "first1000", subset_seed: int = 0):
    # Flatten each image to 784 features & scale pixels to [0, 1]
    Xtr = read_idx(data_dir / "train-images-idx3-ubyte").reshape(-1, 784)
    Xtr = Xtr.astype(np.float32) / 255.0
    ytr = read_idx(data_dir / "train-labels-idx1-ubyte").astype(int)
    Xte = read_idx(data_dir / "t10k-images-idx3-ubyte").reshape(-1, 784)
    Xte = Xte.astype(np.float32) / 255.0
    yte = read_idx(data_dir / "t10k-labels-idx1-ubyte").astype(int)

    # Pick the 10,000 training images
    if subset == "first1000":
        # First 1,000 images of every class, in dataset order
        idx = np.sort(
            np.concatenate(
                [np.where(ytr == c)[0][:N_PER_CLASS] for c in range(N_CLASSES)]
            )
        )
    else:
        # Uniform sample w/o replacement, ignoring class balance
        idx = np.sort(
            np.random.RandomState(subset_seed).choice(len(ytr), N_TRAIN, replace=False)
        )
    Xtr, ytr = Xtr[idx], ytr[idx]

    # One-hot labels => one regression target per class
    return Xtr, np.eye(N_CLASSES)[ytr], Xte, np.eye(N_CLASSES)[yte]


# --------------
# fit & evaluate
# --------------


def tree_preds(forest, X):
    # Stack each tree's own prediction => (n_trees, n_test[, n_out])
    return np.stack([t.predict(X) for t in forest.estimators_])


def run(variant: str, seed: int, Xtr, Ytr, Xte, Yte, n_jobs: int = -1, leaves=None):
    rows = []

    for L in leaves or LEAVES:
        t0 = time.time()

        # Settings shared by every forest in the paper's tree experiment
        kw = {
            "n_estimators": N_TREES,
            "max_features": "sqrt",
            "max_depth": None,
            "bootstrap": False,
            "max_leaf_nodes": L,
            "random_state": seed,
            "n_jobs": n_jobs,
        }

        if variant == "multi":
            # One multi-output forest predicts all 10 classes jointly
            rf = RandomForestRegressor(**kw).fit(Xtr, Ytr)
            P = tree_preds(rf, Xte)  # (100, n, 10)
            leaves = np.array([t.get_n_leaves() for t in rf.estimators_], float)
        else:
            # One-vs-all: ten independent forests, one per class
            Ps, lv = [], []
            for c in range(N_CLASSES):
                rf = RandomForestRegressor(**kw).fit(Xtr, Ytr[:, c])
                Ps.append(tree_preds(rf, Xte))
                lv.append([t.get_n_leaves() for t in rf.estimators_])
            P = np.stack(Ps, axis=-1)  # (100, n, 10)
            # Per-tree leaf count, averaged over the classes
            leaves = np.array(lv, float).mean(axis=0)

        # Running sum over trees => the first k trees' mean is cum[k - 1] / k
        cum = np.cumsum(P, axis=0)

        for k in ENS:
            pred = cum[k - 1] / k
            err = ((pred - Yte) ** 2).mean(axis=0).sum()
            ml = leaves[:k].mean()
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "p_leaf": ("max" if L is None else L),
                    "p_ens": k,
                    "test_err": err,
                    "mean_leaves_firstk": ml,
                    "mean_leaves_100": leaves.mean(),
                    "total_leaves": leaves.mean() * k,
                }
            )

        # rows[-len(ENS)] is the single-tree entry, rows[-1] the full ensemble
        print(
            f"{variant} seed={seed} P_leaf={L}: leaves={leaves.mean():.1f} "
            f"err1={rows[-len(ENS)]['test_err']:.6f} "
            f"err100={rows[-1]['test_err']:.6f} ({time.time() - t0:.1f}s)",
            flush=True,
        )

    return rows


# ----
# main
# ----


def parse_args():
    parser = argparse.ArgumentParser(description="re-run figure 2 (trees on MNIST)")
    parser.add_argument("--variant", required=True, choices=["ova", "multi"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--subset", default="first1000", choices=["first1000", "random"]
    )
    parser.add_argument(
        "--subset-seed-from-seed",
        action="store_true",
        help="redraw the random subset with each run's seed",
    )
    parser.add_argument(
        "--leaves",
        type=int,
        nargs="+",
        default=None,
        help="leaf caps to run instead of the full grid (e.g. 200 for figure 3)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    all_rows = []
    data = None
    t_start = time.time()

    for seed in args.seeds:
        # Reload only when the subset depends on the seed
        if data is None or args.subset_seed_from_seed:
            data = load(args.data_dir, args.subset, subset_seed=seed)

        t0 = time.time()
        all_rows += run(args.variant, seed, *data, leaves=args.leaves)
        print(f"== {args.variant} seed {seed} wall {time.time() - t0:.1f}s", flush=True)

    # Write one row per (seed, P^leaf, P^ens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"total wall {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
