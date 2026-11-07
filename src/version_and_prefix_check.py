"""
re-not-double-descent | version_and_prefix_check.py | ishan kunam
-----------------------------------------------------------------
sanity checks on the forest experiment, using the loader from run_fig2.py
version: errors of capped forests (run it under different scikit-learn versions)
prefix: first k trees of a big forest vs a fresh k-tree forest
"""

import time

import numpy as np

# Forest implementation behind the experiment
from sklearn.ensemble import RandomForestRegressor

# -------------------
# input configuration
# -------------------

SEED = 42  # random_state for every forest
N_TREES = 100  # trees in the big forest

VERSION_LEAVES_OVA = [50, 75, 100, 200, 500, None]  # includes the out-of-grid cap 200
VERSION_LEAVES_MULTI = [500, 2000, None]
PREFIX_LEAVES = [100, None]
PREFIX_TREES = 5  # size of the fresh forest in the prefix check
PREFIX_CLASS = 3  # class used for the one-vs-all prefix check
ENS_CHECKED = (1, 50, 100)  # P^ens values reported

# -----------
# fit forests
# -----------


def rf(L, n: int, X, y):
    # Same settings as the main experiment, w/ n trees and a leaf cap of L
    return RandomForestRegressor(
        n_estimators=n,
        max_features="sqrt",
        bootstrap=False,
        max_leaf_nodes=L,
        random_state=SEED,
        n_jobs=-1,
    ).fit(X, y)


def err_ova(L, data, ks=ENS_CHECKED):
    Xtr, Ytr, Xte, Yte = data

    # Ten independent forests, one per class => (100, n, 10) predictions
    P = []
    for c in range(Ytr.shape[1]):
        f = rf(L, N_TREES, Xtr, Ytr[:, c])
        P.append(np.stack([t.predict(Xte) for t in f.estimators_]))
    cum = np.cumsum(np.stack(P, -1), 0)

    # Squared error of the first k trees' mean, summed over classes
    return {k: ((cum[k - 1] / k - Yte) ** 2).mean(0).sum() for k in ks}


def err_multi(L, data, ks=ENS_CHECKED):
    Xtr, Ytr, Xte, Yte = data

    # One multi-output forest
    f = rf(L, N_TREES, Xtr, Ytr)
    cum = np.cumsum(np.stack([t.predict(Xte) for t in f.estimators_]), 0)

    return {k: ((cum[k - 1] / k - Yte) ** 2).mean(0).sum() for k in ks}


# ------
# checks
# ------


def version_check(data):
    t0 = time.time()

    for L in VERSION_LEAVES_OVA:
        print(
            "OVA  ",
            L,
            {k: round(v, 6) for k, v in err_ova(L, data).items()},
            flush=True,
        )
    for L in VERSION_LEAVES_MULTI:
        print(
            "MULTI",
            L,
            {k: round(v, 6) for k, v in err_multi(L, data).items()},
            flush=True,
        )

    print(f"version block {time.time() - t0:.1f}s")
