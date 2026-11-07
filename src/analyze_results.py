"""
re-not-double-descent | analyze_results.py | ishan kunam
--------------------------------------------------------
compare the re-run against the digitized original figure 2
write a comparison table, then print error minima, differences, up-steps along P^ens & seed statistics
"""

from pathlib import Path

import pandas as pd

# -------------------
# input configuration
# -------------------

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA_DIR = ROOT / "data"  # digitized original figure
RESULTS_DIR = ROOT / "results"  # re-run outputs of run_fig2.py

ENS = [1, 2, 5, 10, 20, 50, 100]  # P^ens grid

# Probe points for the comparison table: (label, panel, series, P^leaf, P^ens)
CENTER_LEAVES = [5, 10, 20, 50, 75, 100]  # single-tree curve (center panel)
RIGHT_ENS = [2, 5, 10, 20, 50, 100]  # P^leaf = 500 curve (right panel)
CENTER_ENS = [5, 10, 50]  # center-panel ensembles, read at P^leaf = 100
RIGHT_LEAVES = [20, 50, 100]  # right-panel leaf caps, read at P^ens = 100

# Seed statistics: (label, P^leaf, P^ens)
SEED_PROBES = [
    ("P_leaf=2,P_ens=1", 2, 1),
    ("P_leaf=75,P_ens=1", 75, 1),
    ("P_leaf=500,P_ens=1", 500, 1),
    ("P_leaf=max,P_ens=1", "max", 1),
    ("P_leaf=500,P_ens=50", 500, 50),
    ("P_leaf=500,P_ens=100", 500, 100),
    ("P_leaf=max,P_ens=100", "max", 100),
]

# ------------
# read results
# ------------


def load_results(path: Path):
    d = pd.read_csv(path)
    # P^leaf holds both integers & "max" => compare as strings
    d["p_leaf"] = d.p_leaf.astype(str)
    return d


def get(d, L, k):
    # Test error at one (P^leaf, P^ens) point
    q = d[(d.p_leaf == str(L)) & (d.p_ens == k)]
    return float(q.test_err.iloc[0])


def og(orig, panel: str, series: str, L=None, k=None):
    # Digitized original error (figure 3 calibration) for one series & point
    q = orig[(orig.panel == panel) & (orig.series.str.startswith(series))]
    if L is not None:
        q = q[q.p_leaf.astype(str) == str(L)]
    if k is not None:
        q = q[q.p_ens == k]
    return float(q.err_fig3cal.iloc[0])


# ----------------
# comparison table
# ----------------


def add_row(rows, results, name: str, original: float, L, k):
    r = {"quantity": name, "original": original}
    for v, d in results.items():
        # The paper's "max" sits at a leaf cap of 500; also record full depth
        r[f"{v}_cap500"] = get(d, L, k)
        r[f"{v}_full"] = get(d, "max" if L == 500 else L, k)
    rows.append(r)


def comparison_table(results, orig):
    rows = []

    # Single-tree curve (P^ens = 1) as the leaf cap grows
    add_row(
        rows, results, "P_leaf=2, P_ens=1", og(orig, "center", "P_ens=1", 2, 1), 2, 1
    )
    for L in CENTER_LEAVES:
        add_row(
            rows,
            results,
            f"P_leaf={L}, P_ens=1",
            og(orig, "center", "P_ens=1", L, 1),
            L,
            1,
        )

    # Cap fixed at 500 as trees are added
    add_row(
        rows,
        results,
        "P_leaf=500 ('max'), P_ens=1",
        og(orig, "center", "P_ens=1", 500, 1),
        500,
        1,
    )
    for k in RIGHT_ENS:
        add_row(
            rows,
            results,
            f"P_leaf=500 ('max'), P_ens={k}",
            og(orig, "right", "P_leaf=500", 500, k),
            500,
            k,
        )

    # Remaining probe points on the center & right panels
    for k in CENTER_ENS:
        add_row(
            rows,
            results,
            f"center P_ens={k} at P_leaf=100",
            og(orig, "center", f"P_ens={k}", 100, k),
            100,
            k,
        )
    for L in RIGHT_LEAVES:
        add_row(
            rows,
            results,
            f"right P_leaf={L} at P_ens=100",
            og(orig, "right", f"P_leaf={L}", L, 100),
            L,
            100,
        )

    # Differences from the original (cap 500 only; full depth has no counterpart)
    T = pd.DataFrame(rows)
    for v in results:
        T[f"d_{v}_cap500"] = T[f"{v}_cap500"] - T.original
    return T
