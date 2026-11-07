"""
re-not-double-descent | plot_figures.py | ishan kunam
-----------------------------------------------------
draw the re-run of figure 2 (three panels each)
overlay: one-vs-all curves w/ the digitized original vertices on top
multi: multi-output forest
seeds: one-vs-all seed 42 line w/ a mean +/- 1 sd band over seeds 0-3
"""

from pathlib import Path

# Plotting (Agg backend => no display needed)
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")

# -------------------
# input configuration
# -------------------

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA_DIR = ROOT / "data"  # digitized original figure
RESULTS_DIR = ROOT / "results"  # re-run outputs of run_fig2.py
FIGURES_DIR = ROOT / "figures"  # where the png files are written

ENS = [1, 2, 5, 10, 20, 50, 100]  # P^ens grid
CENTER_LEAVES = [
    2,
    5,
    10,
    20,
    50,
    75,
    100,
    500,
]  # P^leaf grid (500 == the paper's "max")

# Figure 2's y-axis under the figure 3 calibration (see digitize_fig2.py)
YLIM = (0.0994, 0.9006)  # frame of the original figure
YT = [
    0.15,
    0.3833,
    0.6167,
    0.85,
]  # tick positions (printed 0.1, 0.4, 0.6, 0.8 in the paper)

BLUE, RED = "#0459ED", "#ED1C24"
ENS_COLS = {1: "#0259ED", 5: "#3B8BC2", 10: "#7FB9DA", 50: "#C6DBEF"}  # center panel
LEAF_COLS = {
    20: "#FCBBA1",
    50: "#FB7A5A",
    100: "#E7332A",
    500: "#ED1C24",
}  # right panel

# ----------------
# read digitized x
# ----------------


def left_panel_x(orig):
    # The left panel's x-axis (leaves, then trees) is drawn in the original's own coordinates
    left = orig[orig.panel == "left"]
    blue_x = left[left.series.str.startswith("P_ens=1")].x_px
    red_x = left[left.series.str.startswith("P_leaf=max")].x_px

    blue = dict(zip([2, 5, 10, 20, 50, 75, 100, "max"], blue_x))
    red = dict(zip(ENS, red_x))
    return blue, red


def load_results(path: Path):
    d = pd.read_csv(path)
    # P^leaf holds both integers & "max" => compare as strings
    d["p_leaf"] = d.p_leaf.astype(str)
    return d


def get(d, L, k):
    q = d[(d.p_leaf == str(L)) & (d.p_ens == k)]
    return float(q.test_err.iloc[0])


# ------------------
# three-panel figure
# ------------------


def three_panel(axs, d, maxkey: str, left_x, alpha: float = 1.0):
    a0, a1, a2 = axs
    blue_x, red_x = left_x

    # Left: leaves grow w/ one tree (blue), then trees grow at the cap (red)
    bl = [2, 5, 10, 20, 50, 75, 100]
    xb = [blue_x[L] for L in bl] + [blue_x["max"]]
    yb = [get(d, L, 1) for L in bl] + [get(d, maxkey, 1)]
    xr = [red_x[k] for k in ENS]
    yr = [get(d, maxkey, k) for k in ENS]
    a0.plot(xb, yb, color=BLUE, lw=1.6, alpha=alpha)
    a0.plot(xr, yr, color=RED, lw=1.6, alpha=alpha)
    a0.axvline(blue_x["max"], color="#0C5DA5", ls="--", lw=0.6)
    a0.set_xticks([159.2, 187.7, 211.6])
    a0.set_xticklabels(["200×1", "max×1", "max×50"], fontsize=7)
    a0.set_xlim(135.5, 240.1)
    a0.set_xlabel(r"$P^{leaf}\times P^{ens}$", fontsize=8)
    a0.set_ylabel("Squared test error", fontsize=8)
    a0.set_title("Double Descent in Trees", fontsize=8)

    # Center: error vs leaf cap for fixed ensemble sizes
    for k, c in ENS_COLS.items():
        y = [get(d, (maxkey if L == 500 else L), k) for L in CENTER_LEAVES]
        a1.plot(
            CENTER_LEAVES,
            y,
            color=c,
            lw=1.6,
            ls="-" if k == 1 else "--",
            label=f"$P^{{ens}}$={k}",
            alpha=alpha,
        )
    a1.set_xlim(-11, 537)
    a1.set_xticks(range(0, 501, 100))
    a1.set_xlabel(r"$P^{leaf}$", fontsize=8)
    a1.set_title(r"Error by $P^{leaf}$, $P^{ens}$ fixed", fontsize=8)

    # Right: error vs ensemble size for fixed leaf caps
    for L, c in sorted(LEAF_COLS.items(), reverse=True):
        y = [get(d, (maxkey if L == 500 else L), k) for k in ENS]
        a2.plot(
            ENS,
            y,
            color=c,
            lw=1.6,
            ls="-" if L == 500 else "--",
            label=f"$P^{{leaf}}$={L}",
            alpha=alpha,
        )
    a2.set_xlim(-4, 105)
    a2.set_xticks(range(0, 101, 20))
    a2.set_xlabel(r"$P^{ens}$", fontsize=8)
    a2.set_title(r"Error by $P^{ens}$, $P^{leaf}$ fixed", fontsize=8)

    # Shared y-axis in the original's calibrated units
    for a in axs:
        a.set_ylim(*YLIM)
        a.set_yticks(YT)
        a.tick_params(labelsize=7)
        a.set_yticklabels([f"{t:.2f}" for t in YT])
    a1.legend(fontsize=6, frameon=False, loc="upper right")
    a2.legend(fontsize=6, frameon=False, loc="upper right")
