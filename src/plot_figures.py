"""
re-not-double-descent | plot_figures.py | ishan kunam
-----------------------------------------------------
draw the re-run of figure 2 (three panels each)
overlay: one-vs-all curves w/ the digitized original vertices on top
multi: multi-output forest
seeds: one-vs-all seed 42 line w/ a mean +/- 1 sd band over seeds 0-3
"""

import argparse
from pathlib import Path

# Plotting (Agg backend => no display needed)
import matplotlib.pyplot as plt
import numpy as np
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


def bands(axs, S, maxkey: str, left_x):
    a0, a1, a2 = axs
    blue_x, red_x = left_x

    def ms(L, k):
        # Mean & sample standard deviation over seeds
        x = S[(S.p_leaf == str(L)) & (S.p_ens == k)].test_err
        return x.mean(), x.std(ddof=1)

    # Left panel: both curves
    bl = [2, 5, 10, 20, 50, 75, 100]
    xb = [blue_x[L] for L in bl] + [blue_x["max"]]
    mb = np.array([ms(L, 1) for L in bl] + [ms(maxkey, 1)])
    a0.fill_between(
        xb, mb[:, 0] - mb[:, 1], mb[:, 0] + mb[:, 1], color=BLUE, alpha=0.25, lw=0
    )
    mr = np.array([ms(maxkey, k) for k in ENS])
    a0.fill_between(
        [red_x[k] for k in ENS],
        mr[:, 0] - mr[:, 1],
        mr[:, 0] + mr[:, 1],
        color=RED,
        alpha=0.25,
        lw=0,
    )

    # Center panel: one band per ensemble size
    for k, c in ENS_COLS.items():
        m = np.array([ms(maxkey if L == 500 else L, k) for L in CENTER_LEAVES])
        a1.fill_between(
            CENTER_LEAVES,
            m[:, 0] - m[:, 1],
            m[:, 0] + m[:, 1],
            color=c,
            alpha=0.3,
            lw=0,
        )

    # Right panel: one band per leaf cap
    for L, c in LEAF_COLS.items():
        m = np.array([ms(maxkey if L == 500 else L, k) for k in ENS])
        a2.fill_between(
            ENS, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], color=c, alpha=0.3, lw=0
        )


# -----------
# draw & save
# -----------


def draw_figure(d, maxkey: str, title: str, out: Path, left_x, S=None):
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    three_panel(axs, d, maxkey, left_x)

    # Optional seed band on top of the seed-42 lines
    if S is not None:
        bands(axs, S, maxkey, left_x)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def draw_overlay(d, orig, out: Path, left_x):
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    three_panel(axs, d, "500", left_x, alpha=0.8)

    # Digitized original vertices as open circles (the left panel is in pixel x)
    style = {"s": 10, "facecolors": "none", "edgecolors": "k", "lw": 0.7, "zorder": 5}
    left = orig[orig.panel == "left"]
    axs[0].scatter(left.x_px, left.err_fig3cal, label="original (digitized)", **style)
    center = orig[orig.panel == "center"]
    axs[1].scatter(center.p_leaf.astype(int), center.err_fig3cal, **style)
    right = orig[orig.panel == "right"]
    axs[2].scatter(right.p_ens, right.err_fig3cal, **style)
    axs[0].legend(fontsize=6, frameon=False, loc="upper right")

    fig.suptitle(
        "Ours OVA (lines) vs original Fig. 2 digitized vertices (circles), "
        "y calibrated via Fig. 3 inset",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


# ----
# main
# ----


def main():
    parser = argparse.ArgumentParser(description="draw the figure 2 re-runs")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    # Digitized original (the figure 3 inset is only a calibration reference)
    orig = pd.read_csv(args.data_dir / "fig2_original_digitized.csv")
    orig = orig[orig.panel != "fig3_inset"]
    left_x = left_panel_x(orig)

    # Seed-42 re-runs of both model formulations
    ova = load_results(args.results_dir / "results_ova_seed42.csv")
    multi = load_results(args.results_dir / "results_multi_seed42.csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    draw_overlay(ova, orig, args.out_dir / "fig2_overlay_ova.png", left_x)
    draw_figure(
        multi,
        "500",
        "Ours (re-run), MULTI: one multi-output forest, seed 42; '500' = max_leaf_nodes=500",
        args.out_dir / "fig2_multi.png",
        left_x,
    )

    # Seed band: mean ± 1 sample sd over seeds 0-3 (the seed-42 run is only the line)
    seeds = load_results(args.results_dir / "results_ova_seeds0-3.csv")
    draw_figure(
        ova,
        "500",
        "Ours OVA: seed 42 line; band = mean ± 1 sd over seeds 0,1,2,3",
        args.out_dir / "fig2_seeds_ova.png",
        left_x,
        S=seeds,
    )

    print(f"\ndone. figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
