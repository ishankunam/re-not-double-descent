"""Compact, measured-only figure for the course report's new-dataset result."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.plot_results import load_main

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/section5-overleaf/figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.7, "axes.linewidth": .6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(5.32, 4.30))
    for row, kind in enumerate(["trees", "boosting"]):
        df = load_main(kind)
        tree = kind == "trees"
        grow = "max_leaf_nodes" if tree else "n_boost"
        ens = "n_estimators" if tree else "n_ens"
        left, right = axes[row]
        members = sorted(df[ens].unique())
        colors = plt.cm.Blues(np.linspace(.9, .35, len(members)))
        for idx, m in enumerate(members):
            p = df[df[ens].eq(m)].copy()
            p["x"] = p[grow].fillna(4000) if tree else p[grow]
            p = p.sort_values("x")
            left.plot(p.x, p.test_error_mse,
                      color="#0060ff" if m == 1 else colors[idx],
                      lw=1.5 if m == 1 else 1,
                      ls="-" if m == 1 else (0, (1.5, 1.5)),
                      label=f"$M={m}$")
        legend_location = {} if tree else {"bbox_to_anchor": (1, .68)}
        left.legend(frameon=False, ncol=2,
                    loc="upper right" if tree else "center right",
                    columnspacing=.8, handlelength=1.5, borderaxespad=.25,
                    **legend_location)
        if tree:
            left.set_xscale("log")
            left.set_xticks([2, 20, 100, 1000, 4000],
                            ["2", "20", "100", "1000", "full"])
            left.set_xlabel("Maximum leaves $L$")
        else:
            left.set_xlabel("Boosting stages $T$")
            left.set_xticks([1, 50, 100, 150, 200])
        left.set_ylabel("Squared test error")
        name = "Trees" if tree else "Boosting"
        symbol = "L" if tree else "T"
        left.set_title(f"({'a' if tree else 'c'}) {name}: vary ${symbol}$, fix $M$", loc="left")
        values = [None, 100, 50, 20] if tree else [200, 50, 25, 10]
        red_colors = ["#ff1818", "#b22323", "#d36d6d", "#a36c6c"]
        for i, v in enumerate(values):
            p = df[df[grow].isna() if v is None else df[grow].eq(v)].sort_values(ens)
            vlabel = r"\mathrm{full}" if v is None else str(v)
            right.plot(p[ens], p.test_error_mse, color=red_colors[i],
                       lw=1.5 if i == 0 else 1,
                       ls="-" if i == 0 else (0, (1.5, 1.5)),
                       marker="o", ms=2.1, label=f"${symbol}={vlabel}$")
        right.set_xlabel("Averaged members $M$")
        right.set_title(f"({'b' if tree else 'd'}) {name}: vary $M$, fix ${symbol}$", loc="left")
        right.legend(frameon=False, loc="upper right", handlelength=1.8,
                     borderaxespad=.25)
        right.set_xticks([1, 20, 50, 100] if tree else [1, 5, 10, 15, 20])
        for ax in (left, right):
            ax.tick_params(length=2.7, width=.55)
            ax.set_ylim(.18, .87 if tree else .52)
    fig.subplots_adjust(left=.11, right=.985, bottom=.105, top=.94,
                        wspace=.25, hspace=.48)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"fashion-mnist-axes.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
