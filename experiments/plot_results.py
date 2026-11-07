"""Rebuild paper-style figures from measured Fashion-MNIST results only.

The public author repository contains no plotting source. Layout, CM-style
math, inward ticks, blue/red mechanism coding and small-multiple composition
are reconstructed from the paper. Limits adapt to the measured new dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

from experiments.common import FIGURES, RESULTS, atomic_json

BLUE = "#0060ff"
RED = "#ff1818"


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 11,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": .65, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    FIGURES.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def load_main(kind):
    path = RESULTS / f"{kind}_main.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "pred_idx" in df:
        df = df[df.pred_idx.astype(str).eq("all")].copy()
    if "max_leaf_nodes" in df:
        df["max_leaf_nodes"] = pd.to_numeric(df.max_leaf_nodes, errors="coerce")
    return df


def leaf_coordinates(df):
    finite = sorted(df.max_leaf_nodes.dropna().unique())
    endpoint = float(max(finite) * 1.12)
    return finite, endpoint


def main_figure(kind, df):
    is_tree = kind == "trees"
    ens = "n_estimators" if is_tree else "n_ens"
    first = "max_leaf_nodes" if is_tree else "n_boost"
    title = "Trees" if is_tree else "Gradient Boosting"
    symbol = r"P^{\rm leaf}" if is_tree else r"P^{\rm boost}"
    members = sorted(df[ens].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.7))
    ax, mid, right = axes
    if is_tree:
        finite, switch = leaf_coordinates(df)
        full = df[df[first].isna()].sort_values(ens)
        front = df[df[ens].eq(1)].copy()
        front["axis_x"] = front[first].fillna(switch)
        front = front.sort_values("axis_x")
    else:
        switch = float(df[first].max())
        full = df[df[first].eq(switch)].sort_values(ens)
        front = df[df[ens].eq(1)].sort_values(first).copy()
        front["axis_x"] = front[first]
    ax.plot(front.axis_x, front.test_error_mse, color=BLUE)
    after_x = switch + (full[ens].to_numpy()-1)/(max(members)-1)*switch
    ax.plot(after_x, full.test_error_mse, color=RED)
    ax.axvline(switch, color=BLUE, lw=.7, ls="--", alpha=.65)
    max_label = r"\max" if is_tree else str(int(switch))
    ax.set_xticks([float(front.axis_x.iloc[0]), switch, 2*switch],
                  [r"$2\times1$" if is_tree else r"$1\times1$",
                   f"${max_label}\\times1$", f"${max_label}\\times{int(max(members))}$"])
    ax.set_xlabel(f"${symbol}\\times P^{{\\rm ens}}$")
    ax.set_ylabel("Squared test error")
    ax.set_title(f"Double Descent in {title}")

    colors = plt.cm.Blues(np.linspace(.95, .28, len(members)))
    for i, m in enumerate(members):
        part = df[df[ens].eq(m)].copy()
        part["axis_x"] = part[first].fillna(switch) if is_tree else part[first]
        part = part.sort_values("axis_x")
        mid.plot(part.axis_x, part.test_error_mse, color=BLUE if m == 1 else colors[i],
                 ls="-" if m == 1 else (0, (1, 1)), label=f"$P^{{\\rm ens}}={int(m)}$")
    mid.set_xlabel(f"${symbol}$")
    mid.set_title(f"Error by ${symbol}$, $P^{{\\rm ens}}$ fixed")
    legend_options = ({"loc": "upper right"} if is_tree else
                      {"loc": "center right", "bbox_to_anchor": (1, .62)})
    mid.legend(frameon=False, handlelength=1.8, ncol=2,
               columnspacing=.8, fontsize=7.5, **legend_options)
    if is_tree:
        mid.set_xticks([0, 500, 1000, switch], ["0", "500", "1000", "max"])

    values = ([None, 100, 50, 20] if is_tree else [200, 50, 25, 10])
    for i, value in enumerate(values):
        part = df[df[first].isna() if value is None else df[first].eq(value)].sort_values(ens)
        if part.empty:
            continue
        label = r"\max" if value is None else str(value)
        color = plt.cm.Reds(.72-i*.13)
        right.plot(part[ens], part.test_error_mse, color=RED if i == 0 else color,
                   ls="-" if i == 0 else (0, (1, 1)), label=f"${symbol}={label}$")
    right.set_xlabel(r"$P^{\rm ens}$")
    right.set_title(f"Error by $P^{{\\rm ens}}$, ${symbol}$ fixed")
    right.legend(frameon=False, loc="best", handlelength=2)
    lo = min(df.test_error_mse.min(), front.test_error_mse.min())
    hi = max(df.test_error_mse.max(), front.test_error_mse.max())
    pad = max((hi-lo)*.06, .005)
    for axis in axes:
        axis.set_ylim(max(0, lo-pad), hi+pad)
    fig.subplots_adjust(wspace=.22, bottom=.23, top=.83)
    save(fig, f"{kind}-decomposition")
    return {"main_grid_rows": len(df),
            "single_model_min_test_squared_error": float(front.test_error_mse.min()),
            "single_model_max_complexity_test_squared_error": float(full.iloc[0].test_error_mse),
            "largest_ensemble_test_squared_error": float(full.iloc[-1].test_error_mse),
            "max_complexity_training_squared_error": float(full.iloc[0].train_error_mse)}


def standardize_effective(df):
    df = df.copy()
    if "p_train" not in df:
        df["p_train"] = 10000*df.l2_train_sq
    if "p_test" not in df:
        df["p_test"] = 10000*df.l2_test_sq
    if "max_leaf_nodes" in df:
        df["max_leaf_nodes"] = pd.to_numeric(df.max_leaf_nodes, errors="coerce")
    return df


def effective_frames():
    out = {}
    p = RESULTS / "trees_effective.csv"
    if p.exists():
        tree = pd.read_csv(p)
        if len(tree) == 1100:
            out["Trees"] = standardize_effective(tree)
    p = RESULTS / "boosting_effective_ensemble.csv"
    if p.exists():
        ensemble = standardize_effective(pd.read_csv(p))
        p_single = RESULTS / "boosting_effective_single.csv"
        if p_single.exists():
            single = standardize_effective(pd.read_csv(p_single))
            single["n_ens"] = 1
            # Original normalized ensemble at M=1 is intentionally different
            # at small T. Keep both values, use single stage curve for axis 1.
            out["Boosting"] = (single, ensemble)
    # Load only a complete, validated 54-setting RFF run. Never invent a third
    # panel while cluster measurements are pending.
    from experiments.plot_rff import get_effective_frame
    rff = get_effective_frame()
    if rff is not None:
        out["Linear Regression"] = rff
    return out


def plot_effective(frames):
    if not frames:
        return
    fig7, a7 = plt.subplots(1, len(frames), figsize=(3.55*len(frames), 2.6), squeeze=False)
    fig8, a8 = plt.subplots(1, len(frames), figsize=(3.55*len(frames), 2.7), squeeze=False)
    fig9, a9 = plt.subplots(1, len(frames), figsize=(3.55*len(frames), 2.9), squeeze=False)
    for col, (label, data) in enumerate(frames.items()):
        ax7, ax8, ax9 = a7[0,col], a8[0,col], a9[0,col]
        if label == "Trees":
            df = data
            ens, first = "n_estimators", "max_leaf_nodes"
            _, switch = leaf_coordinates(df)
            df = df.copy()
            df["first_x"] = df[first].fillna(switch)
            front = df[df[ens].eq(1)].sort_values("first_x")
            tail = df[df[first].isna()].sort_values(ens)
            members = [1,2,5,10,20,50,100]
            first_symbol = r"P^{\rm leaf}"
        elif label == "Boosting":
            single, df = data
            ens, first = "n_ens", "n_boost"
            switch = float(single[first].max())
            df = df.copy()
            df["first_x"] = df[first]
            front = single.sort_values(first).copy()
            front["first_x"] = front[first]
            tail = df[df[first].eq(switch)].sort_values(ens)
            members = [1,2,5,10,20,50]
            first_symbol = r"P^{\rm boost}"
        else:
            df = data[data.kind.eq("middle")].copy()
            df["test_error_mse"] = df.class0_test_mse
            original = data[data.kind.eq("left")].sort_values("N").copy()
            original["test_error_mse"] = original.class0_test_mse
            original["first_x"] = original.N
            original["p_excess"] = np.maximum(0, original.N-10000)
            ens, first = "p_excess", "p_ls"
            switch = 10000.
            df["first_x"] = df.p_ls
            front = original[original.N.le(10000)]
            tail = original[original.N.ge(10000)]
            members = [0,2500,5000,10000,20000,50000]
            first_symbol = r"P_{\rm ls}"
        max_m = tail[ens].max()
        x_tail = (tail.N if label == "Linear Regression" else
                  switch + (tail[ens]-1)/(max_m-1)*switch)
        for field, color in [("p_train", "#ef8b00"), ("p_test", "#3ca343")]:
            ax7.plot(front.first_x, front[field], color=color, label=r"$\mathcal{I}_{\rm train}$" if field == "p_train" else r"$\mathcal{I}_{\rm test}$")
            ax7.plot(x_tail, tail[field], color=color)
        ax7.axvline(switch, color=BLUE, ls="--", lw=.7)
        ax7.set_yscale("log")
        if label == "Linear Regression":
            ax7.set_xticks([0,10000,60000], ["0", "10,000", "60,000"])
            ax7.set_xlabel(r"Raw RFF columns $N$ (bias excluded)")
        else:
            endpoint_label = "max" if label == "Trees" else str(int(switch))
            initial_label = "2" if label == "Trees" else "1"
            ax7.set_xticks([float(front.first_x.iloc[0]), switch, 2*switch],
                          [f"{initial_label} × 1", f"{endpoint_label} × 1", f"{endpoint_label} × {max_m:g}"])
            ax7.set_xlabel(f"${first_symbol}\\times P^{{\\rm ens}}$")
        ax7.set_title(label)
        ax7.legend(frameon=False, loc="best")

        for i,m in enumerate(members):
            part = df[df[ens].eq(m)].sort_values("first_x")
            if part.empty:
                continue
            color = plt.cm.Blues(.95-.65*i/max(1,len(members)-1))
            ax8.plot(part.first_x, part.p_test, color=".55", ls=(0,(1,1)), lw=1.4)
            axis_label = r"P^{\rm ex}" if label == "Linear Regression" else r"P^{\rm ens}"
            ax9.plot(part.p_test, part.test_error_mse, color=color, ls=(0,(1,1)), lw=1.7,
                     label=f"${axis_label}={m}$")
        front_marks = (front[front.n_boost.isin([1,2,10,25,50,100,200])]
                       if label == "Boosting" else front)
        ax8.plot(front_marks.first_x, front_marks.p_test, "o", ms=3.8, color=BLUE)
        highlighted = tail if label == "Linear Regression" else tail[tail[ens].isin(members)]
        ax8.plot(np.full(len(highlighted), switch), highlighted.p_test, "o", ms=3.8, color=RED)
        ax8.set_yscale("log")
        ax8.set_xlabel(f"${first_symbol}$")
        ax8.set_title(label)
        ax9.plot(front_marks.p_test, front_marks.test_error_mse, "o", ms=3.7, color=BLUE)
        # Select the same ensemble sizes used for the overlaid contour curves.
        points = tail if label == "Linear Regression" else tail[tail[ens].isin(members)]
        ax9.plot(points.p_test, points.test_error_mse, "o", ms=4.2, color=RED)
        ax9.set_xscale("log")
        ax9.set_xlabel(r"$p_{\hat s}^{\rm test}$ (log scale)")
        ax9.set_title(label)
        ax9.legend(frameon=False, loc="upper left" if label == "Boosting" else "upper center", handlelength=1.8,
                   ncol=2, columnspacing=.8, fontsize=7.5)
    a7[0,0].set_ylabel(r"$p_{\hat s}^{0}$ (log scale)")
    a8[0,0].set_ylabel(r"$p_{\hat s}^{\rm test}$ (log scale)")
    a9[0,0].set_ylabel("Squared test error (0-vs-all)")
    for fig,name in [(fig7,"effective-parameters"),(fig8,"effective-two-axes"),(fig9,"back-to-u")]:
        fig.subplots_adjust(wspace=.27,bottom=.23,top=.88)
        if name in {"effective-two-axes", "back-to-u"} and "Boosting" in frames:
            fig.text(.5, -.015,
                     "Boosting contours retain the released row normalization; blue markers use the single-model smoother.",
                     ha="center", va="top", fontsize=7)
        save(fig,name)


def main():
    setup()
    summary = {}
    for kind in ["trees", "boosting"]:
        df = load_main(kind)
        if df is not None and not df.empty:
            summary[kind] = main_figure(kind,df)
    plot_effective(effective_frames())
    from experiments.plot_rff import get_effective_frame, plot_decomposition
    rff = get_effective_frame()
    if rff is not None:
        plot_decomposition(rff)
        summary["rff"] = {"grid_rows": len(rff),
                          "min_test_squared_error": float(rff.test_error_mse.min())}
    atomic_json(FIGURES / "summary.json", summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
