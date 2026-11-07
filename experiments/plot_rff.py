#!/usr/bin/env python3
"""Plot measured, complete author-protocol RFF results in the style of Fig. 5.

No incomplete curve is drawn. --no-output checks labels and declared grids
without reading results, creating a figure, or running an experiment.

get_effective_frame() returns a validated 54-row DataFrame (or None while
pending), retaining all-class MSE and class-0 one-vs-all MSE separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.mathtext import MathTextParser
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

from experiments.common import FIGURES, RESULTS, atomic_json
from experiments.plot_results import BLUE, RED, save, setup

E_GRID = [0, 2500, 5000, 10000, 20000, 50000]
P_GRID = [25, 500, 1000, 2500, 5000, 7500, 10000]
N_GRID = [25, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 40000, 60000]
PAPER_RIGHT_P = [10000, 7500, 5000, 2500]
N_TRAIN = 10000
DEFAULT_CSV = RESULTS / "rff" / "rff_results.csv"
AXIS_NOTE = (
    "Released-code counts: left N excludes the intercept; middle P_ls includes it.\n"
    "For nominal excess E, the middle/right basis uses 10,000 + E raw features; E = 0 skips PCA."
)
MATH_LABELS = [r"$N$", r"$P_{\rm ls}$", r"$P^{\rm ex}$",
               r"$P^{\rm ex}=0$", r"$P_{\rm ls}=10000$",
               r"$p_{\hat s}^{\rm test}$", r"$\mathcal{I}_{\rm train}$",
               r"$\mathcal{I}_{\rm test}$"]


class PendingResults(Exception):
    """Expected measured grid is not yet complete; no figure should be emitted."""


def expected_keys():
    return ({("middle", e, p) for e in E_GRID for p in P_GRID}
            | {("left", None, n) for n in N_GRID})


def validate_frame(frame):
    """Require the exact 42+12 grid, dimensions, and finite measured metrics."""
    required = {"kind", "p_excess", "p_ls", "N", "actual_columns", "n_train", "n_test",
                "train_error_mse", "test_error_mse", "p_train", "p_test",
                "class0_train_mse", "class0_test_mse"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"RFF CSV lacks columns: {sorted(missing_columns)}")
    df = frame.copy()
    df["kind"] = df["kind"].astype(str)
    numeric = required - {"kind"}
    for key in numeric:
        df[key] = pd.to_numeric(df[key], errors="raise")
    keys = []
    for row in df.itertuples(index=False):
        if row.kind not in {"left", "middle"}:
            raise ValueError(f"Unknown RFF setting kind: {row.kind}")
        if row.n_train != N_TRAIN or row.n_test != 10000:
            raise ValueError("Only the full Fashion-MNIST protocol (10,000 train/test) may be plotted here.")
        if row.kind == "middle":
            if pd.isna(row.p_excess) or pd.isna(row.p_ls):
                raise ValueError("Middle row has no p_excess or p_ls")
            if row.p_excess != int(row.p_excess) or row.p_ls != int(row.p_ls):
                raise ValueError("Nonintegral middle parameter count")
            key = ("middle", int(row.p_excess), int(row.p_ls))
            if row.N != N_TRAIN + row.p_excess or row.actual_columns != row.p_ls:
                raise ValueError(f"Middle dimensions do not follow the released code: {key}")
        else:
            if pd.isna(row.N) or row.N != int(row.N):
                raise ValueError("Left row has missing/nonintegral N")
            if pd.notna(row.p_excess) or pd.notna(row.p_ls):
                raise ValueError("Left rows must not masquerade as middle-axis settings")
            key = ("left", None, int(row.N))
            if row.actual_columns != row.N + 1:
                raise ValueError(f"Left dimensions do not preserve the intercept offset: {key}")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate RFF settings: refusing to average or silently drop them")
    observed = set(keys)
    unexpected = observed - expected_keys()
    if unexpected:
        raise ValueError(f"Unexpected RFF settings: {sorted(unexpected, key=str)}")
    absent = expected_keys() - observed
    if absent:
        middle_count = sum(key[0] == "middle" for key in observed)
        left_count = sum(key[0] == "left" for key in observed)
        raise PendingResults(f"RFF pending: middle {middle_count}/42; left {left_count}/12; "
                             f"missing {len(absent)} settings. No figure written.")
    metrics = ["train_error_mse", "test_error_mse", "p_train", "p_test",
               "class0_train_mse", "class0_test_mse"]
    if not np.isfinite(df[metrics].to_numpy()).all():
        raise ValueError("Nonfinite measured RFF metric; refusing to plot")
    if (df[metrics] < 0).any().any():
        raise ValueError("Negative MSE/effective parameter count")
    if (df.test_error_mse <= 0).any():
        raise ValueError("Logarithmic test-error plot requires strictly positive measured MSE; no epsilon is invented")
    return df


def load_complete_frame(path=DEFAULT_CSV):
    path = Path(path)
    if not path.exists():
        raise PendingResults(f"RFF pending: {path} does not exist. No figure written.")
    manifest = path.parent / "manifest.json"
    if manifest.exists():
        protocol = json.loads(manifest.read_text()).get("protocol", {})
        if protocol.get("smoke") or protocol.get("dataset") != "Fashion-MNIST":
            raise ValueError("Refusing to plot smoke/other-dataset results as the formal Fashion-MNIST experiment")
    return validate_frame(pd.read_csv(path))


def get_effective_frame(path=DEFAULT_CSV):
    """Validated 54 rows for later Fig. 7-9 integration, or None if pending.

    Filter kind='left' for the 12-point original path; kind='middle' gives
    6x7 settings, grouped by p_excess. p_train/p_test already include n=10,000.
    Main Fig. 5 uses test_error_mse (sum of ten one-vs-all losses). For paper
    effective-parameter comparisons use class0_test_mse, also provided through
    the explicit effective_test_error_mse alias; no existing field is changed.
    """
    try:
        df = load_complete_frame(path)
    except PendingResults:
        return None
    df["effective_test_error_mse"] = df.class0_test_mse
    df["effective_train_error_mse"] = df.class0_train_mse
    df.attrs["source_csv"] = str(Path(path).resolve())
    df.attrs["axis_note"] = AXIS_NOTE
    df.attrs["effective_parameter_scale"] = "p_train and p_test are n * mean squared smoother-row norm"
    return df


def plot_decomposition(frame, *, right_p_ls=PAPER_RIGHT_P):
    """Validate again before writing a paper-style three-panel figure."""
    df = validate_frame(frame)
    invalid_p = set(right_p_ls) - set(P_GRID)
    if invalid_p:
        raise ValueError(f"Unsupported fixed p_ls values: {invalid_p}")
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    left, mid, right = axes

    original = df[df.kind.eq("left")].sort_values("N")
    before = original[original.N.le(N_TRAIN)]
    after = original[original.N.ge(N_TRAIN)]
    left.plot(before.N, before.test_error_mse, color=BLUE)
    left.plot(after.N, after.test_error_mse, color=RED)
    left.axvline(N_TRAIN, color=BLUE, lw=.7, ls="--", alpha=.65)
    left.set_xlim(0, max(N_GRID) * 1.025)
    left.set_xticks([0, N_TRAIN, 35000, 60000], ["0", "10,000", "35,000", "60,000"])
    left.set_xlabel(r"Raw RFF columns $N$ (bias excluded)")
    left.set_ylabel("Squared test error (log scale)")
    left.set_title("Double Descent in RFF Regression")

    for i, excess in enumerate(E_GRID):
        part = df[df.kind.eq("middle") & df.p_excess.eq(excess)].sort_values("p_ls")
        color = BLUE if excess == 0 else plt.cm.Blues(.9 - .6 * i / (len(E_GRID) - 1))
        mid.plot(part.p_ls, part.test_error_mse, color=color,
                 ls="-" if excess == 0 else (0, (1, 1)),
                 label=f"$P^{{\\rm ex}}={excess}$")
    mid.set_xlim(0, max(P_GRID) * 1.025)
    mid.set_xticks([0, 2500, 5000, 7500, 10000])
    mid.set_xlabel(r"$P_{\rm ls}$ (includes bias)")
    mid.set_title(r"Error by $P_{\rm ls}$, nominal $P^{\rm ex}$ fixed")
    mid.legend(frameon=False, loc="best", handlelength=2, fontsize=7)

    for i, p_ls in enumerate(right_p_ls):
        part = df[df.kind.eq("middle") & df.p_ls.eq(p_ls)].sort_values("p_excess")
        color = RED if p_ls == max(P_GRID) else plt.cm.Reds(.76 - .38 * i / max(1, len(right_p_ls) - 1))
        right.plot(part.p_excess, part.test_error_mse, color=color,
                   ls="-" if p_ls == max(P_GRID) else (0, (1, 1)),
                   label=f"$P_{{\\rm ls}}={p_ls}$")
    right.set_xlim(-max(E_GRID) * .025, max(E_GRID) * 1.025)
    right.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
    right.set_xlabel(r"Nominal $P^{\rm ex}$")
    right.set_title(r"Error by nominal $P^{\rm ex}$, $P_{\rm ls}$ fixed")
    right.legend(frameon=False, loc="best", handlelength=2, fontsize=7)

    lo, hi = df.test_error_mse.min(), df.test_error_mse.max()
    log_pad = max(.055 * np.log10(hi / lo), .06)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(lo / 10 ** log_pad, hi * 10 ** log_pad)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    fig.subplots_adjust(wspace=.18, bottom=.29, top=.86)
    fig.text(.5, .025, AXIS_NOTE, ha="center", va="bottom", fontsize=7)
    save(fig, "rff-decomposition")
    return {"rows": len(df), "left_points": len(original), "middle_points": 42,
            "right_p_ls": list(right_p_ls), "scale": "log y; linear nominal parameter x",
            "main_loss": "mean over test examples of sum of ten classwise squared residuals",
            "axis_note": AXIS_NOTE,
            "left_min_squared_test_error": float(original.test_error_mse.min()),
            "left_largest_N_squared_test_error": float(original.iloc[-1].test_error_mse)}


def no_output_check():
    assert len(expected_keys()) == 54
    assert len(E_GRID) * len(P_GRID) == 42 and len(N_GRID) == 12
    parser = MathTextParser("agg")
    for label in MATH_LABELS:
        parser.parse(label, dpi=100)
    return {"status": "passed", "expected_settings": 54, "expected_middle": 42,
            "expected_left": 12, "math_labels_parsed": len(MATH_LABELS),
            "result_data_read": False, "figures_written": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--no-output", action="store_true", help="Check declared grids/math labels; do not read data or draw")
    parser.add_argument("--all-fixed-dimensions", action="store_true", help="Show all seven right-panel p_ls curves; default matches the paper's four")
    args = parser.parse_args()
    if args.no_output:
        print(json.dumps(no_output_check(), indent=2))
        return
    try:
        df = load_complete_frame(args.csv)
    except PendingResults as pending:
        print(str(pending))
        return
    summary = plot_decomposition(df, right_p_ls=list(reversed(P_GRID)) if args.all_fixed_dimensions else PAPER_RIGHT_P)
    summary["source_csv"] = str(args.csv.resolve())
    summary["source_csv_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    atomic_json(FIGURES / "rff-decomposition-provenance.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
