"""
re-not-double-descent | analyze_results.py | ishan kunam
--------------------------------------------------------
compare the re-run against the digitized original figure 2
write a comparison table, then print error minima, differences, up-steps along P^ens & seed statistics
"""

import argparse
from pathlib import Path

import numpy as np
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


# ----------------
# printed analysis
# ----------------


def print_minima(results, orig):
    # Where the single-tree curve bottoms out
    for v, d in results.items():
        q = d[d.p_ens == 1]
        i = q.test_err.idxmin()
        print(f"{v}: P_ens=1 min {q.test_err[i]:.6f} at P_leaf={q.p_leaf[i]}")

    q = orig[(orig.panel == "center") & (orig.series == "P_ens=1")]
    i = q.err_fig3cal.idxmin()
    print(f"original: P_ens=1 min {q.err_fig3cal[i]:.6f} at P_leaf={q.p_leaf[i]}")


def print_deltas(results, orig):
    # Re-run minus original at every digitized vertex ("max" -> cap 500)
    dd = []
    for _, r in orig.iterrows():
        L = "500" if str(r.p_leaf) in ("500", "max") else str(r.p_leaf)
        for v, d in results.items():
            delta = get(d, L, int(r.p_ens)) - r.err_fig3cal
            dd.append(
                {
                    "variant": v,
                    "panel": r.panel,
                    "p_leaf": L,
                    "p_ens": r.p_ens,
                    "delta": delta,
                }
            )
    dd = pd.DataFrame(dd)

    by_variant = dd.groupby("variant").delta
    print(
        by_variant.agg(lambda x: np.abs(x).max()).rename(
            "max|delta| over all digitized points"
        )
    )
    print(by_variant.agg(lambda x: np.abs(x).mean()).rename("mean|delta|"))


def print_upsteps(results, orig):
    # Steps along the P^ens grid where the error goes up (right panel)
    def ups_of(e):
        return [
            (ENS[i], ENS[i + 1], e[i + 1] - e[i]) for i in range(6) if e[i + 1] > e[i]
        ]

    def fmt(ups):
        return ";".join(f"{a}->{b}:+{x:.2e}" for a, b, x in ups)

    lines = []
    for v, d in results.items():
        for L in ["20", "50", "100", "500", "max"]:
            ups = ups_of([get(d, L, k) for k in ENS])
            lines.append({"source": v, "p_leaf": L, "n_up": len(ups), "ups": fmt(ups)})
    for L in [20, 50, 100, 500]:
        series = orig[(orig.panel == "right") & (orig.series == f"P_leaf={L}")]
        ups = ups_of(series.sort_values("p_ens").err_fig3cal.values)
        lines.append(
            {"source": "original", "p_leaf": str(L), "n_up": len(ups), "ups": fmt(ups)}
        )
    print(pd.DataFrame(lines).to_string())


def print_seed_stats(variant: str, S):
    # Mean & sample standard deviation across seeds at the probe points
    print(f"\n{variant.upper()} seeds 0-3 mean +/- sd:")
    for name, L, k in SEED_PROBES:
        x = S[(S.p_leaf == str(L)) & (S.p_ens == k)].test_err
        seeds = ", ".join(f"{y:.4f}" for y in x)
        print(f"  {name:24s} {x.mean():.4f} +/- {x.std(ddof=1):.4f}   (seeds: {seeds})")

    # Best single-tree error (& its leaf cap) for each seed
    print("  P_ens=1 min per seed:")
    for seed, g in S[S.p_ens == 1].groupby("seed"):
        i = g.test_err.idxmin()
        print(f"    seed {seed}: {g.test_err[i]:.6f} at P_leaf={g.p_leaf[i]}")


# ----
# main
# ----


def main():
    parser = argparse.ArgumentParser(description="compare the re-run w/ the original")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    # Digitized original (the figure 3 inset is only a calibration reference)
    orig = pd.read_csv(args.data_dir / "fig2_original_digitized.csv")
    orig = orig[orig.panel != "fig3_inset"]

    # Seed-42 re-runs of both model formulations
    results = {
        "OVA": load_results(args.results_dir / "results_ova_seed42.csv"),
        "MULTI": load_results(args.results_dir / "results_multi_seed42.csv"),
    }

    # Comparison table => csv & stdout
    T = comparison_table(results, orig)
    T.to_csv(args.results_dir / "comparison_original_vs_ours.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(6).to_string())

    print_minima(results, orig)
    print_deltas(results, orig)
    print_upsteps(results, orig)

    # Seeds 0-3 are optional (only present if those runs were done)
    for variant in ["ova", "multi"]:
        path = args.results_dir / f"results_{variant}_seeds0-3.csv"
        if path.exists():
            print_seed_stats(variant, load_results(path))

    print(f"\ndone. wrote {args.results_dir / 'comparison_original_vs_ours.csv'}")


if __name__ == "__main__":
    main()
