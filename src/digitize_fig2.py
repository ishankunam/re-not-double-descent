"""
re-not-double-descent | digitize_fig2.py | ishan kunam
------------------------------------------------------
digitize the vector curves of figure 2 (page 3) of curth et al. (2023) & calibrate the y-axis
figure 2's y tick labels (0.8, 0.6, 0.4, 0.1) sit on evenly spaced ticks, so they cannot all be right
figure 3's inset (same page) plots the same curves w/ a self-consistent axis
fit a linear map from figure 2 pixels to figure 3 pixels on all shared vertices, then use figure 3's ticks
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Reads the paper's PDF & exposes its vector drawings
import pymupdf

# -------------------
# input configuration
# -------------------

ROOT = Path(__file__).resolve().parent.parent  # repository root
OUT_FILE = ROOT / "data" / "fig2_original_digitized.csv"  # digitized vertices

PAGE_INDEX = 2  # figure 2 is on page 3 (zero-indexed)

# Windows (x0, x1, y0, y1 in PDF points) that isolate each region of the page
LEFT_PANEL = (135, 241, 75, 139)
CENTER_PANEL = (261, 366, 75, 139)
RIGHT_PANEL = (386, 492, 75, 139)
FIG2_YTICKS = (135, 138, 75, 139)  # left-axis ticks of the left panel
CENTER_XTICKS = (261, 366, 135, 139)
RIGHT_XTICKS = (386, 492, 135, 139)
FIG3_YTICKS = (408, 411, 573, 641)  # y ticks of the figure 3 inset
FIG3_INSET = (409, 497, 574, 640)

# Figure 3 inset major ticks (pixels), labelled 0.1, 0.3, 0.5 & 0.7 in the PDF's text spans
FIG3_MAJOR = [640.0, 622.38, 604.76, 587.14]
FIG3_MAJOR_VALUES = [0.1, 0.3, 0.5, 0.7]

# Stroke colours (RGB, rounded to 2 decimals) used to tell the curves apart
BLUE = (0.01, 0.35, 0.93)
RED = (0.93, 0.11, 0.14)
L100 = (0.91, 0.2, 0.16)  # right panel: P^leaf = 100
L50 = (0.99, 0.48, 0.35)  # right panel: P^leaf = 50 (P^leaf = 100 in the inset)
L20 = (0.99, 0.73, 0.63)  # right panel: P^leaf = 20 (P^leaf = 50 in the inset)
CENTER_ENS = {
    BLUE: 1,
    (0.23, 0.55, 0.76): 5,
    (0.5, 0.73, 0.86): 10,
    (0.78, 0.86, 0.94): 50,
}
RIGHT_LEAF = {RED: 500, L100: 100, L50: 50, L20: 20}

LEAF_GRID = [2, 5, 10, 20, 50, 75, 100, "max"]  # left panel, blue curve
ENS_GRID = [1, 2, 5, 10, 20, 50, 100]  # left panel, red curve

# ----------------------------------
# read vertices from the PDF drawing
# ----------------------------------


def verts(g):
    # Collect the start & end point of every line / curve segment
    out = []
    for it in g["items"]:
        if it[0] == "l":
            out += [it[1], it[2]]
        elif it[0] == "c":
            out += [it[1], it[4]]

    # Drop consecutive duplicates (segments share endpoints)
    r = []
    for q in out:
        if not r or abs(r[-1][0] - q.x) > 1e-3 or abs(r[-1][1] - q.y) > 1e-3:
            r.append((q.x, q.y))
    return np.array(r)


def col(g):
    return tuple(round(c, 2) for c in (g.get("color") or ()))


def strokes(drawings, window, minpts: int = 5):
    x0, x1, y0, y1 = window

    # Coloured strokes (not the black axes) lying fully inside the window
    res = []
    for g in drawings:
        r = g["rect"]
        inside = x0 <= r.x0 and r.x1 <= x1 and y0 <= r.y0 and r.y1 <= y1
        if g["type"] == "s" and inside and col(g) != (0.0, 0.0, 0.0):
            v = verts(g)
            if len(v) >= minpts:
                res.append((col(g), v))
    return res


def ticks(drawings, window, horizontal: bool):
    x0, x1, y0, y1 = window

    # Tick marks are zero-height (horizontal) or zero-width (vertical) filled rectangles
    t = []
    for g in drawings:
        r = g["rect"]
        inside = x0 <= r.x0 and r.x1 <= x1 and y0 <= r.y0 and r.y1 <= y1
        if g["type"] == "fs" and inside:
            if horizontal and abs(r.y1 - r.y0) < 1e-3:
                t.append(round(r.y0, 3))
            if not horizontal and abs(r.x1 - r.x0) < 1e-3:
                t.append(round(r.x0, 3))
    return sorted(set(t))


def find(curves, color):
    # Vertices of every curve drawn in this colour
    return [v for cc, v in curves if cc == color]


# ----------------
# calibrate y-axis
# ----------------


def calibrate(drawings, left, right, inset):
    # Figure 3 inset: pixel -> value from its labelled major ticks
    f3_ticks = ticks(drawings, FIG3_YTICKS, True)
    print("Fig3 inset y ticks (px) incl. minor:", f3_ticks)
    assert all(any(abs(m - t) < 0.01 for t in f3_ticks) for m in FIG3_MAJOR)
    k3 = np.polyfit(FIG3_MAJOR, FIG3_MAJOR_VALUES, 1)

    def v3(y):
        return np.polyval(k3, y)

    # Curves shared by figure 2 & the figure 3 inset
    f2_blue, f2_red = find(left, BLUE)[0], find(left, RED)[0]
    f3_blue, f3_red = find(inset, BLUE)[0], find(inset, RED)[0]
    f2_100, f2_50 = find(right, L100)[0], find(right, L50)[0]
    f3_100, f3_50 = find(inset, L50)[0], find(inset, L20)[0]

    # The inset's blue curve has an extra P^leaf = 200 vertex (index 7)
    f3_blue_no200 = np.delete(f3_blue, 7, axis=0)

    # Linear map from figure 2 pixels to figure 3 pixels
    a = np.concatenate([f2_blue[:, 1], f2_red[:, 1], f2_50[:, 1], f2_100[:, 1]])
    b = np.concatenate([f3_blue_no200[:, 1], f3_red[:, 1], f3_50[:, 1], f3_100[:, 1]])
    m = np.polyfit(a, b, 1)
    resid = np.abs(np.polyval(m, a) - b).max()
    print(
        f"Fig2->Fig3 px map: y3 = {m[0]:.5f} y2 + {m[1]:.3f}; "
        f"n={len(a)}, max resid {resid:.4f} px"
    )

    def v2(y):
        return v3(np.polyval(m, y))

    # The tick labels as printed read "bottom tick = 0.2"; kept as a comparison
    f2_yt = ticks(drawings, FIG2_YTICKS, True)
    print("Fig2 y ticks (px):", f2_yt, "spacings", np.diff(f2_yt))
    k_alt = np.polyfit(f2_yt[::-1], [0.2, 0.4, 0.6, 0.8], 1)

    def v2_alt(y):
        return np.polyval(k_alt, y)

    print(
        f"Fig2 frame under Fig3 calibration: top {v2(75.4):.4f} bottom {v2(138.6):.4f}"
    )
    print(
        "Fig2 tick positions under Fig3 calibration:",
        [round(float(v2(t)), 4) for t in f2_yt],
    )
    return v3, v2, v2_alt


# ---------------------
# build digitized table
# ---------------------


def build_table(drawings, left, center, right, inset, v3, v2, v2_alt):
    # x ticks of the center & right panels give pixel -> P^leaf / P^ens
    cx_t = ticks(drawings, CENTER_XTICKS, False)
    rx_t = ticks(drawings, RIGHT_XTICKS, False)
    print("center x ticks", cx_t, "right x ticks", rx_t)

    def cx(x):
        return (x - cx_t[0]) / (cx_t[-1] - cx_t[0]) * 500

    def rx(x):
        return (x - rx_t[0]) / (rx_t[-1] - rx_t[0]) * 100

    f2_blue, f2_red = find(left, BLUE)[0], find(left, RED)[0]
    f3_blue = find(inset, BLUE)[0]

    rows = []

    # Left panel: blue (P^ens = 1, leaves grow) then red (P^leaf = max, trees grow)
    for i, (x, y) in enumerate(f2_blue):
        rows.append(
            {
                "panel": "left",
                "series": "P_ens=1 (blue)",
                "p_leaf": LEAF_GRID[i],
                "p_ens": 1,
                "x_px": x,
                "y_px": y,
            }
        )
    for i, (x, y) in enumerate(f2_red):
        rows.append(
            {
                "panel": "left",
                "series": "P_leaf=max (red)",
                "p_leaf": "max",
                "p_ens": ENS_GRID[i],
                "x_px": x,
                "y_px": y,
            }
        )

    # Center panel: error vs P^leaf for each fixed P^ens
    for c, v in center:
        for x, y in v:
            rows.append(
                {
                    "panel": "center",
                    "series": f"P_ens={CENTER_ENS[c]}",
                    "p_leaf": int(np.round(cx(x))),
                    "p_ens": CENTER_ENS[c],
                    "x_px": x,
                    "y_px": y,
                }
            )

    # Right panel: error vs P^ens for each fixed P^leaf
    for c, v in right:
        for x, y in v:
            rows.append(
                {
                    "panel": "right",
                    "series": f"P_leaf={RIGHT_LEAF[c]}",
                    "p_leaf": RIGHT_LEAF[c],
                    "p_ens": int(np.round(rx(x))),
                    "x_px": x,
                    "y_px": y,
                }
            )

    # Figure 3 inset: kept as the calibration reference
    for x, y in f3_blue:
        rows.append(
            {
                "panel": "fig3_inset",
                "series": "P_ens=1 (blue)",
                "p_leaf": None,
                "p_ens": 1,
                "x_px": x,
                "y_px": y,
            }
        )

    # Convert pixels to error values under both readings of the y-axis
    df = pd.DataFrame(rows)
    df["err_fig3cal"] = [
        float(v3(y)) if p == "fig3_inset" else float(v2(y))
        for p, y in zip(df.panel, df.y_px)
    ]
    df["err_if_bottom_tick_0.2"] = [
        np.nan if p == "fig3_inset" else float(v2_alt(y))
        for p, y in zip(df.panel, df.y_px)
    ]
    return df


# ----
# main
# ----


def main():
    parser = argparse.ArgumentParser(
        description="digitize figure 2 from the paper's PDF"
    )
    parser.add_argument(
        "--pdf", type=Path, required=True, help="path to the paper's PDF"
    )
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    args = parser.parse_args()

    # Vector drawings on the page that holds figures 2 & 3
    drawings = pymupdf.open(args.pdf)[PAGE_INDEX].get_drawings()

    # Curves in each region
    left = strokes(drawings, LEFT_PANEL)
    center = strokes(drawings, CENTER_PANEL)
    right = strokes(drawings, RIGHT_PANEL)
    inset = strokes(drawings, FIG3_INSET)

    v3, v2, v2_alt = calibrate(drawings, left, right, inset)
    df = build_table(drawings, left, center, right, inset, v3, v2, v2_alt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string())
    print(f"\ndone. wrote {len(df)} vertices to {args.out}")


if __name__ == "__main__":
    main()
