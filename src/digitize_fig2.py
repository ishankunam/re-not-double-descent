"""
re-not-double-descent | digitize_fig2.py | ishan kunam
------------------------------------------------------
digitize the vector curves of figure 2 (page 3) of curth et al. (2023) & calibrate the y-axis
figure 2's y tick labels (0.8, 0.6, 0.4, 0.1) sit on evenly spaced ticks, so they cannot all be right
figure 3's inset (same page) plots the same curves w/ a self-consistent axis
fit a linear map from figure 2 pixels to figure 3 pixels on all shared vertices, then use figure 3's ticks
"""

from pathlib import Path

import numpy as np

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
