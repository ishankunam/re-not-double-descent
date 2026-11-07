"""
re-not-double-descent | typeset_comparison.py | ishan kunam
-----------------------------------------------------------
typeset the paper's figure 2 (left) next to the one-vs-all re-run (right) w/ latex & pgfplots
the original panels are cropped from the paper's pdf at run time, so the paper's figure is never committed
the re-run curves are written as pgfplots coordinates from the results csv
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd

# Reads the paper's PDF & copies page regions into new vector PDFs
import pymupdf

# Readers & grids shared w/ the matplotlib figures (src/ is on the import path when run as a script)
from plot_figures import ENS, YLIM, get, left_panel_x, load_results

# -------------------
# input configuration
# -------------------

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA_DIR = ROOT / "data"  # digitized original figure
RESULTS_DIR = ROOT / "results"  # re-run outputs of run_fig2.py
OUT_DIR = ROOT / "figures" / "fig2_comparison"  # generated files (ignored by git)

RESULTS_FILE = "results_ova_seed42.csv"  # the one-vs-all seed-42 re-run
MAX_KEY = "500"  # the paper's "max" is a leaf cap of 500, not unrestricted growth

PAGE_INDEX = 2  # figure 2 is on page 3 (zero-indexed)

# Regions of that page (PDF points) holding each panel w/ its axis labels
CROPS = [
    pymupdf.Rect(117, 75, 242, 159),  # (a) combined growth
    pymupdf.Rect(247, 75, 372, 159),  # (b) leaf cap varied
    pymupdf.Rect(374, 75, 499, 159),  # (c) ensemble size varied
]

LEFT_LEAVES = [2, 5, 10, 20, 50, 75, 100]  # blue curve of the combined panel
CENTER_LEAVES = [2, 5, 10, 20, 50, 75, 100, 500]  # P^leaf grid of panel (b)
CENTER_ENS = [1, 5, 10, 50]  # one curve per P^ens in panel (b)
RIGHT_LEAVES = [500, 100, 50, 20]  # one curve per P^leaf in panel (c)

# pgfplots line styles: full strength first, then progressively lighter dashed curves
BLUE_STYLES = [
    "bluecurve,line width=1.2pt",
    "bluecurve!75,line width=1.2pt,dashed",
    "bluecurve!50,line width=1.2pt,dashed",
    "bluecurve!25,line width=1.2pt,dashed",
]
RED_STYLES = [
    "redcurve,line width=1.2pt",
    "redcurve!75,line width=1.2pt,dashed",
    "redcurve!50,line width=1.2pt,dashed",
    "redcurve!25,line width=1.2pt,dashed",
]

# --------------
# latex template
# --------------

# Text between @...@ is filled in by the functions below
PREAMBLE = r"""% Compile: pdflatex @NAME@.tex
% Original panels are vector crops of Curth et al. (2023), Figure 2.
% Reproduction coordinates come from @RESULTS@.
\documentclass[border=6pt,varwidth=7.2in]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx,amsmath,pgfplots,caption,array}
\pgfplotsset{compat=1.18}
\definecolor{bluecurve}{RGB}{4,89,237}
\definecolor{redcurve}{RGB}{237,28,36}
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false}
\begin{document}
\centering
\begin{tabular}{@{}m{3.5in}@{\hspace{0.2in}}m{3.5in}@{}}
\centering\textbf{Original: Curth et al. (2023)} & \centering\arraybackslash\textbf{Reproduction: one-vs-all, seed 42} \\[6pt]"""

AXIS = r"""\begin{tikzpicture}[baseline=(current bounding box.center)]
\begin{axis}[scale only axis,width=2.85in,height=1.72in,
font=\small,tick label style={font=\small},
ymin=@YMIN@,ymax=@YMAX@,ytick={0.15,0.383333,0.616667,0.85},
yticklabels={0.15,0.38,0.62,0.85},ylabel={Squared test error},
axis line style={line width=0.4pt},tick style={line width=0.4pt},
legend style={draw=none,fill=none,font=\small,cells={anchor=west},at={(0.98,0.98)},anchor=north east},
@XOPTS@]"""

ROW = r"""\multicolumn{2}{@{}l}{\small @LABEL@} \\[3pt]
\centering\raisebox{-0.5\height}{\includegraphics[width=3.40in]{@CROP@}} & @PLOT@ \\[8pt]"""

CAPTION = r"""\captionof{figure}{\textbf{Reproducing the tree experiment in Figure 2.}
Original panels (left) and our reproduction (right) show squared test error on MNIST.
Rows compare (a) the combined growth axis, (b) leaf caps at fixed ensemble sizes, and
(c) ensemble sizes at fixed leaf caps. Original panels retain their published tick labels;
reproduction values use the Figure 3 inset calibration. In the reproduction,
$\mathrm{max}$ denotes a leaf cap of 500, not unrestricted growth.}"""

# Axis options that differ between the three panels (x-axis only)
XOPTS_LEFT = (
    r"xmin=135.5,xmax=240.1,xtick={159.2,187.7,211.6},"
    r"xticklabels={$200\!\times\!1$,$\mathrm{max}\!\times\!1$,$\mathrm{max}\!\times\!50$},"
    r"xlabel={$P^{\mathrm{leaf}}\times P^{\mathrm{ens}}$}"
)
XOPTS_CENTER = (
    r"xmin=-11,xmax=537,xtick={0,100,200,300,400,500},xlabel={$P^{\mathrm{leaf}}$}"
)
XOPTS_RIGHT = r"xmin=-4,xmax=105,xtick={0,20,40,60,80,100},xlabel={$P^{\mathrm{ens}}$}"

# ------------------------
# crop the original panels
# ------------------------


def crop_panels(pdf: Path, out_dir: Path):
    src = pymupdf.open(pdf)

    # One new single-page PDF per panel, copied as vectors (no rasterizing)
    for i, clip in enumerate(CROPS, start=1):
        doc = pymupdf.open()
        page = doc.new_page(width=clip.width, height=clip.height)
        page.show_pdf_page(page.rect, src, PAGE_INDEX, clip=clip)
        doc.save(out_dir / f"original-{i}.pdf")


# ---------------------------------
# build the reproduction's pgfplots
# ---------------------------------


def coords(xs, ys):
    # pgfplots coordinate list: (x,y) pairs
    return " ".join(f"({x:.8f},{y:.10f})" for x, y in zip(xs, ys))


def pgf_axis(xopts: str, curves, extra: str = ""):
    # Axis options shared by all panels, then one \addplot per curve
    lines = [
        AXIS.replace("@YMIN@", str(YLIM[0]))
        .replace("@YMAX@", str(YLIM[1]))
        .replace("@XOPTS@", xopts)
    ]
    for style, points, legend in curves:
        lines.append(f"\\addplot[{style}] coordinates {{{points}}};")
        if legend:
            lines.append(f"\\addlegendentry{{{legend}}}")
    if extra:
        lines.append(extra)
    lines.append(r"\end{axis}\end{tikzpicture}")
    return "\n".join(lines)


def panel_left(d, left_x):
    blue_x, red_x = left_x

    # Leaves grow w/ one tree (blue), then trees grow at the cap (red)
    xb = [blue_x[L] for L in LEFT_LEAVES] + [blue_x["max"]]
    yb = [get(d, L, 1) for L in LEFT_LEAVES] + [get(d, MAX_KEY, 1)]
    xr = [red_x[k] for k in ENS]
    yr = [get(d, MAX_KEY, k) for k in ENS]
    curves = [
        (BLUE_STYLES[0], coords(xb, yb), None),
        (RED_STYLES[0], coords(xr, yr), None),
    ]

    # Dashed line where growth switches from leaves to trees
    x = float(blue_x["max"])
    divider = (
        f"\\draw[gray,densely dashed,thin] "
        f"(axis cs:{x},{YLIM[0]})--(axis cs:{x},{YLIM[1]});"
    )
    return pgf_axis(XOPTS_LEFT, curves, divider)


def panel_center(d):
    # Error vs leaf cap, one curve per ensemble size
    curves = []
    for style, k in zip(BLUE_STYLES, CENTER_ENS):
        ys = [get(d, L, k) for L in CENTER_LEAVES]
        legend = rf"$P^{{\mathrm{{ens}}}}={k}$"
        curves.append((style, coords(CENTER_LEAVES, ys), legend))
    return pgf_axis(XOPTS_CENTER, curves)


def panel_right(d):
    # Error vs ensemble size, one curve per leaf cap
    curves = []
    for style, L in zip(RED_STYLES, RIGHT_LEAVES):
        ys = [get(d, L, k) for k in ENS]
        legend = rf"$P^{{\mathrm{{leaf}}}}={L}$"
        curves.append((style, coords(ENS, ys), legend))
    return pgf_axis(XOPTS_RIGHT, curves)


# ---------------
# write & compile
# ---------------


def build_tex(d, left_x, name: str, caption: bool):
    labels = [
        "(a) Combined growth: leaves, then trees",
        "(b) Leaf cap varied; ensemble size fixed",
        "(c) Ensemble size varied; leaf cap fixed",
    ]
    plots = [panel_left(d, left_x), panel_center(d), panel_right(d)]

    # Header, then one row per panel (original crop | reproduction)
    parts = [PREAMBLE.replace("@NAME@", name).replace("@RESULTS@", RESULTS_FILE)]
    for i, (label, plot) in enumerate(zip(labels, plots), start=1):
        row = ROW.replace("@LABEL@", label).replace("@CROP@", f"original-{i}.pdf")
        parts.append(row.replace("@PLOT@", plot))
    parts.append(r"\end{tabular}")
    if caption:
        parts.append(CAPTION)
    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"


def compile_tex(tex: Path):
    # Run from the output folder so the crops are found
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name]
    result = subprocess.run(
        cmd, cwd=tex.parent, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        raise SystemExit(f"pdflatex failed on {tex}")


# ----
# main
# ----


def main():
    parser = argparse.ArgumentParser(description="typeset the figure 2 comparison")
    parser.add_argument(
        "--pdf", type=Path, required=True, help="path to the paper's PDF"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--name", default="fig2_comparison", help="output file stem")
    parser.add_argument(
        "--no-caption", action="store_true", help="leave out the caption"
    )
    parser.add_argument(
        "--compile", action="store_true", help="run pdflatex afterwards"
    )
    args = parser.parse_args()

    # Digitized original (only its left-panel x positions are needed)
    orig = pd.read_csv(args.data_dir / "fig2_original_digitized.csv")
    left_x = left_panel_x(orig[orig.panel != "fig3_inset"])
    d = load_results(args.results_dir / RESULTS_FILE)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop_panels(args.pdf, args.out_dir)

    tex = args.out_dir / f"{args.name}.tex"
    tex.write_text(build_tex(d, left_x, args.name, not args.no_caption))
    print("wrote", tex)

    if args.compile:
        compile_tex(tex)
        print("wrote", tex.with_suffix(".pdf"))

    print(f"\ndone. figure files written to {args.out_dir}")


if __name__ == "__main__":
    main()
