# re-not-double-descent

## Files

| Path | Description |
|---|---|
| `src/run_fig2.py` | The experiment: forests for every leaf cap and ensemble size, one-vs-all or multi-output |
| `src/digitize_fig2.py` | Reads the original figure's curves out of the paper's PDF and calibrates the y-axis |
| `src/analyze_results.py` | Compares the re-run with the original; writes `results/comparison_original_vs_ours.csv` |
| `src/plot_figures.py` | Matplotlib figures in `figures/` |
| `src/typeset_comparison.py` | LaTeX figure: the original panels next to the re-run |
| `src/version_and_prefix_check.py` | Checks across scikit-learn versions and of the first-k-trees shortcut |
| `data/fig2_original_digitized.csv` | The original figure's curves (see [data/README.md](data/README.md)) |
| `results/results_{ova,multi}_seed42.csv` | Main runs, one row per (leaf cap, ensemble size) |
| `results/results_{ova,multi}_seeds0-3.csv` | Seed sensitivity |
| `results/results_ova_randomsubset_seed42.csv` | Random training subset |
| `results/comparison_original_vs_ours.csv` | Re-run vs. the digitized original at 20 points |
| `results/check_sklearn{1.9.1,1.3.2}.txt` | Version check output |
| `figures/fig2_overlay_ova.png` | One-vs-all re-run with the original's digitized vertices on top |
| `figures/fig2_multi.png` | Multi-output re-run |
| `figures/fig2_seeds_ova.png` | Seed 42 with a mean ± 1 sd band over seeds 0-3 |
| `requirements.txt` | Pinned Python packages |

## Setup

Python 3.13 with the pinned packages:

```bash
pip install -r requirements.txt
```

Download MNIST into `data/mnist/` as described in [data/README.md](data/README.md). The LaTeX figure also needs `pdflatex` with the `standalone`, `pgfplots`, `lmodern`, `caption` and `varwidth` packages, and the digitizer and LaTeX figure need the paper's PDF ([OpenReview](https://openreview.net/forum?id=O0Lz8XZT2b)), which is not committed.

## Reproduce

Run from the repository root.

| Step | Command | Time* |
|---|---|---|
| One-vs-all, seed 42 | `python src/run_fig2.py --variant ova --seeds 42 --out results/results_ova_seed42.csv` | 68 s |
| Multi-output, seed 42 | `python src/run_fig2.py --variant multi --seeds 42 --out results/results_multi_seed42.csv` | 14 s |
| One-vs-all, seeds 0-3 | `python src/run_fig2.py --variant ova --seeds 0 1 2 3 --out results/results_ova_seeds0-3.csv` | 5.7 min |
| Multi-output, seeds 0-3 | `python src/run_fig2.py --variant multi --seeds 0 1 2 3 --out results/results_multi_seeds0-3.csv` | 44 s |
| Random training subset | `python src/run_fig2.py --variant ova --seeds 42 --subset random --out results/results_ova_randomsubset_seed42.csv` | 68 s |
| Digitize the original | `python src/digitize_fig2.py --pdf <path to the paper's PDF>` | 2 s |
| Compare | `python src/analyze_results.py` | 1 s |
| Matplotlib figures | `python src/plot_figures.py` | 2 s |
| LaTeX figure | `python src/typeset_comparison.py --pdf <path to the paper's PDF> --compile` | 3 s |
| Version and prefix checks | `python src/version_and_prefix_check.py` | 70 s |

\*Apple M2 Pro, 10 cores, `n_jobs=-1`.

- `--data-dir` points `run_fig2.py` at another data folder (e.g. `data/fashion`); `--leaves` runs only some leaf caps (e.g. `--leaves 200`).
- The LaTeX figure is written to the git-ignored `figures/fig2_comparison/`; add `--no-caption` to use it in a document with its own caption.
- `results/check_sklearn1.3.2.txt` came from Python 3.11 with scikit-learn 1.3.2 and NumPy 1.26.4, which needs its own environment.

## Citation

```bibtex
@inproceedings{curth2023a,
  title     = {A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning},
  author    = {Alicia Curth and Alan Jeffares and Mihaela van der Schaar},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
  year      = {2023},
  url       = {https://openreview.net/forum?id=O0Lz8XZT2b}
}
```

Authors' code: [alanjeffares/not-double-descent](https://github.com/alanjeffares/not-double-descent)
