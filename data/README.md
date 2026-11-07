# Data

## Files

| Path | Description |
|---|---|
| `fig2_original_digitized.csv` | Figure 2 of Curth et al. (2023), one row per curve vertex (committed) |
| `mnist/` | The four decompressed MNIST IDX files (downloaded, git-ignored) |
| `fashion/` | Fashion-MNIST in the same format (optional, downloaded, git-ignored) |

Columns of `fig2_original_digitized.csv`:

| Column | Meaning |
|---|---|
| `panel` | `left`, `center`, `right`, or `fig3_inset` (the Figure 3 inset, used only to calibrate the y-axis) |
| `series` | Curve label, e.g. `P_ens=1 (blue)` or `P_leaf=500` |
| `p_leaf`, `p_ens` | Grid point the vertex sits on (`max` is the paper's largest leaf cap) |
| `x_px`, `y_px` | Vertex position on page 3 of the PDF, in points |
| `err_fig3cal` | Squared test error under the Figure 3 inset calibration (the values used everywhere) |
| `err_if_bottom_tick_0.2` | The same vertex read with Figure 2's printed tick labels, kept for comparison |

## Setup

`src/run_fig2.py` reads MNIST from `data/mnist/` by default:

```
data/mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

Download and decompress them from the [CVDF mirror](https://github.com/cvdfoundation/mnist#download), from the repository root:

```bash
mkdir -p data/mnist && cd data/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -LO "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz" && gunzip "$f.gz"
done
```

For Fashion-MNIST, put the same four files from [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist#get-the-data) in `data/fashion/` and pass `--data-dir data/fashion` to `src/run_fig2.py`.

## Reproduce

Regenerate `fig2_original_digitized.csv` from the paper's PDF ([OpenReview](https://openreview.net/forum?id=O0Lz8XZT2b), not committed):

```bash
python src/digitize_fig2.py --pdf <path to the paper's PDF>
```

## Citation

```bibtex
@article{lecun1998gradient,
  title   = {Gradient-based learning applied to document recognition},
  author  = {LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal = {Proceedings of the IEEE},
  volume  = {86},
  number  = {11},
  pages   = {2278--2324},
  year    = {1998}
}

@article{xiao2017fashion,
  title   = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  author  = {Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
  journal = {arXiv preprint arXiv:1708.07747},
  year    = {2017}
}
```
