# Fashion-MNIST tree and boosting experiments

This folder extends the tree and gradient boosting experiments in Curth et al., *A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning*, to Fashion-MNIST. The reference code is [author commit 76c6ec5](https://github.com/alanjeffares/not-double-descent/tree/76c6ec5d6181380baa4554378bf87d32abaf6606).

## Included files

- `experiments/`: dataset adapter, download script, training drivers and plotting code.
- `upstream/not-double-descent/`: unchanged reference source, with the commit recorded in `SOURCE.json`.
- `results/fashion-mnist/`: complete main CSVs, environment manifest and selected training indices.
- `output/`: three generated figures in PDF, PNG and SVG, plus a numerical summary.
- `logs/`: original main-run logs; the workspace path is replaced with `<WORKSPACE>`.
- `verification/`: independent comparison records and packaging checks.

## Use a separate environment

Run every command below from this folder, after `cd fashion_mnist_extension`. This keeps these package versions separate from the repository's other experiments.

The recorded environment used Python 3.11.18, NumPy 1.26.0, scikit-learn 1.3.2, SciPy 1.11.4, PyTorch 2.1.1 and torchvision 0.16.1. The authors pinned NumPy/PyTorch/torchvision; our additional package versions are recorded in `results/fashion-mnist/manifest.json`.

```sh
python3.11 -m venv .venv-fashion
source .venv-fashion/bin/activate
python -m pip install -r experiments/requirements-reproduction.txt
```

## Redraw the included results

These commands read the included CSVs and require no dataset download or model training:

```sh
python -m experiments.plot_results
python -m experiments.plot_section5
```

The tree and boosting triptychs are written to `output/fashion-mnist/`. The four-panel figure is written to `output/section5-overleaf/figures/`. Matching copies are included at the repository root in `figures/fashion_mnist/` for use in the report. Copy newly generated figures there after a rerun if you want to update those copies.

`plot_rff.py` is included because the shared plotting module imports it. There are no fitted RFF results in this package. Missing RFF and effective-parameter result tables are skipped by the plotting command.

## Small implementation checks

```sh
python -m experiments.run_trees --smoke
python -m experiments.run_boosting --smoke
```

These use small synthetic arrays to check model prefixes, predictions and smoother computations. Full-data comparison records from the original experiment are in `verification/`. Independent refits cover selected settings, not every configuration. The tree verification record also describes auxiliary effective-parameter checks; those result tables are outside the scope of the three main figures included here.

## Rerun the main experiments

The raw image files and fitted model caches are omitted. First download and verify the official Fashion-MNIST files:

```sh
python -m experiments.download_data
python -m experiments.common
python -m experiments.run_trees --phase main --workers 3
python -m experiments.run_boosting --mode main --workers 3
python -m experiments.plot_results
python -m experiments.plot_section5
```

The downloader uses the official [Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist) and checks the four published file checksums. The manifest command records the current environment and selected indices for the rerun. Training writes model checkpoints under `results/fashion-mnist/`; those generated caches are ignored by `.gitignore`.

## Experimental setup

Training uses the first 1,000 examples of each class in original order, concatenated by class, for 10,000 examples. Testing uses all 10,000 official test images. Pixels are divided by 255 and flattened to 784 features. Main labels are uncentered one-hot vectors. This follows the released sampling code; the paper describes random subsampling.

The main tree experiment uses one shared-split multi-output forest per configuration, with no bootstrap, square-root feature subsampling and no depth cap. Leaf caps are 2, 5, 10, 20, 50, 75, 100, 500, 1,000, 2,000 and uncapped. Ensemble sizes are 1, 2, 5, 10, 20, 50 and 100, giving 77 configurations.

Boosting fits ten one-vs-rest regressors per member, with 10-leaf base trees, learning rate 0.85, square-root feature subsampling and zero initialization. Stages are 1, 2, 10, 25, 50, 100 and 200; averaged member counts are 1, 2, 5, 10 and 20, giving 35 configurations. Base seed is 42; boosting member j, starting at one, uses seed 42j.

Smaller forests reuse ordered tree prefixes, and shorter boosting models reuse staged predictions. This reuse was compared with independent refits at selected settings. The underlying preprocessing, random seeds, model settings and evaluated grids are retained.

## Reading the result tables

Select `pred_idx == "all"` for the ten-class results: 77 rows in `trees_main.csv` and 35 rows in `boosting_main.csv`. The remaining rows report individual class components. An empty tree leaf-cap cell means no cap.

- `test_error_mse` is the sum of ten classwise mean squared errors.
- For `pred_idx=all`, `test_error_binary` is the ten-class argmax misclassification fraction, not a percentage.
- Each main experiment uses 10,000 training and 10,000 test examples.

The figures distinguish growth within a model from averaging several models. At 200 boosting stages, squared training error is still 0.0125889617724493. The switch to averaging marks the end of the released stage grid, not measured squared-loss interpolation.

## Scope and runtime

These are new-dataset results on one fixed training subset and one base-seed schedule. Test-curve minima describe these runs and are not validation-based model selections. No fitted RFF result or Great Lakes computation contributes to the included results.

Main-run logs span about 42 seconds for trees and 352 seconds for boosting, including predictions, evaluation and cache writes. Each main experiment used three worker processes; each forest used `n_jobs=1`. Jobs overlapped. These durations are single-run observations of the prefix-reuse implementation, not averaged timings of independently refitting every grid point. The historical manifest records macOS ARM64 but does not record a CPU model or core count.
