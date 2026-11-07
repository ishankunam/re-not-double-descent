"""Complete author tree grids on Fashion-MNIST, with exact prefix reuse.

The main experiment retains the repository's shared-split multi-output forest.
The effective experiment separately fits the paper's centered 0-vs-all model.
Effective norms use the exact leaf-intersection identity, not Monte Carlo:
<s_t(x),s_u(x)> = |leaf_t(x) intersect leaf_u(x)| / (|leaf_t(x)| |leaf_u(x)|).
This avoids allocating n-by-n smoother matrices without changing the models.
Binary errors threshold the forest prediction at zero. The author's dense S@y
can acquire signed roundoff at exact zero; smoke reports these boundary cases.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

from experiments.common import RESULTS, load_data, register_upstream

SEED = 42
N_TREES = 100
TREE_GRID = [1, 2, 5, 10, 20, 50, 100]
LEAF_GRID = [2, 5, 10, 20, 50, 75, 100, 500, 1000, 2000, None]
RF_PARAMS = {"max_features": "sqrt", "max_depth": None, "bootstrap": False}
DATASET = "Fashion-MNIST"
CACHE = RESULTS / "trees_cache"
CACHE_VERSION = 1

MAIN_HEADER = ["seed", "n_estimators", "max_leaf_nodes", "train_error_binary",
               "test_error_binary", "train_error_mse", "test_error_mse",
               "pred_idx", "dataset"]
EFFECTIVE_HEADER = ["pred_idx", "seed", "max_leaf_nodes", "dataset", "n_trees_tot",
                    "n_estimators", "train_error_mse", "test_error_mse",
                    "train_error_binary", "test_error_binary", "eff_p_tr",
                    "l2_train", "l2_test", "l2_train_sq", "l2_test_sq",
                    "p_train", "p_test"]
_DATA = None


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] [trees pid={os.getpid()}] {message}", flush=True)


def _leaf_name(max_leaf_nodes):
    return "max" if max_leaf_nodes is None else str(max_leaf_nodes)


def _array_signature(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _forest(max_leaf_nodes, n_estimators=N_TREES):
    # Parallelism is across leaf configurations. Ordered, single-job tree output
    # accumulation matches the author's default n_jobs=None prediction path.
    return RandomForestRegressor(n_estimators=n_estimators,
                                 max_leaf_nodes=max_leaf_nodes,
                                 random_state=SEED, n_jobs=1, **RF_PARAMS)


def _metadata(mode, max_leaf_nodes, signature):
    return {"cache_version": CACHE_VERSION, "mode": mode, "dataset": DATASET,
            "seed": SEED, "n_estimators": N_TREES,
            "max_leaf_nodes": max_leaf_nodes, "rf_params": RF_PARAMS,
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
            "data_sha256": signature,
            "effective_algorithm": "exact_leaf_intersections_v1",
            "effective_binary_policy": "forest prediction > 0; dense S@y may differ only at signed-zero roundoff"}


def _initialize_data():
    global _DATA
    xtr, ytr, xte, yte = load_data(pred_idx="all", center=False)
    xtr0, ytr0, xte0, yte0 = load_data(0, True)
    if not np.array_equal(xtr, xtr0) or not np.array_equal(xte, xte0):
        raise AssertionError("Main and centered 0-vs-all loaders returned different inputs")
    if not np.array_equal(ytr0, ytr[:, 0] - 0.5):
        raise AssertionError("Centered training labels differ from author loader semantics")
    if not np.array_equal(yte0, yte[:, 0] - 0.5):
        raise AssertionError("Centered test labels differ from author loader semantics")
    if xtr.shape != (10000, 784) or xte.shape != (10000, 784):
        raise AssertionError(f"Full-run shape mismatch: {xtr.shape}, {xte.shape}")
    _DATA = {"xtr": xtr, "xte": xte, "ytr": ytr, "yte": yte,
             "ytr0": ytr0, "yte0": yte0,
             "main_signature": _array_signature(xtr, ytr, xte, yte),
             "effective_signature": _array_signature(xtr, ytr0, xte, yte0)}
    log("Loaded exact author-style 10000/10000 arrays; data signatures computed")


def _fit_or_load(mode, max_leaf_nodes, xtr, ytr, metadata):
    stem = f"{mode}_seed{SEED}_leaves{_leaf_name(max_leaf_nodes)}_n{N_TREES}"
    model_path = CACHE / f"{stem}.joblib"
    if model_path.exists():
        payload = joblib.load(model_path)
        if payload.get("metadata") == metadata:
            log(f"Loaded cached {stem}")
            return payload["forest"], stem
        log(f"Cache metadata changed; refitting {stem}")
    started = time.monotonic()
    forest = _forest(max_leaf_nodes)
    forest.fit(xtr, ytr)
    temporary = model_path.with_name(model_path.name + f".{os.getpid()}.tmp")
    joblib.dump({"metadata": metadata, "forest": forest}, temporary, compress=3)
    temporary.replace(model_path)
    log(f"Fit and cached {stem} in {time.monotonic() - started:.2f}s")
    return forest, stem


def _main_metrics(ytrue, ypred):
    component_mse = np.mean((ytrue - ypred) ** 2, axis=0)
    component_binary = np.mean((ypred > 0.5) != ytrue, axis=0)
    multiclass_binary = np.mean(np.argmax(ypred, axis=1) != np.argmax(ytrue, axis=1))
    return component_mse, component_binary, float(component_mse.sum()), float(multiclass_binary)


def main_prefix_metrics(forest, xtr, ytr, xte, yte, max_leaf_nodes,
                        tree_grid=TREE_GRID):
    """Ordered sums preserve the forest's first-k sequence and metric meanings."""
    train_sum = np.zeros_like(ytr, dtype=np.float64)
    test_sum = np.zeros_like(yte, dtype=np.float64)
    rows, cached_train, cached_test = [], [], []
    requested = set(tree_grid)
    for index, tree in enumerate(forest.estimators_, 1):
        train_sum += tree.predict(xtr)
        test_sum += tree.predict(xte)
        if index not in requested:
            continue
        train_pred = train_sum / index
        test_pred = test_sum / index
        tr_mse, tr_binary, tr_total, tr_error = _main_metrics(ytr, train_pred)
        te_mse, te_binary, te_total, te_error = _main_metrics(yte, test_pred)
        for target in range(ytr.shape[1]):
            rows.append(dict(zip(MAIN_HEADER,
                                 [SEED, index, max_leaf_nodes, float(tr_binary[target]),
                                  float(te_binary[target]), float(tr_mse[target]),
                                  float(te_mse[target]), target, DATASET])))
        rows.append(dict(zip(MAIN_HEADER,
                             [SEED, index, max_leaf_nodes, tr_error, te_error,
                              tr_total, te_total, "all", DATASET])))
        cached_train.append(train_pred.copy())
        cached_test.append(test_pred.copy())
    return rows, {"n_estimators": np.asarray(tree_grid),
                  "train_predictions": np.stack(cached_train),
                  "test_predictions": np.stack(cached_test)}


def _tree_leaves(tree, xtr, xte):
    """Compact leaf ids; bootstrap=False means every leaf has training support."""
    train_nodes = tree.apply(xtr)
    test_nodes = tree.apply(xte)
    leaves, train_ids, counts = np.unique(train_nodes, return_inverse=True, return_counts=True)
    test_ids = np.searchsorted(leaves, test_nodes)
    if np.any(test_ids >= len(leaves)) or not np.array_equal(leaves[test_ids], test_nodes):
        raise AssertionError("A test leaf has no training support")
    return train_ids, test_ids, counts


def _pair_intersections(current, previous):
    """Exact cross-tree row inner products, with bounded auxiliary storage."""
    tr_a, te_a, counts_a = current
    tr_b, te_b, counts_b = previous
    n_b = len(counts_b)
    train_codes = tr_a * n_b + tr_b
    test_codes = te_a * n_b + te_b
    cells = len(counts_a) * n_b
    if cells <= 2_000_000:
        contingency = np.bincount(train_codes, minlength=cells)
        train_intersection = contingency[train_codes]
        test_intersection = contingency[test_codes]
    else:
        # Sparse exact fallback avoids a potentially n_leaves**2 dense table.
        codes, inverse, counts = np.unique(train_codes, return_inverse=True, return_counts=True)
        train_intersection = counts[inverse]
        positions = np.searchsorted(codes, test_codes)
        valid = positions < len(codes)
        valid[valid] &= codes[positions[valid]] == test_codes[valid]
        test_intersection = np.zeros(len(test_codes), dtype=np.int64)
        test_intersection[valid] = counts[positions[valid]]
    train_inner = train_intersection / (counts_a[tr_a] * counts_b[tr_b])
    test_inner = test_intersection / (counts_a[te_a] * counts_b[te_b])
    return train_inner, test_inner


def effective_prefix_metrics(forest, xtr, ytr, xte, yte, max_leaf_nodes,
                             progress=False):
    """All 1..M exact norms via leaf intersections, no row subsampling.

    At prefix k, squared_norm_sum[x] is ||sum_{t<=k} s_t(x)||^2.
    Each new tree adds its self term and twice every earlier cross term.
    Averaging gives ||s_forest(x)||^2 = squared_norm_sum[x] / k**2.
    """
    n_train, n_test = len(xtr), len(xte)
    squared_sum_train = np.zeros(n_train, dtype=np.float64)
    squared_sum_test = np.zeros(n_test, dtype=np.float64)
    pred_sum_train = np.zeros_like(ytr, dtype=np.float64)
    pred_sum_test = np.zeros_like(yte, dtype=np.float64)
    leaf_cache = []
    trace_sum = 0
    rows, cached_train, cached_test = [], [], []
    n_trees = len(forest.estimators_)
    for index, tree in enumerate(forest.estimators_, 1):
        current = _tree_leaves(tree, xtr, xte)
        tr_ids, te_ids, counts = current
        increment_train = 1.0 / counts[tr_ids]
        increment_test = 1.0 / counts[te_ids]
        for previous in leaf_cache:
            cross_train, cross_test = _pair_intersections(current, previous)
            increment_train += 2.0 * cross_train
            increment_test += 2.0 * cross_test
        squared_sum_train += increment_train
        squared_sum_test += increment_test
        leaf_cache.append(current)
        trace_sum += len(counts)
        row_norm_sq_train = squared_sum_train / (index * index)
        row_norm_sq_test = squared_sum_test / (index * index)
        if np.any(row_norm_sq_train < 0) or np.any(row_norm_sq_test < 0):
            raise AssertionError("Negative exact squared smoother norm")
        pred_sum_train += tree.predict(xtr)
        pred_sum_test += tree.predict(xte)
        pred_train = pred_sum_train / index
        pred_test = pred_sum_test / index
        l2_train_sq = float(np.mean(row_norm_sq_train))
        l2_test_sq = float(np.mean(row_norm_sq_test))
        row = dict(zip(EFFECTIVE_HEADER,
                       [0, SEED, max_leaf_nodes, DATASET, n_trees, index,
                        float(np.mean((ytr - pred_train) ** 2)),
                        float(np.mean((yte - pred_test) ** 2)),
                        float(np.mean((pred_train > 0) != (ytr > 0))),
                        float(np.mean((pred_test > 0) != (yte > 0))),
                        float(trace_sum / index),
                        float(np.mean(np.sqrt(row_norm_sq_train))),
                        float(np.mean(np.sqrt(row_norm_sq_test))),
                        l2_train_sq, l2_test_sq,
                        n_train * l2_train_sq, n_train * l2_test_sq]))
        if not (1 - 1e-10 <= row["p_train"] <= n_train + 1e-8):
            raise AssertionError("Training effective parameter count outside averaging bounds")
        if not (1 - 1e-10 <= row["p_test"] <= n_train + 1e-8):
            raise AssertionError("Test effective parameter count outside averaging bounds")
        rows.append(row)
        cached_train.append(pred_train.copy())
        cached_test.append(pred_test.copy())
        if progress and (index == 1 or index % 10 == 0 or index == n_trees):
            log(f"effective0 leaves={_leaf_name(max_leaf_nodes)} prefix={index}/{n_trees} "
                f"p_train={row['p_train']:.6g} p_test={row['p_test']:.6g}")
    return rows, {"n_estimators": np.arange(1, n_trees + 1),
                  "train_predictions": np.stack(cached_train),
                  "test_predictions": np.stack(cached_test)}


def _run_one_mode(mode, max_leaf_nodes):
    if _DATA is None:
        raise RuntimeError("Worker data are not initialized")
    xtr, xte = _DATA["xtr"], _DATA["xte"]
    if mode == "main":
        ytr, yte = _DATA["ytr"], _DATA["yte"]
    else:
        ytr, yte = _DATA["ytr0"], _DATA["yte0"]
    metadata = _metadata(mode, max_leaf_nodes, _DATA[f"{mode}_signature"])
    stem = f"{mode}_seed{SEED}_leaves{_leaf_name(max_leaf_nodes)}_n{N_TREES}"
    row_path, pred_path = CACHE / f"{stem}_metrics.json", CACHE / f"{stem}_predictions.npz"
    model_path = CACHE / f"{stem}.joblib"
    if row_path.exists() and pred_path.exists() and model_path.exists():
        saved = json.loads(row_path.read_text())
        if saved.get("metadata") == metadata:
            log(f"Reusing completed {stem}")
            return saved["rows"]
    forest, stem = _fit_or_load(mode, max_leaf_nodes, xtr, ytr, metadata)
    if mode == "main":
        rows, predictions = main_prefix_metrics(forest, xtr, ytr, xte, yte, max_leaf_nodes)
    else:
        rows, predictions = effective_prefix_metrics(forest, xtr, ytr, xte, yte,
                                                     max_leaf_nodes, progress=True)
    _atomic_npz(pred_path, **predictions)
    _atomic_json(row_path, {"metadata": metadata, "rows": rows})
    log(f"Completed and checkpointed {stem}: {len(rows)} rows")
    return rows


def _run_leaf(max_leaf_nodes, phase):
    started = time.monotonic()
    result = {}
    for mode in (["main", "effective"] if phase == "all" else [phase]):
        result[mode] = _run_one_mode(mode, max_leaf_nodes)
    log(f"Finished leaves={_leaf_name(max_leaf_nodes)} phase={phase} "
        f"in {time.monotonic() - started:.1f}s")
    return max_leaf_nodes, result


def _write_csv(mode, rows_by_leaf):
    header = MAIN_HEADER if mode == "main" else EFFECTIVE_HEADER
    path = RESULTS / ("trees_main.csv" if mode == "main" else "trees_effective.csv")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for leaf in LEAF_GRID:
            for row in rows_by_leaf.get(leaf, []):
                # csv writes None as an empty cell, matching the author's CSV.
                writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    log(f"Flushed {path.name}: {sum(map(len, rows_by_leaf.values()))} rows")


def run_full(workers=3, phase="all"):
    CACHE.mkdir(parents=True, exist_ok=True)
    modes = ["main", "effective"] if phase == "all" else [phase]
    accumulated = {mode: {} for mode in modes}
    log(f"Starting complete tree grids, phase={phase}, workers={workers}; "
        "None/max_leaf_nodes is written as an empty CSV cell")
    if workers == 1:
        _initialize_data()
        for leaf in LEAF_GRID:
            completed_leaf, result = _run_leaf(leaf, phase)
            for mode, rows in result.items():
                accumulated[mode][completed_leaf] = rows
                _write_csv(mode, accumulated[mode])
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_data) as pool:
            pending = {pool.submit(_run_leaf, leaf, phase): leaf for leaf in LEAF_GRID}
            for future in as_completed(pending):
                completed_leaf, result = future.result()
                for mode, rows in result.items():
                    accumulated[mode][completed_leaf] = rows
                    _write_csv(mode, accumulated[mode])
    expected = {"main": len(LEAF_GRID) * len(TREE_GRID) * 11,
                "effective": len(LEAF_GRID) * N_TREES}
    for mode in modes:
        row_count = sum(map(len, accumulated[mode].values()))
        if row_count != expected[mode]:
            raise AssertionError(f"Incomplete {mode} grid: {row_count} != {expected[mode]}")
    log("All requested tree configurations completed")


def run_smoke():
    """Independent synthetic validation; never writes formal result CSVs."""
    register_upstream()
    from src import eff_p_utils, eval_utils
    rng = np.random.RandomState(20261102)
    xtr = rng.normal(size=(90, 13)).astype(np.float32)
    xte = rng.normal(size=(45, 13)).astype(np.float32)
    train_classes = np.arange(90) % 10
    test_classes = np.arange(45) % 10
    # Informative labels need not be natural images to verify exact algorithms.
    xtr[:, 0] += train_classes * 0.25
    xte[:, 0] += test_classes * 0.25
    ytr = np.eye(10)[train_classes]
    yte = np.eye(10)[test_classes]
    centered_train, centered_test = ytr[:, 0] - 0.5, yte[:, 0] - 0.5
    n_trees, small_grid, small_leaves = 7, [1, 2, 5, 7], [2, 5, None]
    checks = {"prefix_forests": 0, "tree_structures": 0,
              "main_metric_rows": 0, "dense_effective_prefixes": 0,
              "sparse_leaf_intersection_cases": 0}
    binary_boundary_differences = []
    started = time.monotonic()
    for leaf in small_leaves:
        forest = _forest(leaf, n_trees)
        forest.fit(xtr, ytr)
        rows, predictions = main_prefix_metrics(forest, xtr, ytr, xte, yte, leaf,
                                                tree_grid=small_grid)
        for position, k in enumerate(small_grid):
            original = RandomForestRegressor(n_estimators=k, max_leaf_nodes=leaf,
                                              random_state=SEED, **RF_PARAMS)
            original.fit(xtr, ytr)
            for cached_tree, original_tree in zip(forest.estimators_[:k], original.estimators_):
                if cached_tree.random_state != original_tree.random_state:
                    raise AssertionError("Prefix estimator seeds differ")
                for attribute in ["children_left", "children_right", "feature", "threshold", "value"]:
                    np.testing.assert_array_equal(getattr(cached_tree.tree_, attribute),
                                                  getattr(original_tree.tree_, attribute))
                checks["tree_structures"] += 1
            np.testing.assert_array_equal(predictions["train_predictions"][position],
                                          original.predict(xtr))
            np.testing.assert_array_equal(predictions["test_predictions"][position],
                                          original.predict(xte))
            for row in [r for r in rows if r["n_estimators"] == k]:
                target = row["pred_idx"]
                for split, true, pred in [("train", ytr, original.predict(xtr)),
                                          ("test", yte, original.predict(xte))]:
                    target_pred = pred if target == "all" else pred[:, target]
                    mse, error = eval_utils.compute_prediction_metrics(true, target_pred,
                                                                       pred_idx=target)
                    np.testing.assert_allclose(row[f"{split}_error_mse"], mse,
                                               rtol=1e-13, atol=1e-14)
                    np.testing.assert_allclose(row[f"{split}_error_binary"], error,
                                               rtol=0, atol=1e-15)
                checks["main_metric_rows"] += 1
            checks["prefix_forests"] += 1
        single = _forest(leaf, n_trees)
        single.fit(xtr, centered_train)
        effective_rows, effective_predictions = effective_prefix_metrics(
            single, xtr, centered_train, xte, centered_test, leaf)
        for k, row in enumerate(effective_rows, 1):
            prefix = copy.copy(single)
            prefix.estimators_ = single.estimators_[:k]
            prefix.n_estimators = k
            s_train, s_test = eff_p_utils.create_S_from_forest(prefix, xtr, xte)
            np.testing.assert_allclose(s_train.sum(axis=1), 1, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(s_test.sum(axis=1), 1, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(s_train @ centered_train, prefix.predict(xtr),
                                       rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(s_test @ centered_train, prefix.predict(xte),
                                       rtol=1e-12, atol=1e-13)
            np.testing.assert_array_equal(effective_predictions["train_predictions"][k - 1],
                                          prefix.predict(xtr))
            np.testing.assert_array_equal(effective_predictions["test_predictions"][k - 1],
                                          prefix.predict(xte))
            dense = eff_p_utils.compute_metrics_from_S(s_train, s_test,
                                                       centered_train, centered_test)
            fields = ["train_error_mse", "test_error_mse", "train_error_binary",
                      "test_error_binary", "eff_p_tr", "l2_train", "l2_test",
                      "l2_train_sq", "l2_test_sq"]
            for field, reference in zip(fields, dense):
                if field.endswith("_binary"):
                    # A mathematically zero centered prediction can become
                    # +/-1e-17 under dense BLAS, changing a strict >0 decision.
                    # Verify every differing decision is only this boundary;
                    # never hide non-boundary classification discrepancies.
                    split = "train" if field.startswith("train") else "test"
                    smooth = s_train if split == "train" else s_test
                    actual = effective_predictions[f"{split}_predictions"][k - 1]
                    dense_pred = smooth @ centered_train
                    decisions_differ = (actual > 0) != (dense_pred > 0)
                    if np.any(decisions_differ):
                        np.testing.assert_allclose(actual[decisions_differ], 0,
                                                   rtol=0, atol=1e-13)
                        np.testing.assert_allclose(dense_pred[decisions_differ], 0,
                                                   rtol=0, atol=1e-13)
                        binary_boundary_differences.append(
                            {"leaf": leaf, "prefix": k, "split": split,
                             "n_decisions": int(np.sum(decisions_differ)),
                             "forest_error": row[field], "dense_error": float(reference)})
                        continue
                np.testing.assert_allclose(row[field], reference, rtol=1e-12, atol=1e-13,
                                           err_msg=f"leaf={leaf} prefix={k} field={field}")
            np.testing.assert_allclose(row["p_train"], len(xtr) * dense[-2],
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(row["p_test"], len(xtr) * dense[-1],
                                       rtol=1e-12, atol=1e-12)
            checks["dense_effective_prefixes"] += 1
    # Exercise the exact sparse-table fallback without a large smoother matrix.
    # Singleton partitions make the independent reference particularly simple.
    sparse_n = 1500
    identity = np.arange(sparse_n)
    permutation = rng.permutation(sparse_n)
    test_a = rng.randint(0, sparse_n, 80)
    test_b = rng.randint(0, sparse_n, 80)
    test_b[:10] = permutation[test_a[:10]]
    sparse_train, sparse_test = _pair_intersections(
        (identity, test_a, np.ones(sparse_n, dtype=np.int64)),
        (permutation, test_b, np.ones(sparse_n, dtype=np.int64)))
    np.testing.assert_array_equal(sparse_train, np.ones(sparse_n))
    np.testing.assert_array_equal(sparse_test, (test_b == permutation[test_a]).astype(float))
    checks["sparse_leaf_intersection_cases"] += 1
    report = {"status": "passed", "scope": "synthetic only; no formal CSV rows written",
              "n_train": len(xtr), "n_test": len(xte), "n_features": xtr.shape[1],
              "seed": SEED, "leaves": small_leaves, "tree_grid": small_grid,
              "checks": checks, "sklearn_version": sklearn.__version__,
              "numpy_version": np.__version__, "elapsed_seconds": time.monotonic() - started,
              "binary_zero_roundoff_cases": binary_boundary_differences,
              "tolerance": "prefix trees and ordered predictions exact; dense MSE/norm/trace rtol=1e-12; all differing binary decisions verified within 1e-13 of zero"}
    CACHE.mkdir(parents=True, exist_ok=True)
    _atomic_json(CACHE / "smoke_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Synthetic validation only; no full data run")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent leaf configurations, each using single-job forests (default: 3)")
    parser.add_argument("--phase", choices=["all", "main", "effective"], default="all",
                        help="Run both grids or one resumable phase")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if arguments.smoke:
        run_smoke()
    else:
        run_full(arguments.workers, arguments.phase)


if __name__ == "__main__":
    main()
