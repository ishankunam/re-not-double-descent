#!/usr/bin/env python3
"""Exact-grid Fashion-MNIST boosting reproduction; upstream remains untouched.

Default mode runs the 10-target decomposition grid. Effective-parameter modes
are explicit because they require many exact smoother recurrences. A column
block limits memory without sampling any inputs, labels, or grid points.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import joblib
import numpy as np
import sklearn
from joblib import Parallel, delayed, parallel_backend
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

try:
    from .common import RESULTS, atomic_json, load_data, register_upstream
except ImportError:
    from common import RESULTS, atomic_json, load_data, register_upstream

register_upstream()
from src.boosting.main_boost import BOOST_PARAMS

MEMBERS = [1, 2, 5, 10, 20]
STAGES = [1, 2, 10, 25, 50, 100, 200]
SEED = 42
DATASET = "Fashion-MNIST"
CACHE_VERSION = 1
MAIN_HEADER = ["seed", "n_ens", "n_boost", "train_error_binary",
               "test_error_binary", "train_error_mse", "test_error_mse",
               "pred_idx", "dataset"]


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message, flush=True)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp.replace(path)


def atomic_npy(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value)
    tmp.replace(path)


def atomic_csv(path, rows, header):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def fingerprint(xtr, ytr, xte):
    h = hashlib.sha256()
    for value in (xtr, ytr, xte):
        h.update(str((value.shape, str(value.dtype))).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    h.update(json.dumps({"params": BOOST_PARAMS, "seed": SEED,
                         "stages": STAGES, "version": CACHE_VERSION,
                         "sklearn": sklearn.__version__, "numpy": np.__version__},
                        sort_keys=True).encode())
    return h.hexdigest()


def fit_cached(cache, kind, target, member, xtr, ytr, xte, data_id,
               need_leaves=False):
    """Each completed model is its own restart checkpoint."""
    directory = Path(cache) / f"{kind}_target{target}_member{member:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    ready = directory / "ready.json"
    pred_path = directory / "predictions.npz"
    model_path = directory / "model.joblib"
    leaf_paths = [directory / "leaves_train.npy", directory / "leaves_test.npy"]
    if ready.exists():
        info = json.loads(ready.read_text())
        if info["data_id"] != data_id:
            raise RuntimeError(f"Cache fingerprint mismatch at {directory}; use another --cache-dir")
        if pred_path.exists() and model_path.exists() and (
                not need_leaves or all(p.exists() for p in leaf_paths)):
            return str(directory)
    started = time.monotonic()
    log(f"boosting fit {kind} target={target} member={member + 1} seed={SEED * (member + 1)}")
    with threadpool_limits(limits=1):
        if ready.exists() and model_path.exists():
            model = joblib.load(model_path)
        else:
            model = GradientBoostingRegressor(n_estimators=200,
                      random_state=SEED * (member + 1), **BOOST_PARAMS)
            model.fit(xtr, ytr)
            temp_model = model_path.with_name(model_path.name + ".tmp")
            joblib.dump(model, temp_model)
            temp_model.replace(model_path)
        selected = set(STAGES)
        ptr = np.stack([v for t, v in enumerate(model.staged_predict(xtr), 1)
                        if t in selected])
        pte = np.stack([v for t, v in enumerate(model.staged_predict(xte), 1)
                        if t in selected])
        atomic_npz(pred_path, train=ptr, test=pte, stages=np.asarray(STAGES))
        if need_leaves:
            atomic_npy(leaf_paths[0], np.stack([tree[0].apply(xtr)
                                              for tree in model.estimators_]).astype(np.int32))
            atomic_npy(leaf_paths[1], np.stack([tree[0].apply(xte)
                                              for tree in model.estimators_]).astype(np.int32))
    atomic_json(ready, {"data_id": data_id, "kind": kind, "target": target,
                       "member": member + 1, "seed": SEED * (member + 1),
                       "n_boost": 200, "need_leaves": need_leaves,
                       "seconds": time.monotonic() - started})
    log(f"boosting cached {directory.name} in {time.monotonic() - started:.1f}s")
    return str(directory)


def prepare_models(cache, kind, xtr, ytr, xte, workers, need_leaves=False,
                   n_members=20):
    data_id = fingerprint(xtr, ytr, xte)
    targets = range(ytr.shape[1]) if ytr.ndim == 2 else [0]
    tasks = [(target, member) for target in targets for member in range(n_members)]
    with parallel_backend("loky", inner_max_num_threads=1):
        directories = Parallel(n_jobs=workers, max_nbytes="1M", mmap_mode="r",
                          temp_folder=str(Path(cache) / "joblib_tmp"))(
            delayed(fit_cached)(cache, kind, target, member, xtr,
                               ytr[:, target] if ytr.ndim == 2 else ytr,
                               xte, data_id, need_leaves)
            for target, member in tasks)
    return {(target, member): Path(directory)
            for (target, member), directory in zip(tasks, directories)}, data_id


def run_main(args):
    xtr, ytr, xte, yte = load_data("all", False)
    log(f"main grid: {len(xtr)} train / {len(xte)} test; 10 targets, 20 seeded members, 200 stages")
    directories, data_id = prepare_models(args.cache_dir, "main", xtr, ytr,
                                          xte, args.workers)
    predictions = {}
    for key, directory in directories.items():
        with np.load(directory / "predictions.npz") as cached:
            predictions[key] = (cached["train"], cached["test"])
    rows = []
    for count in MEMBERS:
        for tidx, stage in enumerate(STAGES):
            train_targets, test_targets = [], []
            for target in range(10):
                # Same member order/reduction as VotingRegressor.predict.
                ptr = np.average(np.asarray([predictions[target, m][0][tidx]
                                             for m in range(count)]).T, axis=1)
                pte = np.average(np.asarray([predictions[target, m][1][tidx]
                                             for m in range(count)]).T, axis=1)
                train_targets.append(ptr)
                test_targets.append(pte)
                rows.append(dict(zip(MAIN_HEADER, [SEED, count, stage,
                    float(np.mean((ptr > .5) != ytr[:, target])),
                    float(np.mean((pte > .5) != yte[:, target])),
                    float(mean_squared_error(ytr[:, target], ptr)),
                    float(mean_squared_error(yte[:, target], pte)), target, DATASET])))
            ptr, pte = np.asarray(train_targets).T, np.asarray(test_targets).T
            rows.append(dict(zip(MAIN_HEADER, [SEED, count, stage,
                float(np.mean(np.argmax(ptr, axis=1) != np.argmax(ytr, axis=1))),
                float(np.mean(np.argmax(pte, axis=1) != np.argmax(yte, axis=1))),
                float(sum(mean_squared_error(ytr[:, c], ptr[:, c]) for c in range(10))),
                float(sum(mean_squared_error(yte[:, c], pte[:, c]) for c in range(10))),
                "all", DATASET])))
    assert len(rows) == 385
    path = args.results_dir / "boosting_main.csv"
    atomic_csv(path, rows, MAIN_HEADER)
    atomic_json(args.results_dir / "boosting_main_status.json",
                {"state": "complete", "rows": len(rows), "data_id": data_id,
                 "members": MEMBERS, "stages": STAGES, "seed": SEED})
    log(f"wrote {path} ({len(rows)} rows)")


def leaf_info(train_ids, test_ids):
    unique, itr = np.unique(train_ids, return_inverse=True)
    ite = np.searchsorted(unique, test_ids)
    if not np.all(unique[ite] == test_ids):
        raise ValueError("Test leaf has no training member")
    # These are compact leaf indices, not node IDs. For ten-leaf trees uint8
    # suffices; keeping all 10,000 fitted trees' int64 indices wastes gigabytes.
    index_dtype = np.min_scalar_type(len(unique) - 1)
    itr, ite = itr.astype(index_dtype), ite.astype(index_dtype)
    groups = [np.flatnonzero(itr == i).astype(np.int32) for i in range(len(unique))]
    counts = np.asarray([len(g) for g in groups])
    return itr, ite, groups, counts


def update_block(current_train, current_test, info, columns, eta=.85):
    """Original leaf residual correction, restricted to independent columns."""
    itr, ite, groups, counts = info
    correction = np.stack([current_train[g].sum(axis=0) / len(g) for g in groups])
    base = (np.arange(len(groups))[:, None] == itr[columns][None, :]) / counts[:, None]
    delta = eta * (base - correction)
    current_train += delta[itr]
    current_test += delta[ite]


def empty_accumulators(n, m, members, single, ensemble):
    output = {}
    if single:
        output.update(single_qtr=np.zeros((200, n)), single_qte=np.zeros((200, m)),
                      single_ptr=np.zeros((200, n)), single_pte=np.zeros((200, m)),
                      single_trace=np.zeros(200))
    if ensemble:
        shape_tr, shape_te = (7, members, n), (7, members, m)
        for key, shape in [("qtr", shape_tr), ("qte", shape_te),
                           ("sumtr", shape_tr), ("sumte", shape_te),
                           ("ptr", shape_tr), ("pte", shape_te), ("diag", shape_tr)]:
            output["ensemble_" + key] = np.zeros(shape)
    return output


def metrics_from_rows(qtr, qte, ptr, pte, trace, ytr, yte):
    return {"train_error_mse": float(mean_squared_error(ytr, ptr)),
            "test_error_mse": float(mean_squared_error(yte, pte)),
            "train_error_binary": float(np.mean((ptr > 0) != (ytr > 0))),
            "test_error_binary": float(np.mean((pte > 0) != (yte > 0))),
            "eff_p_tr": float(trace),
            "l2_train": float(np.mean(np.sqrt(np.maximum(qtr, 0)))),
            "l2_test": float(np.mean(np.sqrt(np.maximum(qte, 0)))),
            "l2_train_sq": float(np.mean(qtr)), "l2_test_sq": float(np.mean(qte)),
            "p_train": float(np.sum(qtr)), "p_test": float(len(ytr) * np.mean(qte))}


def write_effective_outputs(args, accum, ytr, yte, members, single, ensemble):
    shared = {"pred_idx": 0, "seed": SEED, "dataset": DATASET,
              "target_encoding": "one_hot_minus_0.5", "computation": "exact_column_blocks",
              "binary_threshold": "strict_score_gt_0",
              "numerical_note": "block_reduction_can_change_sign_of_roundoff_at_exact_zero"}
    if single:
        rows = []
        for i in range(200):
            rows.append({**shared, "max_leaf_nodes": 10, "lr": .85,
                "n_boost_tot": 200, "n_boost": i + 1,
                "effective_variant": "actual_single_zero_init",
                **metrics_from_rows(accum["single_qtr"][i], accum["single_qte"][i],
                                    accum["single_ptr"][i], accum["single_pte"][i],
                                    accum["single_trace"][i], ytr, yte)})
        atomic_csv(args.results_dir / "boosting_effective_single.csv", rows, list(rows[0]))
    if ensemble:
        rows = []
        for i, stage in enumerate(STAGES):
            for j in range(members):
                sums_tr, sums_te = accum["ensemble_sumtr"][i, j], accum["ensemble_sumte"][i, j]
                rawptr, rawpte = accum["ensemble_ptr"][i, j], accum["ensemble_pte"][i, j]
                qtr, qte = accum["ensemble_qtr"][i, j], accum["ensemble_qte"][i, j]
                diag = accum["ensemble_diag"][i, j]
                original = metrics_from_rows(qtr / sums_tr**2, qte / sums_te**2,
                    rawptr / sums_tr, rawpte / sums_te, np.sum(diag / sums_tr), ytr, yte)
                actual = metrics_from_rows(qtr / (j + 1)**2, qte / (j + 1)**2,
                    rawptr / (j + 1), rawpte / (j + 1), np.sum(diag) / (j + 1), ytr, yte)
                rows.append({**shared, "n_boost": stage, "n_ens_tot": members,
                    "n_ens": j + 1, "effective_variant": "upstream_row_normalized",
                    "row_sum_before_member_average": float(np.mean(sums_tr) / (j + 1)),
                    **original, **{"actual_average_" + k: v for k, v in actual.items()}})
        atomic_csv(args.results_dir / "boosting_effective_ensemble.csv", rows, list(rows[0]))


_EFFECTIVE_INFO_CACHE = {}


def load_effective_infos(directories):
    """Build compact leaf groups once per worker, never pickle large groups."""
    key = tuple(str(Path(directory).resolve()) for directory in directories)
    if key not in _EFFECTIVE_INFO_CACHE:
        arrays = [(np.load(Path(d) / "leaves_train.npy", mmap_mode="r"),
                   np.load(Path(d) / "leaves_test.npy", mmap_mode="r")) for d in directories]
        infos = [[leaf_info(a[t], b[t]) for t in range(len(a))] for a, b in arrays]
        _EFFECTIVE_INFO_CACHE.clear()
        _EFFECTIVE_INFO_CACHE[key] = infos
    return _EFFECTIVE_INFO_CACHE[key]


def accumulate_effective_block(accum, infos, ytr, yte, start, stop, single, ensemble):
    """One independent exact column block; no input arrays are modified."""
    n, m, members = len(ytr), len(yte), len(infos)
    columns = np.arange(start, stop)
    width = len(columns)
    total_tr = np.zeros((7, n, width)) if ensemble else None
    total_te = np.zeros((7, m, width)) if ensemble else None
    stage_index = {t: i for i, t in enumerate(STAGES)}
    for member in range(members):
        current_tr, current_te = np.zeros((n, width)), np.zeros((m, width))
        for t in range(1, len(infos[member]) + 1):
            update_block(current_tr, current_te, infos[member][t - 1], columns)
            if single and member == 0:
                idx = t - 1
                accum["single_qtr"][idx] += np.einsum("ij,ij->i", current_tr, current_tr)
                accum["single_qte"][idx] += np.einsum("ij,ij->i", current_te, current_te)
                accum["single_ptr"][idx] += current_tr @ ytr[columns]
                accum["single_pte"][idx] += current_te @ ytr[columns]
                accum["single_trace"][idx] += np.sum(current_tr[columns, np.arange(width)])
            if ensemble and t in stage_index:
                idx = stage_index[t]
                total_tr[idx] += current_tr
                total_te[idx] += current_te
                tr, te = total_tr[idx], total_te[idx]
                accum["ensemble_qtr"][idx, member] += np.einsum("ij,ij->i", tr, tr)
                accum["ensemble_qte"][idx, member] += np.einsum("ij,ij->i", te, te)
                accum["ensemble_sumtr"][idx, member] += np.sum(tr, axis=1)
                accum["ensemble_sumte"][idx, member] += np.sum(te, axis=1)
                accum["ensemble_ptr"][idx, member] += tr @ ytr[columns]
                accum["ensemble_pte"][idx, member] += te @ ytr[columns]
                accum["ensemble_diag"][idx, member, columns] += tr[columns, np.arange(width)]
        if (member + 1) % 5 == 0:
            log(f"effective block {start}:{stop}: completed member {member + 1}/{members}")


def effective_block_task(directories, ytr, yte, start, stop, single, ensemble, spec, path):
    """File checkpoint is the result: avoid transferring giant arrays via IPC."""
    path = Path(path)
    if path.exists():
        with np.load(path) as saved:
            if (str(saved["spec"].item()) != spec or int(saved["start"]) != start
                    or int(saved["stop"]) != stop):
                raise RuntimeError(f"Effective block cache mismatch: {path}")
        log(f"using completed effective block {start}:{stop}")
        return str(path)
    began = time.monotonic()
    log(f"parallel effective block {start}:{stop}/{len(ytr)} starting")
    infos = load_effective_infos(directories)
    delta = empty_accumulators(len(ytr), len(yte), len(directories), single, ensemble)
    with threadpool_limits(limits=1):
        accumulate_effective_block(delta, infos, ytr, yte, start, stop, single, ensemble)
    atomic_npz(path, spec=np.asarray(spec), start=np.asarray(start), stop=np.asarray(stop), **delta)
    log(f"parallel effective block {start}:{stop} cached in {time.monotonic() - began:.1f}s")
    return str(path)


def merge_effective_block(accum, path, spec, expected_start):
    with np.load(path) as saved:
        if str(saved["spec"].item()) != spec or int(saved["start"]) != expected_start:
            raise RuntimeError("Effective blocks must be merged in original start-column order")
        for key in accum:
            accum[key] += saved[key]
        return int(saved["stop"])


def run_effective(args):
    single = args.mode in ("effective-single", "effective-all", "all")
    ensemble = args.mode in ("effective-ensemble", "effective-all", "all")
    members = 50 if ensemble else 1
    xtr, ytr, xte, yte = load_data(0, True)
    directories, data_id = prepare_models(args.cache_dir, "centered", xtr, ytr,
                         xte, args.workers, need_leaves=True, n_members=members)
    directory_list = [directories[0, member] for member in range(members)]
    n, m = len(xtr), len(xte)
    state_name = "both" if single and ensemble else ("single" if single else "ensemble")
    checkpoint = args.cache_dir / f"effective_{state_name}_accumulator.npz"
    spec = json.dumps({"data_id": data_id, "single": single, "ensemble": ensemble,
                       "members": members, "version": CACHE_VERSION}, sort_keys=True)
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if str(saved["spec"].item()) != spec:
                raise RuntimeError("Effective checkpoint configuration mismatch")
            next_col = int(saved["next_col"])
            accum = {key: saved[key] for key in saved.files if key not in ("spec", "next_col")}
        log(f"resuming exact effective computation at training-label column {next_col}/{n}")
    else:
        next_col = 0
        accum = empty_accumulators(n, m, members, single, ensemble)
    effective_workers = getattr(args, "effective_workers", 1)

    def commit(stop, seconds):
        atomic_npz(checkpoint, spec=np.asarray(spec), next_col=np.asarray(stop), **accum)
        atomic_json(args.results_dir / f"boosting_effective_{state_name}_status.json",
                    {"state": "running" if stop < n else "complete", "next_col": stop,
                     "n_columns": n, "block_size": args.block_size, "data_id": data_id,
                     "effective_workers": effective_workers, "last_block_seconds": seconds})
        log(f"effective checkpoint {stop}/{n}; committed in column order")

    if effective_workers == 1:
        infos = load_effective_infos(directory_list)
        for start in range(next_col, n, args.block_size):
            stop = min(n, start + args.block_size)
            began = time.monotonic()
            log(f"effective block columns {start}:{stop}/{n}; {members} members")
            with threadpool_limits(limits=1):
                accumulate_effective_block(accum, infos, ytr, yte, start, stop, single, ensemble)
            commit(stop, time.monotonic() - began)
    elif next_col < n:
        block_dir = args.cache_dir / f"effective_{state_name}_blocks" / hashlib.sha256(spec.encode()).hexdigest()[:16]
        block_dir.mkdir(parents=True, exist_ok=True)
        starts = range(next_col, n, args.block_size)
        log(f"parallel effective computation: {effective_workers} workers, ordered checkpoint merge")
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=effective_workers, return_as="generator", batch_size=1,
                               pre_dispatch=effective_workers)(
                delayed(effective_block_task)(directory_list, ytr, yte, start,
                    min(n, start + args.block_size), single, ensemble, spec,
                    block_dir / f"{start:08d}-{min(n, start + args.block_size):08d}.npz")
                for start in starts)
            expected_start = next_col
            for path in results:
                began = time.monotonic()
                stop = merge_effective_block(accum, path, spec, expected_start)
                commit(stop, time.monotonic() - began)
                expected_start = stop
                # Commit precedes deletion. Unmerged block files survive a crash;
                # merged files need not consume many gigabytes after a long run.
                Path(path).unlink()
    write_effective_outputs(args, accum, ytr, yte, members, single, ensemble)
    log("effective CSV output complete")


def smoke_parallel_blocks():
    """Resume a serial prefix using parallel block deltas, bit for bit."""
    import tempfile
    from types import SimpleNamespace
    rng = np.random.RandomState(983)
    xtr, xte = rng.normal(size=(13, 4)), rng.normal(size=(9, 4))
    ytr = (rng.randint(2, size=len(xtr)) - .5).astype(float)
    yte = (rng.randint(2, size=len(xte)) - .5).astype(float)
    with tempfile.TemporaryDirectory(prefix="boosting-parallel-smoke-") as temp:
        root = Path(temp)
        directories = []
        for member in range(3):
            directory = root / f"member{member}"
            directory.mkdir()
            model = GradientBoostingRegressor(n_estimators=200,
                        random_state=SEED * (member + 1), **BOOST_PARAMS).fit(xtr, ytr)
            atomic_npy(directory / "leaves_train.npy", np.stack([
                tree[0].apply(xtr) for tree in model.estimators_]).astype(np.int32))
            atomic_npy(directory / "leaves_test.npy", np.stack([
                tree[0].apply(xte) for tree in model.estimators_]).astype(np.int32))
            directories.append(directory)
        infos = load_effective_infos(directories)
        serial = empty_accumulators(len(ytr), len(yte), 3, True, True)
        prefix = empty_accumulators(len(ytr), len(yte), 3, True, True)
        with threadpool_limits(limits=1):
            for start in (0, 5, 10):
                accumulate_effective_block(serial, infos, ytr, yte, start,
                                           min(start + 5, len(ytr)), True, True)
            accumulate_effective_block(prefix, infos, ytr, yte, 0, 5, True, True)
        spec = "parallel_smoke_serial_checkpoint_v1"
        checkpoint = root / "old_serial_checkpoint.npz"
        atomic_npz(checkpoint, spec=np.asarray(spec), next_col=np.asarray(5), **prefix)
        with np.load(checkpoint) as saved:
            resumed = {key: saved[key] for key in prefix}
            next_col = int(saved["next_col"])
        with parallel_backend("loky", inner_max_num_threads=1):
            paths = Parallel(n_jobs=2, batch_size=1)(delayed(effective_block_task)(
                directories, ytr, yte, start, min(start + 5, len(ytr)), True, True,
                spec, root / f"block{start}.npz") for start in (5, 10))
        for path in paths:
            next_col = merge_effective_block(resumed, path, spec, next_col)
            atomic_npz(checkpoint, spec=np.asarray(spec), next_col=np.asarray(next_col), **resumed)
        assert next_col == len(ytr)
        for key in serial:
            np.testing.assert_array_equal(serial[key], resumed[key], err_msg=key)
        # Reusing a finished block must be a read-only cache hit.
        assert effective_block_task(directories, ytr, yte, 5, 10, True, True,
                                    spec, root / "block5.npz") == str(root / "block5.npz")
        with np.load(checkpoint) as saved:
            for key in serial:
                np.testing.assert_array_equal(serial[key], saved[key])
        for name, accum in (("serial", serial), ("parallel", resumed)):
            output = root / name
            output.mkdir()
            write_effective_outputs(SimpleNamespace(results_dir=output), accum,
                                    ytr, yte, 3, True, True)
        for filename in ("boosting_effective_single.csv", "boosting_effective_ensemble.csv"):
            assert (root / "serial" / filename).read_bytes() == (root / "parallel" / filename).read_bytes()
    _EFFECTIVE_INFO_CACHE.clear()
    return True


def smoke(args):
    """Small deterministic verification; does not touch production checkpoints."""
    from src import eff_p_utils
    from sklearn.ensemble import VotingRegressor
    rng = np.random.RandomState(123)
    xtr, xte = rng.normal(size=(41, 6)), rng.normal(size=(17, 6))
    ytr = (rng.randint(0, 2, len(xtr)) - .5).astype(float)
    yte = (rng.randint(0, 2, len(xte)) - .5).astype(float)
    models = [GradientBoostingRegressor(n_estimators=7,
                  random_state=SEED * (member + 1), **BOOST_PARAMS).fit(xtr, ytr)
              for member in range(3)]
    exact_prefix = True
    for member, model in enumerate(models):
        staged = list(model.staged_predict(xte))
        for count in (1, 2, 4, 7):
            short = GradientBoostingRegressor(n_estimators=count,
                         random_state=SEED * (member + 1), **BOOST_PARAMS).fit(xtr, ytr)
            np.testing.assert_array_equal(short.predict(xte), staged[count - 1])
            for a, b in zip(short.estimators_, model.estimators_[:count]):
                np.testing.assert_array_equal(a[0].tree_.threshold, b[0].tree_.threshold)
                np.testing.assert_array_equal(a[0].tree_.value, b[0].tree_.value)
    max_difference = 0.0
    all_dense = []
    for model in models:
        dense_tr, dense_te = eff_p_utils.create_S_from_gbtregressor(model, xtr, xte)
        blocked_tr, blocked_te = np.zeros_like(dense_tr), np.zeros_like(dense_te)
        infos = [leaf_info(tree[0].apply(xtr), tree[0].apply(xte)) for tree in model.estimators_]
        for start in range(0, len(xtr), 9):
            columns = np.arange(start, min(start + 9, len(xtr)))
            tr, te = np.zeros((len(xtr), len(columns))), np.zeros((len(xte), len(columns)))
            for info in infos:
                update_block(tr, te, info, columns)
            blocked_tr[:, columns], blocked_te[:, columns] = tr, te
        np.testing.assert_allclose(blocked_tr, dense_tr, atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose(blocked_te, dense_te, atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose(dense_tr @ ytr, model.predict(xtr), atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose(dense_te @ ytr, model.predict(xte), atol=2e-14, rtol=2e-13)
        max_difference = max(max_difference, float(np.max(np.abs(blocked_tr - dense_tr))),
                             float(np.max(np.abs(blocked_te - dense_te))))
        all_dense.append((dense_tr, dense_te))
    for members in (1, 2, 3):
        raw_tr = sum(pair[0] for pair in all_dense[:members])
        raw_te = sum(pair[1] for pair in all_dense[:members])
        avg_prediction = np.average(np.asarray([model.predict(xte) for model in models[:members]]).T, axis=1)
        voter = VotingRegressor([(f"est_{member}", GradientBoostingRegressor(
                    n_estimators=7, random_state=SEED * (member + 1), **BOOST_PARAMS))
                    for member in range(members)]).fit(xtr, ytr)
        np.testing.assert_array_equal(voter.predict(xte), avg_prediction)
        np.testing.assert_allclose(raw_te @ ytr / members, avg_prediction, atol=2e-14)
        normalized_tr, normalized_te = raw_tr / raw_tr.sum(axis=1)[:, None], raw_te / raw_te.sum(axis=1)[:, None]
        dense_metrics = eff_p_utils.compute_metrics_from_S(normalized_tr, normalized_te, ytr, yte)
        ours = metrics_from_rows(np.sum(normalized_tr**2, axis=1), np.sum(normalized_te**2, axis=1),
                   normalized_tr @ ytr, normalized_te @ ytr, np.trace(normalized_tr), ytr, yte)
        ordered = [ours[k] for k in ("train_error_mse", "test_error_mse", "train_error_binary",
                    "test_error_binary", "eff_p_tr", "l2_train", "l2_test", "l2_train_sq", "l2_test_sq")]
        np.testing.assert_allclose(ordered, dense_metrics, atol=2e-14)
    first = GradientBoostingRegressor(n_estimators=1, random_state=SEED, **BOOST_PARAMS).fit(xtr, ytr)
    first_s, _ = eff_p_utils.create_S_from_gbtregressor(first, xtr, xte)
    first_normalized = first_s / first_s.sum(axis=1)[:, None]
    ratio = float(np.sum(first_normalized**2) / np.sum(first_s**2))
    np.testing.assert_allclose(ratio, 1 / .85**2, atol=1e-14)
    parallel_resume = smoke_parallel_blocks()
    report = {"status": "passed", "exact_standalone_stage_prefixes": exact_prefix,
              "exact_voting_member_prefixes": True,
              "parallel_block_serial_checkpoint_resume_bitwise_equal": parallel_resume,
              "dense_block_max_absolute_difference": max_difference,
              "first_stage_upstream_norm_inflation": ratio,
              "sklearn": sklearn.__version__, "numpy": np.__version__,
              "train_samples": len(xtr), "test_samples": len(xte), "members": 3,
              "stage_prefixes_checked": [1, 2, 4, 7]}
    atomic_json(args.results_dir / "boosting_smoke.json", report)
    log(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["main", "effective-single", "effective-ensemble", "effective-all", "all"], default="main")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--effective-workers", type=int, default=1,
                        help="parallel exact column blocks; existing checkpoints remain compatible")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.effective_workers < 1 or args.block_size < 1:
        parser.error("workers and block-size must be positive")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = args.cache_dir or (args.results_dir / "boosting_cache")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke(args)
        return
    if args.mode in ("main", "all"):
        run_main(args)
    if args.mode != "main":
        run_effective(args)


if __name__ == "__main__":
    main()
