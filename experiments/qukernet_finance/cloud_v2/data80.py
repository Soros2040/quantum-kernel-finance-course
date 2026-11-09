"""Prepare private 80-to-40-feature snapshots; no cloud calls or model execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from .features80 import FEATURES, FEATURE_DICTIONARY, build_features
except ImportError:
    from features80 import FEATURES, FEATURE_DICTIONARY, build_features

QUARTERS = ("2022Q1", "2023Q1", "2024Q1")
SIZES = (16, 32, 150)
DATA_SEED = 20261110


def _network(path):
    if str(path).startswith(("\\\\", "//")):
        return True
    p = Path(path)
    if os.name == "nt" and p.drive:
        import ctypes
        return ctypes.windll.kernel32.GetDriveTypeW(p.anchor) == 4
    return False


def local_path(value):
    if _network(value):
        raise ValueError("Network paths are not admitted")
    path = Path(value).resolve()
    if _network(path):
        raise ValueError("Resolved path is not local")
    return path


def file_identity(path):
    path = local_path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed during hashing")
    return {"bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_inputs(panel, calendar):
    identities, paths = {}, {}
    if isinstance(panel, pd.DataFrame):
        raw = panel.copy(deep=True)
    else:
        paths["panel"] = local_path(panel)
        identities["panel"] = file_identity(paths["panel"])
        raw = pd.read_csv(paths["panel"], dtype={"symbol": str})
    if isinstance(calendar, pd.DataFrame):
        dates = calendar["date"]
    elif isinstance(calendar, (str, os.PathLike)):
        paths["calendar"] = local_path(calendar)
        identities["calendar"] = file_identity(paths["calendar"])
        dates = pd.read_csv(paths["calendar"])["date"]
    else:
        dates = calendar
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        raise ValueError("Missing required columns: "+", ".join(sorted(required-set(raw.columns))))
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    cal = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
    if raw.date.isna().any() or cal.hasnans or raw.date.dt.tz is not None or cal.tz is not None:
        raise ValueError("Dates must be nonmissing, timezone-naive daily market dates")
    if not raw.date.eq(raw.date.dt.normalize()).all() or not cal.equals(cal.normalize()):
        raise ValueError("Daily dates must be normalized")
    if not raw.date.dt.year.between(2019, 2024).all() or not ((cal.year >= 2019) & (cal.year <= 2024)).all():
        raise ValueError("Only frozen 2019-2024 inputs are admitted; reject 2025")
    if cal.has_duplicates or raw.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate calendar dates or security-date keys")
    if raw.symbol.isna().any() or raw.symbol.astype(str).str.strip().eq("").any():
        raise ValueError("Missing security identifiers")
    raw["symbol"] = raw.symbol.astype(str)
    if raw.symbol.nunique() != 20:
        raise ValueError("The fixed protocol requires exactly 20 securities")
    cal = cal.sort_values()
    if not raw.date.isin(cal).all():
        raise ValueError("Input contains dates absent from market calendar")
    for key, path in paths.items():
        if file_identity(path) != identities[key]:
            raise RuntimeError("Input changed while reading: "+key)
    return raw, cal, identities


def _discretize(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Quantile bins require finite nonempty values")
    edges = np.unique(np.quantile(values, np.arange(1, 10)/10))
    edges = edges[(edges > values.min()) & (edges < values.max())]
    return np.searchsorted(edges, values, side="right"), edges.tolist()


def _mi(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    counts = np.zeros((int(a.max())+1, int(b.max())+1), dtype=float)
    np.add.at(counts, (a, b), 1.)
    joint = counts/counts.sum()
    product = joint.sum(axis=0)[None, :]*joint.sum(axis=1)[:, None]
    use = joint > 0
    return float(np.sum(joint[use]*np.log(joint[use]/product[use])))


def mrmr_select(x, y, names=FEATURES, count=40, task="regression"):
    """Remove train-constant/exact-duplicate columns, then empirical MI mRMR.

    Nonconstant means a strictly positive observed range (no unit-dependent
    absolute threshold). Duplicate means exact elementwise equality on T only.
    """
    x, y = np.asarray(x, float), np.asarray(y)
    if x.shape != (len(y), len(names)) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("mRMR requires finite aligned training arrays")
    kept, removed = [], []
    for j, name in enumerate(names):
        if np.ptp(x[:, j]) == 0.:
            removed.append({"feature": name, "reason": "constant_on_training"})
            continue
        duplicate = next((k for k in kept if np.array_equal(x[:, j], x[:, k])), None)
        if duplicate is not None:
            removed.append({"feature": name, "reason": "exact_duplicate_on_training", "of": names[duplicate]})
        else:
            kept.append(j)
    if len(kept) < count:
        raise ValueError(f"Only {len(kept)} nonconstant, nonduplicate training features; require {count}")
    discrete, edges = {}, {}
    for j in kept:
        discrete[j], edges[names[j]] = _discretize(x[:, j])
    if task == "classification":
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Classification training data must contain both labels")
        target, target_edges = y.astype(int), None
    elif task == "regression":
        if np.ptp(y) == 0.:
            raise ValueError("Constant regression target")
        target, target_edges = _discretize(y)
    else:
        raise ValueError("Unknown task")
    relevance = {j: _mi(discrete[j], target) for j in kept}
    redundancy = {(j, k): _mi(discrete[j], discrete[k]) for j in kept for k in kept if j <= k}
    selected, steps = [], []
    for _ in range(count):
        scores = {j: relevance[j]-(np.mean([redundancy[min(j, k), max(j, k)] for k in selected]) if selected else 0.) for j in kept if j not in selected}
        best = max(scores.values())
        choice = next(j for j in kept if j in scores and scores[j] >= best-1e-12)
        selected.append(choice)
        steps.append({"feature": names[choice], "score_nats": float(scores[choice])})
    return selected, {"method": "10-quantile-bin empirical MI relevance minus mean redundancy", "fit_rows": len(y),
                      "fit_split": "controlled training only", "eligible_count": len(kept), "removed": removed,
                      "target_kind": task, "target_bin_edges": target_edges, "feature_bin_edges": edges,
                      "relevance_nats": {names[j]: relevance[j] for j in kept}, "steps": steps,
                      "tie_rule": "declared feature order within 1e-12 nats"}


def fit_scaler(x):
    x = np.asarray(x, float)
    median = np.median(x, axis=0)
    iqr = np.quantile(x, .75, axis=0)-np.quantile(x, .25, axis=0)
    return {"median": median.tolist(), "iqr": iqr.tolist(), "fit_rows": len(x),
            "zero_iqr_columns": np.flatnonzero(iqr == 0.).tolist(),
            "formula": "pi*clip((x-median)/IQR,-3,3)/3; zero-IQR columns -> 0", "angle_range": [-float(np.pi), float(np.pi)]}


def apply_scaler(x, scaler):
    x = np.asarray(x, float)
    median, iqr = np.asarray(scaler["median"]), np.asarray(scaler["iqr"])
    varying = iqr > 0.
    result = np.clip((x-median)/np.where(varying, iqr, 1.), -3., 3.)*np.pi/3.
    result[:, ~varying] = 0.
    return result


def balanced_sample(pool, days, size, seed):
    bins, per_date = {16: (8, 2), 32: (16, 2), 150: (15, 10)}[size]
    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(sorted(pool.symbol.unique()), 0)
    chosen_indices, trace = [], []
    for number, box in enumerate(np.array_split(days, bins)):
        part = pool[pool.j.isin(box)]
        eligible_dates = sorted(int(j) for j, rows in part.groupby("j") if len(rows) >= per_date)
        if not eligible_dates:
            raise ValueError(f"Date bin {number} lacks {per_date} feature-valid securities")
        j = eligible_dates[len(eligible_dates)//2]
        cross = part[part.j == j].sort_values("symbol").copy()
        cross["_exposure"], cross["_tie"] = cross.symbol.map(counts), rng.random(len(cross))
        chosen = cross.sort_values(["_exposure", "_tie", "symbol"]).head(per_date)
        chosen_indices.extend(chosen.index.tolist())
        for symbol in chosen.symbol:
            counts[symbol] += 1
        trace.append({"bin": number, "first_j": int(box[0]), "last_j": int(box[-1]), "chosen_j": j,
                      "eligible_dates": len(eligible_dates), "selected_symbols": chosen.symbol.tolist()})
    result = pool.loc[chosen_indices].sort_values(["j", "symbol"]).copy()
    if len(result) != size or result.duplicated(["symbol", "date"]).any():
        raise RuntimeError("Balanced sample size or uniqueness failure")
    # Selection has already finished without accessing target availability/value.
    if not result.target_valid.all():
        raise ValueError("Selected feature-valid sample has missing matured targets; no label-based substitution")
    return result, {"seed": int(seed), "date_bins": trace, "security_counts": counts,
                    "date_rule": "upper-middle eligible date in each equal calendar bin; labels not consulted",
                    "security_rule": "least previous exposure, seeded random tie, stable identifier"}


def _metadata(rows, calendar):
    return [{"symbol": str(row.symbol), "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
             "label_start": calendar[int(row.label_start_j)].strftime("%Y-%m-%d"),
             "label_end": calendar[int(row.label_end_j)].strftime("%Y-%m-%d")} for row in rows.itertuples()]


def prepare_window(panel, calendar, quarter, train_size=32, task="regression", seed=DATA_SEED, *, _prepared=None):
    """Return immutable arrays plus metadata for one size/task/quarter.

    T/V use the same requested size. O includes every eligible five-session
    decision date whose t+6 endpoint remains in the quarter. D0 label:
    open[t+6]/open[t+1]-1; classification is strictly positive -> 1.
    """
    start = time.perf_counter()
    if quarter not in QUARTERS or train_size not in SIZES or task not in ("regression", "classification"):
        raise ValueError("Unsupported quarter, sample size or task")
    if _prepared is None:
        raw, cal, identities = load_inputs(panel, calendar)
        frame = build_features(raw, cal)
    else:
        raw, cal, identities, frame = _prepared
    period = pd.Period(quarter, freq="Q")
    outer = np.flatnonzero(cal.to_period("Q") == period)
    if not len(outer) or outer[0] < 252:
        raise ValueError("Insufficient market-calendar history")
    cutoff = int(outer[0])-1
    prior = np.arange(cutoff-251, cutoff+1)
    t_days, v_days = prior[:168], prior[168:]
    t_pool = frame[frame.j.isin(t_days) & (frame.label_end_j < v_days[0]) & frame.feature_valid]
    v_pool = frame[frame.j.isin(v_days) & (frame.label_end_j <= cutoff) & frame.feature_valid]
    # Partition the deterministic maturity-purged calendar, not the original
    # block: otherwise the final 5-day V bin would lie entirely in the 6-day
    # embargo for size 32. This depends on no observed target values.
    t_mature_days = t_days[t_days+6 < v_days[0]]
    v_mature_days = v_days[v_days+6 <= cutoff]
    train, t_sampling = balanced_sample(t_pool, t_mature_days, train_size, seed)
    valid, v_sampling = balanced_sample(v_pool, v_mature_days, train_size, seed+1)
    decision_days = [int(j) for j in outer[::5] if j+6 < len(cal) and cal[j+6].to_period("Q") == period]
    test_grid = frame[frame.j.isin(decision_days)]
    test = test_grid[test_grid.feature_valid & test_grid.target_valid].sort_values(["j", "symbol"]).copy()
    if test.empty:
        raise ValueError("No valid outer evaluation records")
    if not (train.label_end_j.max() < v_days[0] and valid.label_end_j.max() <= cutoff < test.j.min()):
        raise RuntimeError("Maturity purge boundary violated")
    if not all(cal[int(j)].to_period("Q") == period for j in test.label_end_j):
        raise RuntimeError("Outer labels leave the quarter")
    y_raw = train.y_raw.to_numpy(dtype=float)
    y = y_raw if task == "regression" else (y_raw > 0.).astype(np.int64)
    indices, selection = mrmr_select(train[FEATURES].to_numpy(), y, task=task)
    names = [FEATURES[j] for j in indices]
    scaler = fit_scaler(train[names].to_numpy())
    preprocessing = {"selected_features": names, "selected_indices": indices, "feature_dictionary_sha256": _json_hash(FEATURE_DICTIONARY),
                     "mrmr": selection, "scaler": scaler, "training_target_mean": float(y_raw.mean()),
                     "training_target_sd": float(y_raw.std(ddof=1)), "fit_scope": "controlled T sample only"}
    result = {}
    for split, rows in (("train", train), ("valid", valid), ("test", test)):
        raw_y = rows.y_raw.to_numpy(dtype=float)
        result["x_"+split] = apply_scaler(rows[names].to_numpy(), scaler)
        result["y_"+split] = raw_y.copy() if task == "regression" else (raw_y > 0.).astype(np.int64)
        result["y_raw_"+split] = raw_y
        result["meta_"+split] = _metadata(rows, cal)
    quality = {}
    for split in ("train", "valid", "test"):
        x = result["x_"+split]
        quality[split] = {"rows": len(x), "features": x.shape[1], "finite": bool(np.isfinite(x).all()),
                          "unique_angle_rows": int(len(np.unique(x, axis=0))), "varying_angle_columns": int(np.count_nonzero(np.ptp(x, axis=0) > 0)),
                          "centered_matrix_rank": int(np.linalg.matrix_rank(x-x.mean(axis=0))),
                          "target_sd_raw": float(result["y_raw_"+split].std(ddof=1)),
                          "positive_targets": int(np.count_nonzero(result["y_raw_"+split] > 0))}
    if quality["train"]["unique_angle_rows"] < 2 or quality["train"]["varying_angle_columns"] == 0:
        raise ValueError("Degenerate transformed training sample")
    audit = {"quarter": quarter, "task": task, "train_size": train_size, "data_seed": seed,
             "scope": "D0 raw-price diagnostic; G4 unverified; not total-return or strategy-performance evidence",
             "classification_rule": "open[t+6]/open[t+1]-1 > 0 -> 1; otherwise 0",
             "input_identities": identities, "rows": {s: len(result["y_"+s]) for s in ("train", "valid", "test")},
             "quarter_first_market_date": cal[outer[0]].strftime("%Y-%m-%d"), "fit_cutoff": cal[cutoff].strftime("%Y-%m-%d"),
             "T": {"first": cal[t_days[0]].strftime("%Y-%m-%d"), "last": cal[t_days[-1]].strftime("%Y-%m-%d"), "calendar_days": 168},
             "V": {"first": cal[v_days[0]].strftime("%Y-%m-%d"), "last": cal[v_days[-1]].strftime("%Y-%m-%d"), "calendar_days": 84},
             "sampling": {"train": t_sampling, "valid": v_sampling}, "eligible_T_records": len(t_pool), "eligible_V_records": len(v_pool),
             "sampling_calendar_days_after_maturity_purge": {"train": len(t_mature_days), "valid": len(v_mature_days)},
             "outer_decision_dates": [cal[j].strftime("%Y-%m-%d") for j in decision_days], "outer_grid_records": len(test_grid),
             "outer_missing_features": int((~test_grid.feature_valid).sum()),
             "outer_feature_valid_missing_target": int((test_grid.feature_valid & ~test_grid.target_valid).sum()),
             "feature_history": "120 preceding market sessions plus t: 121 valid chronological bars; no fill",
             "outer_selection": "finite feature and matured target required for retrospective scoring; unavailable records counted",
             "financial_time_status": "2022-2024 development replay; not a newly unseen holdout",
             "quality": quality, "boundaries_verified": True, "preparation_seconds": time.perf_counter()-start}
    result.update(selected_features=names, preprocessing=preprocessing, preprocessing_hash=_json_hash(preprocessing), audit=audit)
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return result


def private_output(root, inputs=()):
    destination = local_path(root)
    script = Path(__file__).resolve()
    protected = next((p for p in script.parents if (p/"课程大纲.md").is_file()),
                     next((p for p in script.parents if p.name.casefold() == "code"), script.parent))
    if destination == protected or destination.is_relative_to(protected):
        raise ValueError("Prepared financial artifacts must remain outside the public source tree")
    for value in inputs:
        parent = local_path(value).parent
        if destination == parent or destination.is_relative_to(parent):
            raise ValueError("Output overlaps a read-only input directory")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def save_window(window, output, stem, inputs=()):
    destination = private_output(output, inputs)
    if Path(stem).name != stem or not stem:
        raise ValueError("Invalid snapshot filename")
    npz_path, json_path = destination/(stem+".npz"), destination/(stem+".json")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("Frozen snapshot already exists")
    arrays = {key: value for key, value in window.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in window.items() if not isinstance(value, np.ndarray)}
    metadata["schema_version"] = "cloud_v2_data80_v1"
    metadata["created_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["source_code"] = {name: file_identity(Path(__file__).parent/name) for name in ("features80.py", "data80.py")}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False))
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata["npz"] = {"file": npz_path.name, **file_identity(npz_path),
                       "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}}
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"npz": npz_path.name, "manifest": json_path.name, "sha256": metadata["npz"]["sha256"],
            "quarter": window["audit"]["quarter"], "task": window["audit"]["task"], "train_size": window["audit"]["train_size"],
            "rows": window["audit"]["rows"], "quality": window["audit"]["quality"]}


def load_window(path):
    with np.load(local_path(path), allow_pickle=False) as archive:
        result = json.loads(str(archive["metadata_json"].item()))
        for key in archive.files:
            if key != "metadata_json":
                result[key] = archive[key].copy()
                result[key].flags.writeable = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarter", action="append", choices=QUARTERS)
    parser.add_argument("--size", action="append", type=int, choices=SIZES)
    parser.add_argument("--task", action="append", choices=("regression", "classification"))
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    args = parser.parse_args()
    inputs = [args.input, args.calendar]
    output = private_output(args.output, inputs)
    started = time.perf_counter()
    raw, cal, identities = load_inputs(*inputs)
    frame = build_features(raw, cal)
    prepared = (raw, cal, identities, frame)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": "data preparation only; no model or quantum execution",
               "input_identities": identities, "feature_dictionary_sha256": _json_hash(FEATURE_DICTIONARY),
               "quality": {"calendar_sessions": len(cal), "securities": raw.symbol.nunique(), "source_rows": len(raw),
                           "calendar_grid_rows": len(frame), "valid_bars": int(frame.bar_valid.sum()),
                           "complete_history_rows": int(frame.history_complete.sum()), "all80_finite_eligible_rows": int(frame.feature_valid.sum()),
                           "nonfinite_by_feature": {name: int((~np.isfinite(frame[name])).sum()) for name in FEATURES}},
               "windows": [], "failures": []}
    for quarter in args.quarter or QUARTERS:
        for size in args.size or SIZES:
            for task in args.task or ("regression", "classification"):
                try:
                    window = prepare_window(None, None, quarter, size, task, args.seed, _prepared=prepared)
                    row = save_window(window, output, f"{quarter}_{task}_n{size:03d}", inputs)
                    summary["windows"].append(row)
                    print(json.dumps({key: row[key] for key in ("quarter", "task", "train_size", "rows", "npz")}, ensure_ascii=False), flush=True)
                except Exception as error:
                    failure = {"quarter": quarter, "task": task, "train_size": size, "type": type(error).__name__, "message": str(error)}
                    summary["failures"].append(failure)
                    print(json.dumps(failure, ensure_ascii=False), flush=True)
    summary["input_identities_after"] = {name: file_identity(path) for name, path in zip(("panel", "calendar"), inputs)}
    summary["inputs_unchanged"] = summary["input_identities_after"] == identities
    summary["seconds"] = time.perf_counter()-started
    summary["passed"] = not summary["failures"] and summary["inputs_unchanged"]
    with (output/"preparation_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
