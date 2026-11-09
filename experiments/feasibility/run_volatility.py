"""D0 raw-price one-day variance forecasting, not an economic backtest.

Protocol: 2019--2024 input only; forecast targets in 2022--2024. Rolling-60
mean squared log return, EWMA(0.94), and quarterly zero-mean normal GARCH(1,1).
GARCH fits the contiguous finite-return suffix (252--756 returns) ending before
the quarter's first target. It uses percentage returns and rescale=False. The
mean of the first 60 training squared returns is supplied as arch backcast,
not silently as arch's default. This is not a forced h[0] assignment: arch uses
h[0]=omega+(alpha+beta)*backcast before numerical variance bounds. The full
training filter and the first out-of-sample variance are checked independently.
The one-step acceptance reference is volatility.forecast with the same explicit
backcast; result.forecast recomputes a default backcast in arch 7.2 and is logged
separately, including any initialization-induced difference.
Predictions are made before observing the target return. Missing returns reset
rolling/EWMA state and disable that quarter's GARCH state; no forward filling.
Failures stay NA. Scores use the same valid target points for all three models.

Example (all locations supplied by the operator):
  python run_volatility.py --panel INPUT_CSV --calendar CALENDAR_CSV --output PRIVATE_DIR
  python run_volatility.py --self-test

The self-test uses analytic arrays only and is not a financial experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from safe_io import OutputRoot

MODELS = ("rolling60", "ewma094", "garch11")
FLOOR = 1e-12
DECAY = 0.94
MIN_TRAIN = 252
MAX_TRAIN = 756
FORECAST_ATOL_PCT2 = 1e-10
FORECAST_RTOL = 1e-8


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path):
    path = Path(path)
    before = path.stat()
    value = digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed during provenance read")
    return {"sha256": value, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def reject_dates(dates, label):
    dates = pd.DatetimeIndex(dates)
    if dates.isna().any() or dates.tz is not None:
        raise ValueError(label + ": invalid or timezone-aware dates")
    if len(dates) == 0 or (dates < pd.Timestamp("2019-01-01")).any() or (dates >= pd.Timestamp("2025-01-01")).any():
        raise ValueError(label + ": only 2019--2024 records are admitted")
    if (dates != dates.normalize()).any():
        raise ValueError(label + ": daily timestamps must be midnight")
    return dates


def log_returns(close):
    close = np.asarray(close, dtype=float)
    result = np.full(close.shape, np.nan)
    valid = np.isfinite(close) & (close > 0)
    pairs = valid[1:] & valid[:-1]
    positions = np.flatnonzero(pairs) + 1
    result[positions] = np.log(close[positions] / close[positions - 1])
    return result


def load_inputs(panel_path, calendar_path):
    raw = pd.read_csv(panel_path, dtype={"symbol": str})
    required = {"symbol", "date", "close"}
    if not required.issubset(raw.columns):
        raise ValueError("Panel lacks symbol/date/close")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    reject_dates(raw["date"], "panel")
    if raw["symbol"].isna().any() or raw.duplicated(["symbol", "date"]).any():
        raise ValueError("Missing symbol or duplicate symbol-date key")
    calendar_frame = pd.read_csv(calendar_path)
    if "date" not in calendar_frame:
        raise ValueError("Calendar lacks date")
    calendar = reject_dates(pd.to_datetime(calendar_frame["date"], errors="raise"), "calendar")
    if calendar.has_duplicates:
        raise ValueError("Duplicate calendar date")
    calendar = calendar.sort_values()
    if not raw["date"].isin(calendar).all():
        raise ValueError("Panel dates outside supplied market calendar")
    series = {}
    for symbol, group in raw.groupby("symbol", sort=True):
        close = pd.to_numeric(group.set_index("date")["close"], errors="coerce").reindex(calendar).to_numpy(dtype=float)
        series[symbol] = log_returns(close)
    targets = np.flatnonzero((calendar.year >= 2022) & (calendar.year <= 2024))
    if not len(targets):
        raise ValueError("No 2022--2024 target dates")
    return series, calendar


def trailing_training(returns, cutoff):
    """cutoff is the first target index, and is excluded from the training set."""
    begin = int(cutoff)
    while begin > 0 and cutoff - begin < MAX_TRAIN and np.isfinite(returns[begin - 1]):
        begin -= 1
    return begin, np.asarray(returns[begin:cutoff], dtype=float)


def parameter_errors(omega, alpha, beta, convergence):
    errors = []
    if convergence != 0:
        errors.append("convergence_flag_nonzero")
    if not np.isfinite([omega, alpha, beta]).all():
        errors.append("nonfinite_parameters")
        return errors
    if omega <= 0:
        errors.append("omega_not_positive")
    if alpha < 0 or beta < 0:
        errors.append("negative_arch_or_garch_coefficient")
    if alpha + beta > 1 - 1e-6:
        errors.append("persistence_outside_strict_protocol_bound")
    return errors


def garch_step(omega, alpha, beta, observed_return_pct, variance_pct2):
    return float(omega + alpha * observed_return_pct ** 2 + beta * variance_pct2)


def manual_training_filter(y_pct, omega, alpha, beta, backcast, variance_bounds):
    """Independent p=q=1,power=2 arch 7.2 recursion including numeric bounds."""
    y_pct = np.asarray(y_pct, dtype=float)
    bounds = np.asarray(variance_bounds, dtype=float)
    if bounds.shape != (len(y_pct), 2):
        raise ValueError("Unexpected variance-bound shape")
    values = np.empty(len(y_pct), dtype=float)
    bound_count = 0
    for t in range(len(y_pct)):
        raw = (omega + alpha * backcast + beta * backcast) if t == 0 else garch_step(omega, alpha, beta, y_pct[t - 1], values[t - 1])
        lower, upper = bounds[t]
        if raw < lower:
            values[t] = lower
            bound_count += 1
        elif raw > upper:
            values[t] = upper + (np.log(raw / upper) if not np.isinf(raw) else 1000)
            bound_count += 1
        else:
            values[t] = raw
    return values, bound_count


def fit_garch(returns, cutoff, calendar, symbol, arch_model):
    started = time.perf_counter()
    begin, train = trailing_training(returns, cutoff)
    target = calendar[cutoff]
    info = {"symbol": symbol, "quarter": str(target.to_period("Q")),
            "first_target_date": target.date().isoformat(),
            "train_n": int(len(train)), "train_start_date": calendar[begin].date().isoformat() if len(train) else None,
            "train_end_date": calendar[cutoff - 1].date().isoformat() if len(train) else None,
            "status": "not_fitted", "warnings": [], "fit_seconds": 0.0,
            "scale": "100 * raw-price log return; forecast variances divided by 10000"}
    if len(train) < MIN_TRAIN:
        info["status"] = "insufficient_contiguous_history"
        return None, info
    try:
        y = train * 100.0
        if not np.isfinite(y).all() or np.mean(y * y) <= 0:
            raise ValueError("Nonfinite or all-zero training returns")
        backcast = float(np.mean(y[:60] ** 2))
        if not np.isfinite(backcast) or backcast < 0:
            raise ValueError("Invalid first-60 training squared-return backcast")
        info.update({"backcast_pct2": backcast, "backcast_decimal2": backcast / 10000.0,
                     "backcast_observations": 60, "backcast_start_date": calendar[begin].date().isoformat(),
                     "backcast_end_date": calendar[begin + 59].date().isoformat(),
                     "backcast_interpretation": "first-60 mean supplied to arch pre-sample backcast; not forced h[0]"})
        model = arch_model(y, mean="Zero", vol="GARCH", p=1, o=0, q=1,
                           power=2.0, dist="normal", rescale=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(update_freq=0, disp="off", show_warning=True, backcast=backcast,
                               options={"maxiter": 1000, "ftol": 1e-8})
        info["warnings"] = [type(w.message).__name__ + ": " + str(w.message)[:1000] for w in caught]
        omega, alpha, beta = (float(result.params[key]) for key in ("omega", "alpha[1]", "beta[1]"))
        convergence = int(result.convergence_flag)
        info.update({"omega_pct2": finite_float(omega), "alpha": finite_float(alpha),
                     "beta": finite_float(beta), "persistence": finite_float(alpha + beta),
                     "convergence_flag": convergence})
        errors = parameter_errors(omega, alpha, beta, convergence)
        if errors:
            info.update(status="rejected_fit", rejection_reasons=errors)
            return None, info
        conditional = np.asarray(result.conditional_volatility, dtype=float)
        actual_filter = conditional ** 2
        variance_bounds = model.volatility.variance_bounds(y, power=2.0)
        manual_filter, bound_count = manual_training_filter(y, omega, alpha, beta, backcast, variance_bounds)
        full_filter_passed = (actual_filter.shape == manual_filter.shape and np.isfinite(actual_filter).all()
                              and np.isfinite(manual_filter).all() and (actual_filter > 0).all()
                              and np.allclose(manual_filter, actual_filter, atol=FORECAST_ATOL_PCT2, rtol=FORECAST_RTOL))
        full_error = float(np.max(np.abs(manual_filter - actual_filter)))
        info.update({"initial_unbounded_h0_pct2": finite_float(omega + (alpha + beta) * backcast),
                     "arch_initial_h0_pct2": finite_float(actual_filter[0]),
                     "manual_initial_h0_pct2": finite_float(manual_filter[0]),
                     "training_filter_observations_checked": len(y),
                     "training_filter_bound_application_count": bound_count,
                     "training_filter_max_absolute_error_pct2": finite_float(full_error),
                     "training_filter_max_absolute_error_decimal2": finite_float(full_error / 10000.0),
                     "training_filter_check_passed": bool(full_filter_passed)})
        if not full_filter_passed:
            info["status"] = "training_filter_check_failed"
            return None, info
        last_variance = float(actual_filter[-1])
        if not np.isfinite(last_variance) or last_variance <= 0:
            raise ValueError("Invalid final conditional variance")
        manual = garch_step(omega, alpha, beta, y[-1], last_variance)
        # arch 7.2 HARX.forecast (mean.py:959-972) recomputes its own backcast.
        # The public volatility.forecast interface lets this check share the exact
        # fitted parameters, residuals, explicit backcast and variance bounds.
        controlled = model.volatility.forecast(np.array([omega, alpha, beta]), y,
                                              backcast, variance_bounds, start=len(y) - 1,
                                              horizon=1, method="analytic")
        controlled_array = np.asarray(controlled.forecasts, dtype=float)
        if controlled_array.shape != (1, 1):
            raise ValueError("Unexpected controlled one-step forecast shape")
        official = float(controlled_array[0, 0])
        absolute_error = abs(manual - official)
        passed = (np.isfinite([manual, official]).all() and min(manual, official) > 0
                  and np.isclose(manual, official, atol=FORECAST_ATOL_PCT2, rtol=FORECAST_RTOL))
        info.update({"manual_first_variance_pct2": finite_float(manual),
                     "arch_first_variance_pct2": finite_float(official),
                     "first_step_reference_api": "model.volatility.forecast with explicit fit backcast, residuals, bounds and fitted parameters",
                     "first_step_absolute_error_pct2": finite_float(absolute_error),
                     "first_step_absolute_error_decimal2": finite_float(absolute_error / 10000.0),
                     "first_step_relative_error": finite_float(absolute_error / max(abs(official), 1e-300)),
                     "first_step_check_passed": bool(passed)})
        # Retain the convenient result-level API as a separate diagnostic. Its
        # default initialization is intentionally not the acceptance reference.
        try:
            default_backcast = float(model.volatility.backcast(y))
            default_forecast = result.forecast(horizon=1, method="analytic", align="origin", reindex=False)
            default_variance = float(default_forecast.variance.iloc[-1, 0])
            default_residual = float(default_forecast.residual_variance.iloc[-1, 0])
            default_error = abs(manual - default_variance)
            info.update({"result_forecast_default_backcast_pct2": finite_float(default_backcast),
                         "result_forecast_default_backcast_rule": "arch 7.2 first min(75,n) squared residuals, normalized 0.94**arange weights",
                         "result_forecast_default_variance_pct2": finite_float(default_variance),
                         "result_forecast_default_residual_variance_pct2": finite_float(default_residual),
                         "result_forecast_default_manual_absolute_error_pct2": finite_float(default_error),
                         "result_forecast_default_vs_controlled_absolute_error_pct2": finite_float(abs(default_variance - official)),
                         "result_forecast_default_same_tolerance_agreement": bool(np.isfinite(default_variance) and np.isclose(manual, default_variance, atol=FORECAST_ATOL_PCT2, rtol=FORECAST_RTOL)),
                         "result_forecast_default_is_acceptance_reference": False})
        except Exception as diagnostic_error:
            info["result_forecast_default_diagnostic_error"] = type(diagnostic_error).__name__ + ": " + str(diagnostic_error)[:1000]
        if not passed:
            info["status"] = "first_step_check_failed"
            return None, info
        info["status"] = "accepted"
        return {"omega": omega, "alpha": alpha, "beta": beta, "next_variance_pct2": manual}, info
    except Exception as exc:
        info.update(status="fit_or_forecast_exception", error=type(exc).__name__ + ": " + str(exc)[:1000])
        return None, info
    finally:
        info["fit_seconds"] = time.perf_counter() - started


def past_only_baselines(returns):
    """Forecast t from at most t-1, then observe t; missing returns reset state."""
    rolling = np.full(len(returns), np.nan)
    ewma = np.full(len(returns), np.nan)
    history = deque(maxlen=60)
    state = None
    for j, value in enumerate(returns):
        if not np.isfinite(value):
            history.clear()
            state = None
            continue
        if len(history) == 60:
            rolling[j] = float(np.mean(history))
            ewma[j] = state
        squared = float(value * value)
        warmed = len(history) == 60
        history.append(squared)
        if warmed:
            state = DECAY * state + (1 - DECAY) * squared
        elif len(history) == 60:
            state = float(np.mean(history))
    return rolling, ewma


def forecast_symbol(returns, calendar, symbol, arch_model):
    rolling, ewma = past_only_baselines(returns)
    targets = np.flatnonzero((calendar.year >= 2022) & (calendar.year <= 2024))
    quarter_first = {}
    for j in targets:
        quarter_first.setdefault(str(calendar[j].to_period("Q")), int(j))
    rows, fits, failures = [], [], []
    state = None
    garch_reason = "not_initialized"
    for j in targets:
        date = calendar[j]
        quarter = str(date.to_period("Q"))
        if j == quarter_first[quarter]:
            state, fit = fit_garch(returns, int(j), calendar, symbol, arch_model)
            fits.append(fit)
            garch_reason = "available" if state is not None else fit["status"]
        value = returns[j]
        valid_target = bool(np.isfinite(value))
        predicted = np.nan
        if not valid_target:
            if state is not None:
                failures.append({"symbol": symbol, "date": date.date().isoformat(), "quarter": quarter,
                                 "reason": "missing_return_disabled_garch_until_next_quarter"})
            state = None
            garch_reason = "gap_reset_wait_next_quarter"
        elif state is not None:
            predicted = state["next_variance_pct2"] / 10000.0
            if not np.isfinite(predicted) or predicted <= 0:
                failures.append({"symbol": symbol, "date": date.date().isoformat(), "quarter": quarter,
                                 "reason": "invalid_garch_recursion"})
                state = None
                garch_reason = "invalid_garch_recursion"
                predicted = np.nan
        row = {"symbol": symbol, "date": date.date().isoformat(), "year": int(date.year),
               "quarter": quarter, "target_valid": valid_target,
               "log_return": float(value) if valid_target else np.nan,
               "proxy_squared_log_return": float(value * value) if valid_target else np.nan,
               "rolling60_variance": rolling[j], "ewma094_variance": ewma[j],
               "garch11_variance": predicted, "garch_status": garch_reason,
               "baseline_status": "available" if np.isfinite(rolling[j]) else ("target_missing" if not valid_target else "warmup_after_start_or_gap")}
        rows.append(row)
        # The target return becomes observed only after this target's forecast is recorded.
        if state is not None and valid_target:
            state["next_variance_pct2"] = garch_step(state["omega"], state["alpha"], state["beta"],
                                                     float(value) * 100.0, state["next_variance_pct2"])
    return rows, fits, failures


def metrics(frame):
    columns = [model + "_variance" for model in MODELS]
    predictions = frame[columns].to_numpy(dtype=float)
    available = np.isfinite(predictions) & (predictions >= 0)
    # Rolling/EWMA zero forecasts are permitted and floored for scoring; GARCH is strictly positive.
    common = frame["target_valid"].to_numpy(dtype=bool) & available.all(axis=1)
    result = []
    groups = [("all", "2022-2024", np.ones(len(frame), dtype=bool))]
    groups += [("year", str(year), (frame.year == year).to_numpy()) for year in (2022, 2023, 2024)]
    groups += [("quarter", str(quarter), (frame.quarter == quarter).to_numpy()) for quarter in sorted(frame.quarter.unique())]
    for level, key, mask in groups:
        use = mask & common
        eligible = mask & frame["target_valid"].to_numpy(dtype=bool)
        y = frame.loc[use, "proxy_squared_log_return"].to_numpy(dtype=float)
        for k, model in enumerate(MODELS):
            raw_h = predictions[use, k]
            h = np.maximum(raw_h, FLOOR)
            result.append({"group_type": level, "group": key, "model": model,
                           "candidate_targets": int(mask.sum()), "valid_proxy_targets": int(eligible.sum()),
                           "model_available_with_valid_proxy": int((mask & eligible & available[:, k]).sum()),
                           "common_valid_points": int(use.sum()),
                           "common_coverage_of_valid_proxy": float(use.sum() / eligible.sum()) if eligible.sum() else None,
                           "model_coverage_of_valid_proxy": float((eligible & available[:, k]).sum() / eligible.sum()) if eligible.sum() else None,
                           "variance_floor_count_on_common": int((raw_h < FLOOR).sum()),
                           "mse": float(np.mean((h - y) ** 2)) if len(y) else None,
                           "qlike_log_h_plus_y_over_h": float(np.mean(np.log(h) + y / h)) if len(y) else None})
    frame["common_valid"] = common
    return result


def peak_rss_bytes():
    if os.name == "nt":
        return None
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def run(args, out):
    started = time.perf_counter()
    import arch
    import scipy
    if arch.__version__ != "7.2.0":
        raise ValueError("This protocol requires arch 7.2.0; observed " + arch.__version__)
    from arch import arch_model
    identities = {"panel": file_identity(args.panel), "calendar": file_identity(args.calendar)}
    series, calendar = load_inputs(args.panel, args.calendar)
    predictions, fits, failures, symbol_timing = [], [], [], []
    for symbol, returns in series.items():
        point = time.perf_counter()
        rows, logs, problems = forecast_symbol(returns, calendar, symbol, arch_model)
        predictions.extend(rows); fits.extend(logs); failures.extend(problems)
        symbol_timing.append({"symbol": symbol, "seconds": time.perf_counter() - point})
        print(json.dumps({"symbol": symbol, "quarters": len(logs), "accepted_fits": sum(x["status"] == "accepted" for x in logs),
                          "seconds": symbol_timing[-1]["seconds"]}), flush=True)
    after = {"panel": file_identity(args.panel), "calendar": file_identity(args.calendar)}
    if identities != after:
        raise ValueError("Input changed during calculation; outputs withheld")
    frame = pd.DataFrame(predictions)
    scores = metrics(frame)
    finite_scores = [row[key] for row in scores for key in ("mse", "qlike_log_h_plus_y_over_h") if row[key] is not None]
    if not np.isfinite(finite_scores).all():
        raise ValueError("Nonfinite score after variance flooring")
    frame.to_csv(out.path("volatility_predictions.csv"), index=False, na_rep="NA")
    pd.DataFrame(scores).to_csv(out.path("volatility_metrics.csv"), index=False, na_rep="NA")
    pd.DataFrame(fits).to_csv(out.path("volatility_fit_log.csv"), index=False, na_rep="NA")
    out.json("volatility_fit_log.json", fits)
    out.json("volatility_runtime_failures.json", failures)
    errors = [x["first_step_absolute_error_pct2"] for x in fits if x.get("first_step_absolute_error_pct2") is not None]
    counts = dict(Counter(x["status"] for x in fits))
    filter_errors = [x["training_filter_max_absolute_error_pct2"] for x in fits if x.get("training_filter_max_absolute_error_pct2") is not None]
    default_errors = [x["result_forecast_default_manual_absolute_error_pct2"] for x in fits if x.get("result_forecast_default_manual_absolute_error_pct2") is not None]
    summary = {
        "status": "complete_with_fit_failures" if any(x["status"] != "accepted" for x in fits) or failures else "complete",
        "scope": "D0 raw-price diagnostic; no total-return, latent-volatility accuracy, trading or economic conclusion",
        "proxy": "squared one-market-day raw-close log return; decimal-return-squared units",
        "prediction_alignment": "target date t, information through previous supplied-calendar date only",
        "input_period": "2019-2024 only; records dated 2025 or later rejected in both panel and calendar",
        "target_period": "2022-2024; every supplied market date for every fixed security",
        "securities": len(series), "calendar_dates": len(calendar), "candidate_targets": len(frame),
        "valid_proxy_targets": int(frame.target_valid.sum()), "common_valid_points": int(frame.common_valid.sum()),
        "quarterly_fit_status_counts": counts, "runtime_failure_count": len(failures),
        "baseline_target_status_counts": {str(k): int(v) for k, v in frame.baseline_status.value_counts().items()},
        "garch_target_status_counts": {str(k): int(v) for k, v in frame.garch_status.value_counts().items()},
        "training_filter_comparisons": len(filter_errors),
        "maximum_training_filter_absolute_error_pct2": max(filter_errors) if filter_errors else None,
        "first_step_comparisons": len(errors), "maximum_first_step_absolute_error_pct2": max(errors) if errors else None,
        "maximum_first_step_absolute_error_decimal2": max(errors) / 10000 if errors else None,
        "first_step_tolerance": {"atol_pct2": FORECAST_ATOL_PCT2, "rtol": FORECAST_RTOL},
        "first_step_reference_api": "arch volatility.forecast, explicit same-initialization arguments",
        "result_forecast_default_diagnostic": {"not_an_acceptance_reference": True,
            "comparisons": len(default_errors),
            "maximum_manual_absolute_error_pct2": max(default_errors) if default_errors else None,
            "same_tolerance_disagreements": sum(x.get("result_forecast_default_same_tolerance_agreement") is False for x in fits),
            "reason": "arch 7.2 HARX.forecast recomputes GARCH default backcast rather than reusing fit(backcast=...)"},
        "rolling60": "mean of 60 immediately preceding valid contiguous squared log returns; target excluded",
        "ewma094": "initial next-day variance is mean of first 60 contiguous squared returns, then h_next=.94*h+.06*r_observed^2; reset after gap",
        "garch": {"mean": "Zero", "p": 1, "q": 1, "o": 0, "power": 2, "distribution": "normal", "rescale": False,
                  "fit_units": "percentage returns",
                  "backcast": "explicit mean(first 60 training percentage-return squares), passed to arch fit(backcast=...)",
                  "initial_variance_semantics": "arch h[0]=omega+(alpha+beta)*backcast before arch numeric variance bounds; full training filter manually checked",
                  "controlled_forecast": "model.volatility.forecast(parameters,y,backcast,variance_bounds,start=n-1,horizon=1,method=analytic)",
                  "D0_initialisation_difference": "using the specified first-60 mean as arch pre-sample backcast is not literal h[0]=that mean; this diagnostic does not claim that literal initial-state convention",
                  "train_contiguous_min": MIN_TRAIN, "train_contiguous_max": MAX_TRAIN,
                  "refit": "first target of each calendar quarter; training ends on immediately previous market date",
                  "acceptance": "convergence_flag=0; omega>0; alpha,beta>=0; alpha+beta<=1-1e-6; finite positive conditional variance; full training-filter and first-step forecast agreement",
                  "failure_policy": "NA, no parameter clipping or model substitution; gap disables GARCH until next quarterly refit"},
        "scoring": {"common_points_only": True, "variance_floor": FLOOR,
                    "MSE": "mean((max(h,floor)-r_t^2)^2)", "QLIKE": "mean(log(max(h,floor))+r_t^2/max(h,floor))",
                    "aggregation": "equal weight per common security-date observation; coverage also reported by year/quarter",
                    "uncertainty": "no iid significance or latent-volatility ranking guarantee claimed"},
        "inputs_before": identities, "inputs_after": after, "input_files_unchanged": True,
        "script_sha256": digest(__file__), "created_utc": datetime.now(timezone.utc).isoformat(),
        "resources": {"wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_rss_bytes(),
                      "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
                      "pandas": pd.__version__, "scipy": scipy.__version__, "arch": arch.__version__,
                      "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                      "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                      "execution_location": os.environ.get("EXECUTION_LOCATION", "unspecified"), "symbol_seconds": symbol_timing},
        "metrics": scores,
    }
    out.json("volatility_summary.json", summary)
    return summary


def protected_source_root(script_path):
    """Find an actual course by markers; do not infer it from deployment depth."""
    script = Path(script_path).resolve()
    ancestors = list(script.parents)
    for directory in ancestors:
        if (directory / "课程大纲.md").is_file() and (directory / "README.md").is_file():
            return directory, "course_markers"
    # A private deployment may be ROOT/code/feasibility/script.py with ROOT/runs.
    for directory in ancestors:
        if directory.name.casefold() == "code":
            return directory, "deployment_code_directory"
    return script.parent, "script_directory"


def private_output(output_path, panel_path, calendar_path, script_path):
    output = Path(output_path).resolve()
    protected, reason = protected_source_root(script_path)
    if output == protected or output.is_relative_to(protected):
        raise ValueError("Financial results overlap protected source tree (" + reason + ")")
    for input_path in (panel_path, calendar_path):
        parent = Path(input_path).resolve().parent
        if output == parent or output.is_relative_to(parent):
            raise ValueError("Output overlaps read-only input directory")
    # OutputRoot also rejects network shares and escaping relative output paths.
    return OutputRoot(output, Path(panel_path).resolve().parent)


def self_test():
    checks = []
    r = log_returns([100.0, 110.0, np.nan, 120.0, 132.0])
    assert np.isnan(r[[0, 2, 3]]).all() and np.allclose(r[[1, 4]], np.log(1.1))
    checks.append("missing calendar close breaks both adjacent return pairs")
    values = np.arange(1.0, 71.0) / 1000
    rolling, ewma = past_only_baselines(values)
    assert np.isnan(rolling[:60]).all() and np.isclose(rolling[60], np.mean(values[:60] ** 2))
    changed = values.copy(); changed[60:] = 10.0
    r2, e2 = past_only_baselines(changed)
    assert rolling[60] == r2[60] and ewma[60] == e2[60]
    assert np.isclose(ewma[61], .94 * ewma[60] + .06 * values[60] ** 2)
    checks.append("rolling/EWMA first target excludes that target and later observations")
    gap = np.ones(130) * .01; gap[65] = np.nan
    rg, eg = past_only_baselines(gap)
    assert np.isnan(rg[65:126]).all() and np.isnan(eg[65:126]).all() and np.isfinite(rg[126])
    checks.append("gap clears baseline history and requires 60 observed returns to warm up")
    v = np.ones(1000); v[900] = np.nan
    begin, train = trailing_training(v, 999)
    assert begin == 901 and len(train) == 98
    begin, train = trailing_training(np.ones(1000), 999)
    assert begin == 243 and len(train) == 756
    checks.append("training suffix is contiguous, capped at 756, and excludes target")
    assert parameter_errors(.1, .1, .8, 0) == []
    assert parameter_errors(.1, .1, .9, 0) and parameter_errors(0, .1, .8, 0)
    assert parameter_errors(.1, -.1, .8, 0) and parameter_errors(.1, .1, .8, 1)
    assert np.isclose(garch_step(.1, .2, .7, 2.0, 3.0), 3.0)
    checks.append("GARCH parameter gates and hand recurrence")
    filtered, active = manual_training_filter(np.array([2.0, -1.0]), .1, .2, .7, 4.0,
                                             np.array([[0.0, np.inf], [0.0, np.inf]]))
    assert np.allclose(filtered, [3.7, 3.49]) and active == 0
    assert not np.isclose(filtered[0], 4.0)
    bounded, active = manual_training_filter(np.array([2.0]), .1, .2, .7, 4.0, np.array([[0.0, 3.0]]))
    assert active == 1 and np.isclose(bounded[0], 3.0 + np.log(3.7 / 3.0))
    checks.append("explicit backcast differs from h0 and manual full-filter bounds match arch convention")
    fixed_returns = np.zeros(3)
    bounds = np.tile([0.0, np.inf], (3, 1))
    first, _ = manual_training_filter(fixed_returns, .1, 0.0, .99, 4.0, bounds)
    second, _ = manual_training_filter(fixed_returns, .1, 0.0, .99, 8.0, bounds)
    f1 = garch_step(.1, 0.0, .99, 0.0, first[-1])
    f2 = garch_step(.1, 0.0, .99, 0.0, second[-1])
    assert np.isclose(f2 - f1, .99 ** 4 * (8.0 - 4.0))
    checks.append("different backcasts alone produce persistent one-step differences at high beta")
    try:
        reject_dates(pd.to_datetime(["2024-12-31", "2025-01-02"]), "analytic calendar")
    except ValueError:
        pass
    else:
        raise AssertionError("2025 was admitted")
    checks.append("2025 input is rejected")
    test = pd.DataFrame({"year": [2022, 2022], "quarter": ["2022Q1"] * 2,
                         "target_valid": [True, True], "proxy_squared_log_return": [.01, .02],
                         "rolling60_variance": [0.0, .01], "ewma094_variance": [.01, .01],
                         "garch11_variance": [.01, np.nan]})
    scores = metrics(test)
    assert all(x["common_valid_points"] == 1 for x in scores if x["group_type"] == "all")
    assert all(x["qlike_log_h_plus_y_over_h"] is None or np.isfinite(x["qlike_log_h_plus_y_over_h"]) for x in scores)
    checks.append("scores share one mask and zero forecast uses the declared floor")
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        public = base / "public_course"
        public_script = public / "experiments" / "feasibility" / "run_volatility.py"
        public_script.parent.mkdir(parents=True)
        (public / "README.md").write_text("course marker", encoding="utf-8")
        (public / "课程大纲.md").write_text("course marker", encoding="utf-8")
        remote = base / "private_deployment"
        remote_script = remote / "code" / "feasibility" / "run_volatility.py"
        remote_script.parent.mkdir(parents=True)
        inputs = remote / "data"
        inputs.mkdir()
        panel = inputs / "panel.csv"
        calendar = inputs / "calendar.csv"
        for item in (panel, calendar):
            item.write_text("read-only input fixture", encoding="utf-8")
        original = panel.read_bytes()
        assert protected_source_root(public_script) == (public.resolve(), "course_markers")
        assert protected_source_root(remote_script) == ((remote / "code").resolve(), "deployment_code_directory")
        local_out = private_output(base / "private_runs", panel, calendar, public_script)
        remote_out = private_output(remote / "runs" / "volatility_v2", panel, calendar, remote_script)
        assert local_out.root.exists() and remote_out.root.exists()
        for destination, script in [(public / "results", public_script),
                                    (remote / "code" / "results", remote_script),
                                    (inputs / "results", remote_script)]:
            try:
                private_output(destination, panel, calendar, script)
            except ValueError:
                pass
            else:
                raise AssertionError("Protected destination was admitted")
            assert not destination.exists()
        try:
            remote_out.path("../../data/overwrite.csv")
        except ValueError:
            pass
        else:
            raise AssertionError("Output-root escape was admitted")
        assert panel.read_bytes() == original
    checks.append("public-course markers protect course tree; private deployment admits sibling runs but rejects code/input/escape")
    return {"passed": True, "checks": checks,
            "scope": "analytic and temporary-directory checks only; no arch fit, financial experiment or persistent output"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--calendar", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False)); return
    if any(value is None for value in (args.panel, args.calendar, args.output)):
        parser.error("--panel, --calendar and --output are required")
    out = private_output(args.output, args.panel, args.calendar, Path(__file__))
    try:
        result = run(args, out)
        print(json.dumps({"status": result["status"], "candidate_targets": result["candidate_targets"],
                          "common_valid_points": result["common_valid_points"],
                          "quarterly_fit_status_counts": result["quarterly_fit_status_counts"]}, allow_nan=False), flush=True)
    except Exception as exc:
        out.json("volatility_failure.json", {"status": "failed", "error": type(exc).__name__ + ": " + str(exc),
                                             "scope": "D0 diagnostic; no economic conclusion", "created_utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
