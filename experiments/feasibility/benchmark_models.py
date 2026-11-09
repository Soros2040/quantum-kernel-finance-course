"""Bounded four-candidate kernel/fit benchmark; analytic inputs, no finance data."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from benchmark_quantum import THREAD_VARIABLES, peak_rss_bytes, windows_memory_job
from safe_io import OutputRoot


def worker(args) -> int:
    if sys.stdin.readline().strip() != "GO":
        return 2
    for name in THREAD_VARIABLES:
        os.environ[name] = str(args.threads)
    import numpy as np
    import scipy
    from quantum_core import CANDIDATES, state_features, fidelity_kernel, real_density_features
    from kernel_models import kernel_map, predict_mapped, explicit_ridge
    n = args.size
    row = np.arange(1, 2*n+1, dtype=float)[:, None]
    col = np.arange(1, 5, dtype=float)[None, :]
    x = np.sin(row*np.sqrt(col)+0.17*col**2)
    y = np.sin(np.arange(n)*0.31)+0.2*np.cos(np.arange(n)*0.17)
    results = []
    overall = time.perf_counter()
    for candidate in CANDIDATES:
        a, depth = candidate["a"], candidate["depth"]
        timing = {}
        start = time.perf_counter()
        st, sv = state_features(x[:n], a, depth), state_features(x[n:], a, depth)
        timing["state_seconds"] = time.perf_counter()-start
        start = time.perf_counter()
        train, cross = fidelity_kernel(st), fidelity_kernel(sv, st)
        timing["kernel_seconds"] = time.perf_counter()-start
        start = time.perf_counter()
        zt, zv, diagnostic = kernel_map(train, cross)
        timing["spectral_map_seconds"] = time.perf_counter()-start
        start = time.perf_counter()
        pred = predict_mapped(zt, zv, y, args.regularization)
        timing["dual_fit_predict_seconds"] = time.perf_counter()-start
        start = time.perf_counter()
        ft, fv = real_density_features(st), real_density_features(sv)
        timing["density_features_seconds"] = time.perf_counter()-start
        start = time.perf_counter()
        pe = explicit_ridge(ft, fv, y, args.regularization, match_spectral_threshold=True)
        timing["matched_explicit_fit_predict_seconds"] = time.perf_counter()-start
        error = float(np.max(np.abs(pred-pe)))
        results.append({
            "candidate": candidate["name"], "a": a, "depth": depth,
            "training_records": n, "evaluation_records": n, "feature_dimension": ft.shape[1],
            "state_array_bytes": int(st.nbytes+sv.nbytes),
            "train_and_cross_kernel_bytes": int(train.nbytes+cross.nbytes),
            "explicit_feature_bytes": int(ft.nbytes+fv.nbytes),
            "retained_rank": diagnostic["retained_rank"],
            "prediction_equivalence_max_error": error,
            "prediction_equivalence_passed": bool(np.allclose(pred, pe, atol=1e-8, rtol=1e-8)),
            **timing})
    elapsed = time.perf_counter()-overall
    report = {
        "status": "success" if all(r["prediction_equivalence_passed"] for r in results) else "numerical_failure",
        "n_train": n, "n_evaluation": n, "qubits": 4, "candidate_count": 4,
        "regularization": args.regularization, "total_workload_seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes(), "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "state_dtype": "complex128", "kernel_dtype": "float64",
        "candidates": results,
        "prediction_equivalence_max_error": max(r["prediction_equivalence_max_error"] for r in results),
    }
    for key in ("state_seconds", "kernel_seconds", "spectral_map_seconds",
                "dual_fit_predict_seconds", "density_features_seconds",
                "matched_explicit_fit_predict_seconds"):
        report[key] = sum(r[key] for r in results)
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["status"] == "success" else 1


def run_point(args, size, repetition):
    command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker",
               "--size", str(size), "--threads", str(args.threads),
               "--regularization", str(args.regularization)]
    env = os.environ.copy()
    env.update({name: str(args.threads) for name in THREAD_VARIABLES})
    limit = int(args.memory_gib*1024**3)
    kwargs = {}
    if os.name != "nt":
        def limit_address_space():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        kwargs["preexec_fn"] = limit_address_space
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8",
                               env=env, **kwargs)
    close_job = None
    row = {"n_train": size, "n_evaluation": size, "repetition": repetition,
           "observed_utc": datetime.now(timezone.utc).isoformat(), "thread_limit": args.threads,
           "timeout_seconds": args.timeout, "memory_limit_bytes": limit}
    try:
        if os.name == "nt":
            close_job = windows_memory_job(process, limit)
        output, error = process.communicate("GO\n", timeout=args.timeout)
        row["returncode"] = process.returncode
        row["process_seconds"] = time.perf_counter()-started
        if output.strip():
            row.update(json.loads(output.strip()))
        else:
            row["status"] = "worker_failure"
        if error.strip():
            row["stderr"] = error[-4000:]
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        row.update(status="timeout", process_seconds=time.perf_counter()-started)
    except Exception as exc:
        if process.poll() is None:
            process.kill()
        process.communicate()
        row.update(status="runner_failure", error=f"{type(exc).__name__}: {exc}",
                   process_seconds=time.perf_counter()-started)
    finally:
        if close_job:
            close_job()
    return row


def save(args, rows):
    summaries = []
    for n in args.sizes:
        group = [r for r in rows if r["n_train"] == n]
        successes = [r for r in group if r.get("status") == "success"]
        times = [r["total_workload_seconds"] for r in successes]
        peaks = [r["peak_rss_bytes"] for r in successes if r.get("peak_rss_bytes") is not None]
        summaries.append({"n_train": n, "n_evaluation": n, "completed_repetitions": len(group),
                          "successful_repetitions": len(successes),
                          "median_total_workload_seconds": statistics.median(times) if times else None,
                          "minimum_total_workload_seconds": min(times) if times else None,
                          "maximum_total_workload_seconds": max(times) if times else None,
                          "maximum_peak_rss_bytes": max(peaks) if peaks else None})
    source = Path(__file__).resolve().parent
    report = {
        "status": "complete" if len(rows) == len(args.sizes)*args.repetitions else "partial",
        "all_runs_successful": bool(rows) and all(r.get("status") == "success" for r in rows),
        "input_kind": "deterministic_analytic_nonfinancial",
        "python_version": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "parameters": {"sizes": args.sizes, "repetitions": args.repetitions, "threads": args.threads,
                       "timeout_seconds": args.timeout, "memory_gib": args.memory_gib,
                       "regularization": args.regularization, "loss_convention": "mean_loss_n_lambda"},
        "source_sha256": {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                          for name in ("benchmark_models.py", "benchmark_quantum.py",
                                       "kernel_models.py", "quantum_core.py", "safe_io.py")},
        "summaries": summaries, "runs": rows,
        "limitations": [
            "All four candidates are processed sequentially; peak RSS is not simultaneous storage of four pipelines.",
            "Total workload includes both kernel and explicit-feature implementations and their comparison.",
            "Kernel timings include dense spectral repair; they are not minimal possible exact-feature ridge cost.",
            "Fit time uses one prespecified lambda and excludes model selection.",
            "Process startup and imports are excluded from workload time but included in process time.",
            "No financial data, predictive validity, GPU, cloud or quantum-device execution is tested.",
        ],
    }
    args.out.json("model_benchmark.json", report)
    fields = ["n_train", "n_evaluation", "repetition", "status", "total_workload_seconds",
              "process_seconds", "state_seconds", "kernel_seconds", "spectral_map_seconds",
              "dual_fit_predict_seconds", "density_features_seconds",
              "matched_explicit_fit_predict_seconds", "peak_rss_bytes",
              "prediction_equivalence_max_error", "thread_limit", "memory_limit_bytes",
              "timeout_seconds", "observed_utc"]
    with args.out.path("model_benchmark.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, default=128, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    args.out = OutputRoot(args.output_dir)
    if not 1 <= args.threads <= 4 or not 0 < args.memory_gib <= 16 or not 0 < args.timeout <= 300:
        parser.error("require 1..4 threads, 0<memory-gib<=16, and 0<timeout<=300")
    if args.repetitions < 1 or any(n < 2 or n > 1024 for n in args.sizes) or args.regularization <= 0:
        parser.error("require positive repetitions/lambda and 2<=training sizes<=1024")
    rows = []
    for n in args.sizes:
        for repetition in range(1, args.repetitions+1):
            row = run_point(args, n, repetition)
            rows.append(row)
            save(args, rows)
            print(json.dumps({key: row.get(key) for key in
                              ("n_train", "repetition", "status", "total_workload_seconds")}), flush=True)
            # Preserve this failure and stop before another repeat or larger workload.
            if row.get("status") != "success":
                return 1
    return 0 if all(r.get("status") == "success" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
