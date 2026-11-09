"""Bounded state-only benchmark, independent workers and analytic inputs.

Defaults: q=4,8,12,16,20; depth=1,2,4; one input; three repetitions per point.
Each child has a 300-second total-process timeout, four numerical-library
threads and a 16-GiB process memory ceiling. No large unitary is constructed.
The timed region is state_features only, not process startup or NumPy import.
"""
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
from safe_io import OutputRoot

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def peak_rss_bytes() -> int | None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                 "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                 "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),
                                      ctypes.byref(counter), counter.cb):
            return int(counter.PeakWorkingSetSize)
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def windows_memory_job(process: subprocess.Popen, limit: int):
    """Attach a kill-on-close Windows job with a hard process commit ceiling."""
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                               ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x100 | 0x2000
    info.ProcessMemoryLimit = limit
    try:
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        if not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(process._handle)):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    except Exception:
        kernel.CloseHandle(handle)
        raise
    return lambda: kernel.CloseHandle(handle)


def worker(args) -> int:
    # Parent holds this worker before imports until the Windows job is attached.
    if sys.stdin.readline().strip() != "GO":
        return 2
    for key in THREAD_VARIABLES:
        os.environ[key] = str(args.threads)
    import numpy as np
    from quantum_core import BIT_ORDER, state_features
    x = np.sin(np.arange(args.samples*args.qubit, dtype=float)
               .reshape(args.samples, args.qubit)*0.37+0.2)
    started = time.perf_counter()
    states = state_features(x, a=args.scale, depth=args.layer)
    elapsed = time.perf_counter()-started
    normalization_error = float(np.max(np.abs(np.sum(np.abs(states)**2, axis=1)-1)))
    report = {
        "status": "success" if normalization_error <= 1e-10 else "numerical_failure",
        "qubits": args.qubit, "depth": args.layer, "samples": args.samples,
        "a": args.scale, "state_seconds": elapsed,
        "state_array_bytes": int(states.nbytes), "peak_rss_bytes": peak_rss_bytes(),
        "normalization_error": normalization_error,
        "maximum_imaginary_amplitude": float(np.max(np.abs(states.imag))),
        "ry_gates_per_state": args.qubit*args.layer,
        "cnot_gates_per_state": (args.qubit-1)*(args.layer-1),
        "numpy_version": np.__version__, "dtype": str(states.dtype),
        "bit_order": BIT_ORDER,
    }
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "success" else 1


def run_point(args, q: int, depth: int, repetition: int) -> dict:
    cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker",
           "--qubit", str(q), "--layer", str(depth), "--samples", str(args.samples),
           "--scale", str(args.scale), "--threads", str(args.threads)]
    env = os.environ.copy()
    env.update({key: str(args.threads) for key in THREAD_VARIABLES})
    limit_bytes = int(args.memory_gib*1024**3)
    kwargs = {}
    if os.name != "nt":
        def limit_address_space():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        kwargs["preexec_fn"] = limit_address_space
    started = time.perf_counter()
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8",
                               env=env, **kwargs)
    close_job = None
    row = {"qubits": q, "depth": depth, "repetition": repetition,
           "samples": args.samples, "a": args.scale,
           "observed_utc": datetime.now(timezone.utc).isoformat(),
           "timeout_seconds": args.timeout, "memory_limit_bytes": limit_bytes,
           "thread_limit": args.threads}
    try:
        if os.name == "nt":
            close_job = windows_memory_job(process, limit_bytes)
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


def save_report(args, rows: list[dict]) -> None:
    summaries = []
    for q in args.qubits:
        for depth in args.depths:
            group = [r for r in rows if r["qubits"] == q and r["depth"] == depth]
            times = [r["state_seconds"] for r in group if r.get("status") == "success"]
            peaks = [r["peak_rss_bytes"] for r in group
                     if r.get("status") == "success" and r.get("peak_rss_bytes") is not None]
            summaries.append({"qubits": q, "depth": depth, "completed_repetitions": len(group),
                              "successful_repetitions": len(times),
                              "state_median_seconds": statistics.median(times) if times else None,
                              "state_min_seconds": min(times) if times else None,
                              "state_max_seconds": max(times) if times else None,
                              "maximum_peak_rss_bytes": max(peaks) if peaks else None})
    script_dir = Path(__file__).resolve().parent
    report = {
        "status": "complete" if len(rows) == len(args.qubits)*len(args.depths)*args.repetitions else "partial",
        "all_runs_successful": bool(rows) and all(r.get("status") == "success" for r in rows),
        "input_kind": "deterministic_analytic_nonfinancial",
        "python_version": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "parameters": {"qubits": args.qubits, "depths": args.depths, "samples": args.samples,
                       "repetitions": args.repetitions, "a": args.scale,
                       "timeout_seconds": args.timeout, "memory_gib": args.memory_gib,
                       "threads": args.threads},
        "source_sha256": {name: hashlib.sha256((script_dir/name).read_bytes()).hexdigest()
                          for name in ("quantum_core.py", "benchmark_quantum.py", "safe_io.py")},
        "summaries": summaries, "runs": rows,
        "limitations": [
            "State timing excludes interpreter startup and NumPy import; process time includes both.",
            "Peak RSS includes interpreter, imports and post-timing normalization checks.",
            "Memory ceilings are Windows process commit limits or POSIX virtual-address-space limits, not equivalent RSS limits.",
            "Three repetitions describe this machine and workload, not a universal speed guarantee.",
            "No dense unitary, density-feature expansion, financial data, KRR, device or cloud execution is included.",
        ],
    }
    out = args.output_guard
    out.json("state_benchmark.json", report)
    fields = ["qubits", "depth", "repetition", "samples", "a", "status",
              "state_seconds", "process_seconds", "state_array_bytes", "peak_rss_bytes",
              "normalization_error", "maximum_imaginary_amplitude", "thread_limit",
              "memory_limit_bytes", "timeout_seconds", "observed_utc"]
    with out.path("state_benchmark.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--qubit", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--layer", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    args.output_guard = OutputRoot(args.output_dir)
    if not 1 <= args.threads <= 4 or not 0 < args.memory_gib <= 16:
        parser.error("local limits require 1..4 threads and 0<memory-gib<=16")
    if not 0 < args.timeout <= 300 or args.repetitions < 1 or args.samples < 1:
        parser.error("require 0<timeout<=300 and positive repetitions/samples")
    if any(q < 1 or q > 20 for q in args.qubits) or any(d < 1 for d in args.depths):
        parser.error("this feasibility benchmark requires 1<=q<=20 and positive depths")
    rows = []
    for q in args.qubits:
        for depth in args.depths:
            for repetition in range(1, args.repetitions+1):
                row = run_point(args, q, depth, repetition)
                rows.append(row)
                save_report(args, rows)
                print(json.dumps({k: row.get(k) for k in
                                  ("qubits", "depth", "repetition", "status", "state_seconds")}),
                      flush=True)
                # Preserve this failure and stop before another repeat or larger workload.
                if row.get("status") != "success":
                    return 1
    return 0 if all(r.get("status") == "success" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
