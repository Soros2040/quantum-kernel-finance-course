"""Serial 20-qubit analytic capacity benchmark, with zero cloud shots.

Each point includes 24 exact states, kernels, course KRR, and two independent
overlap checks. The inputs are reproducible analytic fixtures, not market data.
All output must be explicitly directed outside the public source tree.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import pennylane as qml

try:
    from .quantum_engine import states, gram
    from .modeling import krr
    from .data_pipeline import private_output
except ImportError:
    from quantum_engine import states, gram
    from modeling import krr
    from data_pipeline import private_output


class PointTimeout(TimeoutError):
    pass


@contextmanager
def point_deadline(seconds=300):
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("The frozen benchmark requires Linux SIGALRM deadlines")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError("A conflicting SIGALRM timer is already active")
    old_handler = signal.getsignal(signal.SIGALRM)

    def expire(signum, frame):
        raise PointTimeout(f"Benchmark point exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _rss_bytes():
    # ru_maxrss is KiB on the target Linux environment and cumulative over the
    # process lifetime. It includes interpreter/import baseline and prior points.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def analytic_fixture():
    rng = np.random.default_rng(20261109)
    x = rng.uniform(-0.6, 0.6, size=(24, 20)).astype(np.float64)
    y = np.sin(x).mean(axis=1) + 0.1 * (x[:, :-1] * x[:, 1:]).mean(axis=1)
    return x, y


def _hash_array(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def independent_overlap(x, xp, depth):
    """Overlap via qml.adjoint, independent of the engine's state Gram path."""
    device = qml.device("default.qubit", wires=20, shots=None)

    def feature_map(row):
        for _ in range(depth):
            for wire in range(20):
                (qml.RX if wire % 2 == 0 else qml.RY)(row[wire], wires=wire)
            for wire in range(19):
                qml.CNOT(wires=[wire, wire + 1])

    def circuit():
        feature_map(x)
        qml.adjoint(feature_map)(xp)
        return qml.probs(wires=range(20))

    node = qml.QNode(circuit, device, interface=None, diff_method=None)
    return float(node()[0])


def run_point(depth, repetition, x, y):
    layout = {"q": 20, "depth": depth, "axes": [0, 1], "edges": [1] * 19,
              "id": f"q20_d{depth}_a01_e" + "1" * 19}
    started = time.perf_counter()
    record = {"depth": depth, "repetition": repetition, "layout": layout,
              "status": "running", "wall_seconds": {}, "point_deadline_seconds": 300,
              "rss_peak_before_bytes": _rss_bytes(), "shots": 0,
              "state_preparations_completed": 0, "overlap_executions_completed": 0,
              "current_stage": "not_started"}
    state_cache = None
    try:
        with point_deadline(300):
            record["current_stage"] = "states"
            stage = time.perf_counter()
            state_cache = states(x, layout)
            record["wall_seconds"]["states"] = time.perf_counter() - stage
            record["state_preparations_completed"] = len(x)
            record["state_cache_shape"] = list(state_cache.shape)
            record["state_cache_dtype"] = str(state_cache.dtype)
            record["state_cache_bytes"] = int(state_cache.nbytes)
            record["max_state_norm_error"] = float(np.max(np.abs(np.sum(np.abs(state_cache)**2, axis=1) - 1)))

            record["current_stage"] = "gram"
            stage = time.perf_counter()
            K = gram(state_cache)
            record["wall_seconds"]["gram"] = time.perf_counter() - stage
            off_diagonal = K[~np.eye(len(K), dtype=bool)]
            record["kernel"] = {"shape": list(K.shape), "sha256": _hash_array(K),
                                 "diagonal_max_error": float(np.max(np.abs(np.diag(K) - 1))),
                                 "symmetry_max_error": float(np.max(np.abs(K - K.T))),
                                 "all_entries_variance": float(np.var(K)),
                                 "off_diagonal_variance": float(np.var(off_diagonal)),
                                 "minimum_eigenvalue": float(np.linalg.eigvalsh(K).min())}
            if record["max_state_norm_error"] > 1e-10 or record["kernel"]["diagonal_max_error"] > 1e-10:
                raise AssertionError("State normalization or kernel diagonal check failed")

            record["current_stage"] = "krr"
            stage = time.perf_counter()
            prediction, diagnostics = krr(K[:16, :16], K[16:, :16], y[:16], regularization=0.01)
            record["wall_seconds"]["krr"] = time.perf_counter() - stage
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite analytic KRR predictions")
            record["krr"] = {"train_samples": 16, "evaluation_samples": 8, "lambda": 0.01,
                              "prediction_sha256": _hash_array(prediction),
                              "analytic_fixture_mse": float(np.mean((prediction - y[16:])**2)),
                              "interpretation": "Functional check on analytic targets; not financial predictive evidence",
                              "mapping_diagnostics": diagnostics}

            record["current_stage"] = "independent_overlaps"
            stage = time.perf_counter()
            checks = []
            for i, j in [(0, 16), (7, 23)]:
                pair_start = time.perf_counter()
                reference = independent_overlap(x[i], x[j], depth)
                record["overlap_executions_completed"] += 1
                observed = float(K[i, j])
                error = abs(reference - observed)
                nontrivial = 1e-8 < observed < 1 - 1e-8
                checks.append({"indices": [i, j], "state_kernel": observed,
                               "adjoint_overlap": reference, "absolute_error": error,
                               "nontrivial": nontrivial, "atol": 1e-10,
                               "passed": bool(nontrivial and error <= 1e-10),
                               "wall_seconds": time.perf_counter() - pair_start})
            record["wall_seconds"]["independent_overlaps"] = time.perf_counter() - stage
            record["overlap_checks"] = checks
            if not all(check["passed"] for check in checks):
                raise AssertionError("Independent nontrivial overlap check failed")
            record["current_stage"] = "complete"
            record["status"] = "passed"
    except Exception as error:
        record["status"] = "timeout" if isinstance(error, PointTimeout) else "failed"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
    finally:
        record["wall_seconds"]["total"] = time.perf_counter() - started
        record["rss_peak_after_bytes"] = _rss_bytes()
        record["rss_interpretation"] = "Linux ru_maxrss: cumulative process peak including baseline and previous points; not exclusive point memory"
        del state_cache
        gc.collect()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Private local directory outside the public source tree")
    args = parser.parse_args()
    output = private_output(args.output, script_path=__file__)
    summary_path = output / "benchmark20.json"
    if summary_path.exists() or any(output.glob("benchmark20_d*_r*.json")):
        raise FileExistsError("Benchmark records already exist; refuse overwrite or silent resume")
    x, y = analytic_fixture()
    source_files = [Path(__file__), Path(__file__).with_name("quantum_engine.py"),
                    Path(__file__).with_name("modeling.py")]
    summary = {"status": "running", "seed": 20261109, "source": "analytic fixture only",
               "angles": {"shape": list(x.shape), "distribution": "uniform[-0.6,0.6] radians",
                          "sha256": _hash_array(x)}, "target_sha256": _hash_array(y),
               "pennylane_version": qml.__version__, "numpy_version": np.__version__,
               "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
               "global_order": "depth1 reps1..3 then depth2 reps1..3; caller serializes with search",
               "expected_points": 6, "points": [], "cloud_jobs": 0, "shots": 0,
               "memory_note": "24 complex128 states require 384MiB retained storage; temporary arrays and runtime add memory"}
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    stop = False
    for depth in (1, 2):
        for repetition in (1, 2, 3):
            point = run_point(depth, repetition, x, y)
            name = f"benchmark20_d{depth}_r{repetition}.json"
            with (output / name).open("x", encoding="utf-8") as handle:
                json.dump(point, handle, ensure_ascii=False, indent=2, allow_nan=False)
            summary["points"].append({"file": name, "depth": depth, "repetition": repetition,
                                      "status": point["status"], "wall_seconds": point["wall_seconds"]["total"]})
            print(json.dumps(summary["points"][-1]), flush=True)
            if point["status"] != "passed":
                stop = True
                break
        if stop:
            break
    summary["status"] = "failed" if stop else "passed"
    summary["completed_points"] = len(summary["points"])
    summary["unrun_points"] = 6 - len(summary["points"])
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 1 if stop else 0


if __name__ == "__main__":
    sys.exit(main())
