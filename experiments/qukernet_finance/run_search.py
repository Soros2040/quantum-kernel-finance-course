"""Bounded, resumable QuKerNet financial research; no cloud operations.

All outputs, including candidate kernels and inference bundles, are private.
Use ``--preflight`` before the frozen full protocol. Importing runs no experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
COURSE = next((p for p in HERE.parents if (p / "课程大纲.md").is_file()), HERE.parent)
sys.path.insert(0, str(HERE.parent / "feasibility"))
from safe_io import OutputRoot

LAMBDAS = (.001, .01, .1)
SEEDS = (20261109, 20261110, 20261111)
QUARTERS = ("2022Q1", "2023Q1", "2024Q1")


def utc():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def output_root(path, source=None):
    resolved = Path(path).resolve()
    if resolved == COURSE or resolved.is_relative_to(COURSE):
        raise ValueError("Research outputs must be outside the public course tree")
    return OutputRoot(resolved, source)


def write_json(out, name, data):
    target = out.path(name)
    temporary = out.path(str(name) + ".writing")
    temporary.write_text(json.dumps(plain(data), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def saved_stage(out, name, run_hash, resume, computation):
    target = out.path(name)
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if not resume or previous.get("state") != "complete" or previous.get("run_hash") != run_hash:
            raise RuntimeError(f"Stage cannot be automatically rerun: {name}")
        if digest(previous["result"]) != previous["result_sha256"]:
            raise RuntimeError(f"Stage payload checksum mismatch: {name}")
        return previous["result"]
    record = {"state": "running", "run_hash": run_hash, "started_utc": utc()}
    write_json(out, name, record)
    started = time.perf_counter()
    try:
        result = plain(computation())
        record.update(state="complete", result=result, result_sha256=digest(result),
                      observed_seconds=time.perf_counter() - started, finished_utc=utc())
        write_json(out, name, record)
        return result
    except BaseException as exc:
        record.update(state="failed", error=repr(exc), finished_utc=utc())
        write_json(out, name, record)
        raise


def code_hashes():
    paths = list(HERE.glob("*.py")) + [HERE / "plan.md", HERE.parent / "feasibility" / "safe_io.py",
                                      HERE.parent / "feasibility" / "quantum_core.py"]
    return {p.name: file_hash(p) for p in sorted(paths) if p.is_file()}


def environment():
    packages = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "PennyLane", "jax", "jaxlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "packages": packages, "cpu_only": True,
            "thread_limit": os.environ.get("OMP_NUM_THREADS")}


def _request_signature(value):
    if isinstance(value, np.ndarray):
        return {"array_hash": array_hash(value), "shape": value.shape, "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {k: _request_signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_request_signature(v) for v in value]
    return plain(value)


def _worker(connection):
    """Spawned once: no fork of an initialized JAX/PennyLane runtime."""
    try:
        import quantum_engine as qe
        connection.send({"ready": True})
        while True:
            request = connection.recv()
            if request is None:
                return
            begin = time.perf_counter()
            try:
                if request["kind"] == "finetune":
                    result = qe.finetune(**request["arguments"])
                elif request["kind"] == "kernel":
                    a = request["arguments"]
                    st = qe.states(a["x_train"], a["layout"], a.get("mask"), a.get("params"))
                    K = qe.gram(st)
                    result = {"status": "ok", "K": K, "training_states": len(st),
                              "evaluation_states": 0}
                    try:
                        result["alignment"] = qe.alignment(K, a["y_train"], a["task"])
                    except qe.DegenerateKernel as exc:
                        result.update(status="invalid", invalid_reason=str(exc), alignment=None)
                    if result["status"] == "ok" and a.get("x_evaluation") is not None:
                        se = qe.states(a["x_evaluation"], a["layout"], a.get("mask"), a.get("params"))
                        result["B"] = qe.gram(se, st)
                        result["evaluation_states"] = len(se)
                else:
                    raise ValueError("Unknown worker operation")
                connection.send({"ok": True, "result": result,
                                 "worker_seconds": time.perf_counter() - begin})
            except Exception as exc:
                connection.send({"ok": False, "error_type": type(exc).__name__,
                                 "error": str(exc), "traceback": traceback.format_exc(),
                                 "worker_seconds": time.perf_counter() - begin})
    except BaseException as exc:
        try:
            connection.send({"ready": False, "error": repr(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        connection.close()


class BoundedEngine:
    def __init__(self, out, run_hash, resume=False):
        self.out, self.run_hash, self.resume = out, run_hash, resume
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        if not self.connection.poll(120):
            self.close(force=True)
            raise TimeoutError("Quantum worker initialization exceeded 120 seconds")
        ready = self.connection.recv()
        if not ready.get("ready"):
            self.close(force=True)
            raise RuntimeError(f"Quantum worker initialization failed: {ready}")
        self.actual_calls = 0
        self.cached_calls = 0
        self.accounting = []

    def call(self, name, request, timeout):
        invocation_start = time.perf_counter()
        unit_hash = digest({"run_hash": self.run_hash, "request": _request_signature(request)})
        relative = f"units/{name}/{unit_hash}.json"
        state_path = self.out.path(relative)
        if state_path.exists():
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            if not self.resume:
                raise RuntimeError("Existing unit requires explicit --resume")
            if prior.get("state") != "complete" or prior.get("unit_hash") != unit_hash:
                raise RuntimeError(f"Unfinished/failed unit cannot be automatically rerun: {name}")
            if digest(prior["result"]) != prior.get("result_sha256"):
                raise RuntimeError("Cached result payload checksum mismatch")
            data_path = self.out.path(prior["arrays_file"])
            if file_hash(data_path) != prior["arrays_sha256"]:
                raise RuntimeError("Cached array checksum mismatch")
            with np.load(data_path, allow_pickle=False) as data:
                def restore(v):
                    if isinstance(v, dict) and set(v) == {"__array__"}:
                        return np.array(data[v["__array__"]], copy=True)
                    if isinstance(v, dict):
                        return {k: restore(x) for k, x in v.items()}
                    if isinstance(v, list):
                        return [restore(x) for x in v]
                    return v
                result = restore(prior["result"])
            self.cached_calls += 1
            self.accounting.append({"name": name, "unit_hash": unit_hash, "cached": True,
                    "original_worker_seconds": prior["worker_seconds"],
                    "original_observed_seconds": prior["observed_seconds"],
                    "this_invocation_seconds": time.perf_counter() - invocation_start})
            return result
        record = {"state": "running", "unit_hash": unit_hash, "name": name,
                  "started_utc": utc(), "timeout_seconds": timeout,
                  "request": _request_signature(request)}
        write_json(self.out, relative, record)
        begin = time.perf_counter()
        self.connection.send(request)
        if not self.connection.poll(timeout):
            record.update(state="timeout", finished_utc=utc(), observed_seconds=time.perf_counter() - begin)
            write_json(self.out, relative, record)
            self.close(force=True)
            raise TimeoutError(f"Stopped after {timeout}s: {name}")
        answer = self.connection.recv()
        if not answer.get("ok"):
            record.update(state="failed", finished_utc=utc(), response=answer,
                          observed_seconds=time.perf_counter() - begin)
            write_json(self.out, relative, record)
            raise RuntimeError(f"Quantum operation failed: {name}: {answer}")
        arrays = {}
        def package(v):
            if isinstance(v, np.ndarray):
                key = f"a{len(arrays)}"
                arrays[key] = v
                return {"__array__": key}
            if isinstance(v, dict):
                return {k: package(x) for k, x in v.items()}
            if isinstance(v, list):
                return [package(x) for x in v]
            return plain(v)
        packed = package(answer["result"])
        arrays_name = f"units/{name}/{unit_hash}.npz"
        np.savez_compressed(self.out.path(arrays_name), **arrays)
        record.update(state="complete", finished_utc=utc(), result=packed,
                      result_sha256=digest(packed),
                      arrays_file=arrays_name, arrays_sha256=file_hash(self.out.path(arrays_name)),
                      observed_seconds=time.perf_counter() - begin,
                      worker_seconds=answer["worker_seconds"])
        write_json(self.out, relative, record)
        self.actual_calls += 1
        self.accounting.append({"name": name, "unit_hash": unit_hash, "cached": False,
                    "original_worker_seconds": record["worker_seconds"],
                    "original_observed_seconds": record["observed_seconds"],
                    "this_invocation_seconds": time.perf_counter() - invocation_start})
        return answer["result"]

    def close(self, force=False):
        if getattr(self, "process", None) is not None:
            if self.process.is_alive() and not force:
                try:
                    self.connection.send(None)
                except (BrokenPipeError, EOFError):
                    pass
                self.process.join(3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(10)
            self.connection.close()


def layout_images(layouts, max_depth=5):
    images = np.zeros((len(layouts), 5, 8, 2 * max_depth), dtype=np.float64)
    for k, layout in enumerate(layouts):
        for layer in range(layout["depth"]):
            for wire in range(8):
                images[k, layout["axes"][wire % 2], wire, 2 * layer] = 1
            for wire, edge in enumerate(layout["edges"]):
                if edge:
                    images[k, 3, wire, 2 * layer + 1] = 1
                    images[k, 4, wire + 1, 2 * layer + 1] = 1
    return images.reshape(len(layouts), -1)


def fit_predictor(layouts, labels, candidate_layouts, seed):
    function_start = time.perf_counter()
    import torch
    from torch import nn
    if len(labels) < 2 or np.std(labels) <= 1e-12:
        raise ValueError("Insufficient nondegenerate training labels for the neural predictor")
    torch.manual_seed(int(seed))
    torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    network = nn.Sequential(nn.Linear(400, 128), nn.ReLU(), nn.Linear(128, 1)).double()
    X = torch.from_numpy(layout_images(layouts))
    Y = torch.as_tensor(np.asarray(labels) * 10, dtype=torch.float64).reshape(-1, 1)
    optimizer = torch.optim.Adam(network.parameters(), lr=.01)
    criterion = nn.SmoothL1Loss()
    generator = torch.Generator().manual_seed(int(seed))
    history = []
    training_start = time.perf_counter()
    for epoch in range(30):
        permutation = torch.randperm(len(Y), generator=generator)
        accumulated = 0.
        for batch in permutation.split(32):
            optimizer.zero_grad()
            loss = criterion(network(X[batch]), Y[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MLP loss")
            loss.backward()
            optimizer.step()
            accumulated += float(loss.detach()) * len(batch)
        history.append({"epoch": epoch + 1, "smooth_l1": accumulated / len(Y)})
    training_seconds = time.perf_counter() - training_start
    candidate_encoding_start = time.perf_counter()
    candidate_tensor = torch.from_numpy(layout_images(candidate_layouts))
    candidate_encoding_seconds = time.perf_counter() - candidate_encoding_start
    inference_start = time.perf_counter()
    with torch.no_grad():
        predicted = network(candidate_tensor).numpy().ravel() / 10
    inference_seconds = time.perf_counter() - inference_start
    return predicted, {"dtype": "float64", "training_count": len(Y), "history": history,
                       "state_dict": {k: v.detach().numpy().tolist() for k, v in network.state_dict().items()},
                       "input_shape": [5, 8, 10], "target_multiplier": 10,
                       "timings": {"training_seconds": training_seconds,
                                   "candidate_encoding_seconds": candidate_encoding_seconds,
                                   "inference_seconds": inference_seconds,
                                   "function_total_seconds": time.perf_counter() - function_start,
                                   "compilation_seconds": None, "compilation_note": "Eager PyTorch; no explicit JIT compilation"}}


def predictor_metrics(actual, predicted):
    from scipy.stats import kendalltau, pearsonr
    a, p = np.asarray(actual), np.asarray(predicted)
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(p) <= 1e-12:
        return {"status": "insufficient_evidence", "n": len(a)}
    return {"status": "ok", "n": len(a), "mse": float(np.mean((a - p) ** 2)),
            "pearson": float(pearsonr(a, p).statistic), "kendall": float(kendalltau(a, p).statistic)}


def date_accuracy(y, prediction, metadata):
    dates = np.asarray([str(m["date"]) for m in metadata])
    return float(np.mean([np.mean(np.asarray(y)[dates == d] == np.asarray(prediction)[dates == d])
                          for d in np.unique(dates)]))


def evaluate_kernel(K, Bv, Bt, window, task):
    import modeling as model
    y, yv, yt = window["y_train"], window["y_valid"], window["y_test"]
    if task == "regression":
        regularization, fitted, grid = model.select_lambda(K, Bv, y, yv, window["meta_valid"], LAMBDAS)
        pv, pt = model.predict_krr(fitted, Bv), model.predict_krr(fitted, Bt)
        restored = json.loads(json.dumps(plain(fitted), allow_nan=False))
        restored_prediction = model.predict_krr(restored, Bt)
        restore_error = float(np.max(np.abs(restored_prediction - pt))) if len(pt) else 0.
        if not np.allclose(restored_prediction, pt, atol=1e-8, rtol=1e-8):
            raise AssertionError("Restored KRR bundle changed predictions")
        variance = float(np.var(y, ddof=1))
        if variance <= 1e-24:
            raise ValueError("Degenerate regression training variance")
        valid_raw = model.date_equal_mse(yv, pv, window["meta_valid"])
        test_raw = model.date_equal_mse(yt, pt, window["meta_test"])
        valid_score, test_score = valid_raw / variance, test_raw / variance
        detail = {"lambda": regularization, "lambda_grid": grid, "model": fitted,
                  "validation_standardized_date_mse": valid_score,
                  "test_standardized_date_mse": test_score,
                  "validation_raw_date_mse": valid_raw, "test_raw_date_mse": test_raw,
                  "restored_model_max_abs_error": restore_error,
                  "test_rank_ic": model.rank_ic(yt, pt, window["meta_test"])}
    else:
        pv, dv = model.fit_predict(K, Bv, y, task="classification", C=1.)
        pt, dt = model.fit_predict(K, Bt, y, task="classification", C=1.)
        valid_score = 1 - date_accuracy(yv, pv, window["meta_valid"])
        test_score = 1 - date_accuracy(yt, pt, window["meta_test"])
        detail = {"C": 1., "validation_date_accuracy": 1 - valid_score,
                  "test_date_accuracy": 1 - test_score, "validation_fit": dv, "test_fit": dt,
                  "model": {"type": "reconstruct_precomputed_SVC", "C": 1.,
                            "train_kernel": np.asarray(K).tolist(), "labels": np.asarray(y).tolist()}}
    detail.update(validation_selection_loss=valid_score, test_loss=test_score,
                  validation_predictions=np.asarray(pv).tolist(), test_predictions=np.asarray(pt).tolist())
    return detail


def run_baselines(window, task):
    """All transformations are already fitted by the financial training window."""
    import modeling as model
    from quantum_core import state_features, fidelity_kernel
    zt, zv, ze = [np.asarray(window[k]) / np.pi for k in ("x_train", "x_valid", "x_test")]
    y, yv, yt = [np.asarray(window[k]) for k in ("y_train", "y_valid", "y_test")]
    results = {}
    # Linear kernel uses exactly the same compatible map and downstream model.
    linear_name = "ridge_8_features_64_rows" if task == "regression" else "linear_kernel_SVC_8_features_64_rows"
    results[linear_name] = evaluate_kernel(zt @ zt.T, zv @ zt.T, ze @ zt.T, window, task)
    rbf_trials = []
    for gamma in (.1, 1., 10.):
        rbf_trials.append(dict(evaluate_kernel(model.rbf(zt, zt, gamma),
                  model.rbf(zv, zt, gamma), model.rbf(ze, zt, gamma), window, task), gamma=gamma))
    chosen_rbf = min(rbf_trials, key=lambda row: (row["validation_selection_loss"], row["gamma"]))
    results["rbf_8_features_64_rows"] = dict(chosen_rbf, gamma_trials=rbf_trials,
                   gamma_grid=[.1, 1., 10.], gamma_units="z=angles/pi; no variance rescaling")
    extra = window["extras"]
    for name, scale, depth in (("Q1", .5, 1), ("Q2", 1., 1), ("Q3", .5, 2), ("Q4", 1., 2)):
        states = [state_features(np.asarray(extra["x4_" + role]) / np.pi, a=scale, depth=depth)
                  for role in ("train", "valid", "test")]
        results["prior_four_feature_" + name] = evaluate_kernel(fidelity_kernel(states[0]),
                       fidelity_kernel(states[1], states[0]), fidelity_kernel(states[2], states[0]), window, task)
    if task == "regression":
        xf, yf = np.asarray(window["x_full"]) / np.pi, np.asarray(window["y_full"])
        rows = []
        for regularization in LAMBDAS:
            prediction = model.explicit_ridge(xf, zv, yf, regularization)
            rows.append({"lambda": regularization, "validation_date_mse": model.date_equal_mse(yv, prediction, window["meta_valid"])})
        chosen = min(rows, key=lambda x: (x["validation_date_mse"], -x["lambda"]))["lambda"]
        pv = model.explicit_ridge(xf, zv, yf, chosen)
        pt = model.explicit_ridge(xf, ze, yf, chosen)
        variance = float(np.var(y, ddof=1))
        results["full_training_window_ridge"] = {"lambda": chosen, "grid": rows, "training_rows": len(yf),
              "validation_standardized_date_mse": model.date_equal_mse(yv, pv, window["meta_valid"]) / variance,
              "test_standardized_date_mse": model.date_equal_mse(yt, pt, window["meta_test"]) / variance,
              "validation_raw_date_mse": model.date_equal_mse(yv, pv, window["meta_valid"]),
              "test_raw_date_mse": model.date_equal_mse(yt, pt, window["meta_test"]),
              "validation_predictions": pv.tolist(), "test_predictions": pt.tolist(),
              "test_rank_ic": model.rank_ic(yt, pt, window["meta_test"]),
              "normalization_for_reporting": "Same 64-row training target variance as other methods",
              "selected_features": window["selected_features"], "preprocessing": window["preprocessing"]}
        momentum = np.asarray(extra["momentum20_test"])
        results["momentum20_ranking_only"] = {"test_rank_ic": model.rank_ic(yt, momentum, window["meta_test"]),
                    "test_predictions": momentum.tolist(), "scope": "Ranking score; not a calibrated 5-day return forecast"}
    return results


def kernel_request(window, layout, task, mask=None, params=None, evaluation=False):
    a = {"x_train": np.asarray(window["x_train"]), "y_train": np.asarray(window["y_train"]),
         "layout": layout, "task": task, "mask": mask, "params": params}
    if evaluation:
        a["x_evaluation"] = np.vstack((window["x_valid"], window["x_test"]))
    return {"kind": "kernel", "arguments": a}


def score_candidate(engine, prefix, window, layout, task, mask=None, params=None):
    result = engine.call(prefix, kernel_request(window, layout, task, mask, params, True), 300)
    if result["status"] != "ok":
        return {"status": "invalid", "layout": layout, "mask": mask or [], "params": params or [],
                "invalid_reason": result.get("invalid_reason")}, None
    nv = len(window["y_valid"])
    if task == "regression":
        from modeling import kernel_map
        try:
            kernel_map(result["K"], result["B"][:nv])
        except ValueError as exc:
            if str(exc) not in ("No retained positive eigenspace", "Degenerate centered kernel"):
                raise
            return {"status": "invalid", "layout": layout, "mask": mask or [], "params": params or [],
                    "invalid_reason": "Downstream mapping scientifically degenerate: " + str(exc)}, None
    metrics = evaluate_kernel(result["K"], result["B"][:nv], result["B"][nv:], window, task)
    bundle = {"status": "ok", "layout": layout, "mask": mask or [], "params": params or [],
              "preprocessing": window["preprocessing"], "selected_features": window["selected_features"],
              "x_train_angles": np.asarray(window["x_train"]).tolist(),
              "training_labels": np.asarray(window["y_train"]).tolist(),
              "training_target_mean": float(np.mean(window["y_train"])),
              "training_target_sd": float(np.std(window["y_train"], ddof=1)),
              "task": task, "alignment": result["alignment"], "metrics": metrics}
    bundle["candidate_hash"] = digest({"layout": layout, "mask": bundle["mask"], "params": bundle["params"]})
    return bundle, {"K": result["K"], "B": result["B"][:nv]}


def prediction_rows(bundle, window, method, stage, seed, quarter):
    if bundle["status"] != "ok":
        return []
    return [dict(meta, y_true=float(y), prediction=float(p), method=method, stage=stage,
                 seed=seed, quarter=quarter, candidate_hash=bundle["candidate_hash"],
                 code_bundle_sha256=digest(code_hashes()))
            for meta, y, p in zip(window["meta_test"], window["y_test"], bundle["metrics"]["test_predictions"])]


def run_window(engine, out, window, quarter, seed, task, run_hash, preflight=False):
    import quantum_engine as qe
    prefix = f"{task}/{quarter}/seed_{seed}"
    begin = time.perf_counter()
    completion_path = out.path(prefix + "/completion.json")
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not engine.resume or completion.get("run_hash") != run_hash:
            raise RuntimeError("Completed window requires matching --resume")
        for relative, expected in completion["artifact_hashes"].items():
            if file_hash(out.path(relative)) != expected:
                raise RuntimeError("Completed window artifact checksum mismatch: " + relative)
        # Preserve original observations; a cache visit never overwrites timings.
        return json.loads(out.path(prefix + "/summary.json").read_text(encoding="utf-8"))
    window_accounting_start = len(engine.accounting)
    layouts = qe.all_layouts(8, 5)
    if len(layouts) != 5760 or len({digest(x) for x in layouts}) != 5760:
        raise AssertionError("Expected 5760 unique canonical circuit codes")
    order = np.random.default_rng(seed).permutation(len(layouts))
    shuffled = [layouts[i] for i in order]
    labels_pool = shuffled[:8 if preflight else 160]
    candidates = shuffled[160:160 + 4096]
    write_json(out, prefix + "/window.json", {"run_hash": run_hash, "task": task, "seed": seed,
               "data_seed": 20261109, "quarter": quarter, "preprocessing": window["preprocessing"],
               "selected_features": window["selected_features"], "data_audit": window["audit"],
               "meta_train": window["meta_train"], "meta_valid": window["meta_valid"],
               "meta_test": window["meta_test"], "label_layouts": labels_pool,
               "candidate_layouts": [] if preflight else candidates})
    t = time.perf_counter()
    label_accounting_start = len(engine.accounting)
    labels = []
    for i, layout in enumerate(labels_pool):
        result = engine.call(prefix + f"/labels/{i:03d}", kernel_request(window, layout, task), 300)
        labels.append({"layout": layout, **result})
    label_invocation_seconds = time.perf_counter() - t
    label_seconds = sum(r["original_observed_seconds"] for r in engine.accounting[label_accounting_start:])
    valid = [r for r in labels if r["status"] == "ok"]
    if preflight:
        if not valid:
            return {"status": "insufficient_evidence", "reason": "No admissible preflight labels"}
        selected = valid[0]["layout"]
        mask = qe.make_mask(8, selected["depth"], seed, 0)
        result = engine.call(prefix + "/preflight_finetune", {"kind": "finetune", "arguments": {
            "x_angles": window["x_train"], "y": window["y_train"], "layout": selected,
            "mask": mask, "seed": seed, "task": task, "steps": 30, "batch_size": 16, "lr": .2}}, 1800)
        preflight_passed = (result["status"] == "ok" and result["completed_steps"] == 30
                            and np.isfinite(result["final_alignment"]) and np.isfinite(result["initial_alignment"])
                            and all(np.isfinite(r["gradient_norm"]) for r in result["trace"]))
        summary = {"status": "complete" if preflight_passed else "scientific_invalid", "mode": "preflight",
                   "preflight_passed": bool(preflight_passed), "label_requested": 8,
                   "valid_labels": len(valid), "invalid_labels": len(labels) - len(valid),
                   "finetune": result, "seconds": time.perf_counter() - begin,
                   "does_not_certify_full_protocol": True}
        write_json(out, prefix + "/summary.json", summary)
        return summary
    training = [r for r in labels[:128] if r["status"] == "ok"]
    held = []
    for i, row in enumerate(labels[128:], start=128):
        duplicate = row["status"] == "ok" and any(np.max(np.abs(row["K"] - tr["K"])) <= 1e-10 for tr in training)
        held.append({"index": i, "layout": row["layout"], "status": row["status"],
                     "alignment": row.get("alignment"), "duplicates_training_kernel": bool(duplicate),
                     "invalid_reason": row.get("invalid_reason")})
    admitted_held = [h for h in held if h["status"] == "ok" and not h["duplicates_training_kernel"]]
    if len(training) < 2 or np.std([r["alignment"] for r in training]) <= 1e-12:
        summary = {"status": "insufficient_evidence", "reason": "Predictor labels degenerate",
                   "valid_train_labels": len(training), "heldout": held, "label_seconds": label_seconds}
        write_json(out, prefix + "/summary.json", summary)
        return summary
    t = time.perf_counter()
    def train_predict():
        p, m = fit_predictor([r["layout"] for r in training], [r["alignment"] for r in training],
                            candidates + [r["layout"] for r in admitted_held], seed)
        return {"predicted": p, "predictor": m}
    checkpoint = saved_stage(out, prefix + "/predictor_stage.json", run_hash, engine.resume, train_predict)
    predicted, predictor = np.asarray(checkpoint["predicted"]), checkpoint["predictor"]
    prediction_invocation_seconds = time.perf_counter() - t
    predictor_record = json.loads(out.path(prefix + "/predictor_stage.json").read_text(encoding="utf-8"))
    prediction_seconds = predictor_record["observed_seconds"]
    predictor["heldout_metrics"] = predictor_metrics([r["alignment"] for r in admitted_held], predicted[len(candidates):])
    predictor["heldout_audit"] = held
    predictor["heldout_policy"] = "Original 32 retained in audit; equivalent-to-training kernels excluded only from evaluation; no replacements"
    write_json(out, prefix + "/predictor.json", predictor)
    write_json(out, prefix + "/candidate_predictions.json", [{"layout": layout, "predicted_alignment": float(p)}
              for layout, p in zip(candidates, predicted)])
    selected = np.argsort(-predicted[:len(candidates)], kind="stable")[:5]
    selected_layouts = [candidates[i] for i in selected]
    shortlisted_exact = []
    for j, layout in enumerate(selected_layouts):
        result = engine.call(prefix + f"/qukernet_extra/{j}", kernel_request(window, layout, task), 300)
        shortlisted_exact.append({"layout": layout, "status": result["status"],
                                  "alignment": result.get("alignment"), "invalid_reason": result.get("invalid_reason")})
    write_json(out, prefix + "/qukernet_shortlist_exact.json", shortlisted_exact)
    random_indices = np.random.default_rng(np.random.SeedSequence([seed, 991])).choice(len(candidates), 5, replace=False)
    random_new = []
    for j, i in enumerate(random_indices):
        res = engine.call(prefix + f"/random_extra/{j}", kernel_request(window, candidates[i], task), 300)
        random_new.append({"layout": candidates[i], **res})
    random_all = [r for r in labels + random_new if r["status"] == "ok"]
    random_all.sort(key=lambda row: (-row["alignment"], row["layout"]["id"]))
    arms = {"qukernet": selected_layouts, "random": [r["layout"] for r in random_all[:5]]}
    summaries, all_predictions = {}, []
    for method, shortlist in arms.items():
        arm_start = time.perf_counter()
        arm_accounting_start = len(engine.accounting)
        prior_arm_path = out.path(prefix + f"/{method}/summary.json")
        prior_arm = json.loads(prior_arm_path.read_text(encoding="utf-8")) if prior_arm_path.exists() else {}
        pre, post, post_raw, config_records = [], [], [], []
        for j, layout in enumerate(shortlist):
            bundle, _ = score_candidate(engine, prefix + f"/{method}/pre/{j}", window, layout, task)
            pre.append(bundle)
            write_json(out, prefix + f"/{method}/pre_{j}.json", bundle)
            if bundle["status"] != "ok":
                post.append({"status": "invalid", "layout": layout, "invalid_reason": "Shortlist kernel degenerate; no replacement"})
                post_raw.append(None)
                continue
            choices = []
            for config in range(5):
                mask = qe.make_mask(8, layout["depth"], seed, config)
                ftseed = int(np.random.SeedSequence([seed, layout["depth"], config, 731]).generate_state(1)[0])
                result = engine.call(prefix + f"/{method}/finetune/{j}/{config}", {"kind": "finetune", "arguments": {
                    "x_angles": window["x_train"], "y": window["y_train"], "layout": layout, "mask": mask,
                    "seed": ftseed, "task": task, "steps": 30, "batch_size": 16, "lr": .2}}, 1800)
                config_records.append({"shortlist_index": j, "config": config, "layout": layout, **result})
                if result["status"] == "ok":
                    if "final_train_kernel" in result:
                        checked = qe.alignment(np.asarray(result["final_train_kernel"]), window["y_train"], task)
                        if abs(checked - result["final_alignment"]) > 1e-10:
                            raise AssertionError("Microtuning endpoint alignment mismatch")
                    choices.append((result["final_alignment"], config, result))
            if not choices:
                post.append({"status": "invalid", "layout": layout, "invalid_reason": "All five configurations invalid; no replacement"})
                post_raw.append(None)
                continue
            _, chosen_config, chosen = sorted(choices, key=lambda x: (-x[0], x[1]))[0]
            tuned, raw = score_candidate(engine, prefix + f"/{method}/post/{j}", window, layout, task,
                                         chosen["mask"], chosen["params"])
            tuned["chosen_configuration"] = chosen_config
            post.append(tuned)
            post_raw.append(raw)
            write_json(out, prefix + f"/{method}/post_{j}.json", tuned)
        selections = {}
        for stage, bundles in (("pre", pre), ("post", post)):
            good = [(i, b) for i, b in enumerate(bundles) if b["status"] == "ok"]
            if good:
                index, best = min(good, key=lambda item: (item[1]["metrics"]["validation_selection_loss"], item[0]))
                selections[stage] = {"index": index, "candidate_hash": best["candidate_hash"], "metrics": best["metrics"]}
                all_predictions.extend(prediction_rows(best, window, method, stage, seed, quarter))
        arm_records = engine.accounting[arm_accounting_start:]
        arm_invocation_seconds = time.perf_counter() - arm_start
        original_arm_wall = prior_arm.get("arm_observed_seconds") if any(r["cached"] for r in arm_records) else arm_invocation_seconds
        scientific_complete = (len(pre) == len(post) == 5 and all(b["status"] == "ok" for b in post)
                               and len(config_records) == 25 and all(r["status"] == "ok" and r["completed_steps"] == 30 for r in config_records))
        summaries[method] = {"status": "complete" if scientific_complete else "partial_scientific_invalid",
                             "workflow_status": "complete", "scientific_status": "all_requested_configurations_valid" if scientific_complete else "rejections_or_incomplete_configurations",
                             "pre": pre, "post": post, "selections": selections,
                             "arm_observed_seconds": original_arm_wall,
                             "arm_this_invocation_seconds": arm_invocation_seconds,
                             "arm_quantum_task_seconds_original_sum": sum(r["original_observed_seconds"] for r in arm_records),
                             "quantum_task_accounting": arm_records,
                             "nominal_label_cost_per_method": 160, "additional_search_full_kta": 5,
                             "full_kta_search_quota": 165, "finetune_configuration_quota": 25,
                             "finetune_step_quota": 750, "configs_completed": sum(r["status"] == "ok" for r in config_records),
                             "configurations_attempted": len(config_records),
                             "steps_completed": sum(r["completed_steps"] for r in config_records),
                             "post_search_evaluation_full_train_kernel_calls": len(pre) + sum(b["status"] == "ok" for b in post),
                             "all_configuration_records": config_records,
                             "timing_warning": "Shared label computation charged to each method's nominal ledger; wall times are not independent equal-time runs"}
        write_json(out, prefix + f"/{method}/summary.json", summaries[method])
        if method == "qukernet" and task == "regression" and quarter == "2024Q1" and seed == SEEDS[0]:
            if len(post_raw) == 5 and all(x is not None for x in post_raw):
                finite_name = prefix + "/finite_input.npz"
                np.savez_compressed(out.path(finite_name), K=np.stack([r["K"] for r in post_raw]),
                                    B=np.stack([r["B"] for r in post_raw]), y_train=window["y_train"],
                                    y_valid=window["y_valid"],
                                    dates=np.asarray([str(x["date"]) for x in window["meta_valid"]], dtype="U32"))
                write_json(out, prefix + "/finite_input.json", {"status": "complete", "run_hash": run_hash,
                           "task": task, "quarter": quarter, "seed": seed,
                           "arrays_file": finite_name, "arrays_sha256": file_hash(out.path(finite_name)),
                           "candidate_hashes": [b["candidate_hash"] for b in post], "candidates": post,
                           "pool_was_selected_using_exact_kernels": True, "selector_isolation_applies_only_to_final_selection": True})
            else:
                write_json(out, prefix + "/finite_input.json", {"status": "insufficient_evidence",
                           "reason": "All five frozen tuned candidates must be admissible; no replacements"})
    scientifically_complete = all(arm["status"] == "complete" for arm in summaries.values())
    window_records = engine.accounting[window_accounting_start:]
    summary = {"status": "complete" if scientifically_complete else "complete_with_scientific_invalid",
               "workflow_status": "complete", "scientific_status": "all_configurations_valid" if scientifically_complete else "see_per_method_rejections",
               "run_hash": run_hash, "task": task, "quarter": quarter, "seed": seed,
               "label_seconds": label_seconds, "predictor_seconds": prediction_seconds,
               "label_time_definition": "Sum of original completed kernel-task observed times; excludes cache-read acceleration",
               "label_this_invocation_seconds": label_invocation_seconds,
               "predictor_this_invocation_seconds": prediction_invocation_seconds,
               "observed_seconds": None if any(r["cached"] for r in window_records) else time.perf_counter() - begin,
               "this_invocation_seconds": time.perf_counter() - begin, "label_requested": 160,
               "valid_label_count": len(valid), "invalid_label_count": 160 - len(valid),
               "predictor_holdout": predictor["heldout_metrics"], "methods": summaries,
               "input_data_audit": window["audit"], "code_hashes": code_hashes(),
               "financial_scope": "D0 raw-price diagnostic; no total-return or executable-profit claim"}
    baseline_start = time.perf_counter()
    baselines = saved_stage(out, prefix + "/baseline_stage.json", run_hash, engine.resume,
                           lambda: run_baselines(window, task))
    baseline_invocation_seconds = time.perf_counter() - baseline_start
    baseline_record = json.loads(out.path(prefix + "/baseline_stage.json").read_text(encoding="utf-8"))
    write_json(out, prefix + "/baselines.json", {"status": "complete", "models": baselines,
               "observed_seconds": baseline_record["observed_seconds"],
               "this_invocation_seconds": baseline_invocation_seconds,
               "baseline_inputs": "Eight selected angles/pi; prior Q1-Q4 use their original four-feature mapping"})
    summary["baselines_file"] = prefix + "/baselines.json"
    if not any(r["cached"] for r in window_records):
        summary["observed_seconds"] = time.perf_counter() - begin
    summary["this_invocation_seconds"] = time.perf_counter() - begin
    write_json(out, prefix + "/predictions.json", all_predictions)
    write_json(out, prefix + "/summary.json", summary)
    artifacts = {}
    for directory in (out.root / prefix, out.root / "units" / prefix):
        for p in directory.rglob("*"):
            if p.is_file() and p.name != "completion.json" and not p.name.endswith(".writing"):
                artifacts[str(p.relative_to(out.root))] = file_hash(p)
    write_json(out, prefix + "/completion.json", {"run_hash": run_hash, "completed_utc": utc(),
               "artifact_hashes": artifacts, "workflow_status": "complete", "scientific_status": summary["scientific_status"]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Read-only local panel CSV")
    parser.add_argument("--calendar", required=True, help="Read-only market calendar CSV")
    parser.add_argument("--output", required=True, help="Private local output root outside the public course")
    parser.add_argument("--quarters", nargs="+", choices=QUARTERS, default=list(QUARTERS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--preflight", action="store_true", help="Eight labels and one full 30-step configuration only")
    parser.add_argument("--resume", action="store_true", help="Reuse only complete checksum-matching units")
    parser.add_argument("--skip-classification", action="store_true")
    args = parser.parse_args()
    from data_pipeline import local_path
    args.input, args.calendar = local_path(args.input), local_path(args.calendar)
    source = args.input.parent
    output_resolved = Path(args.output).resolve()
    if output_resolved == args.calendar.parent or output_resolved.is_relative_to(args.calendar.parent):
        raise ValueError("Output overlaps the read-only calendar directory")
    out = output_root(args.output, source)
    contract = {"code_hashes": code_hashes(), "input_sha256": file_hash(args.input),
                "calendar_sha256": file_hash(args.calendar), "quarters": args.quarters, "seeds": args.seeds,
                "preflight": args.preflight, "skip_classification": args.skip_classification,
                "data_seed": 20261109, "environment": environment(), "lambda_grid": LAMBDAS}
    run_hash = digest(contract)
    manifest_path = out.path("manifest.json")
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume or prior.get("run_hash") != run_hash:
            raise RuntimeError("Output already exists or run/source/environment hash changed; use a new private output root")
    else:
        write_json(out, "manifest.json", {"run_hash": run_hash, "contract": contract, "created_utc": utc()})
    engine = None
    try:
        from data_pipeline import prepare_window
        engine = BoundedEngine(out, run_hash, args.resume)
        jobs = [(q, s, "regression") for q in args.quarters for s in args.seeds]
        if args.preflight:
            jobs = jobs[:1]
        elif not args.skip_classification:
            jobs.append(("2024Q1", SEEDS[0], "classification"))
        completed = []
        for quarter, seed, task in jobs:
            print(json.dumps({"stage": "window_start", "quarter": quarter, "seed": seed, "task": task, "utc": utc()}), flush=True)
            window = prepare_window(args.input, args.calendar, quarter, task=task, seed=20261109)
            result = run_window(engine, out, window, quarter, seed, task, run_hash, args.preflight)
            if args.preflight and not result.get("preflight_passed", False):
                raise RuntimeError("Preflight did not produce one valid complete 30-step configuration; no automatic expansion")
            completed.append({"quarter": quarter, "seed": seed, "task": task, "status": result["status"]})
            write_json(out, "progress.json", {"state": "running", "completed": completed, "updated_utc": utc()})
        write_json(out, "progress.json", {"state": "complete", "completed": completed, "updated_utc": utc(),
                   "worker_actual_calls_this_invocation": engine.actual_calls, "worker_cached_calls": engine.cached_calls})
        return 0
    except BaseException as exc:
        write_json(out, "failure.json", {"state": "stopped", "utc": utc(), "type": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(),
                   "automatic_expansion_stopped": True, "run_hash": run_hash})
        print(f"Stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
