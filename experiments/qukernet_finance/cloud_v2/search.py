"""Cloud-only quantum kernels and an auditable complete QuKerNet chain.

The injected provider is the ONLY implementation of quantum probabilities.
Classical preprocessing, MLP fitting and downstream estimators run locally.
Unknown or failed cloud intents are never retried implicitly.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from . import circuits as qc

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from feasibility.safe_io import OutputRoot
import modeling


class ScientificInvalid(ValueError):
    """A defined numerical/scientific rejection, not a transport exception."""


def now():
    return datetime.now(timezone.utc).isoformat()


def array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def source_hash():
    files = sorted(HERE.parent.glob("*.py")) + [HERE.parents[1]/"modeling.py"]
    return qc.digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def is_test_execution(execution_identity, predictor_override=False):
    """Test markers are field values; an is_mock=false key is not a marker."""
    test_kinds = {"mock_control_flow_only", "analytic_unit_test_only", "injected_test_provider"}
    return bool(predictor_override or execution_identity.get("is_mock") is True or
                execution_identity.get("kind") in test_kinds)


class Checkpoints:
    def __init__(self, root, identity, source=None):
        candidate = Path(root).resolve()
        code_root = next((p for p in HERE.parents if p.name == "code"), HERE.parent)
        course = next((p for p in HERE.parents if (p/"课程大纲.md").exists()), code_root)
        if candidate == course or candidate.is_relative_to(course):
            raise ValueError("Results must be outside the public course/code directory")
        if any(part.casefold() == "share2" for part in candidate.parts):
            raise ValueError("Outputs cannot target share2")
        self.out = OutputRoot(candidate, source=source)
        self.identity = _plain(identity)
        self.identity_hash = qc.digest(self.identity)
        path = self.out.path("identity.json")
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != self.identity:
                raise ValueError("Resume identity changed: code, input, task, or configuration differs")
        else:
            self.write("identity.json", self.identity)

    def write(self, name, data):
        path = self.out.path(name)
        temporary = self.out.path(name+".tmp")
        temporary.write_text(json.dumps(_plain(data), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        os.replace(temporary, path)

    def path_for(self, key):
        return "checkpoints/"+qc.digest(key)+".json"

    def get(self, key, request):
        path = self.out.path(self.path_for(key))
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("identity_hash") != self.identity_hash or record.get("request_hash") != qc.digest(request):
            raise ValueError("Checkpoint request or source identity mismatch")
        if record.get("status") != "complete":
            raise RuntimeError("Incomplete/unknown checkpoint requires explicit resolution; will not repeat: "+str(key))
        result = record["result"]
        if qc.digest(result) != record.get("result_sha256"):
            raise ValueError("Checkpoint payload checksum mismatch")
        return result

    def intent(self, key, request):
        record = {"key": key, "identity_hash": self.identity_hash, "request_hash": qc.digest(request),
                  "status": "intent", "started_utc": now()}
        self.write(self.path_for(key), record)
        return record

    def finish(self, key, record, result):
        result = _plain(result)
        record.update(status="complete", finished_utc=now(), result=result, result_sha256=qc.digest(result))
        self.write(self.path_for(key), record)
        return result


class CloudKernel:
    def __init__(self, provider, checkpoints, shots=128, kernel_seconds=300., batch_limit=20):
        if shots < 1 or kernel_seconds <= 0 or not 1 <= batch_limit <= 20:
            raise ValueError("Invalid execution limits")
        self.provider, self.store = provider, checkpoints
        self.shots, self.kernel_seconds, self.batch_limit = int(shots), float(kernel_seconds), int(batch_limit)
        self.invocation_calls = 0

    def deadline(self, outer=None):
        own = time.monotonic()+self.kernel_seconds
        return min(own, outer) if outer is not None else own

    def probabilities(self, specs, key, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError("Quantum operation deadline expired before submission")
        request = {"specifications": specs, "shots": self.shots}
        saved = self.store.get(key, request)
        if saved is not None:
            return np.asarray(saved["p0"], float)
        intent = self.store.intent(key, request)
        try:
            results = self.provider.probability(specs, self.shots,
                context={"phase": "main", "operation": key, "run_identity_sha256": self.store.identity_hash}, deadline=deadline)
            if len(results) != len(specs):
                raise RuntimeError("Cloud result count differs from submitted specifications")
            values = []
            for spec, result in zip(specs, results):
                if result.get("circuit_id", result.get("id")) != spec["circuit_id"]:
                    raise RuntimeError("Cloud results are not in the submitted circuit order")
                value = float(result["p0"])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise RuntimeError("Invalid all-zero probability")
                if int(result.get("shots", 0)) != self.shots:
                    raise RuntimeError("Observed shots differ from the requested budget")
                values.append(value)
            self.invocation_calls += len(specs)
            self.store.finish(key, intent, {"p0": values, "provider_results": results,
                "probability_circuits": len(specs), "nominal_shots": len(specs)*self.shots})
            return np.asarray(values, float)
        except Exception as exc:
            intent.update(status="failed_or_unknown", error=type(exc).__name__+": "+str(exc), stopped_utc=now())
            self.store.write(self.store.path_for(key), intent)
            raise

    def matrix(self, x_left, x_right, layout, mask=(), parameters=(), key="kernel", deadline=None):
        left = np.asarray(x_left, float)
        symmetric = x_right is None
        right = left if symmetric else np.asarray(x_right, float)
        q, p, _ = qc.validate_layout(layout)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != p or right.shape[1] != p:
            raise ValueError("Kernel input dimensions do not match feature map")
        limit = self.deadline(deadline)
        result = np.eye(len(left)) if symmetric else np.empty((len(left), len(right)), float)
        pending, indices = [], []
        batch_index = 0
        def flush():
            nonlocal batch_index
            if not pending:
                return
            values = self.probabilities(list(pending), f"{key}/batch{batch_index}", limit)
            for (i, j), value in zip(indices, values):
                result[i, j] = value
                if symmetric:
                    result[j, i] = value
            pending.clear(); indices.clear(); batch_index += 1
        for i in range(len(left)):
            for j in range(i+1 if symmetric else 0, len(right)):
                gates = qc.overlap_gates(left[i], right[j], layout, mask, parameters)
                pending.append(qc.specification(gates, q, f"{key}/{i}/{j}"))
                indices.append((i, j))
                if len(pending) == self.batch_limit:
                    flush()
        flush()
        return result

    def derivatives(self, x, layout, mask, parameters, key, deadline=None):
        """Independent physical occurrence shifts on BOTH overlap ends."""
        x = np.asarray(x, float)
        derivative = np.zeros((len(mask), len(x), len(x)), float)
        limit, q = self.deadline(deadline), layout["qubits"]
        pending, terms, batch_index = [], [], 0
        def flush():
            nonlocal batch_index
            if not pending:
                return
            values = self.probabilities(list(pending), f"{key}/batch{batch_index}", limit)
            for (parameter, i, j, weight), value in zip(terms, values):
                derivative[parameter, i, j] += weight*value
            pending.clear(); terms.clear(); batch_index += 1
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                gates = qc.overlap_gates(x[i], x[j], layout, mask, parameters)
                for parameter in range(len(mask)):
                    occurrences = qc.shift_occurrences(gates, parameter)
                    if {gates[index]["end"] for index, _ in occurrences} != {"left", "right"}:
                        raise ValueError("Gradient lost an overlap end")
                    for index, coefficient in occurrences:
                        for sign in (-1, 1):
                            request_key = f"{key}/{i}/{j}/p{parameter}/g{index}/s{sign}"
                            pending.append(qc.specification(gates, q, request_key, (index, sign*math.pi/2)))
                            terms.append((parameter, i, j, sign*.5*coefficient))
                            if len(pending) == self.batch_limit:
                                flush()
        flush()
        for parameter in range(len(mask)):
            derivative[parameter] += derivative[parameter].T
        return derivative


def alignment_and_gradient(kernel, y, task="classification", derivatives=None):
    k, y = np.asarray(kernel, float), np.asarray(y, float)
    if k.shape != (len(y), len(y)) or not np.isfinite(k).all() or not np.isfinite(y).all():
        raise ValueError("Finite square kernel and matching targets required")
    dk = None if derivatives is None else np.asarray(derivatives, float)
    if dk is not None and (dk.ndim != 3 or dk.shape[1:] != k.shape or not np.isfinite(dk).all()):
        raise ValueError("Invalid probability derivative tensor")
    if task == "classification":
        if not set(np.unique(y)).issubset({0., 1.}):
            raise ValueError("Binary 0/1 labels required")
        if len(np.unique(y)) != 2:
            raise ScientificInvalid("Single-class alignment batch; rejected without replacement")
        target = np.where(y[:, None] == y[None, :], 1., -1.)
    elif task == "regression":
        y = y-y.mean()
        target = np.outer(y, y)
        h = np.eye(len(y))-np.ones((len(y), len(y)))/len(y)
        k = h@k@h
        if dk is not None:
            dk = np.asarray([h@d@h for d in dk])
    else:
        raise ValueError("Unknown task")
    nk, nt = np.linalg.norm(k), np.linalg.norm(target)
    if nk <= 1e-12 or nt <= 1e-12:
        raise ScientificInvalid("Degenerate alignment denominator")
    value = float(np.sum(k*target)/(nk*nt))
    gradient = None if dk is None else np.asarray([
        np.sum(d*target)/(nk*nt)-value*np.sum(k*d)/(nk*nk) for d in dk])
    return value, gradient


def reject_constant_kernel(k):
    k = np.asarray(k, float)
    centered = k-k.mean(0)[None, :]-k.mean(1)[:, None]+k.mean()
    if np.linalg.norm(centered) <= 1e-12:
        raise ScientificInvalid("Numerically constant centered kernel")


def spectral_classification_map(train, cross):
    """Positive training spectrum and compatible cross extension; no centering."""
    k, b = np.asarray(train, float), np.asarray(cross, float)
    values, vectors = np.linalg.eigh((k+k.T)*.5)
    tolerance = 1e-10*max(1., float(np.max(np.abs(values))))
    keep = values > tolerance
    if not keep.any():
        raise ScientificInvalid("No positive classification eigenspace")
    basis = vectors[:, keep]
    kp, bp = (basis*values[keep])@basis.T, (b@basis)@basis.T
    return kp, bp, {"basis": basis.tolist(), "negative_eigenvalues": int(np.sum(values < -tolerance)),
        "retained_rank": int(keep.sum()), "tolerance": tolerance,
        "correction_frobenius": float(np.linalg.norm(kp-k)),
        "cross_discard_frobenius": float(np.linalg.norm(bp-b)),
        "centering": False, "normalization": False}


def evaluate_validation(k, cross, y, y_valid, dates, task):
    reject_constant_kernel(k)
    if task == "classification":
        from sklearn.svm import SVC
        kp, bp, mapping = spectral_classification_map(k, cross)
        fitted = SVC(C=1., kernel="precomputed").fit(kp, y)
        prediction = fitted.predict(bp)
        decision = np.asarray(fitted.decision_function(bp))
        bundle = {"kind": "svc_binary", "C": 1., "mapping": mapping,
                  "classes": fitted.classes_.tolist(), "support": fitted.support_.tolist(),
                  "dual": fitted.dual_coef_[0].tolist(), "intercept": float(fitted.intercept_[0])}
        manual = predict_bundle(bundle, cross)[0]
        if not np.array_equal(prediction, manual):
            raise RuntimeError("Portable SVC bundle does not reproduce validation predictions")
        score = modeling.date_equal_accuracy(y_valid, prediction, dates)
        return {"selection_loss": -score, "validation_accuracy": score,
                "validation_predictions": prediction.tolist(), "validation_decisions": decision.tolist(),
                "bundle": bundle}
    if task == "regression":
        # Same centered/scaled spectral-feature KRR as the course, with this
        # protocol's frozen 1e-10 spectral threshold (old module uses 1e-8).
        kp, bp, mapping = spectral_classification_map(k, cross)
        basis = np.asarray(mapping["basis"])
        eigenvalues = np.diag(basis.T@kp@basis)
        roots = np.sqrt(eigenvalues)
        z = basis*roots
        mean = z.mean(0)
        scale = float(np.mean(np.sum((z-mean)**2, axis=1)))
        if scale <= 1e-12:
            raise ScientificInvalid("Degenerate centered regression kernel")
        zt, zv = (z-mean)/math.sqrt(scale), (np.asarray(cross)@basis/roots-mean)/math.sqrt(scale)
        mapping.update(sqrt_eigenvalues=roots.tolist(), feature_mean=mean.tolist(), centered_scale=scale,
                       centering=True, normalization=True)
        ym = float(np.mean(y))
        grid, fits = [], []
        for regularization in modeling.LAMBDAS:
            alpha = np.linalg.solve(zt@zt.T+len(y)*regularization*np.eye(len(y)), np.asarray(y)-ym)
            prediction = zv@(zt.T@alpha)+ym
            loss = modeling.date_equal_mse(y_valid, prediction, dates)
            grid.append({"lambda": regularization, "date_equal_mse_raw": loss})
            fits.append((loss, -regularization, alpha, prediction))
        loss, negative_lambda, alpha, prediction = min(fits, key=lambda row: row[:2])
        regularization = -negative_lambda
        bundle = {"kind": "cloud_krr", "mapping": mapping, "regularization": regularization,
                  "n_train": len(y), "target_mean": ym, "alpha": alpha.tolist(), "z_train": zt.tolist(),
                  "objective": "mean squared loss + lambda RKHS norm squared; K+n*lambda*I",
                  "lambda_tie_rule": "largest lambda for exactly equal validation loss"}
        loss = modeling.date_equal_mse(y_valid, prediction, dates)
        return {"selection_loss": loss, "validation_mse_raw": loss,
                "validation_mse_standardized": loss/float(np.var(y, ddof=1)),
                "lambda": regularization, "lambda_grid": grid,
                "validation_predictions": prediction.tolist(), "bundle": bundle}
    raise ValueError("Unknown task")


def predict_bundle(bundle, cross):
    if bundle["kind"] == "svc_binary":
        basis = np.asarray(bundle["mapping"]["basis"])
        b = (np.asarray(cross)@basis)@basis.T
        score = b[:, bundle["support"]]@np.asarray(bundle["dual"])+bundle["intercept"]
        return np.asarray(bundle["classes"])[(score >= 0).astype(int)], score
    if bundle["kind"] == "cloud_krr":
        mapping = bundle["mapping"]
        z = np.asarray(cross)@np.asarray(mapping["basis"])/np.asarray(mapping["sqrt_eigenvalues"])
        z = (z-np.asarray(mapping["feature_mean"]))/math.sqrt(mapping["centered_scale"])
        return z@(np.asarray(bundle["z_train"]).T@np.asarray(bundle["alpha"]))+bundle["target_mean"], None
    return modeling.predict_krr(bundle, cross), None


class MLPScorer:
    def __init__(self, state, maximum_depth):
        self.state, self.maximum_depth = state, maximum_depth

    def predict(self, layouts):
        result = []
        w1, b1, w2, b2 = (np.asarray(self.state[k], float) for k in ("w1", "b1", "w2", "b2"))
        for start in range(0, len(layouts), 256):
            x = np.asarray([qc.encode_layout(layout, self.maximum_depth) for layout in layouts[start:start+256]])
            values = np.maximum(x@w1.T+b1, 0.)@w2.T+b2
            result.extend((values.ravel()/10.).tolist())
        return np.asarray(result)


def train_predictor(layouts, labels, maximum_depth, seed):
    import torch
    if len(labels) < 2 or np.std(labels) <= 1e-12:
        raise ScientificInvalid("Insufficient nonconstant predictor training labels")
    x = np.asarray([qc.encode_layout(layout, maximum_depth) for layout in layouts])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).double()
        optimizer = torch.optim.Adam(net.parameters(), lr=.01)
        inputs, targets = torch.from_numpy(x), torch.as_tensor(np.asarray(labels)[:, None]*10., dtype=torch.float64)
        losses = []
        generator = torch.Generator().manual_seed(seed)
        for _ in range(30):
            order = torch.randperm(len(labels), generator=generator)
            total = 0.
            for part in order.split(32):
                optimizer.zero_grad()
                loss = torch.nn.functional.smooth_l1_loss(net(inputs[part]), targets[part])
                loss.backward(); optimizer.step()
                total += float(loss.detach())*len(part)
            losses.append(total/len(labels))
        state = {"w1": net[0].weight.detach().numpy().tolist(), "b1": net[0].bias.detach().numpy().tolist(),
                 "w2": net[2].weight.detach().numpy().tolist(), "b2": net[2].bias.detach().numpy().tolist()}
    return MLPScorer(state, maximum_depth), {"state": state, "losses": losses,
        "architecture": "flatten-Linear128-ReLU-Linear1", "dtype": "float64",
        "target_multiplier": 10, "optimizer": "Adam", "learning_rate": .01,
        "epochs": 30, "batch_size": 32, "loss": "SmoothL1", "seed": seed}


def finetune(kernel, x, y, layout, mask, seed, configuration, task, steps, batch_size, learning_rate, key, seconds=1800.):
    request = {"layout": layout, "mask": mask, "seed": seed, "configuration": configuration,
               "task": task, "steps": steps, "batch_size": batch_size, "learning_rate": learning_rate}
    saved = kernel.store.get(key+"/complete", request)
    if saved is not None:
        return saved
    # An incomplete configuration cannot silently reset its 1800-second budget.
    # Manual reconciliation is required before resuming an interrupted intent.
    marker = kernel.store.intent(key+"/complete", request)
    start, deadline = time.monotonic(), time.monotonic()+seconds
    params = qc.initial_parameters(x, layout, mask)
    initial = params.copy()
    initial_k = kernel.matrix(x, None, layout, mask, params, key+"/initial", deadline)
    initial_alignment, _ = alignment_and_gradient(initial_k, y, task)
    trace = []
    for step in range(steps):
        if time.monotonic() >= deadline:
            raise TimeoutError("Configuration deadline expired")
        indices = np.random.default_rng(np.random.SeedSequence([seed, configuration, step, 123])).choice(len(x), batch_size, replace=True)
        xb, yb = x[indices], y[indices]
        # Reject a scientifically uninformative sampled batch before spending
        # cloud shots. The sampled indices are retained; no resampling occurs.
        if (task == "classification" and len(np.unique(yb)) != 2) or (task == "regression" and np.var(yb) <= 1e-24):
            kernel.store.write(f"traces/{qc.digest(key)}/rejected_batch.json", {
                "step": step, "batch_indices": indices.tolist(), "reason": "degenerate sampled target", "resampled": False})
            raise ScientificInvalid("Degenerate sampled target; rejected without replacement")
        batch_k = kernel.matrix(xb, None, layout, mask, params, f"{key}/step{step}/base", deadline)
        derivative = kernel.derivatives(xb, layout, mask, params, f"{key}/step{step}/gradient", deadline)
        value, gradient = alignment_and_gradient(batch_k, yb, task, derivative)
        if not np.isfinite(gradient).all():
            raise ScientificInvalid("Nonfinite finite-shot KTA gradient")
        params = params+learning_rate*gradient
        row = {"step": step, "batch_indices": indices.tolist(), "alignment_before": value,
               "gradient": gradient.tolist(), "gradient_norm": float(np.linalg.norm(gradient)),
               "parameters_after": params.tolist()}
        kernel.store.write(f"traces/{qc.digest(key)}/step{step:03}.json", row)
        trace.append(row)
    final_k = kernel.matrix(x, None, layout, mask, params, key+"/final", deadline)
    reject_constant_kernel(final_k)
    final_alignment, _ = alignment_and_gradient(final_k, y, task)
    result = {"status": "valid", "mask": mask, "initial_parameters": initial.tolist(),
              "parameters": params.tolist(), "initial_alignment": initial_alignment,
              "final_alignment": final_alignment, "final_train_kernel": final_k.tolist(),
              "completed_steps": len(trace), "trace": trace, "seconds_this_invocation": time.monotonic()-start}
    return kernel.store.finish(key+"/complete", marker, result)


def _dates(metadata, split, count):
    aliases = ("meta_"+split, split+"_metadata", split+"_meta")
    for key in aliases:
        if key in metadata:
            values = metadata[key]
            if len(values) != count:
                raise ValueError("Date metadata length mismatch")
            return values
    raise ValueError("Date metadata required for "+split)


def run_search(window, profile, provider, output, seed, task="classification", method="qukernet", source=None,
               predictor_factory=None):
    """Run ALL k*c end points; O targets never enter model/parameter selection.

    predictor_factory is a deliberate test seam. Such runs are labelled mock.
    It must never be supplied by the production CLI.
    """
    if task not in ("classification", "regression") or method not in ("qukernet", "random"):
        raise ValueError("Unsupported task or method")
    arrays = {k: np.asarray(window[k]) for k in ("x_train", "x_valid", "x_test", "y_train", "y_valid", "y_test")}
    x, xv, xt, y, yv = (np.asarray(arrays[k], float) for k in ("x_train", "x_valid", "x_test", "y_train", "y_valid"))
    for split_name in ("train", "valid", "test"):
        xx, yy = arrays["x_"+split_name], arrays["y_"+split_name]
        if xx.ndim != 2 or xx.shape[1] != profile["features"] or yy.shape != (len(xx),) or not len(xx):
            raise ValueError("All splits require the registered feature width and one target per row")
        if not np.isfinite(xx).all() or not np.isfinite(yy).all():
            raise ValueError("Finite admitted window required")
        if task == "classification" and not set(np.unique(yy)).issubset({0., 1.}):
            raise ValueError("All classification splits require binary labels")
    if len(x) != profile["n_train"] or len(xv) != profile["n_valid"]:
        raise ValueError("Window sizes differ from the registered stage")
    if task == "classification" and set(np.unique(y)) != {0, 1}:
        raise ValueError("Training classification window must contain both classes")
    if task == "regression" and np.var(y, ddof=1) <= 1e-24:
        raise ScientificInvalid("Degenerate training regression targets")
    metadata = window.get("metadata", {k: v for k, v in window.items() if not isinstance(v, np.ndarray)})
    if metadata.get("audit", {}).get("task") != task:
        raise ValueError("Prepared window task does not match the requested task")
    _dates(metadata, "train", len(x))
    valid_dates, test_dates = _dates(metadata, "valid", len(xv)), _dates(metadata, "test", len(xt))
    versions = {}
    for package in ("numpy", "torch", "scikit-learn", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    execution_identity = provider.execution_identity() if hasattr(provider, "execution_identity") else {"kind": "injected_test_provider", "is_mock": True}
    test_execution = is_test_execution(execution_identity, predictor_factory is not None)
    identity = {"schema": 1, "source_sha256": source_hash(), "package_versions": versions,
                "execution_identity": execution_identity,
                "profile": profile, "seed": seed,
                "task": task, "method": method, "inputs": {k: array_hash(v) for k, v in arrays.items()},
                "metadata_sha256": qc.digest(metadata), "test_override": predictor_factory is not None,
                "provider_class": type(provider).__module__+"."+type(provider).__qualname__}
    store = Checkpoints(output, identity, source=source)
    existing = store.get("workflow/complete", identity)
    if existing is not None:
        return existing
    kernel = CloudKernel(provider, store, profile["shots"], profile.get("kernel_seconds", 300))
    started = time.monotonic()
    summary = {"workflow_status": "running", "scientific_status": "unassessed", "identity": identity,
               "execution_kind": "mock_test" if test_execution else "cloud_provider",
               "task": task, "method": method, "started_utc": now()}
    store.write("summary.json", summary)
    try:
        m, top, count = profile["labels"], profile["top_k"], profile["configurations"]
        labels = qc.sample_unique(m, profile["qubits"], profile["features"], profile["depths"], seed)
        codes = [qc.layout_code(v) for v in labels]
        pool = qc.sample_unique(profile["candidate_pool"], profile["qubits"], profile["features"],
                                profile["depths"], seed+1, excluded=codes)
        store.write("layouts.json", {"label_layouts": labels, "candidate_layouts": pool,
                                     "label_hashes": [qc.digest(v) for v in labels],
                                     "candidate_hashes": [qc.digest(v) for v in pool]})
        labeled, initial_kernels = [], {}
        for i, layout in enumerate(labels):
            k = kernel.matrix(x, None, layout, key=f"labels/{i}")
            initial_kernels[qc.digest(layout)] = k
            row = {"index": i, "layout": layout, "layout_hash": qc.digest(layout)}
            try:
                reject_constant_kernel(k)
                row.update(status="valid", alignment=alignment_and_gradient(k, y, task)[0])
            except ScientificInvalid as exc:
                row.update(status="scientific_invalid", reason=str(exc))
            labeled.append(row)
        store.write("labels.json", labeled)
        split = int(.8*m)
        training = [row for row in labeled[:split] if row["status"] == "valid"]
        predictor_note = {"status": "not_used_by_random_control"}
        if method == "qukernet":
            factory = predictor_factory or train_predictor
            scorer, predictor_note = factory([r["layout"] for r in training], [r["alignment"] for r in training], max(profile["depths"]), seed)
            heldout = []
            for row in labeled[split:]:
                copy = dict(row)
                equivalent = any(np.max(np.abs(initial_kernels[row["layout_hash"]]-initial_kernels[t["layout_hash"]])) <= 1e-10 for t in training)
                copy["measured_train_kernel_equivalent"] = equivalent
                if row["status"] == "valid" and not equivalent:
                    copy["prediction"] = float(scorer.predict([row["layout"]])[0])
                heldout.append(copy)
            predictions = scorer.predict(pool)
            if predictions.shape != (len(pool),) or not np.isfinite(predictions).all():
                raise ScientificInvalid("Predictor returned nonfinite or incorrectly shaped candidate scores")
            if predictions.shape != (len(pool),) or not np.isfinite(predictions).all():
                raise RuntimeError("Predictor returned invalid candidate scores")
            selected_indices = sorted(range(len(pool)), key=lambda i: (-predictions[i], qc.layout_code(pool[i])))[:top]
            selected = [pool[i] for i in selected_indices]
            store.write("predictor.json", {**predictor_note, "original_train_code_count": split,
                "original_heldout_code_count": m-split, "valid_train_count": len(training), "heldout": heldout,
                "selected_pool_indices": selected_indices,
                "pool_predictions": predictions.tolist()})
        else:
            candidates = [r for r in labeled if r["status"] == "valid"]
            extra_rows = []
            for i, layout in enumerate(pool[:top]):
                k = kernel.matrix(x, None, layout, key=f"random_extra/{i}")
                initial_kernels[qc.digest(layout)] = k
                extra = {"index": i, "layout": layout, "layout_hash": qc.digest(layout)}
                try:
                    reject_constant_kernel(k)
                    extra.update(status="valid", alignment=alignment_and_gradient(k, y, task)[0])
                    candidates.append(extra)
                except ScientificInvalid as exc:
                    extra.update(status="scientific_invalid", reason=str(exc))
                extra_rows.append(extra)
            store.write("random_extra_labels.json", extra_rows)
            selected = [r["layout"] for r in sorted(candidates, key=lambda r: (-r["alignment"], qc.layout_code(r["layout"])))[:top]]
        if len(selected) != top:
            raise ScientificInvalid("Fewer valid selected layouts than registered top_k; no replacement")
        pre, finals = [], []
        for i, layout in enumerate(selected):
            k = initial_kernels.get(qc.digest(layout))
            if k is None:
                k = kernel.matrix(x, None, layout, key=f"selected/{i}/train")
            bv = kernel.matrix(xv, x, layout, key=f"selected/{i}/valid")
            try:
                row = evaluate_validation(k, bv, y, yv, valid_dates, task)
                row.update(status="valid", layout=layout, layout_hash=qc.digest(layout))
            except ScientificInvalid as exc:
                row = {"status": "scientific_invalid", "reason": str(exc), "layout": layout}
            pre.append(row)
            for config in range(count):
                mask = qc.make_mask(layout, seed, config, profile["max_parameters"])
                key = f"finetune/layout{i}/config{config}"
                row = {"layout_index": i, "configuration": config, "layout": layout,
                       "layout_hash": qc.digest(layout), "mask": mask}
                config_limit = float(profile.get("configuration_seconds", 1800))
                config_deadline = time.monotonic()+config_limit
                try:
                    tuned = finetune(kernel, x, y, layout, mask, seed, config, task, profile["steps"],
                                     profile["batch_size"], profile["learning_rate"], key, config_limit)
                    row.update(tuned)
                    if tuned["status"] != "valid":
                        raise ScientificInvalid(tuned["reason"])
                    remaining = config_limit-tuned["seconds_this_invocation"]
                    if remaining <= 0:
                        raise TimeoutError("Configuration has no remaining validation time")
                    bv = kernel.matrix(xv, x, layout, mask, tuned["parameters"], key+"/valid",
                                       min(config_deadline, time.monotonic()+remaining))
                    evaluation = evaluate_validation(np.asarray(tuned["final_train_kernel"]), bv, y, yv, valid_dates, task)
                    row.update(evaluation)
                except ScientificInvalid as exc:
                    row.update(status="scientific_invalid", reason=str(exc))
                    path = store.out.path(store.path_for(key+"/complete"))
                    if path.exists():
                        marker = json.loads(path.read_text(encoding="utf-8"))
                        if marker.get("status") == "intent":
                            trace_dir = store.out.root/"traces"/qc.digest(key)
                            completed = len(list(trace_dir.glob("step*.json"))) if trace_dir.exists() else 0
                            row["completed_steps"] = completed
                            store.finish(key+"/complete", marker, {"status": "scientific_invalid", "reason": str(exc),
                                "mask": mask, "completed_steps": completed})
                finals.append(row)
                store.write("final_candidates.json", finals)
        valid = [row for row in finals if row["status"] == "valid"]
        if not valid:
            raise ScientificInvalid("No valid final candidate; no O evaluation allowed")
        winner = min(valid, key=lambda row: (row["selection_loss"], row["layout_hash"], row["configuration"]))
        # Every valid k*c endpoint has now been scored on V. No O value selected it.
        store.write("selection_before_test.json", {"selected_layout_hash": winner["layout_hash"],
            "configuration": winner["configuration"], "validation_selection_loss": winner["selection_loss"],
            "configuration_attempts_terminal": len(finals), "valid_final_endpoints_scored": len(valid),
            "expected_configurations": top*count,
            "selected_utc": now()})
        bt = kernel.matrix(xt, x, winner["layout"], winner["mask"], winner["parameters"], "winner/test")
        prediction, decisions = predict_bundle(winner["bundle"], bt)
        yt = np.asarray(arrays["y_test"], float)
        metric = {"date_equal_accuracy": modeling.date_equal_accuracy(yt, prediction, test_dates)} if task == "classification" else {
            "date_equal_mse_raw": modeling.date_equal_mse(yt, prediction, test_dates),
            "date_equal_mse_standardized": modeling.date_equal_mse(yt, prediction, test_dates)/float(np.var(y, ddof=1)),
            "rank_ic": modeling.rank_ic(yt, prediction, test_dates)}
        if task == "classification":
            from sklearn.metrics import balanced_accuracy_score, roc_auc_score
            metric["balanced_accuracy"] = float(balanced_accuracy_score(yt, prediction))
            metric["roc_auc"] = float(roc_auc_score(yt, decisions)) if len(np.unique(yt)) == 2 else None
            counts = np.bincount(y.astype(int), minlength=2)
            majority = int(np.argmax(counts))  # ties fixed to class zero
            majority_baseline = {"training_class_counts": counts.tolist(), "predicted_class": majority,
                "tie_rule": "class_zero", "validation_date_equal_accuracy": modeling.date_equal_accuracy(yv, np.full(len(yv), majority), valid_dates),
                "test_date_equal_accuracy": modeling.date_equal_accuracy(yt, np.full(len(yt), majority), test_dates),
                "test_balanced_accuracy": float(balanced_accuracy_score(yt, np.full(len(yt), majority)))}
            store.write("majority_class_baseline.json", majority_baseline)
            summary["majority_class_baseline"] = majority_baseline
        store.write("model_bundle.json", {"source_sha256": identity["source_sha256"], "metadata": metadata,
            "layout": winner["layout"], "mask": winner["mask"], "parameters": winner["parameters"],
            "downstream": winner["bundle"], "x_train_angles": x.tolist(), "training_sample_order": list(range(len(x)))})
        store.write("predictions.json", {"metadata": test_dates, "prediction": prediction.tolist(),
            "decision": None if decisions is None else decisions.tolist(), "target": yt.tolist(), "metrics": metric})
        summary.update(workflow_status="complete", scientific_status="complete" if len(valid) == top*count else "partial_scientific_invalid",
            completed_utc=now(), seconds_this_invocation=time.monotonic()-started,
            label_code_count=len(labels), valid_labels=sum(r["status"] == "valid" for r in labeled),
            candidate_code_count=len(pool), selected_layout_count=len(selected), attempted_configurations=len(finals),
            valid_configurations=len(valid), actual_completed_steps=sum(r.get("completed_steps", 0) for r in finals),
            all_configuration_attempts_terminal=len(finals) == top*count,
            valid_final_endpoints_scored=len(valid), expected_configurations=top*count,
            all_configurations_scientifically_valid=len(valid) == top*count,
            selected_layout_hash=winner["layout_hash"],
            selected_configuration=winner["configuration"], test_metrics=metric,
            nominal_complete_alignment_evaluations_per_arm=m+top,
            new_probability_circuits_this_invocation=kernel.invocation_calls)
        store.write("pre_candidates.json", pre)
        store.write("summary.json", summary)
        marker = store.intent("workflow/complete", identity)
        return store.finish("workflow/complete", marker, summary)
    except Exception as exc:
        summary.update(workflow_status="scientific_invalid" if isinstance(exc, ScientificInvalid) else "failed_or_unknown",
                       error=type(exc).__name__+": "+str(exc), stopped_utc=now(),
                       seconds_this_invocation=time.monotonic()-started)
        store.write("summary.json", summary)
        raise
