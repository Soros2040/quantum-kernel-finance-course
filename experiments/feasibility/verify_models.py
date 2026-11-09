"""Analytic checks of mean-loss KRR and compatible spectral extension."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from kernel_models import preprocess, kernel_map, predict_mapped, explicit_ridge, krr
from quantum_core import CANDIDATES, state_features, fidelity_kernel, real_density_features
from safe_io import OutputRoot


def analytic_inputs(n: int, q: int = 4) -> np.ndarray:
    row = np.arange(1, n+1, dtype=float)[:, None]
    col = np.arange(1, q+1, dtype=float)[None, :]
    return np.sin(row*np.sqrt(col)+0.17*col**2)


def run_checks(tolerance: float = 1e-10) -> dict:
    started = time.perf_counter()
    checks = []
    def record(name, error, **context):
        error = float(error)
        checks.append({"name": name, "error": error if np.isfinite(error) else None,
                       "tolerance": tolerance, "passed": bool(np.isfinite(error) and error <= tolerance),
                       **context})
    n, v = 24, 11
    x = analytic_inputs(n+v)
    y = np.sin(np.arange(n)*0.31)+0.2*np.cos(np.arange(n)*0.17)+0.7
    for candidate in CANDIDATES:
        name, a, depth = candidate["name"], candidate["a"], candidate["depth"]
        st, sv = state_features(x[:n], a, depth), state_features(x[n:], a, depth)
        train, cross = fidelity_kernel(st), fidelity_kernel(sv, st)
        ft, fv = real_density_features(st), real_density_features(sv)
        for lam in (0.001, 0.01, 0.1):
            pred, diagnostics = krr(train, cross, y, lam)
            explicit = explicit_ridge(ft, fv, y, lam, match_spectral_threshold=True)
            record("quantum_kernel_vs_matched_explicit_ridge",
                   np.max(np.abs(pred-explicit)), candidate=name, regularization=lam,
                   retained_rank=diagnostics["retained_rank"])
            shifted, _ = krr(train, cross, y+3.25, lam)
            record("unpenalized_intercept_shift",
                   np.max(np.abs(shifted-pred-3.25)), candidate=name, regularization=lam)
    # Verify the n*lambda convention independently in already mapped features.
    zt = analytic_inputs(13, 5)
    zv = analytic_inputs(7, 5)*0.8
    training_mean = zt.mean(axis=0)
    zt, zv = zt-training_mean, zv-training_mean
    yy = np.cos(np.arange(len(zt))*0.4)+0.3
    for lam in (0.001, 0.01, 0.1):
        dual = predict_mapped(zt, zv, yy, lam)
        centered_y = yy-yy.mean()
        beta = np.linalg.solve(zt.T@zt+len(yy)*lam*np.eye(zt.shape[1]), zt.T@centered_y)
        direct_dual = zv@zt.T@np.linalg.solve(
            zt@zt.T+len(yy)*lam*np.eye(len(yy)), centered_y)+yy.mean()
        record("mean_loss_primal_dual", np.max(np.abs(dual-(zv@beta+yy.mean()))),
               regularization=lam)
        record("mapped_prediction_vs_direct_dual", np.max(np.abs(dual-direct_dual)),
               regularization=lam)
    # A genuinely indefinite training matrix and arbitrary rectangular cross block.
    ortho, _ = np.linalg.qr(analytic_inputs(6, 6))
    eig = np.array([3., 1., 0.2, -0.4, -0.1, 0.])
    train = (ortho*eig)@ortho.T
    cross = analytic_inputs(4, 6)
    zt, zv, diagnostics = kernel_map(train, cross)
    d, q = np.linalg.eigh((train+train.T)/2)
    keep = d > 1e-8*max(1., np.max(np.abs(d)))
    kp = (q[:, keep]*d[keep])@q[:, keep].T
    cp = cross@q[:, keep]@q[:, keep].T
    h = np.eye(len(train))-np.ones_like(train)/len(train)
    kc = h@kp@h
    scale = np.trace(kc)/len(train)
    cc = cp-cp.mean(axis=1)[:, None]-kp.mean(axis=0)[None, :]+kp.mean()
    record("spectral_repair_centered_train", np.max(np.abs(zt@zt.T-kc/scale)))
    record("spectral_extension_centered_cross", np.max(np.abs(zv@zt.T-cc/scale)))
    joint = np.vstack((zt, zv))
    record("extended_joint_gram_psd", max(0., -np.linalg.eigvalsh(joint@joint.T).min()))
    record("known_indefinite_retained_rank", abs(diagnostics["retained_rank"]-3))
    record("known_indefinite_negative_count", abs(diagnostics["negative_eigenvalues"]-2))
    # Singular directions below the training threshold must also be removed in
    # explicit out-of-sample features, rather than revived by evaluation data.
    basis, _ = np.linalg.qr(analytic_inputs(9, 4))
    ft = basis@np.diag([3., 1., 1e-5, 1e-9])
    fv = analytic_inputs(5, 4)
    yy = np.sin(np.arange(9)*0.4)
    p, diagnostics = krr(ft@ft.T, fv@ft.T, yy, 0.01)
    explicit = explicit_ridge(ft, fv, yy, 0.01, match_spectral_threshold=True)
    record("near_null_training_spectrum_prediction", np.max(np.abs(p-explicit)))
    record("near_null_training_spectrum_rank", abs(diagnostics["retained_rank"]-2))
    for label, bad_train in (("zero", np.zeros((4, 4))), ("constant", np.ones((4, 4)))):
        rejected = False
        try:
            kernel_map(bad_train, np.ones((2, 4)))
        except ValueError:
            rejected = True
        record("reject_degenerate_"+label+"_kernel", 0 if rejected else 1)
    # An evaluation distribution change cannot refit the training preprocessing.
    tx, ex, target, prep = preprocess(x[:n], x[n:], y)
    tx2, _, target2, prep2 = preprocess(x[:n], x[n:]*100+50, y)
    record("evaluation_does_not_refit_training_transform", np.max(np.abs(tx-tx2)))
    record("evaluation_does_not_refit_target_transform", np.max(np.abs(target-target2)))
    record("evaluation_does_not_refit_training_metadata", 0 if prep == prep2 else 1)
    source = Path(__file__).resolve().parent
    return {
        "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "check_count": len(checks), "checks": checks,
        "maximum_numerical_error": max(c["error"] for c in checks if c["error"] is not None),
        "elapsed_seconds": time.perf_counter()-started,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "input_kind": "deterministic_analytic_nonfinancial",
        "python_version": sys.version, "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "platform": platform.platform(), "thread_limit": 4,
        "source_sha256": {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                          for name in ("verify_models.py", "kernel_models.py",
                                       "quantum_core.py", "safe_io.py")},
        "limitations": [
            "Checks use analytic nonfinancial inputs and do not validate historical data or financial forecasting.",
            "No guarantee of differentiability at spectral truncation boundaries is claimed.",
            "The explicit ridge comparison matches the training spectral threshold, centering and scale.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = OutputRoot(args.output.parent)
    destination = output.path(args.output.name)
    report = run_checks()
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                     ("status", "check_count", "maximum_numerical_error", "elapsed_seconds")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
