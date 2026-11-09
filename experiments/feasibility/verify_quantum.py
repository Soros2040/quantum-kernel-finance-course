"""Independent small-matrix checks of quantum_core; analytic inputs only."""
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
from quantum_core import BIT_ORDER, fidelity_kernel, real_density_features, state_features
from safe_io import OutputRoot


def dense_ry(theta: float, qubit: int, q: int) -> np.ndarray:
    gate = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
    result = np.ones((1, 1), dtype=np.complex128)
    for k in range(q):
        result = np.kron(result, gate if k == qubit else np.eye(2))
    return result


def dense_cnot(control: int, target: int, q: int) -> np.ndarray:
    result = np.zeros((2**q, 2**q), dtype=np.complex128)
    for col in range(2**q):
        bits = list(format(col, f"0{q}b"))
        if bits[control] == "1":
            bits[target] = "1" if bits[target] == "0" else "0"
        result[int("".join(bits), 2), col] = 1
    return result


def chronological_gates(x: np.ndarray, a: float, depth: int) -> list[np.ndarray]:
    q, gates = len(x), []
    for layer in range(depth):
        if layer:
            gates.extend(dense_cnot(k, k+1, q) for k in range(q-1))
        gates.extend(dense_ry(a*np.pi*x[k], k, q) for k in range(q))
    return gates


def dense_state(x: np.ndarray, a: float, depth: int) -> np.ndarray:
    result = np.zeros(2**len(x), dtype=np.complex128)
    result[0] = 1
    for gate in chronological_gates(x, a, depth):
        result = gate @ result
    return result


def run_checks(tolerance: float = 1e-10) -> dict:
    start = time.perf_counter()
    checks = []
    def record(name: str, error: float, **context) -> None:
        checks.append({"name": name, "error": float(error),
                       "tolerance": tolerance, "passed": bool(error <= tolerance),
                       **context})
    for q in (1, 2, 4):
        x = np.sin(np.arange(5*q, dtype=float).reshape(5, q) * 0.37 + 0.2)
        for a in (0.5, 1.0):
            for depth in (1, 2, 4):
                context = {"qubits": q, "a": a, "depth": depth}
                states = state_features(x, a, depth)
                dense = np.stack([dense_state(row, a, depth) for row in x])
                record("state_vs_independent_dense_gates",
                       np.max(np.abs(states-dense)), **context)
                record("normalization", np.max(np.abs(
                    np.sum(np.abs(states)**2, axis=1)-1)), **context)
                kernel = fidelity_kernel(states)
                record("diagonal_one", np.max(np.abs(np.diag(kernel)-1)), **context)
                record("symmetry", np.max(np.abs(kernel-kernel.T)), **context)
                record("positive_semidefinite",
                       max(0.0, -np.linalg.eigvalsh(kernel).min()), **context)
                phi = real_density_features(states)
                record("density_feature_kernel",
                       np.max(np.abs(phi @ phi.T-kernel)), **context)
                record("cross_kernel", np.max(np.abs(
                    fidelity_kernel(states[:2], states[2:])-kernel[:2, 2:])), **context)
                overlap = dense_state(x[0], a, depth)
                for gate in reversed(chronological_gates(x[1], a, depth)):
                    overlap = gate.conj().T @ overlap
                record("inverse_order_overlap_amplitude",
                       abs(overlap[0]-np.vdot(states[1], states[0])), **context)
                record("overlap_zero_probability", abs(
                    abs(overlap[0])**2-kernel[0, 1]), **context)
                if depth == 1:
                    analytic = np.prod(np.cos(a*np.pi*(
                        x[:, None, :]-x[None, :, :])/2)**2, axis=2)
                    record("product_cosine_kernel",
                           np.max(np.abs(analytic-kernel)), **context)
    # A direct bit-order witness: RY(pi) on q0 prepares |1000>.
    witness = state_features(np.array([[1., 0., 0., 0.]]), 1., 1)[0]
    expected = np.zeros(16, dtype=np.complex128)
    expected[8] = 1
    record("big_endian_qubit_zero_witness", np.max(np.abs(witness-expected)))
    # Validation must fail loudly instead of accepting unsupported complex states.
    rejected = False
    try:
        real_density_features(np.array([[1/np.sqrt(2), 1j/np.sqrt(2)]]))
    except ValueError:
        rejected = True
    record("reject_nonreal_density_input", 0 if rejected else 1)
    core = Path(__file__).with_name("quantum_core.py")
    return {
        "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "checks": checks, "check_count": len(checks),
        "maximum_numerical_error": max(c["error"] for c in checks),
        "elapsed_seconds": time.perf_counter()-start,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "input_kind": "deterministic_analytic_nonfinancial",
        "bit_order": BIT_ORDER, "state_dtype": "complex128",
        "python_version": sys.version, "numpy_version": np.__version__,
        "platform": platform.platform(), "thread_limit": 4,
        "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
        "verification_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "safe_io_sha256": hashlib.sha256(Path(__file__).with_name("safe_io.py").read_bytes()).hexdigest(),
        "limitations": [
            "Dense reference unitaries are only constructed for q<=4.",
            "No finance data, fitted KRR, hardware or cloud service is tested.",
            "Passing checks is numerical agreement on these inputs, not a proof for every input.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = OutputRoot(args.output.parent).path(args.output.name)
    report = run_checks()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                     ("status", "check_count", "maximum_numerical_error", "elapsed_seconds")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
