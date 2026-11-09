"""Independent numerical checks for the frozen quantum engine.

Run as a module/script and capture stdout outside the source tree. No datasets,
network calls, cloud submissions, or output files are created by this module.
The dense reference is deliberately only a two-qubit correctness fixture.
"""
import json
import sys
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

try:
    from . import quantum_engine as engine
except ImportError:
    import quantum_engine as engine


def direct_overlap(x, xp, layout, mask=None, params=None):
    """Independent overlap QNode; explicit reverse-order inverse gate list."""
    q = layout["q"]
    replacements = dict(zip(mask or [], params if params is not None else []))
    rotations = (qml.RX, qml.RY, qml.RZ)

    def apply(row, inverse):
        layers = range(layout["depth"] - 1, -1, -1) if inverse else range(layout["depth"])
        for layer in layers:
            if inverse:
                for wire in range(q - 2, -1, -1):
                    if layout["edges"][wire]:
                        qml.CNOT(wires=[wire, wire + 1])
            wires = range(q - 1, -1, -1) if inverse else range(q)
            for wire in wires:
                angle = replacements.get(layer * q + wire, row[wire])
                rotations[layout["axes"][wire % 2]](-angle if inverse else angle, wires=wire)
            if not inverse:
                for wire in range(q - 1):
                    if layout["edges"][wire]:
                        qml.CNOT(wires=[wire, wire + 1])

    def circuit():
        apply(x, False)
        apply(xp, True)
        return qml.probs(wires=range(q))

    node = qml.QNode(circuit, qml.device("default.qubit", wires=q, shots=None),
                     interface=None, diff_method=None)
    return float(node()[0])


def _dense_two_qubit_reference(x, layout, mask=None, params=None):
    if layout["q"] != 2:
        raise ValueError("The independent dense fixture is restricted to two qubits")
    paulis = (np.array([[0, 1], [1, 0]], dtype=complex),
              np.array([[0, -1j], [1j, 0]], dtype=complex),
              np.diag([1, -1]).astype(complex))
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    replacements = dict(zip(mask or [], params if params is not None else []))
    state = np.array([1, 0, 0, 0], dtype=complex)
    for layer in range(layout["depth"]):
        gates = []
        for wire in range(2):
            angle = replacements.get(2 * layer + wire, x[wire])
            gates.append(np.cos(angle / 2) * np.eye(2) -
                         1j * np.sin(angle / 2) * paulis[layout["axes"][wire]])
        state = np.kron(gates[0], gates[1]) @ state
        if layout["edges"][0]:
            state = cnot @ state
    return state


def verify_complex_feature_krr():
    """Dual KRR vs Hermitian density-feature primal and spectral extension."""
    rng = np.random.default_rng(801)
    layout = {"q": 2, "depth": 3, "axes": [0, 1], "edges": [1], "id": "fixture"}
    state = engine.states(rng.normal(size=(9, 2)), layout)
    features = np.stack([np.outer(s, s.conj()).reshape(-1) for s in state])
    train, valid = features[:6], features[6:]
    K = engine.gram(state[:6])
    B = engine.gram(state[6:], state[:6])
    kernel_error = float(np.max(np.abs(train @ train.conj().T - K)))
    mean = train.mean(axis=0)
    train, valid = train - mean, valid - mean
    scale = np.mean(np.sum(np.abs(train) ** 2, axis=1))
    train, valid = train / np.sqrt(scale), valid / np.sqrt(scale)
    y = np.array([0.4, -0.2, 0.1, -0.5, 0.8, -0.3])
    y = y - y.mean()
    n, lam = len(y), 0.03
    dual = np.linalg.solve(train @ train.conj().T + n * lam * np.eye(n), y)
    dual_pred = valid @ train.conj().T @ dual
    weights = np.linalg.solve(train.conj().T @ train + n * lam * np.eye(train.shape[1]),
                              train.conj().T @ y)
    primal_pred = valid @ weights
    values, vectors = np.linalg.eigh(K)
    keep = values > 1e-8 * max(1.0, np.max(np.abs(values)))
    Z = vectors[:, keep] * np.sqrt(values[keep])
    Z_valid = (B @ vectors[:, keep]) / np.sqrt(values[keep])
    zmean = Z.mean(axis=0)
    Z, Z_valid = Z - zmean, Z_valid - zmean
    zscale = np.mean(np.sum(Z * Z, axis=1))
    Z, Z_valid = Z / np.sqrt(zscale), Z_valid / np.sqrt(zscale)
    spectral_pred = Z_valid @ Z.T @ np.linalg.solve(Z @ Z.T + n * lam * np.eye(n), y)
    return {"density_kernel_error": kernel_error,
            "primal_dual_error": float(np.max(np.abs(primal_pred - dual_pred))),
            "spectral_dual_error": float(np.max(np.abs(spectral_pred - dual_pred))),
            "prediction_imaginary_residue": float(np.max(np.abs(np.imag(dual_pred))))}


def run_checks():
    start = time.perf_counter()
    checks = []

    def check(name, error, tolerance=1e-10):
        checks.append({"name": name, "error": float(error), "tolerance": float(tolerance),
                       "passed": bool(np.isfinite(error) and error <= tolerance)})

    layouts = engine.all_layouts()
    check("5760_layouts", abs(len(layouts) - 5760), 0)
    check("canonical_unique_ids", len(layouts) - len({v["id"] for v in layouts}), 0)
    check("5760_unique_structures", len(layouts) - len({(v["depth"], tuple(v["axes"]),
                                                       tuple(v["edges"])) for v in layouts}), 0)
    rng = np.random.default_rng(371)
    for axes in ([0, 1], [1, 2], [2, 0]):
        layout = {"q": 2, "depth": 3, "axes": axes, "edges": [1], "id": "dense_fixture"}
        x = rng.normal(size=(4, 2))
        mask, params = [1, 4], [0.43, -0.72]
        actual = engine.states(x, layout, mask, params)
        reference = np.stack([_dense_two_qubit_reference(row, layout, mask, params) for row in x])
        check(f"dense_state_axes_{axes}", np.max(np.abs(actual - reference)))
        K = engine.gram(actual)
        check(f"PSD_axes_{axes}", max(0, -np.linalg.eigvalsh(K).min()))
        check(f"unit_diagonal_axes_{axes}", np.max(np.abs(np.diag(K) - 1)))
        for i, j in [(0, 0), (0, 1), (2, 3)]:
            overlap = direct_overlap(x[i], x[j], layout, mask, params)
            check(f"independent_overlap_axes_{axes}_{i}_{j}", abs(K[i, j] - overlap))

    layout = {"q": 8, "depth": 1, "axes": [1, 1], "edges": [1] * 7, "id": "analytic"}
    x = rng.normal(size=(3, 8))
    K = engine.gram(engine.states(x, layout))
    analytic = np.prod(np.cos((x[:, None, :] - x[None, :, :]) / 2) ** 2, axis=2)
    check("8q_RY_product_analytic_common_entangler_cancels", np.max(np.abs(K - analytic)))
    marker_layout = {"q": 8, "depth": 1, "axes": [1, 1], "edges": [0] * 7, "id": "marker"}
    marker = np.zeros((1, 8))
    marker[0, 0] = np.pi
    expected = np.zeros(256, dtype=complex)
    expected[128] = 1
    check("wire0_is_most_significant", np.max(np.abs(engine.states(marker, marker_layout)[0] - expected)))

    for depth in range(1, 6):
        for config in range(5):
            mask = engine.make_mask(8, depth, 42, config)
            valid = 1 <= len(mask) <= 7 and len(set(mask)) == len(mask) and 0 <= min(mask) <= max(mask) < 8 * depth
            check(f"mask_valid_d{depth}_c{config}", 0 if valid else 1, 0)
            check(f"mask_reproducible_d{depth}_c{config}",
                  0 if mask == engine.make_mask(8, depth, 42, config) else 1, 0)
        layout = {"q": 8, "depth": depth, "axes": [0, 1], "edges": [1, 0, 1, 1, 0, 1, 0], "id": f"reload_{depth}"}
        mask = engine.make_mask(8, depth, 42, 0)
        params = [float(x[:, position % 8].mean()) for position in mask]
        restored = json.loads(json.dumps({"layout": layout, "mask": mask, "params": params}))
        a = engine.states(x[:2], layout, mask, params)
        b = engine.states(x[:2], restored["layout"], restored["mask"], restored["params"])
        check(f"parameter_reload_depth{depth}", np.max(np.abs(a - b)))
        check(f"complex128_depth{depth}", 0 if a.dtype == np.complex128 else 1, 0)
        check(f"8q_overlap_depth{depth}", abs(engine.gram(a)[0, 1] -
              direct_overlap(x[0], x[1], layout, mask, params)))

    layout = {"q": 2, "depth": 3, "axes": [0, 1], "edges": [1], "id": "gradient"}
    x = rng.normal(size=(7, 2))
    mask = [0, 3]
    parameters = pnp.array([0.23, -0.51], requires_grad=True)
    for task, y in [("regression", np.array([-.7, .2, .9, -.2, .1, .4, -.8])),
                    ("classification", np.array([0, 1, 1, 0, 0, 1, 0]))]:
        objective = lambda p: engine.differentiable_alignment(x, y, layout, mask, p, task)
        autodiff = np.asarray(qml.grad(objective)(parameters))
        finite_difference = []
        for j in range(len(parameters)):
            delta = np.zeros(len(parameters))
            delta[j] = 1e-6
            finite_difference.append((float(objective(parameters + delta)) -
                                      float(objective(parameters - delta))) / 2e-6)
        check(f"shared_parameter_gradient_{task}", np.max(np.abs(autodiff - finite_difference)), 2e-7)
        K = engine.gram(engine.states(x, layout, mask, parameters))
        value = engine.alignment(K, y, task)
        check(f"public_differentiable_alignment_{task}", abs(value - float(objective(parameters))))
        if task == "regression":
            check("centered_alignment_label_shift_scale", abs(value - engine.alignment(K, 3*y + 5, task)))
            check("centered_alignment_kernel_scale", abs(value - engine.alignment(2*K, y, task)))
        else:
            target = 2 * (y[:, None] == y[None, :]) - 1
            reference = np.sum(K * target) / (len(y) * np.linalg.norm(K))
            check("classification_target_formula", abs(value - reference))
        first = engine.finetune(x, y, layout, mask, 903, task, steps=2, batch_size=16)
        second = engine.finetune(x, y, layout, mask, 903, task, steps=2, batch_size=16)
        check(f"finetune_ok_{task}", 0 if first["status"] == second["status"] == "ok" else 1, 0)
        check(f"finetune_reproducible_{task}", np.max(np.abs(np.array(first["params"]) - second["params"])))
        if first["status"] == "ok":
            final = engine.alignment(engine.gram(engine.states(x, layout, mask, first["params"])), y, task)
            check(f"finetune_full_endpoint_{task}", abs(final - first["final_alignment"]))
        expected_initial = [float(x[:, position % 2].mean()) for position in mask]
        check(f"feature_mean_initialization_{task}", np.max(np.abs(np.array(first["initial_params"]) - expected_initial)))

    for name, K, y in [("constant_kernel", np.ones((4, 4)), np.arange(4)),
                       ("constant_target", np.eye(4), np.ones(4))]:
        try:
            engine.alignment(K, y)
            rejected = False
        except engine.DegenerateKernel:
            rejected = True
        check(f"reject_{name}", 0 if rejected else 1, 0)
    for name, error in verify_complex_feature_krr().items():
        check(name, error, 1e-8)
    return {"status": "passed" if all(c["passed"] for c in checks) else "failed",
            "pennylane_version": qml.__version__, "numpy_version": np.__version__,
            "checks_count": len(checks), "failed_count": sum(not c["passed"] for c in checks),
            "seconds": time.perf_counter() - start, "checks": checks}


if __name__ == "__main__":
    report = run_checks()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    sys.exit(0 if report["status"] == "passed" else 1)
