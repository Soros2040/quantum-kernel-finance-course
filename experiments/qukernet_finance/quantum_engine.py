"""QuKerNet-style repeated-block quantum kernels using PennyLane 0.33.1.

Inputs are already rotation angles. Wire 0 is the most significant state bit;
axes[0] applies to zero-based even wires, axes[1] to odd wires. Every block
applies all rotations followed by adjacent CNOTs in increasing wire order.
The identical block layout is repeated ``depth`` times, including its last
entangling layer. The simulator and differentiation are PennyLane primitives.
"""
from functools import lru_cache
from itertools import product
from contextlib import contextmanager
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class DegenerateKernel(ValueError):
    """The requested alignment or fit has no admissible numerical definition."""


KERNEL_VARIANCE_TOL = 1e-12
LABEL_STD_TOL = 1e-12


def _layout_key(layout):
    q, depth = int(layout["q"]), int(layout["depth"])
    axes = tuple(int(a) for a in layout["axes"])
    edges = tuple(int(e) for e in layout["edges"])
    if q < 2 or depth < 1 or len(axes) != 2 or any(a not in (0, 1, 2) for a in axes):
        raise ValueError("Invalid qubit count, depth, or rotation axes")
    if len(edges) != q - 1 or any(e not in (0, 1) for e in edges):
        raise ValueError("edges must have q-1 binary entries")
    return q, depth, axes, edges


def all_layouts(q=8, max_depth=5):
    """Enumerate canonical layouts: depth, even axis, odd axis, edge bits.

    This enumerates layouts, not distinct kernels; equivalent and degenerate
    kernels remain in the search space. There are 5760 layouts for q=8, d<=5.
    """
    if int(q) != q or q < 2 or int(max_depth) != max_depth or max_depth < 1:
        raise ValueError("q>=2 and max_depth>=1 must be integers")
    result = []
    for depth in range(1, int(max_depth) + 1):
        for a0, a1 in product(range(3), repeat=2):
            for edges in product((0, 1), repeat=int(q) - 1):
                result.append({"q": int(q), "depth": depth, "axes": [a0, a1],
                               "edges": list(edges),
                               "id": f"q{q}_d{depth}_a{a0}{a1}_e{''.join(map(str, edges))}"})
    return result


def _mask_tuple(mask, q, depth):
    raw = () if mask is None else tuple(mask)
    if any(isinstance(i, bool) or int(i) != i for i in raw):
        raise ValueError("Mask indices must be integers")
    mask = tuple(int(i) for i in raw)
    if len(set(mask)) != len(mask) or any(i < 0 or i >= q * depth for i in mask):
        raise ValueError("Mask indices must be unique and inside depth*q")
    return mask


def make_mask(q, depth, seed, config_index):
    """Draw 1..7 replacement gates; independent of axes, edges and layout ID."""
    q, depth = int(q), int(depth)
    if q < 2 or depth < 1 or int(seed) < 0 or int(config_index) < 0:
        raise ValueError("Invalid mask generation configuration")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), q, depth, int(config_index)]))
    count = int(rng.integers(1, min(7, q * depth - 1) + 1))
    return sorted(int(i) for i in rng.choice(q * depth, size=count, replace=False))


def _input_angles(x_angles, q):
    x = np.asarray(x_angles, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != q or x.shape[0] < 1 or not np.isfinite(x).all():
        raise ValueError("x_angles must be a finite nonempty [n,q] matrix")
    return x


def _apply_map(x, params, key, mask):
    q, depth, axes, edges = key
    replacements = {position: j for j, position in enumerate(mask)}
    gates = (qml.RX, qml.RY, qml.RZ)
    for layer in range(depth):
        for wire in range(q):
            position = layer * q + wire
            angle = params[replacements[position]] if position in replacements else x[wire]
            gates[axes[wire % 2]](angle, wires=wire)
        for wire, present in enumerate(edges):
            if present:
                qml.CNOT(wires=[wire, wire + 1])


@lru_cache(maxsize=32)
def _state_qnode(key, mask):
    if qml.__version__ != "0.33.1":
        raise RuntimeError(f"This frozen experiment requires PennyLane 0.33.1; got {qml.__version__}")
    device = qml.device("default.qubit", wires=key[0], shots=None)

    def circuit(x, params):
        _apply_map(x, params, key, mask)
        return qml.state()

    return qml.QNode(circuit, device, interface="autograd", diff_method="backprop", cache=False)


def states(x_angles, layout, mask=None, params=None):
    """Return exact PennyLane complex128 states, shape [n,2**q]."""
    key = _layout_key(layout)
    x = _input_angles(x_angles, key[0])
    mask = _mask_tuple(mask, key[0], key[1])
    if params is None:
        if mask:
            raise ValueError("params are required for a nonempty mask")
        params = []
    parameters = np.asarray(params, dtype=np.float64)
    if parameters.shape != (len(mask),) or not np.isfinite(parameters).all():
        raise ValueError("params must be finite and follow the supplied mask order")
    node = _state_qnode(key, mask)
    result = np.stack([np.asarray(node(pnp.array(row, requires_grad=False),
                                         pnp.array(parameters, requires_grad=False))) for row in x])
    if result.dtype != np.complex128:
        raise RuntimeError(f"Expected complex128 simulation, got {result.dtype}")
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite state amplitudes")
    return result


def gram(states_left, states_right=None):
    """Fidelity Gram/cross matrix from cached state rows; no new circuits."""
    left = np.asarray(states_left, dtype=np.complex128)
    right = left if states_right is None else np.asarray(states_right, dtype=np.complex128)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("State arrays must have matching Hilbert dimensions")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise FloatingPointError("Nonfinite cached states")
    overlaps = left.conj() @ right.T
    return np.asarray(np.real(overlaps.conj() * overlaps), dtype=np.float64)


def _checked_labels(y, n, task):
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (n,) or n < 2 or not np.isfinite(y).all():
        raise DegenerateKernel("Finite labels and at least two samples are required")
    if task == "regression":
        if np.std(y, ddof=1) <= LABEL_STD_TOL:
            raise DegenerateKernel("Training/batch labels have negligible variance")
    elif task == "classification":
        if len(np.unique(y)) != 2:
            raise DegenerateKernel("Binary classification requires both label values")
    else:
        raise ValueError("task must be regression or classification")
    return y


def _alignment_autograd(K, y, task):
    # Labels are fixed data, never differentiation arguments. In PL 0.33.1 a
    # requires_grad=False pennylane scalar can intercept ArrayBox arithmetic;
    # keep target arrays as plain NumPy and scalar normalizers as Python float.
    fixed_y = np.asarray(y, dtype=np.float64)
    if task == "regression":
        C = K - pnp.mean(K, axis=0, keepdims=True) - pnp.mean(K, axis=1, keepdims=True) + pnp.mean(K)
        yc = fixed_y - fixed_y.mean()
        target = np.outer(yc, yc)
        label_norm = float(np.dot(yc, yc))
        return pnp.sum(C * target) / (pnp.sqrt(pnp.sum(C * C)) * label_norm)
    target = 2.0 * (fixed_y[:, None] == fixed_y[None, :]) - 1.0
    return pnp.sum(K * target) / (float(len(fixed_y)) * pnp.sqrt(pnp.sum(K * K)))


def alignment(K, y, task="regression"):
    """Raw-kernel alignment; never projects or clips the optimization kernel.

    Regression uses centered kernel/target alignment. Classification uses the
    binary same-class target J=2*same-1 and normalization n*||K||_F. Both reject
    constant centered kernels; this is an explicit numerical admissibility rule.
    """
    K = np.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or not np.isfinite(K).all():
        raise DegenerateKernel("Kernel must be finite and square")
    if np.max(np.abs(np.imag(K))) > 1e-12:
        raise DegenerateKernel("Fidelity kernel has a nonnegligible imaginary part")
    K = np.asarray(np.real(K), dtype=np.float64)
    if not np.allclose(K, K.T, rtol=0, atol=1e-10):
        raise DegenerateKernel("Kernel is not symmetric")
    y = _checked_labels(y, len(K), task)
    C = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    if np.trace(C) / len(K) <= KERNEL_VARIANCE_TOL or np.linalg.norm(C) <= 1e-12:
        raise DegenerateKernel("Centered kernel has negligible feature variance")
    if np.linalg.norm(K) <= 1e-12:
        raise DegenerateKernel("Kernel norm is zero")
    value = float(_alignment_autograd(pnp.array(K, requires_grad=False), y, task))
    if not np.isfinite(value):
        raise DegenerateKernel("Alignment is nonfinite")
    return value


def differentiable_alignment(x_angles, y, layout, mask, params, task="regression"):
    """Autograd scalar for diagnostics and optimization; params are shared.

    Callers must check admissibility separately. Both states in each inner
    product depend on the same params, so autodiff includes both dependencies.
    """
    key = _layout_key(layout)
    mask = _mask_tuple(mask, key[0], key[1])
    node = _state_qnode(key, mask)
    s = pnp.stack([node(pnp.array(row, requires_grad=False), params) for row in x_angles])
    overlaps = pnp.dot(pnp.conj(s), pnp.transpose(s))
    K = pnp.real(pnp.conj(overlaps) * overlaps)
    return _alignment_autograd(K, y, task)


def finetune(x_angles, y, layout, mask, seed, task="regression", steps=30,
             batch_size=16, lr=0.2):
    """Fixed-step minibatch GD; sampling is with replacement.

    Parameters start at the training mean of their replaced feature angle.
    There is no validation-based checkpoint choice or retry. Numerical failure
    returns status=invalid. Timeouts and other execution errors propagate to the
    runner, rather than being reclassified as invalid scientific candidates.
    """
    start = time.perf_counter()
    key = _layout_key(layout)
    x = _input_angles(x_angles, key[0])
    mask = _mask_tuple(mask, key[0], key[1])
    if not mask or steps < 0 or batch_size < 2 or lr <= 0:
        raise ValueError("Nonempty mask, nonnegative steps, batch>=2 and lr>0 required")
    initial = [float(x[:, position % key[0]].mean()) for position in mask]
    params = pnp.array(initial, requires_grad=True)
    trace = []
    counts = {"full_training_state_batches": 0, "diagnostic_batch_state_batches": 0,
              "gradient_calls": 0, "explicit_state_vectors": 0,
              "gradient_state_vectors": 0}
    timings = {"initial_state_kta_seconds": 0.0, "diagnostic_seconds": 0.0,
               "gradient_seconds": 0.0, "update_seconds": 0.0,
               "final_state_kta_seconds": 0.0}

    @contextmanager
    def measured(name):
        tick = time.perf_counter()
        try:
            yield
        finally:
            timings[name] += time.perf_counter() - tick

    result = {"status": "invalid", "mask": list(mask), "initial_params": initial,
              "params": initial[:], "initial_alignment": None, "final_alignment": None,
              "final_train_kernel": None, "resource_counts": counts,
              "timings": timings,
              "execution_mode": {"interface": "autograd", "diff_method": "backprop",
                                 "jit": False, "compilation_seconds": None,
                                 "compilation_note": "Not applicable: no JAX/JIT path is used",
                                 "initial_timing_note": "Initial full batch can include QNode setup; not an isolated first-call compilation measurement"},
              "trace": trace, "history": trace, "seconds": None, "invalid_reason": None,
              "completed_steps": 0, "seed": int(seed), "task": task,
              "steps_requested": int(steps), "batch_size": int(batch_size), "lr": float(lr)}
    try:
        y = _checked_labels(y, len(x), task)
        with measured("initial_state_kta_seconds"):
            initial_states = states(x, layout, mask, params)
            counts["full_training_state_batches"] += 1
            counts["explicit_state_vectors"] += len(x)
            result["initial_alignment"] = alignment(gram(initial_states), y, task)
        rng = np.random.default_rng(int(seed))
        optimizer = qml.GradientDescentOptimizer(stepsize=float(lr))
        for step in range(int(steps)):
            indices = rng.integers(0, len(x), size=int(batch_size))
            xb, yb = x[indices], _checked_labels(y[indices], len(indices), task)
            diagnostic_before = timings["diagnostic_seconds"]
            with measured("diagnostic_seconds"):
                batch_states = states(xb, layout, mask, params)
                counts["diagnostic_batch_state_batches"] += 1
                counts["explicit_state_vectors"] += len(xb)
                before = alignment(gram(batch_states), yb, task)
            diagnostic_seconds = timings["diagnostic_seconds"] - diagnostic_before

            def objective(parameters):
                return -differentiable_alignment(xb, yb, layout, mask, parameters, task)

            gradient_before = timings["gradient_seconds"]
            with measured("gradient_seconds"):
                grad_fn = qml.grad(objective)
                counts["gradient_calls"] += 1
                counts["gradient_state_vectors"] += len(xb)
                gradient = grad_fn(params)
            gradient_seconds = timings["gradient_seconds"] - gradient_before
            if not np.isfinite(np.asarray(gradient)).all():
                raise FloatingPointError("Nonfinite alignment gradient")
            # Supply the computed gradient to the official optimizer; avoid a
            # second gradient evaluation while retaining its update semantics.
            update_before = timings["update_seconds"]
            with measured("update_seconds"):
                params = optimizer.step(objective, params, grad_fn=lambda _: gradient)
            update_seconds = timings["update_seconds"] - update_before
            if not np.isfinite(np.asarray(params)).all():
                raise FloatingPointError("Nonfinite updated parameters")
            trace.append({"step": step + 1, "batch_indices": indices.tolist(),
                          "alignment_before": before,
                          "gradient_norm": float(np.linalg.norm(np.asarray(gradient))),
                          "diagnostic_seconds": diagnostic_seconds,
                          "gradient_seconds": gradient_seconds,
                          "update_seconds": update_seconds})
            result["completed_steps"] = step + 1
        with measured("final_state_kta_seconds"):
            final_states = states(x, layout, mask, params)
            counts["full_training_state_batches"] += 1
            counts["explicit_state_vectors"] += len(x)
            final_kernel = gram(final_states)
            result["final_train_kernel"] = final_kernel.tolist()
            result["final_alignment"] = alignment(final_kernel, y, task)
        result["status"] = "ok"
    except (DegenerateKernel, FloatingPointError) as error:
        result["invalid_reason"] = f"{type(error).__name__}: {error}"
    result["params"] = np.asarray(params, dtype=np.float64).tolist()
    result["seconds"] = time.perf_counter() - start
    return result
