"""Small exact quantum feature maps; NumPy only, with no data or cloud access.

Qubit 0 is the most significant bit of a computational basis index.  An input
row x prepares E(x)|0> at depth 1.  Further layers apply the chronological
CNOT chain 0->1, 1->2, ..., then E(x).  Thus depth 2 is E C E (rightmost first).
All rotations are RY(a*pi*x_j).  These gates preserve real amplitudes, although
states are stored as complex128 for an explicit, conventional state interface.
"""
from __future__ import annotations

import math
import numbers
import numpy as np


CANDIDATES = (
    {"name": "Q1", "a": 0.5, "depth": 1},
    {"name": "Q2", "a": 1.0, "depth": 1},
    {"name": "Q3", "a": 0.5, "depth": 2},
    {"name": "Q4", "a": 1.0, "depth": 2},
)
BIT_ORDER = "qubit_0_is_most_significant_bit"


def _apply_ry(states: np.ndarray, angles: np.ndarray, qubit: int, q: int) -> None:
    """Apply one RY to a batch in place without building a dense unitary."""
    blocks = states.reshape(len(states), 1 << qubit, 2, 1 << (q-qubit-1))
    zero = blocks[:, :, 0, :].copy()
    one = blocks[:, :, 1, :].copy()
    c = np.cos(angles / 2)[:, None, None]
    s = np.sin(angles / 2)[:, None, None]
    blocks[:, :, 0, :] = c * zero - s * one
    blocks[:, :, 1, :] = s * zero + c * one


def _apply_cnot(states: np.ndarray, control: int, target: int, q: int) -> None:
    """Swap target amplitudes for control=1; q0 is the leftmost bit."""
    tensor = states.reshape((len(states),) + (2,) * q)
    low = [slice(None)] * (q + 1)
    high = [slice(None)] * (q + 1)
    low[control + 1] = high[control + 1] = 1
    low[target + 1], high[target + 1] = 0, 1
    saved = tensor[tuple(low)].copy()
    tensor[tuple(low)] = tensor[tuple(high)]
    tensor[tuple(high)] = saved


def state_features(X: np.ndarray, a: float = 1.0, depth: int = 1) -> np.ndarray:
    """Return normalized complex128 states, shape (n, 2**q).

    X must be finite and real, with shape (n, q), q>=1.  depth is a positive
    integer.  No input scaling, normalization or clipping is performed here.
    The caller must budget the exponentially sized state array before use.
    """
    raw = np.asarray(X)
    if raw.ndim != 2 or raw.shape[1] < 1 or np.iscomplexobj(raw):
        raise ValueError("X must be a two-dimensional real array with q>=1")
    x = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("X must contain only finite values")
    if not isinstance(depth, numbers.Integral) or isinstance(depth, bool) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if not isinstance(a, numbers.Real) or not math.isfinite(float(a)):
        raise ValueError("a must be a finite real scalar")
    n, q = x.shape
    if q >= 63:
        raise ValueError("q>=63 cannot be indexed by this in-memory implementation")
    states = np.zeros((n, 1 << q), dtype=np.complex128)
    states[:, 0] = 1.0
    if not n:
        return states
    angles = x * (float(a) * np.pi)
    if not np.isfinite(angles).all():
        raise ValueError("rotation angles overflowed")
    for layer in range(int(depth)):
        if layer:
            for control in range(q - 1):
                _apply_cnot(states, control, control + 1, q)
        for qubit in range(q):
            _apply_ry(states, angles[:, qubit], qubit, q)
    return states


def _states_array(states: np.ndarray) -> np.ndarray:
    arr = np.asarray(states, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("states must have shape (n, 2**q), q>=1")
    d = arr.shape[1]
    if d & (d - 1):
        raise ValueError("state dimension must be a power of two")
    if not np.isfinite(arr).all():
        raise ValueError("states must be finite")
    return arr


def fidelity_kernel(states1: np.ndarray, states2: np.ndarray | None = None) -> np.ndarray:
    """Return |<psi_i|phi_j>|**2 without silently normalizing or clipping.

    Inputs should be normalized states; small rounding errors remain visible
    to verification.  Memory includes the complex overlap and real output.
    """
    left = _states_array(states1)
    right = left if states2 is None else _states_array(states2)
    if left.shape[1] != right.shape[1]:
        raise ValueError("the two state dimensions must agree")
    overlap = left.conj() @ right.T
    return np.square(np.abs(overlap))


def real_density_features(
    states: np.ndarray, *, imaginary_tolerance: float = 1e-12,
    max_output_bytes: int = 512 * 1024**2,
) -> np.ndarray:
    """Return real symmetric density features of shape (n, d*(d+1)//2).

    Columns are d diagonal entries followed by sqrt(2)*psi_i*psi_j for i<j
    in NumPy upper-triangle order. Their dot product equals the fidelity kernel
    for real normalized amplitudes. This interface rejects appreciably complex
    amplitudes, even a non-real global-phase representative; it is not a general
    complex-state density feature map. The default allocation guard prevents
    accidental use on the high-qubit state-only benchmark.
    """
    arr = _states_array(states)
    if imaginary_tolerance < 0 or not math.isfinite(imaginary_tolerance):
        raise ValueError("imaginary_tolerance must be finite and nonnegative")
    if arr.size and np.max(np.abs(arr.imag)) > imaginary_tolerance:
        raise ValueError("real_density_features requires real amplitudes")
    n, d = arr.shape
    p = d * (d + 1) // 2
    if n * p * 8 > max_output_bytes:
        raise MemoryError("requested density feature array exceeds allocation guard")
    result = np.empty((n, p), dtype=np.float64)
    if not n:
        return result
    real = arr.real
    result[:, :d] = real * real
    i, j = np.triu_indices(d, 1)
    result[:, d:] = np.sqrt(2.0) * real[:, i] * real[:, j]
    return result
