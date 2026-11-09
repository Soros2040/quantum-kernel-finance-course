"""QCIS circuits for QuKerNet. No simulator or cloud submission lives here.

Internal wires are zero based; QCIS labels are Q1..Qq. State-vector tests use
wire 0 as the most significant bit. Each feature block has its own sampled
layout, repeated ``depth`` times before moving to the next feature block.
"""
from __future__ import annotations
import hashlib
import json
import math
from typing import Iterable
import numpy as np

AXES = ("RX", "RY", "RZ")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_layout(layout):
    q, p, depth = (int(layout[k]) for k in ("qubits", "features", "depth"))
    if q < 1 or p < 1 or depth < 1:
        raise ValueError("Positive qubits/features/depth required")
    blocks = layout["blocks"]
    if len(blocks) != math.ceil(p / q):
        raise ValueError("One independently specified block per feature chunk required")
    for block in blocks:
        if len(block["axes"]) != q or any(a not in AXES for a in block["axes"]):
            raise ValueError("Invalid rotation axes")
        if len(block["entanglers"]) != max(0, q-1) or any(v not in (0, 1) for v in block["entanglers"]):
            raise ValueError("Only adjacent directed CNOT chain bits are supported")
    return q, p, depth


def layout_code(layout):
    validate_layout(layout)
    return canonical({k: layout[k] for k in ("qubits", "features", "depth", "blocks")})


def sample_layout(rng, qubits, features, depths=(1, 2, 3, 4, 5)):
    depth = int(rng.choice(depths))
    blocks = []
    for _ in range(math.ceil(features / qubits)):
        even, odd = rng.choice(AXES, size=2)
        blocks.append({"axes": [str(even if i % 2 == 0 else odd) for i in range(qubits)],
                       "entanglers": rng.integers(0, 2, size=max(0, qubits-1)).tolist()})
    layout = {"qubits": int(qubits), "features": int(features), "depth": depth, "blocks": blocks}
    validate_layout(layout)
    return layout


def sample_unique(count, qubits, features, depths, seed, excluded=()):
    """Deduplicate circuit codes, not measured or mathematical kernels."""
    if count < 0 or not depths or min(depths) < 1:
        raise ValueError("Invalid sampling request")
    rng, seen, result = np.random.default_rng(seed), set(excluded), []
    for _ in range(max(1000, count * 100)):
        if len(result) == count:
            return result
        layout = sample_layout(rng, qubits, features, depths)
        code = layout_code(layout)
        if code not in seen:
            seen.add(code)
            result.append(layout)
    raise ValueError("Unique layout sampling exhausted; no duplicate padding permitted")


def valid_positions(layout):
    q, p, depth = validate_layout(layout)
    return [(block * depth + repeat) * q + wire
            for block in range(len(layout["blocks"])) for repeat in range(depth)
            for wire in range(q) if block*q + wire < p]


def feature_index(layout, position):
    q, p, depth = validate_layout(layout)
    block, rem = divmod(int(position), depth*q)
    _, wire = divmod(rem, q)
    index = block*q + wire
    if position not in valid_positions(layout) or index >= p:
        raise ValueError("Replacement position is padding or outside the circuit")
    return index


def make_mask(layout, seed, config_index, max_parameters):
    """Same q/p/depth/config seed gives the same mask across all layouts."""
    allowed = valid_positions(layout)
    if not 1 <= max_parameters <= len(allowed):
        raise ValueError("Invalid replacement-parameter cap")
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed), layout["qubits"], layout["features"], layout["depth"], int(config_index)]))
    size = int(rng.integers(1, max_parameters + 1))
    return sorted(int(v) for v in rng.choice(allowed, size=size, replace=False))


def initial_parameters(x_train, layout, mask):
    x = np.asarray(x_train, dtype=float)
    _, p, _ = validate_layout(layout)
    if x.ndim != 2 or x.shape[1] != p or not np.isfinite(x).all():
        raise ValueError("Finite training angle matrix required")
    return np.asarray([x[:, feature_index(layout, pos)].mean() for pos in mask], dtype=float)


def feature_gates(x, layout, mask=(), parameters=()):
    q, p, depth = validate_layout(layout)
    x, parameters = np.asarray(x, float), np.asarray(parameters, float)
    if x.shape != (p,) or not np.isfinite(x).all():
        raise ValueError("Input must contain exactly p finite angles")
    if len(mask) != len(set(mask)) or list(mask) != sorted(mask):
        raise ValueError("Mask must contain sorted unique positions")
    if parameters.shape != (len(mask),) or not np.isfinite(parameters).all():
        raise ValueError("Finite parameters must match mask order")
    if not set(mask).issubset(valid_positions(layout)):
        raise ValueError("Mask cannot replace padding rotations")
    parameter_at = {int(position): i for i, position in enumerate(mask)}
    gates = []
    for b, block in enumerate(layout["blocks"]):
        for repeat in range(depth):
            for wire, axis in enumerate(block["axes"]):
                pos, index = (b*depth + repeat)*q + wire, b*q+wire
                param = parameter_at.get(pos)
                angle = parameters[param] if param is not None else (x[index] if index < p else 0.)
                gates.append({"gate": axis, "wires": [wire], "angle": float(angle),
                              "parameter": param, "coefficient": 1. if param is not None else 0.,
                              "position": pos})
            for wire, enabled in enumerate(block["entanglers"]):
                if enabled:
                    gates.append({"gate": "CNOT", "wires": [wire, wire+1],
                                  "parameter": None, "coefficient": 0.})
    return gates


def invert_gates(gates):
    result = []
    for gate in reversed(gates):
        inv = dict(gate, wires=list(gate["wires"]))
        if inv["gate"] in AXES:
            inv["angle"] = -inv["angle"]
            inv["coefficient"] = -inv["coefficient"]
        elif inv["gate"] != "CNOT":
            raise ValueError("Unknown inverse gate")
        result.append(inv)
    return result


def overlap_gates(x_left, x_right, layout, mask=(), parameters=()):
    left = [dict(g, end="left") for g in feature_gates(x_left, layout, mask, parameters)]
    right = [dict(g, end="right") for g in invert_gates(feature_gates(x_right, layout, mask, parameters))]
    return left + right


def qcis(gates, qubits, occurrence_shift=None):
    """Shift one physical gate occurrence, never the whole repeated parameter."""
    shift_index, shift = occurrence_shift if occurrence_shift is not None else (None, 0.)
    if shift_index is not None and not (0 <= shift_index < len(gates) and gates[shift_index]["gate"] in AXES):
        raise ValueError("Shift must identify a rotation occurrence")
    lines = []
    for i, gate in enumerate(gates):
        wires = [int(w) for w in gate["wires"]]
        if any(w < 0 or w >= qubits for w in wires):
            raise ValueError("Wire outside circuit")
        if gate["gate"] in AXES:
            angle = gate["angle"] + (shift if i == shift_index else 0.)
            if not math.isfinite(angle):
                raise ValueError("Nonfinite gate angle")
            lines.append(f'{gate["gate"]} Q{wires[0]+1} {float(angle):.17g}')
        elif gate["gate"] == "CNOT":
            # Fixed exact decomposition used by the inspected Cqlib CX method.
            # Gate support must still be established by the provider preflight.
            lines.extend((f'H Q{wires[1]+1}', f'CZ Q{wires[0]+1} Q{wires[1]+1}', f'H Q{wires[1]+1}'))
        else:
            raise ValueError("Unsupported QCIS gate")
    lines.extend(f"M Q{i+1}" for i in range(qubits))
    return "\n".join(lines)


def specification(gates, qubits, request_key, occurrence_shift=None):
    text = qcis(gates, qubits, occurrence_shift)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cid = digest({"request_key": request_key, "qcis_sha256": sha})
    return {"circuit_id": cid, "id": cid, "qcis": text, "sha256": sha,
            "qubits": qubits, "measured_qubits": list(range(1, qubits+1)),
            "metadata": {"request_key": request_key, "wire_convention": "QCIS_1_based",
                         "observable": "probability_all_measured_bits_zero",
                         "resources": emitted_resources(text, qubits),
                         "input_logical_gate_count": len(gates)}}


def emitted_resources(text, qubits):
    """ASAP wire depth of emitted QCIS; excludes readout, not hardware duration."""
    clocks, counts = [0]*qubits, {}
    for line in text.splitlines():
        parts = line.split()
        if parts[0] == "M":
            continue
        wires = [int(token[1:])-1 for token in parts[1:] if token.startswith("Q")]
        counts[parts[0]] = counts.get(parts[0], 0)+1
        layer = max(clocks[w] for w in wires)+1
        for wire in wires:
            clocks[wire] = layer
    return {"emitted_gate_count": sum(counts.values()), "emitted_gate_counts": counts,
            "emitted_asap_depth_excluding_measurement": max(clocks),
            "measurement_count": qubits, "cnot_decomposition": "H(target) CZ(control,target) H(target)"}


def shift_occurrences(gates, parameter):
    return [(i, g["coefficient"]) for i, g in enumerate(gates)
            if g.get("parameter") == parameter and g["gate"] in AXES]


def encode_layout(layout, maximum_depth):
    q, _, depth = validate_layout(layout)
    if depth > maximum_depth:
        raise ValueError("Encoding width is smaller than circuit depth")
    image = np.zeros((5, q, 2*len(layout["blocks"])*maximum_depth), dtype=np.float64)
    for b, block in enumerate(layout["blocks"]):
        for r in range(depth):
            column = 2*(b*maximum_depth+r)
            for wire, axis in enumerate(block["axes"]):
                image[AXES.index(axis), wire, column] = 1.
            for wire, flag in enumerate(block["entanglers"]):
                if flag:
                    image[3, wire, column+1] = 1.
                    image[4, wire+1, column+1] = 1.
    return image.ravel()


def budget_estimate(n, v, o, labels, top_k, configurations, steps, batch_size,
                    max_parameters, shots=128, repetitions=1):
    values = (n, v, o, labels, top_k, configurations, steps, batch_size, shots, repetitions)
    if any(int(z) != z or z < 1 for z in values) or max_parameters < 1:
        raise ValueError("Positive integral budget inputs required")
    a, ab = n*(n-1)//2, batch_size*(batch_size-1)//2
    parts = {"label_probabilities": labels*a,
             "preselection_probabilities": top_k*(a+n*v),
             "finetuning_probabilities": top_k*configurations*(2*a+n*v+steps*(1+4*max_parameters)*ab),
             "final_test_probabilities": n*o}
    total = sum(parts.values())
    return {**parts, "probability_circuits_per_arm_seed": total,
            "nominal_shots_per_arm_seed": total*shots,
            "probability_circuits_all_repetitions": total*repetitions,
            "nominal_shots_all_repetitions": total*shots*repetitions,
            "assumptions": ["ideal normalized fidelity diagonal known to be one",
                            "every trainable RX/RY/RZ gate has both overlap ends differentiated",
                            "one independent parameter per replaced gate; 4 shifts per parameter",
                            "includes initial and final full training kernels and ALL final validation kernels",
                            "excludes other method arms, TEK/other baselines, noise, retries and preflight"]}
