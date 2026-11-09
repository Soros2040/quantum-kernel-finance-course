"""Prepare twelve QCIS fixtures offline; this module cannot submit cloud jobs.

Uses only the Python standard library. Output contains a pending manifest and
QCIS text, never credentials or financial data. Example:
    python experiments/feasibility/prepare_cloud.py --output private/cloud_preflight
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from safe_io import OutputRoot


SHOTS = 512
BACKEND = "tianyan_sw"
X4 = (0.10, -0.20, 0.30, -0.40)
Y4 = (0.40, -0.10, 0.00, -0.25)


@dataclass(frozen=True)
class Operation:
    gate: str
    qubits: tuple[int, ...]
    angle: float | None = None

    def qcis(self) -> str:
        parts = [self.gate, *(f"Q{q}" for q in self.qubits)]
        if self.angle is not None:
            if not math.isfinite(self.angle):
                raise ValueError("Non-finite gate angle")
            parts.append(format(self.angle, ".17g"))
        return " ".join(parts)


def encoding(values: Iterable[float], scale: float, depth: int) -> list[Operation]:
    """Time-ordered RY layers, with a directed nearest-neighbour CX chain between."""
    values = tuple(values)
    if not values or depth not in (1, 2) or scale not in (0.5, 1.0):
        raise ValueError("Unsupported frozen fixture parameters")
    if not all(math.isfinite(z) and -1 <= z <= 1 for z in values):
        raise ValueError("Fixture coordinates must be finite and in [-1, 1]")
    operations: list[Operation] = []
    for layer in range(depth):
        operations.extend(Operation("RY", (j + 1,), scale * math.pi * z)
                          for j, z in enumerate(values))
        if layer + 1 < depth:
            operations.extend(Operation("CX", (j, j + 1))
                              for j in range(1, len(values)))
    return operations


def inverse(operations: Iterable[Operation]) -> list[Operation]:
    result = []
    for op in reversed(tuple(operations)):
        if op.gate == "RY":
            result.append(Operation("RY", op.qubits, -op.angle))
        elif op.gate in ("X", "H", "CX"):
            result.append(op)
        else:
            raise ValueError(f"No frozen inverse rule for {op.gate}")
    return result


def overlap(x: Iterable[float], y: Iterable[float], scale: float,
            depth: int) -> list[Operation]:
    """Apply U(x), then U(y)^dagger; no intermediate measurements."""
    x, y = tuple(x), tuple(y)
    if len(x) != len(y):
        raise ValueError("Overlap inputs must have the same width")
    return encoding(x, scale, depth) + inverse(encoding(y, scale, depth))


def resources(operations: Iterable[Operation], qubits: int) -> dict:
    """Count explicit instructions and schedule disjoint gates in parallel."""
    operations = tuple(operations)
    latest = {q: 0 for q in range(1, qubits + 1)}
    measurements = []
    gate_counts: dict[str, int] = {}
    unitary_depth = 0
    measurement_started = False
    for op in operations:
        if not op.qubits or any(q not in latest for q in op.qubits):
            raise ValueError("Qubit label outside Q1..Qq")
        if len(set(op.qubits)) != len(op.qubits):
            raise ValueError("Repeated qubit within one operation")
        if op.gate not in ("X", "H", "RY", "CX", "M"):
            raise ValueError("Gate outside the frozen QCIS fixture set")
        if len(op.qubits) != (2 if op.gate == "CX" else 1):
            raise ValueError("Wrong gate arity")
        if (op.angle is not None) != (op.gate == "RY"):
            raise ValueError("Wrong parameter arity")
        if op.gate == "M":
            measurement_started = True
            measurements.extend(op.qubits)
        elif measurement_started:
            raise ValueError("Unitary operation after measurement")
        layer = 1 + max(latest[q] for q in op.qubits)
        for q in op.qubits:
            latest[q] = layer
        if op.gate != "M":
            unitary_depth = max(unitary_depth, layer)
        gate_counts[op.gate] = gate_counts.get(op.gate, 0) + 1
    if measurements != list(range(1, qubits + 1)):
        raise ValueError("Each qubit must be measured exactly once in Q1..Qq order")
    return {
        "logical_unitary_gate_count": len(operations) - len(measurements),
        "measurement_count": len(measurements),
        "total_operations": len(operations),
        "logical_unitary_depth": unitary_depth,
        "depth_including_measurement": max(latest.values(), default=0),
        "gate_counts": gate_counts,
    }


def build_fixtures() -> list[tuple[dict, str]]:
    specifications = [
        ("zero_4q", 4, [], {"expected_zero_probability": 1.0,
                              "expected_raw_column_outcomes": {"0000": 1.0}}),
        ("marker_4q", 4, [Operation("X", (1,))],
         {"expected_zero_probability": 0.0,
          "expected_raw_column_outcomes": {"1000": 1.0}}),
        ("bell_4q", 4, [Operation("H", (1,)), Operation("CX", (1, 2))],
         {"expected_zero_probability": 0.5,
          "expected_raw_column_outcomes": {"0000": 0.5, "1100": 0.5}}),
    ]
    overlap_specs = [
        ("Q3_same_4q", X4, X4, 0.5, 2, "Q3"),
        ("Q1_xy_4q", X4, Y4, 0.5, 1, "Q1"),
        ("Q2_xy_4q", X4, Y4, 1.0, 1, "Q2"),
        ("Q3_xy_4q", X4, Y4, 0.5, 2, "Q3"),
        ("Q3_yx_4q", Y4, X4, 0.5, 2, "Q3"),
        ("Q4_xy_4q", X4, Y4, 1.0, 2, "Q4"),
        ("Q4_yx_4q", Y4, X4, 1.0, 2, "Q4"),
        ("overlap_8q_L2", X4 * 2, Y4 * 2, 0.5, 2, None),
        ("overlap_12q_L2", X4 * 3, Y4 * 3, 0.5, 2, None),
    ]
    for name, x, y, scale, depth, candidate in overlap_specs:
        expected = 1.0 if x == y else None
        if depth == 1:
            expected = math.prod(math.cos(scale * math.pi * (a - b) / 2) ** 2
                                 for a, b in zip(x, y))
        metadata = {
            "candidate": candidate, "encoding_scale": scale,
            "encoding_layers": depth, "x": list(x), "y": list(y),
            "expected_zero_probability": expected,
            "expected_probability_source": "analytic_identity_or_product" if expected is not None
                                           else "requires_independent_local_reference",
        }
        specifications.append((name, len(x), overlap(x, y, scale, depth), metadata))
    fixtures = []
    for index, (name, qubits, unitary, metadata) in enumerate(specifications, 1):
        operations = unitary + [Operation("M", (q,)) for q in range(1, qubits + 1)]
        text = "\n".join(op.qcis() for op in operations)
        raw = text.encode("utf-8")
        entry = {
            "fixture_id": index, "name": name, "file": f"{index:02d}_{name}.qcis",
            "backend_target": BACKEND, "qubits": qubits, "shots": SHOTS,
            "state": "prepared_not_submitted", "measurement_labels": list(range(1, qubits + 1)),
            "logical_to_qcis_mapping": {str(j): f"Q{j + 1}" for j in range(qubits)},
            "sha256": hashlib.sha256(raw).hexdigest(),
            "qcis_characters": len(text), "qcis_bytes": len(raw),
            **resources(operations, qubits), **metadata,
        }
        fixtures.append((entry, text))
    if len(fixtures) != 12 or sum(item[0]["shots"] for item in fixtures) != 6144:
        raise AssertionError("Frozen task budget changed")
    return fixtures


def prepare(output: Path) -> dict:
    fixtures = build_fixtures()
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline_function_and_scale_preflight",
        "fixture_data": "synthetic_parameters_not_financial_data",
        "sdk_contract": "Cqlib 1.3.11 QCIS; SDK parser validation is a separate check",
        "execution_enabled": False,
        "state": "prepared_not_submitted",
        "max_new_submission_attempts": 12,
        "max_new_requested_shots": 6144,
        "shots_per_fixture": SHOTS,
        "quota_gate": {
            "backend_is_free_and_running": "must_recheck_before_authorized_execution",
            "hardware_quota_deduction_is_zero": "unverified_blocking",
        },
        "qcis_hash_definition": "SHA256 of UTF-8, LF separators, no trailing newline",
        "depth_definition": "explicit logical instructions; no backend compilation or cancellation",
        "circuits": [item[0] for item in fixtures],
    }
    files = {entry["file"]: text.encode("utf-8") for entry, text in fixtures}
    files["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output_root = OutputRoot(output)
    destinations = {name: output_root.path(name) for name in files}
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError("Output artifact exists; choose a new empty output directory")
    for name, data in files.items():
        with destinations[name].open("xb") as stream:
            stream.write(data)
    return {
        "state": manifest["state"], "circuits": len(fixtures),
        "planned_requested_shots": 6144, "actual_submissions": 0,
        "manifest_sha256": hashlib.sha256(files["manifest.json"]).hexdigest(),
        "blocking_evidence": "No confirmed zero deduction from hardware quota",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New local directory for offline QCIS and the pending manifest")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
