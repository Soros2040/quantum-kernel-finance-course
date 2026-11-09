"""Independent analytic verification. These helpers REFUSE more than 4 qubits."""
import math
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cloud_v2 import circuits as c


def rotation(axis, angle):
    pauli = {"RX": np.array([[0, 1], [1, 0]], complex),
             "RY": np.array([[0, -1j], [1j, 0]], complex),
             "RZ": np.diag([1., -1.]).astype(complex)}[axis]
    return np.cos(angle/2)*np.eye(2)-1j*np.sin(angle/2)*pauli


def explicit_state(gates, q):
    if q > 4:
        raise ValueError("Analytic test helper is limited to 4 qubits")
    result = np.eye(2**q, dtype=complex)[:, 0]
    for gate in gates:
        if gate["gate"] == "CNOT":
            control, target = gate["wires"]
            matrix = np.zeros((2**q, 2**q), complex)
            for value in range(2**q):
                dest = value ^ (1 << (q-1-target)) if value & (1 << (q-1-control)) else value
                matrix[dest, value] = 1.
        else:
            matrix = np.array([[1.]], complex)
            for wire in range(q):
                matrix = np.kron(matrix, rotation(gate["gate"], gate["angle"]) if wire == gate["wires"][0] else np.eye(2))
        result = matrix@result
    return result


def qcis_state(text, q):
    """Separate direct-amplitude parser, not the gate-list implementation."""
    if q > 4:
        raise ValueError("Analytic test helper is limited to 4 qubits")
    state = np.zeros(2**q, complex); state[0] = 1.
    measured = []
    for line in text.splitlines():
        parts = line.split()
        if parts[0] == "M":
            measured.append(int(parts[1][1:])); continue
        wire = int(parts[1][1:])-1
        if parts[0] == "CZ":
            target = int(parts[2][1:])-1
            for index in range(len(state)):
                if index & (1 << (q-1-wire)) and index & (1 << (q-1-target)):
                    state[index] *= -1
        elif parts[0] == "H":
            bit = 1 << (q-1-wire)
            for index in range(len(state)):
                if not index & bit:
                    other = index | bit
                    a, b = state[index], state[other]
                    state[index], state[other] = (a+b)/math.sqrt(2), (a-b)/math.sqrt(2)
        elif parts[0] == "CNOT":
            target = int(parts[2][1:])-1
            for index in range(len(state)):
                if index & (1 << (q-1-wire)) and not index & (1 << (q-1-target)):
                    other = index ^ (1 << (q-1-target))
                    state[index], state[other] = state[other], state[index]
        else:
            angle, bit = float(parts[2]), 1 << (q-1-wire)
            cosine, sine = math.cos(angle/2), math.sin(angle/2)
            for index in range(len(state)):
                if index & bit:
                    continue
                other = index | bit
                a, b = state[index], state[other]
                if parts[0] == "RX":
                    state[index], state[other] = cosine*a-1j*sine*b, -1j*sine*a+cosine*b
                elif parts[0] == "RY":
                    state[index], state[other] = cosine*a-sine*b, sine*a+cosine*b
                elif parts[0] == "RZ":
                    state[index], state[other] = np.exp(-.5j*angle)*a, np.exp(.5j*angle)*b
                else:
                    raise ValueError("Unknown QCIS opcode")
    if measured != list(range(1, q+1)):
        raise ValueError("QCIS did not measure the full 1-based wire set")
    return state


class CircuitTests(unittest.TestCase):
    def test_independent_feature_blocks_and_padding(self):
        layout = c.sample_unique(1, 8, 40, [5], 97)[0]
        self.assertEqual(len(layout["blocks"]), 5)
        self.assertGreater(len({c.canonical(b) for b in layout["blocks"]}), 1)
        self.assertEqual(len(c.valid_positions(layout)), 200)
        layout36 = c.sample_unique(1, 36, 40, [1], 97)[0]
        self.assertEqual(len(c.valid_positions(layout36)), 40)
        mask = c.make_mask(layout36, 32, 0, 2)
        self.assertTrue(all(c.feature_index(layout36, position) < 40 for position in mask))
        with self.assertRaises(ValueError):
            c.feature_gates(np.zeros(40), layout36, [71], [0.])

    def test_deduplication_and_mask_pairing(self):
        first = c.sample_unique(10, 2, 4, [1, 2], 12)
        second = c.sample_unique(10, 2, 4, [1, 2], 13, [c.layout_code(v) for v in first])
        self.assertEqual(len({c.layout_code(v) for v in first+second}), 20)
        for layout in first:
            partner = dict(layout, blocks=list(reversed(layout["blocks"])))
            self.assertEqual(c.make_mask(layout, 1, 2, 3), c.make_mask(partner, 1, 2, 3))

    def test_explicit_matrix_qcis_and_overlap(self):
        rng = np.random.default_rng(221)
        for q in (1, 2, 3, 4):
            for depth in (1, 2):
                layout = c.sample_unique(1, q, q+1, [depth], q+depth)[0]
                x, z = rng.normal(size=(2, q+1))
                mask = c.make_mask(layout, 2, 0, 2)
                params = np.array([.23]*len(mask))
                for sample in (x, z):
                    gates = c.feature_gates(sample, layout, mask, params)
                    a, b = explicit_state(gates, q), qcis_state(c.qcis(gates, q), q)
                    np.testing.assert_allclose(a, b, atol=1e-12)
                    self.assertAlmostEqual(float(np.vdot(a, a).real), 1., places=12)
                overlap = c.overlap_gates(x, z, layout, mask, params)
                value = abs(qcis_state(c.qcis(overlap, q), q)[0])**2
                expected = abs(np.vdot(explicit_state(c.feature_gates(z, layout, mask, params), q),
                                       explicit_state(c.feature_gates(x, layout, mask, params), q)))**2
                self.assertAlmostEqual(value, expected, places=11)
                equal = c.overlap_gates(x, x, layout, mask, params)
                self.assertAlmostEqual(abs(qcis_state(c.qcis(equal, q), q)[0])**2, 1., places=11)
                for param in range(len(mask)):
                    occurrences = c.shift_occurrences(overlap, param)
                    self.assertEqual(len(occurrences), 2)
                    self.assertEqual(sorted(coef for _, coef in occurrences), [-1., 1.])

    def test_budget_both_ends(self):
        result = c.budget_estimate(16, 16, 16, 16, 2, 2, 3, 8, 2)
        self.assertEqual(result["probability_circuits_per_arm_seed"], 7936)
        self.assertEqual(result["nominal_shots_per_arm_seed"], 1015808)
        actual_o = c.budget_estimate(16, 16, 220, 16, 2, 2, 3, 8, 2)
        self.assertEqual(actual_o["probability_circuits_per_arm_seed"], 11200)

    def test_cnot_emitted_resources(self):
        gates = [{"gate": "CNOT", "wires": [0, 1], "parameter": None, "coefficient": 0.}]
        text = c.qcis(gates, 2)
        self.assertEqual(text, "H Q2\nCZ Q1 Q2\nH Q2\nM Q1\nM Q2")
        resources = c.emitted_resources(text, 2)
        self.assertEqual(resources["emitted_gate_count"], 3)
        self.assertEqual(resources["emitted_asap_depth_excluding_measurement"], 3)

    def test_no_large_local_simulator(self):
        with self.assertRaises(ValueError):
            qcis_state("", 8)
        with self.assertRaises(ValueError):
            explicit_state([], 8)


if __name__ == "__main__":
    unittest.main()
