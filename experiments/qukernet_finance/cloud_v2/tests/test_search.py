"""Small analytic checks and explicitly labelled mocked workflow tests.

No credential, network, provider SDK or real cloud submission is used.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_v2 import circuits as c, search as s
from test_circuits import qcis_state, explicit_state


class AnalyticUnitProvider:
    """Exact <=4q reference ONLY for unit checks, never a CLI backend."""
    def __init__(self):
        self.calls = 0

    def execution_identity(self):
        return {"kind": "analytic_unit_test_only", "max_qubits": 4, "is_mock": True}

    def probability(self, specs, shots, context=None, deadline=None):
        self.calls += len(specs)
        return [{"circuit_id": spec["circuit_id"], "p0": float(abs(qcis_state(spec["qcis"], spec["qubits"])[0])**2),
                 "shots": shots, "zero_count": None, "execution_provenance": "analytic_test_not_measurement"}
                for spec in specs]


class ExplicitMockProvider:
    def __init__(self):
        self.calls = 0

    def execution_identity(self):
        return {"kind": "mock_control_flow_only", "is_mock": True}

    def probability(self, specs, shots, context=None, deadline=None):
        self.calls += len(specs)
        result = []
        for spec in specs:
            count = 1+int(spec["sha256"][:8], 16) % (shots-2)
            result.append({"circuit_id": spec["circuit_id"], "p0": count/shots,
                           "shots": shots, "zero_count": count, "execution_provenance": "MOCK_NOT_CLOUD"})
        return result


class MockScorer:
    def predict(self, layouts):
        return np.asarray([int(c.digest(v)[:8], 16)/2**32 for v in layouts])


def mock_factory(layouts, labels, maximum_depth, seed):
    return MockScorer(), {"status": "mock_predictor_not_trained", "label_count": len(labels)}


class SearchTests(unittest.TestCase):
    def test_production_identity_false_is_not_a_mock_marker(self):
        self.assertFalse(s.is_test_execution({"backend": "tianyan_sw", "is_mock": False}))
        self.assertTrue(s.is_test_execution({"backend": "tianyan_sw", "is_mock": True}))
        self.assertTrue(s.is_test_execution({"kind": "analytic_unit_test_only"}))
        self.assertTrue(s.is_test_execution({"is_mock": False}, predictor_override=True))

    def test_source_root_is_not_an_output_even_in_flat_deployment(self):
        with self.assertRaisesRegex(ValueError, "public course/code"):
            s.Checkpoints(s.HERE.parents[2], {"test": "output_guard"})

    def test_gradient_both_overlap_ends_and_normalized_kta(self):
        layout = {"qubits": 2, "features": 3, "depth": 2,
                  "blocks": [{"axes": ["RY", "RX"], "entanglers": [1]},
                             {"axes": ["RZ", "RY"], "entanglers": [1]}]}
        x = np.array([[.2, -.6, .8], [.7, .3, -.2], [-.4, .9, .5]])
        mask, params = [0, 2], np.array([.31, -.42])
        provider = AnalyticUnitProvider()
        with tempfile.TemporaryDirectory() as directory:
            kernel = s.CloudKernel(provider, s.Checkpoints(Path(directory)/"out", {"test": "gradient"}), 128)
            k = kernel.matrix(x, None, layout, mask, params, "base")
            dk = kernel.derivatives(x, layout, mask, params, "gradient")
            self.assertEqual(provider.calls, 3+3*4*len(mask))
            for task, y in (("classification", [0, 1, 0]), ("regression", [.2, -.3, .7])):
                _, gradient = s.alignment_and_gradient(k, y, task, dk)
                finite = []
                for parameter in range(len(params)):
                    results = []
                    for sign in (-1, 1):
                        changed = params.copy(); changed[parameter] += sign*1e-6
                        states = [explicit_state(c.feature_gates(row, layout, mask, changed), 2) for row in x]
                        gram = abs(np.asarray(states).conj()@np.asarray(states).T)**2
                        results.append(s.alignment_and_gradient(gram, y, task)[0])
                    finite.append((results[1]-results[0])/2e-6)
                np.testing.assert_allclose(gradient, finite, atol=2e-7, rtol=2e-6)
                self.assertGreater(np.linalg.norm(gradient), 1e-5)

    def test_alignment_invariance_and_degeneracy(self):
        k = np.array([[1., .2, .3], [.2, 1., .1], [.3, .1, 1.]])
        a = s.alignment_and_gradient(k, [0, 1, 0])[0]
        self.assertAlmostEqual(a, s.alignment_and_gradient(3*k, [0, 1, 0])[0])
        with self.assertRaises(s.ScientificInvalid):
            s.alignment_and_gradient(k, [1., 1., 1.], "regression")
        with self.assertRaises(s.ScientificInvalid):
            s.alignment_and_gradient(k, [1., 1., 1.], "classification")

    def test_cross_projection_is_compatible(self):
        k = np.array([[1., 1.8], [1.8, 1.]])
        b = np.array([[.4, .7], [.1, .5]])
        kp, bp, info = s.spectral_classification_map(k, b)
        basis = np.asarray(info["basis"])
        self.assertEqual(info["negative_eigenvalues"], 1)
        np.testing.assert_allclose(bp, b@basis@basis.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(kp).min(), -1e-12)
        self.assertGreater(info["cross_discard_frobenius"], 0.)

    def test_regression_mean_loss_primal_bundle_and_tolerance(self):
        rng = np.random.default_rng(55)
        z, v = rng.normal(size=(6, 4)), rng.normal(size=(3, 4))
        k, b = z@z.T, v@z.T
        y, target = rng.normal(size=6), rng.normal(size=3)
        dates = ["a", "b", "b"]
        row = s.evaluate_validation(k, b, y, target, dates, "regression")
        bundle, mapping = row["bundle"], row["bundle"]["mapping"]
        zt = np.asarray(bundle["z_train"])
        zv = (b@np.asarray(mapping["basis"])/np.asarray(mapping["sqrt_eigenvalues"])-np.asarray(mapping["feature_mean"]))/np.sqrt(mapping["centered_scale"])
        weight = np.linalg.solve(zt.T@zt+len(y)*row["lambda"]*np.eye(zt.shape[1]), zt.T@(y-y.mean()))
        expected = zv@weight+y.mean()
        np.testing.assert_allclose(s.predict_bundle(bundle, b)[0], expected, atol=1e-10)
        self.assertAlmostEqual(mapping["tolerance"], 1e-10*max(1., np.max(np.abs(np.linalg.eigvalsh(k)))))
        self.assertEqual({item["lambda"] for item in row["lambda_grid"]}, {.001, .01, .1})

    def test_batches_cannot_exceed_provider_twenty_circuit_limit(self):
        class BoundedMock(ExplicitMockProvider):
            def probability(self, specs, shots, **kwargs):
                if len(specs) > 20:
                    raise AssertionError("Too many circuits")
                return super().probability(specs, shots, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            provider = BoundedMock()
            kernel = s.CloudKernel(provider, s.Checkpoints(Path(directory)/"out", {"test": "batch"}), 16)
            layout = c.sample_unique(1, 2, 2, [1], 8)[0]
            kernel.matrix(np.ones((8, 2)), None, layout)
            self.assertEqual(provider.calls, 28)

    def test_task_and_split_mismatch_fail_before_cloud(self):
        provider = ExplicitMockProvider()
        window = {"x_"+split: np.ones((4, 4)) for split in ("train", "valid", "test")}
        window.update({"y_"+split: np.array([0., 1., 0., 1.]) for split in ("train", "valid", "test")})
        window["metadata"] = {"audit": {"task": "regression"}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "task"):
                s.run_search(window, {"features": 4, "n_train": 4, "n_valid": 4}, provider, Path(directory)/"out", 2)
            window["x_valid"] = np.ones((4, 3))
            with self.assertRaisesRegex(ValueError, "feature width"):
                s.run_search(window, {"features": 4, "n_train": 4, "n_valid": 4}, provider, Path(directory)/"out", 2)
        self.assertEqual(provider.calls, 0)

    def test_unknown_does_not_resubmit_and_payload_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = s.Checkpoints(Path(directory)/"out", {"test": 2})
            provider = ExplicitMockProvider()
            kernel = s.CloudKernel(provider, store, 16)
            layout = c.sample_unique(1, 2, 2, [1], 4)[0]
            spec = c.specification(c.overlap_gates([.1, .3], [.2, .7], layout), 2, "unit")
            request = {"specifications": [spec], "shots": 16}
            store.intent("unknown", request)
            with self.assertRaises(RuntimeError):
                kernel.probabilities([spec], "unknown", time.monotonic()+30)
            self.assertEqual(provider.calls, 0)
            kernel.probabilities([spec], "good", time.monotonic()+30)
            path = store.out.path(store.path_for("good"))
            record = json.loads(path.read_text(encoding="utf-8"))
            record["result"]["p0"][0] = .99
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                kernel.probabilities([spec], "good", time.monotonic()+30)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn unavailable; must run in the prepared test environment")
    def test_mock_complete_chain_all_final_endpoints_and_resume(self):
        rng = np.random.default_rng(15)
        window = {"x_train": rng.normal(size=(4, 4)), "x_valid": rng.normal(size=(4, 4)),
                  "x_test": rng.normal(size=(4, 4)), "y_train": np.array([0, 1, 0, 1]),
                  "y_valid": np.array([1, 0, 1, 0]), "y_test": np.array([0, 0, 1, 1]),
                  "metadata": {"audit": {"task": "classification"},
                               "meta_train": [{"date": str(i//2)} for i in range(4)],
                               "meta_valid": [{"date": str(i//2)} for i in range(4)],
                               "meta_test": [{"date": str(i//2)} for i in range(4)]}}
        profile = {"qubits": 2, "features": 4, "depths": [1, 2], "n_train": 4, "n_valid": 4,
                   "labels": 8, "candidate_pool": 16, "top_k": 2, "configurations": 2,
                   "steps": 2, "batch_size": 4, "max_parameters": 1, "learning_rate": .2, "shots": 16}
        provider = ExplicitMockProvider()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"out"
            result = s.run_search(window, profile, provider, out, 500, predictor_factory=mock_factory)
            self.assertEqual(result["workflow_status"], "complete")
            self.assertEqual(result["execution_kind"], "mock_test")
            self.assertEqual(result["attempted_configurations"], 4)
            self.assertEqual(result["valid_configurations"], 4)
            self.assertEqual(result["actual_completed_steps"], 8)
            selection = json.loads((out/"selection_before_test.json").read_text())
            self.assertEqual(selection["valid_final_endpoints_scored"], 4)
            self.assertEqual(result["majority_class_baseline"]["predicted_class"], 0)
            calls = provider.calls
            restored = s.run_search(window, profile, provider, out, 500, predictor_factory=mock_factory)
            self.assertEqual(restored, result)
            self.assertEqual(provider.calls, calls)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable; real MLP smoke remains to be run")
    def test_real_mlp_serializable_predictions(self):
        layouts = c.sample_unique(8, 2, 4, [1, 2], 39)
        scorer, note = s.train_predictor(layouts, np.linspace(.1, .8, 8), 2, 91)
        restored = s.MLPScorer(json.loads(json.dumps(note["state"])), 2)
        np.testing.assert_allclose(scorer.predict(layouts), restored.predict(layouts), atol=1e-12)
        self.assertEqual(len(note["losses"]), 30)


if __name__ == "__main__":
    unittest.main()
