"""Offline contract tests. No SDK import, credential access or network traffic.

Run from experiments/qukernet_finance:
    python -B -m unittest discover -s cloud_v2/tests -p test_cloud.py -v
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from cloud_v2.cloud import (TianyanCloudAdapter, UnknownSubmission, InvalidResult,
                           TerminalFailure, SubmissionRejected, validate_raw_result,
                           Cqlib1311Transport, create_provider)
from cloud_v2.ledger import (PersistentLedger, BudgetExceeded, UnresolvedTask,
                            DuplicateSubmission, LedgerError, HARD_LIMITS, utc, public_source_root, private_directory)
from cloud_v2.preflight import (PreflightBlocked, INTERFACE, circuit_spec, normalize_qcis,
                               inspect_current_backend)


class Clock:
    def __init__(self): self.t = 0.0; self.sleeps = []
    def mono(self): return self.t
    def wall(self): return 1_800_000_000.0 + self.t
    def sleep(self, seconds): self.sleeps.append(seconds); self.t += seconds


def spec(ident="c0"):
    qcis = "H Q1\nCX Q1 Q2\nM Q1\nM Q2"
    return {"id": ident, "qcis": qcis, "qubits": 2, "measured_qubits": [1, 2],
            "sha256": hashlib.sha256(qcis.encode()).hexdigest()}


def evidence(ledger, clock):
    return {"account_scope_sha256": ledger.account_hash, "backend": "tianyan_sw",
            "canonical_ledger_path_sha256": hashlib.sha256(str(ledger.path.resolve()).encode()).hexdigest(),
            "interface": INTERFACE, "authorized": True, "free": True,
            "zero_real_device_quota": True, "no_untracked_inflight_jobs": True,
            "source_kind": "test_fixture", "source_locator": "offline-test-fixture",
            "source_sha256": "a" * 64, "observed_utc": utc(clock.wall()-1),
            "valid_until_utc": utc(clock.wall()+400_000),
            "capabilities": {"backend": "tianyan_sw", "running": True, "max_qubits": 36,
                "server_max_shots": None, "approved_probe_shots_limit": 128,
                "approved_main_shots_limit": 128,
                "client_max_circuit_characters": 20_000, "server_max_circuit_characters": None,
                "gate_validation_basis": "installed_cqlib_1.3.11_offline_parser",
                "supported_gates": ["H", "I", "CX", "CZ", "RX", "RY", "RZ", "M"],
                "observed_utc": utc(clock.wall()-1), "valid_until_utc": utc(clock.wall()+400_000),
                "source_locator": "offline-capability-fixture", "source_sha256": "b" * 64}}


class MockTransport:
    is_mock = True
    sdk_version = "1.3.11-mock"
    def __init__(self, ledger, clock):
        self.ledger, self.clock = ledger, clock
        self.submissions = 0; self.queries = 0; self.mode = "valid"
        self.backend_reads = 0; self.backend_fee = 1; self.backend_status = 0
        self.ids = []; self.specs = []; self.shots = 0; self.saw_durable_intent = False
    def submit(self, specs, shots, backend, aid, deadline):
        self.submissions += 1
        entries = self.ledger.snapshot()["entries"]
        self.saw_durable_intent = entries[-1]["state"] == "submission_pending" and entries[-1]["attempt_id"] == aid
        self.specs, self.shots = specs, shots
        self.ids = [str(self.submissions * 100 + i) for i in range(len(specs))]
        if self.mode == "submit_timeout": raise TimeoutError("offline simulated timeout")
        if self.mode == "submit_rejected": raise SubmissionRejected("offline explicit rejection")
        if self.mode == "partial_ids": return self.ids[:1]
        if self.mode == "duplicate_ids": return self.ids[:1] * len(specs)
        return self.ids
    def item(self, i):
        rows = [[0, 1]] * self.shots
        rows[0] = [0, 0]
        item = {"experimentTaskId": self.ids[i], "resultStatus": [[1, 2]] + rows,
                "probability": {"00": 1/self.shots, "10": (self.shots-1)/self.shots}}
        if self.mode == "bad_rows": item["resultStatus"][1] = [False, 0]
        if self.mode == "bad_probability": item["probability"] = {"00": 1.0}
        return item
    def query(self, ids, deadline):
        self.queries += 1
        if self.mode == "query_error": return {"code": 401, "msg": "offline expired token"}
        if self.mode == "terminal": return {"code": 1, "msg": "任务" + ids[0] + "运行失败"}
        items = [self.item(i) for i in range(len(ids))]
        if self.mode == "empty": items = []
        if self.mode == "partial_results" and self.queries == 1: items = items[:1]
        if self.mode == "duplicate_results": items = items + items[:1]
        return {"code": 0, "data": {"experimentResultModelList": items}}
    def returned_circuit(self, jid, deadline):
        if self.mode == "circuit_timeout": self.clock.sleep(100)
        if self.mode == "missing_circuit": return []
        qcis = self.specs[self.ids.index(jid)]["qcis"]
        if self.mode == "wrong_circuit": qcis = qcis.replace("H Q1", "I Q1")
        return [{"qcis": qcis}]
    def backend_snapshot(self, deadline):
        self.backend_reads += 1
        return {"code": 0, "data": [{"code": "tianyan_sw", "isToll": self.backend_fee,
                                      "status": self.backend_status}]}


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.ledger = PersistentLedger(self.temp.name, "non-secret-test-account", clock=self.clock.wall)
        self.transport = MockTransport(self.ledger, self.clock)
        self.proof = evidence(self.ledger, self.clock)
        self.provider = TianyanCloudAdapter(self.transport, self.ledger, self.proof,
            observation_seconds=3, poll_seconds=1, sleep=self.clock.sleep, monotonic=self.clock.mono)
    def tearDown(self): self.temp.cleanup()
    def state(self): return self.ledger.snapshot()["entries"][-1]["state"]
    def test_valid_raw_counts_hash_and_provenance(self):
        out = self.provider.probability([spec()], 4)
        self.assertTrue(self.transport.saw_durable_intent)
        self.assertEqual(out[0]["p0"], .25)
        self.assertEqual(out[0]["probability_bit_order_relative_to_columns"], "reversed")
        self.assertEqual(out[0]["execution_provenance"]["execution_location"], "mock_only")
        self.assertEqual(self.state(), "validated")
    def test_fee_missing_default_blocks_before_reservation(self):
        self.provider.evidence = None
        with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 4)
        self.assertEqual(self.transport.submissions, 0)
        self.assertEqual(self.transport.backend_reads, 0)
        self.assertEqual(self.ledger.snapshot()["entries"], [])
    def test_free_flag_does_not_prove_hardware_quota(self):
        del self.proof["zero_real_device_quota"]
        with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 4)
        self.assertEqual(self.transport.submissions, 0)
    def test_account_interface_backend_proof_binding(self):
        for key in ("account_scope_sha256", "interface", "backend"):
            with self.subTest(key=key):
                self.provider.evidence = copy.deepcopy(self.proof); self.provider.evidence[key] = "wrong"
                with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 4)
        self.assertEqual(self.transport.submissions, 0)
    def test_expired_evidence(self):
        self.proof["valid_until_utc"] = utc(self.clock.wall()-1)
        with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 4)
    def test_sa_and_noise_backend_fail_closed(self):
        for backend in ("tianyan_sa", "tianyan_swn", "tianyan176"):
            self.provider.backend = backend
            with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 4)
        self.assertEqual(self.transport.submissions, 0)
    def test_empty_and_oversized_batch_no_submission(self):
        for values in ([], [spec(str(i)) for i in range(21)]):
            with self.assertRaises(PreflightBlocked): self.provider.probability(values, 4)
        self.assertEqual(self.transport.submissions, 0)
    def test_bad_circuit_hash_no_submission(self):
        s = spec(); s["sha256"] = "0"*64
        with self.assertRaises(PreflightBlocked): self.provider.probability([s], 4)
    def test_partial_measurement_no_submission(self):
        s = spec(); s.pop("sha256"); s["qcis"] = "H Q1\nCX Q1 Q2\nM Q1"
        with self.assertRaises(PreflightBlocked): self.provider.probability([s], 4)
    def test_noncontiguous_label_rejected(self):
        s = spec(); s.pop("sha256"); s["qcis"] = s["qcis"].replace("Q2", "Q999"); s["measured_qubits"] = [1, 999]
        with self.assertRaises(PreflightBlocked): self.provider.probability([s], 4)
    def test_duplicate_send_refused(self):
        self.provider.probability([spec()], 4)
        with self.assertRaises(DuplicateSubmission): self.provider.probability([spec()], 4)
        self.assertEqual(self.transport.submissions, 1)
    def test_explicit_repetition_context_distinguishes_intent(self):
        self.provider.probability([spec()], 4, context={"repetition": 0})
        self.provider.probability([spec()], 4, context={"repetition": 1})
        self.assertEqual(self.transport.submissions, 2)
    def test_submit_timeout_persists_blocks_restart_and_counts_budget(self):
        self.transport.mode = "submit_timeout"
        with self.assertRaises(UnknownSubmission): self.provider.probability([spec()], 4)
        self.assertEqual(self.state(), "submission_unknown")
        restored = PersistentLedger(self.temp.name, "non-secret-test-account", clock=self.clock.wall)
        self.assertEqual(restored.snapshot()["entries"][0]["requested_shots"], 4)
        self.provider.ledger = restored
        with self.assertRaises(UnresolvedTask): self.provider.probability([spec("new")], 4)
        self.assertEqual(self.transport.submissions, 1)
    def test_partial_ids_unknown(self):
        self.transport.mode = "partial_ids"
        with self.assertRaises(UnknownSubmission): self.provider.probability([spec("a"), spec("b")], 4)
        self.assertEqual(self.state(), "submission_unknown")
        self.assertEqual(self.ledger.snapshot()["entries"][-1]["partial_returned_ids"], ["100"])
    def test_duplicate_ids_unknown(self):
        self.transport.mode = "duplicate_ids"
        with self.assertRaises(UnknownSubmission): self.provider.probability([spec("a"), spec("b")], 4)
    def test_empty_result_is_unknown_not_queue_or_failure(self):
        self.transport.mode = "empty"
        with self.assertRaises(UnknownSubmission): self.provider.probability([spec()], 4)
        self.assertEqual(self.state(), "observation_unknown")
        self.assertEqual(self.transport.queries, 2)
        self.assertEqual(self.clock.sleeps, [1, 2])
    def test_explicit_terminal_failure(self):
        self.transport.mode = "terminal"
        with self.assertRaises(TerminalFailure): self.provider.probability([spec()], 4)
        self.assertEqual(self.state(), "terminal_failure")
    def test_query_auth_error_not_task_failure(self):
        self.transport.mode = "query_error"
        with self.assertRaises(UnknownSubmission): self.provider.probability([spec()], 4)
        self.assertEqual(self.state(), "observation_unknown")
    def test_partial_batch_results_can_complete_without_resubmission(self):
        self.transport.mode = "partial_results"
        out = self.provider.probability([spec("a"), spec("b")], 4)
        self.assertEqual([x["id"] for x in out], ["a", "b"])
        self.assertEqual(self.transport.submissions, 1)
    def test_duplicate_results_invalid(self):
        self.transport.mode = "duplicate_results"
        with self.assertRaises(InvalidResult): self.provider.probability([spec()], 4)
        self.assertEqual(self.state(), "received_invalid")
    def test_invalid_rows_and_probabilities(self):
        for mode in ("bad_rows", "bad_probability"):
            with self.subTest(mode=mode):
                s = circuit_spec(spec(), self.proof["capabilities"])
                self.transport.specs = [s]; self.transport.ids = ["100"]; self.transport.shots = 4; self.transport.mode = mode
                with self.assertRaises(InvalidResult): validate_raw_result(self.transport.item(0), "100", s, 4)
    def test_returned_hash_mismatch_blocks_new_tasks(self):
        self.transport.mode = "wrong_circuit"
        with self.assertRaises(InvalidResult): self.provider.probability([spec()], 4)
        self.assertEqual(self.state(), "received_invalid")
        with self.assertRaises(UnresolvedTask): self.provider.probability([spec("new")], 4)
    def test_missing_returned_circuit_is_not_success(self):
        self.transport.mode = "missing_circuit"
        with self.assertRaises(InvalidResult): self.provider.probability([spec()], 4)
    def test_returned_circuit_uses_remaining_deadline(self):
        self.transport.mode = "circuit_timeout"
        with self.assertRaises(UnknownSubmission): self.provider.probability([spec()], 4)
        self.assertEqual(self.state(), "observation_unknown")
    def test_expired_caller_deadline_no_reservation(self):
        with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 4, deadline=0)
        self.assertEqual(self.ledger.snapshot()["entries"], [])
    def test_account_lock_global_serial(self):
        with self.ledger.execution():
            with self.assertRaises(UnresolvedTask): self.provider.probability([spec()], 4)
    def test_rejected_attempt_is_still_charged(self):
        self.transport.mode = "submit_rejected"
        with self.assertRaises(SubmissionRejected): self.provider.probability([spec()], 4)
        self.assertEqual(self.state(), "submission_rejected")
        self.assertEqual(self.ledger.snapshot()["entries"][0]["requested_shots"], 4)
    def test_preflight_subcap_and_global_accounting(self):
        other = PersistentLedger(Path(self.temp.name)/"small", "account", limits={"preflight_shots": 6}, clock=self.clock.wall)
        a = other.reserve(1, 4, "preflight", "a", {}); other.update(a["attempt_id"], "submission_rejected")
        with self.assertRaises(BudgetExceeded): other.reserve(1, 3, "preflight", "b", {})
        b = other.reserve(1, 3, "main", "b", {})
        self.assertEqual(sum(e["requested_shots"] for e in other.snapshot()["entries"]), 7)
    def test_hard_caps_and_deadline_cannot_reset(self):
        with self.assertRaises(LedgerError): PersistentLedger(Path(self.temp.name)/"bad", "a", limits={"shots": HARD_LIMITS["shots"]+1})
        self.clock.sleep(HARD_LIMITS["seconds"])
        restored = PersistentLedger(self.temp.name, "non-secret-test-account", clock=self.clock.wall)
        with self.assertRaises(BudgetExceeded): restored.reserve(1, 1, "main", "later", {})
    def test_public_and_unc_outputs_forbidden(self):
        public = public_source_root()
        with self.assertRaises(LedgerError): PersistentLedger(public / "should-not-exist", "a")
        with self.assertRaises(LedgerError): PersistentLedger(r"\\example\share\should-not-exist", "a")
    def test_flat_remote_deployment_private_sibling_is_allowed(self):
        base = Path(self.temp.name).resolve() / "deployment"
        fake = base / "code/qukernet_finance/cloud_v2/ledger.py"
        with mock.patch("cloud_v2.ledger.__file__", str(fake)):
            self.assertEqual(public_source_root(), base / "code")
            self.assertEqual(private_directory(base / "private/evidence"), base / "private/evidence")
            with self.assertRaises(LedgerError): private_directory(base / "code/private")
    def test_course_layout_protects_entire_course(self):
        base = Path(self.temp.name).resolve() / "course"
        fake = base / "experiments/qukernet_finance/cloud_v2/ledger.py"
        with mock.patch("cloud_v2.ledger.__file__", str(fake)):
            self.assertEqual(public_source_root(), base)
            with self.assertRaises(LedgerError): private_directory(base / "docs/private")
    def test_zero_based_columns_and_symmetric_order_ambiguity(self):
        s = {"qubits": 2, "measured_qubits": [0, 1]}
        item = {"experimentTaskId": "1", "resultStatus": [[1, 0], [0, 0], [1, 1]], "probability": {"00": .5, "11": .5}}
        result = validate_raw_result(item, "1", s, 2)
        self.assertEqual(result["probability_bit_order_relative_to_columns"], "ambiguous_direct_or_reversed")
    def test_probability_only_does_not_pass_as_raw_shots(self):
        with self.assertRaises(InvalidResult): validate_raw_result({"experimentTaskId": "1", "probability": {"00": 1}}, "1", {"qubits": 2, "measured_qubits": [1, 2]}, 4)
    def test_execution_identity_binds_account_backend_and_code(self):
        value = self.provider.execution_identity()
        self.assertEqual(value["account_scope_sha256"], self.ledger.account_hash)
        self.assertTrue(value["is_mock"])
        self.assertEqual(len(value["client_code_sha256"]), 3)
    def test_same_approval_cannot_restart_ledger_in_another_root(self):
        self.provider.ledger = PersistentLedger(Path(self.temp.name)/"fresh", "non-secret-test-account", clock=self.clock.wall)
        with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 4)
        self.assertEqual(self.transport.submissions, 0)
    def test_global_requested_shot_cap(self):
        small = PersistentLedger(Path(self.temp.name)/"caps", "a", limits={"shots": 5}, clock=self.clock.wall)
        a = small.reserve(1, 4, "main", "a", {}); small.update(a["attempt_id"], "submission_rejected")
        with self.assertRaises(BudgetExceeded): small.reserve(1, 2, "main", "b", {})
    def test_global_circuit_cap_counts_each_circuit_not_batches(self):
        small = PersistentLedger(Path(self.temp.name)/"capc", "a", limits={"circuits": 2}, clock=self.clock.wall)
        a = small.reserve(2, 1, "main", "a", {}); small.update(a["attempt_id"], "submission_rejected")
        with self.assertRaises(BudgetExceeded): small.reserve(1, 1, "main", "b", {})
    def test_production_factory_missing_evidence_creates_no_output(self):
        dest = Path(self.temp.name)/"not-created"
        with self.assertRaises(PreflightBlocked): create_provider(platform=object(), account_ledger_root=dest,
            account_scope="a", evidence=None)
        self.assertFalse(dest.exists())
    def test_transport_blocks_sdk_second_send_and_restores_method(self):
        class Platform:
            SUBMIT_EXP_PATH = "/submit"
            def _send_request(self, *args, **kwargs): raise AssertionError("Original transport must never be called")
            def submit_experiment(self, *args, **kwargs):
                self._send_request(self.SUBMIT_EXP_PATH, method="POST", data={})
                self._send_request(self.SUBMIT_EXP_PATH, method="POST", data={})
        t = Cqlib1311Transport.__new__(Cqlib1311Transport)
        t.platform = Platform(); t._submitted_attempts = set()
        original = t.platform._send_request
        t._request = mock.Mock(return_value={"code": 0, "data": {"query_ids": ["1"]}})
        with self.assertRaises(UnknownSubmission): t.submit([spec()], 4, "tianyan_sw", "a", 100)
        self.assertEqual(t._request.call_count, 1)
        self.assertEqual(t.platform._send_request, original)
        with self.assertRaises(UnknownSubmission): t.submit([spec()], 4, "tianyan_sw", "a", 100)
        self.assertEqual(t._request.call_count, 1)
    def test_raw_response_budget_checked_before_submission(self):
        self.transport.max_response_bytes = 10
        with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 4)
        self.assertEqual(self.transport.submissions, 0)
    def test_live_backend_check_before_every_submit(self):
        self.provider.probability([spec()], 4, context={"repetition": 0})
        self.provider.probability([spec()], 4, context={"repetition": 1})
        self.assertEqual(self.transport.backend_reads, 2)
        self.assertEqual(self.transport.submissions, 2)
        checks = [e["metadata"]["live_backend_check"] for e in self.ledger.snapshot()["entries"]]
        self.assertTrue(all(c["running_and_free"] for c in checks))
        self.assertTrue(all(c["quota_isolation_proven_by_this_read"] is False for c in checks))
    def test_live_paid_or_not_running_blocks_without_reservation(self):
        for fee, status in ((2, 0), (1, 1), (None, 0), (True, 0), (1, False)):
            self.transport.backend_fee, self.transport.backend_status = fee, status
            with self.subTest(fee=fee, status=status):
                with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 4)
        self.assertEqual(self.transport.submissions, 0)
        self.assertEqual(self.ledger.snapshot()["entries"], [])
    def test_status_change_between_batches_is_not_cached(self):
        self.provider.probability([spec()], 4)
        self.transport.backend_status = 2
        with self.assertRaises(PreflightBlocked): self.provider.probability([spec("next")], 4)
        self.assertEqual(self.transport.submissions, 1)
        self.assertEqual(self.transport.backend_reads, 2)
    def test_default_poll_schedule_is_5_10_15_then_15_with_bounded_tail(self):
        self.transport.mode = "empty"
        self.provider = TianyanCloudAdapter(self.transport, self.ledger, self.proof,
            observation_seconds=46, sleep=self.clock.sleep, monotonic=self.clock.mono)
        with self.assertRaises(UnknownSubmission): self.provider.probability([spec()], 4)
        self.assertEqual(self.clock.sleeps, [5, 10, 15, 15, 1])
        self.assertEqual(self.transport.submissions, 1)
    def test_readonly_resource_function_requires_explicit_authorization(self):
        with self.assertRaises(PreflightBlocked): inspect_current_backend(self.transport,
            account_scope_sha256=self.ledger.account_hash, deadline=3)
        self.assertEqual(self.transport.backend_reads, 0)
    def test_readonly_resource_record_does_not_claim_quota_or_limits(self):
        result = inspect_current_backend(self.transport, account_scope_sha256=self.ledger.account_hash,
            read_only_authorized=True, deadline=3, output_root=Path(self.temp.name)/"read-only-evidence")
        self.assertTrue(result["running_and_free"])
        self.assertFalse(result["quota_isolation_proven_by_this_read"])
        self.assertFalse(result["limits_inferred_from_this_read"])
        self.assertEqual(result["submitted_circuits"], 0)
        self.assertTrue(Path(result["evidence_file"]).exists())
        self.assertEqual(self.transport.submissions, 0)
    def test_unknown_server_max_allows_explicit_probe_cap(self):
        out = self.provider.probability([spec()], 128, phase="preflight")
        self.assertEqual(out[0]["shots"], 128)
        self.assertIsNone(self.proof["capabilities"]["server_max_shots"])
    def test_probe_and_main_client_caps_are_separate(self):
        self.proof["capabilities"]["approved_main_shots_limit"] = None
        with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 128)
        self.provider.probability([spec()], 128, phase="preflight")
        self.assertEqual(self.transport.submissions, 1)
    def test_explicit_server_max_caps_client_approval_when_known(self):
        self.proof["capabilities"]["server_max_shots"] = 64
        with self.assertRaises(PreflightBlocked): self.provider.probability([spec()], 128, phase="preflight")
        self.assertEqual(self.transport.submissions, 0)


if __name__ == "__main__": unittest.main()
