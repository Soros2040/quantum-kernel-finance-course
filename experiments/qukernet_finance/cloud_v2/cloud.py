"""TianYan finite-shot adapter, Cqlib 1.3.11, with no automatic resubmission.

``probability(specs, shots, phase='main', context=None, deadline=None)`` returns
same-order dictionaries with id/circuit_id, p0, zero_count, shots, job_id and
evidence/provenance. ``deadline`` is an absolute time.monotonic() deadline.
The caller supplies an ALREADY authenticated Cqlib platform; this module never
loads credentials or logs in. SA exact mode and noisy backends are unsupported.

Transport payload/header names follow Cqlib 1.3.11 (Apache-2.0), copyright 2024
China Telecom Quantum Group, QuantumCTek Co., Ltd., and Center for Excellence
in Quantum Information and Quantum Physics. Changes: no retries, exact HTTPS
destination, request wall deadline, durable reservation and raw validation.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import signal
import threading
import time
import types

from .ledger import PersistentLedger, atomic_json, digest, utc
from .preflight import (BACKEND, INTERFACE, CloudError, PreflightBlocked, circuit_spec,
                        normalize_qcis, validate_evidence, inspect_current_backend, approved_shots_limit)


class UnknownSubmission(CloudError): pass
class InvalidResult(CloudError): pass
class TerminalFailure(CloudError): pass
class SubmissionRejected(CloudError): pass


def _pairs(pairs):
    value = {}
    for k, v in pairs:
        if k in value: raise InvalidResult("Duplicate JSON key")
        value[k] = v
    return value


def _decode(value):
    # The API has historically used both JSON objects and JSON-encoded strings.
    for _ in range(2):
        if not isinstance(value, str): return value
        value = json.loads(value, object_pairs_hook=_pairs)
    return value


def sanitize(value, secrets=()):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if any(t in str(k).lower() for t in
                ("token", "password", "openid", "authorization", "secret", "cookie", "credential"))
                else sanitize(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [sanitize(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret: value = value.replace(secret, "[REDACTED]")
    return value


@contextmanager
def _hard_request_deadline(deadline):
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        raise PreflightBlocked("Real transport needs Linux main-thread SIGALRM wall deadlines")
    remaining = deadline - time.monotonic()
    if remaining <= 0: raise TimeoutError("Client request deadline expired; backend state unchanged")
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    start = time.monotonic()
    duration = min(remaining, old_timer[0]) if old_timer[0] > 0 else remaining
    def timeout(signum, frame):
        raise TimeoutError("Client HTTP wall deadline; backend task may continue")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, duration)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, old_timer[0] - (time.monotonic() - start)), old_timer[1])


class Cqlib1311Transport:
    """Use verified SDK submission method with its retrying transport replaced.

    Construct only after independent fee/account checks. No constructor network
    calls. An existing access token is used only in request headers in memory.
    """
    is_mock = False
    sdk_version = "1.3.11"

    def __init__(self, platform, request_timeout=30.0, max_response_bytes=128 * 1024**2):
        if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
            raise PreflightBlocked("Real transport requires Linux main-thread hard wall deadlines")
        if importlib.metadata.version("cqlib") != self.sdk_version:
            raise PreflightBlocked("Only the inspected Cqlib 1.3.11 integration is supported")
        if getattr(platform, "DOMAIN", None) != "qc.zdxlz.com" or getattr(platform, "SCHEME", None) != "https":
            raise PreflightBlocked("Unexpected SDK destination; HTTPS domain may not be bypassed")
        expected = {"SUBMIT_EXP_PATH": "/qccp-quantum/sdk/experiment/submit",
                    "QUERY_EXP_PATH": "/qccp-quantum/sdk/experiment/result/find",
                    "GET_EXP_CIRCUIT_PATH": "/qccp-quantum/sdk/experiment/getQcis/by/taskIds",
                    "MACHINE_LIST_PATH": "/qccp-quantum/sdk/quantumComputer/list"}
        if any(getattr(platform, k, None) != v for k, v in expected.items()):
            raise PreflightBlocked("Installed SDK endpoint contract differs from reviewed version")
        if not getattr(platform, "access_token", None):
            raise PreflightBlocked("Caller must supply an already authenticated SDK platform")
        if not math.isfinite(request_timeout) or not 0 < request_timeout <= 60:
            raise PreflightBlocked("Request timeout must be in (0,60] seconds")
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 128 * 1024**2:
            raise PreflightBlocked("Response byte bound must be a positive integer no larger than 128 MiB")
        self.platform, self.request_timeout = platform, float(request_timeout)
        self.max_response_bytes = max_response_bytes
        self._submitted_attempts = set()
        self.last_submission_response = None

    def sanitized(self, value):
        return sanitize(value, (getattr(self.platform, "access_token", ""),))

    def validate_offline(self, specs):
        from cqlib import Circuit
        checks = []
        for spec in specs:
            parsed = Circuit.load(spec["qcis"])
            if parsed.num_qubits != spec["qubits"]:
                raise PreflightBlocked("Official SDK parser reports a different qubit count")
            checks.append({"id": spec["id"], "sdk_parser": "Cqlib 1.3.11 Circuit.load",
                           "num_qubits": parsed.num_qubits, "input_sha256": spec["sha256"]})
        return checks

    def _request(self, path, data, deadline, method="POST"):
        import requests
        allowed = (self.platform.SUBMIT_EXP_PATH, self.platform.QUERY_EXP_PATH,
                   self.platform.GET_EXP_CIRCUIT_PATH, self.platform.MACHINE_LIST_PATH)
        if path not in allowed: raise PreflightBlocked("Unreviewed endpoint blocked")
        expected_method = "GET" if path == self.platform.MACHINE_LIST_PATH else "POST"
        if method != expected_method: raise PreflightBlocked("Unreviewed HTTP method blocked")
        end = min(deadline, time.monotonic() + self.request_timeout)
        with _hard_request_deadline(end):
            session = requests.Session()
            session.trust_env = False
            # Requests' default adapter retries=0; explicitly retain this contract.
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
            try:
                token = self.platform.access_token
                response = session.request(method, "https://qc.zdxlz.com" + path, json=data,
                    headers={"basicToken": token, "Authorization": "Bearer " + token},
                    timeout=max(.01, end-time.monotonic()), allow_redirects=False, verify=True, stream=True)
                if response.status_code != 200:
                    raise CloudError("HTTP request failed; status=" + str(response.status_code))
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > self.max_response_bytes: raise InvalidResult("Cloud response exceeds bounded size")
                    chunks.append(chunk)
                result = json.loads(b"".join(chunks), object_pairs_hook=_pairs)
                if not isinstance(result, dict): raise InvalidResult("Cloud envelope must be an object")
                return self.sanitized(result)
            finally:
                session.close()

    def submit(self, specs, shots, backend, attempt_id, deadline):
        if backend != BACKEND: raise PreflightBlocked("Unapproved backend")
        if attempt_id in self._submitted_attempts: raise UnknownSubmission("Attempt already sent; retry forbidden")
        self._submitted_attempts.add(attempt_id)  # Set BEFORE entering the SDK method.
        self.last_submission_response = None
        original = self.platform._send_request
        calls = [0]
        def one_request(platform, path, method="GET", data=None, params=None, raise_for_code=True):
            if path != platform.SUBMIT_EXP_PATH or method.upper() != "POST" or params is not None:
                raise PreflightBlocked("SDK attempted an unreviewed request during submission")
            calls[0] += 1
            if calls[0] != 1: raise UnknownSubmission("SDK automatic submission retry blocked")
            response = self._request(path, data, deadline)
            self.last_submission_response = response
            if type(response.get("code")) is not int: raise InvalidResult("Missing API code")
            if response["code"] != 0:
                # Only one explicitly documented/reobserved rejection is classified.
                message = str(response.get("msg", response.get("message", "")))
                if response["code"] == 1 and "机时包最大比特数不支持本任务" in message:
                    raise SubmissionRejected("Account qubit entitlement explicitly rejected")
                raise UnknownSubmission("Submission rejected or uncertain; independent classification required")
            return response
        self.platform._send_request = types.MethodType(one_request, self.platform)
        try:
            ids = self.platform.submit_experiment([s["qcis"] for s in specs],
                name="course-cloud-v2-" + attempt_id, num_shots=shots,
                machine_name=backend, is_verify=True)
            return ids
        finally:
            self.platform._send_request = original

    def query(self, ids, deadline):
        return self._request(self.platform.QUERY_EXP_PATH, {"query_ids": ids}, deadline)

    def backend_snapshot(self, deadline):
        """Read raw flags without SDK's in-place conversion of code/status/isToll."""
        return self._request(self.platform.MACHINE_LIST_PATH, None, deadline, method="GET")

    def returned_circuit(self, job_id, deadline):
        envelope = self._request(self.platform.GET_EXP_CIRCUIT_PATH, {"query_ids": [job_id]}, deadline)
        if type(envelope.get("code")) is not int or envelope["code"] != 0:
            raise InvalidResult("Returned circuit request not successful")
        return envelope.get("data")


def validate_raw_result(item, job_id, spec, shots):
    """Raw integer shot rows are required; probabilities never replace rows."""
    if not isinstance(item, dict): raise InvalidResult("Result item is not an object")
    actual_id = item.get("experimentTaskId")
    if type(actual_id) not in (str, int) or str(actual_id) != job_id:
        raise InvalidResult("Result ID mismatch")
    if "status" in item and (type(item["status"]) is not int or item["status"] != 0):
        raise InvalidResult("Result has no verified completed status")
    try: rows = _decode(item.get("resultStatus"))
    except (ValueError, TypeError) as exc: raise InvalidResult("Malformed resultStatus JSON") from exc
    if not isinstance(rows, list) or len(rows) != shots + 1:
        raise InvalidResult("Missing raw shots or requested/returned shot count mismatch")
    labels = rows[0]
    q = spec["qubits"]
    if (not isinstance(labels, list) or len(labels) != q or any(type(x) is not int for x in labels)
            or set(labels) != set(spec["measured_qubits"])):
        raise InvalidResult("Raw measurement columns differ from requested qubits")
    histogram = Counter()
    for row in rows[1:]:
        if (not isinstance(row, list) or len(row) != q
                or any(type(b) is not int or b not in (0, 1) for b in row)):
            raise InvalidResult("Raw measurement row is not a complete binary integer row")
        histogram["".join(map(str, row))] += 1
    bit_order = "raw_columns_only_probability_map_unavailable"
    if item.get("probability") is not None:
        try: probs = _decode(item["probability"])
        except (ValueError, TypeError) as exc: raise InvalidResult("Malformed probability JSON") from exc
        if (not isinstance(probs, dict) or not probs
                or any(not isinstance(k, str) or len(k) != q or set(k) - {"0", "1"} for k in probs)
                or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values())
                or abs(math.fsum(probs.values()) - 1) > 1e-10):
            raise InvalidResult("Malformed normalized probability map")
        reverse = {k[::-1]: v for k, v in histogram.items()}
        def error(h): return max(abs(probs.get(k, 0) - h.get(k, 0) / shots) for k in set(probs) | set(h))
        direct_ok, reverse_ok = error(histogram) <= 1e-10, error(reverse) <= 1e-10
        if not (direct_ok or reverse_ok): raise InvalidResult("Probability map disagrees with raw frequencies")
        bit_order = "ambiguous_direct_or_reversed" if direct_ok and reverse_ok else "direct" if direct_ok else "reversed"
    zero = histogram.get("0" * q, 0)
    return {"p0": zero / shots, "zero_count": zero, "shots": shots,
            "measurement_columns": labels, "probability_bit_order_relative_to_columns": bit_order,
            "raw_rows_sha256": digest(rows), "histogram_sha256": digest(dict(histogram)),
            "unique_outcomes": len(histogram), "sampling_independence_certified": False}


class TianyanCloudAdapter:
    def __init__(self, transport, ledger, evidence=None, backend=BACKEND,
                 observation_seconds=300.0, poll_seconds=5.0, sleep=time.sleep, monotonic=time.monotonic):
        if not isinstance(ledger, PersistentLedger): raise PreflightBlocked("Persistent account ledger is required")
        if not isinstance(transport, Cqlib1311Transport) and getattr(transport, "is_mock", False) is not True:
            raise PreflightBlocked("Only reviewed Cqlib transport or explicitly labelled offline mock is accepted")
        if (not math.isfinite(observation_seconds) or not 0 < observation_seconds <= 1800
                or not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 20):
            raise PreflightBlocked("Invalid observation/poll bound")
        if isinstance(transport, Cqlib1311Transport) and poll_seconds != 5:
            raise PreflightBlocked("Production polling is fixed at 5/10/15 seconds, then 15 seconds")
        self.transport, self.ledger, self.evidence, self.backend = transport, ledger, evidence, backend
        self.observation_seconds, self.poll_seconds = observation_seconds, poll_seconds
        self.poll_delays = (poll_seconds, poll_seconds * 2, poll_seconds * 3)
        self.sleep, self.monotonic = sleep, monotonic

    def execution_identity(self):
        """Non-secret identity for matrix/checkpoint binding; never includes tokens."""
        return {"platform": "TianYan", "backend": self.backend, "interface": INTERFACE,
                "sdk_version": getattr(self.transport, "sdk_version", "unverified_mock"),
                "p0_mode": "empirical_raw_shot_frequency", "server_precision": "not_disclosed",
                "is_mock": getattr(self.transport, "is_mock", False) is True,
                "account_scope_sha256": self.ledger.account_hash,
                "canonical_ledger_path_sha256": hashlib.sha256(str(self.ledger.path.resolve()).encode()).hexdigest(),
                "client_code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("cloud.py", "ledger.py", "preflight.py")}}

    def _safe(self, value):
        f = getattr(self.transport, "sanitized", sanitize)
        return f(value)

    def probability(self, specs, shots, phase="main", context=None, deadline=None):
        if not isinstance(specs, list) or not 1 <= len(specs) <= 20:
            raise PreflightBlocked("Each explicit batch must contain 1..20 circuits; empty batches are errors")
        if type(shots) is not int or shots <= 0: raise PreflightBlocked("Shots must be a positive integer")
        if context is not None and not isinstance(context, dict): raise PreflightBlocked("Context must be a dictionary")
        context = self._safe(context or {})
        if "phase" in context:
            if phase != "main" and phase != context["phase"]: raise PreflightBlocked("Conflicting phase declarations")
            phase = context["phase"]
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise PreflightBlocked("Deadline must be an absolute monotonic timestamp")
        with self.ledger.execution():
            caps = validate_evidence(self.evidence, self.ledger.account_hash, self.backend,
                now=self.ledger.clock(), mock=getattr(self.transport, "is_mock", False) is True,
                ledger_path_hash=self.execution_identity()["canonical_ledger_path_sha256"])
            if shots > approved_shots_limit(caps, phase):
                raise PreflightBlocked("Shots exceed the approved client/known server limit for this phase")
            canonical = [circuit_spec(s, caps) for s in specs]
            if len({s["id"] for s in canonical}) != len(canonical): raise PreflightBlocked("Duplicate IDs within a batch")
            # Even minimally serialized raw rows cannot fit above this bound.
            # Do not send a batch that is already known to exceed the reader.
            lower_bytes = sum(shots * (2 * s["qubits"] + 2) for s in canonical)
            if lower_bytes > getattr(self.transport, "max_response_bytes", 128 * 1024**2):
                raise PreflightBlocked("Requested raw result cannot fit the bounded response reader; reduce the explicit batch")
            parser_checks = (self.transport.validate_offline(canonical)
                             if isinstance(self.transport, Cqlib1311Transport)
                             else [{"status": "offline_mock_only_no_sdk_parser"}])
            snapshot = self.ledger.snapshot()
            remaining = snapshot["deadline_epoch"] - self.ledger.clock()
            end = min(self.monotonic() + min(self.observation_seconds, remaining),
                      float("inf") if deadline is None else deadline)
            if self.monotonic() >= end: raise PreflightBlocked("No time remains to start a bounded cloud observation")
            fingerprint = digest({"circuits": [(s["id"], s["sha256"]) for s in canonical],
                                  "shots": shots, "backend": self.backend, "phase": phase, "context": context})
            # Re-read before EVERY submission, under the account execution lock.
            # This does not replace the separately required zero-hardware-quota proof.
            live = inspect_current_backend(self.transport, backend=self.backend,
                account_scope_sha256=self.ledger.account_hash, read_only_authorized=True,
                deadline=end, output_root=self.ledger.root / "backend_checks")
            if not live["running_and_free"]:
                raise PreflightBlocked("Current backend free/running check failed; no submission reserved or sent")
            if self.monotonic() >= end:
                raise PreflightBlocked("Observation deadline exhausted by the read-only backend check")
            # Proof may expire during a slow read; recheck before reserving work.
            validate_evidence(self.evidence, self.ledger.account_hash, self.backend,
                now=self.ledger.clock(), mock=getattr(self.transport, "is_mock", False) is True,
                ledger_path_hash=self.execution_identity()["canonical_ledger_path_sha256"])
            record = self.ledger.reserve(len(canonical), shots, phase, fingerprint,
                {"backend": self.backend, "interface": INTERFACE, "circuits": [
                    {k: v for k, v in s.items() if k != "qcis"} for s in canonical],
                 "context": context, "fee_evidence_sha256": digest(self.evidence),
                 "live_backend_check": live})
            aid = record["attempt_id"]
            folder = self.ledger.root / aid
            folder.mkdir()
            atomic_json(folder / "submission_intent.json", {"record": record, "circuits": canonical,
                "account_evidence": self._safe(self.evidence), "sdk_parser_checks": parser_checks,
                "observation_seconds": max(0, end-self.monotonic())})
            ids = None
            def save_submission_outcome():
                atomic_json(folder / "submission_outcome.json", {"returned_ids": self._safe(ids),
                    "response": self._safe(getattr(self.transport, "last_submission_response", None))})
            try:
                ids = self.transport.submit(canonical, shots, self.backend, aid, end)
                save_submission_outcome()
                if (not isinstance(ids, list) or len(ids) != len(canonical)
                        or any(type(v) not in (str, int) or str(v) == "" for v in ids)
                        or len({str(v) for v in ids}) != len(ids)):
                    raise UnknownSubmission("Submission returned incomplete/duplicate/unusable task IDs")
                ids = [str(v) for v in ids]
                self.ledger.update(aid, "submitted", job_ids=ids)
            except SubmissionRejected:
                save_submission_outcome()
                self.ledger.update(aid, "submission_rejected", reason="explicit_account_qubit_entitlement_rejection",
                    submission_evidence_file=str(folder / "submission_outcome.json"))
                raise
            except BaseException as exc:
                save_submission_outcome()
                self.ledger.update(aid, "submission_unknown", error_type=type(exc).__name__,
                    partial_returned_ids=self._safe(ids),
                    submission_evidence_file=str(folder / "submission_outcome.json"),
                    reason="May have reached server; no automatic retry or budget refund")
                if isinstance(exc, (KeyboardInterrupt, SystemExit)): raise
                raise UnknownSubmission("Submission outcome unknown; account ledger blocks new submissions") from None
            try:
                out = self._poll(ids, canonical, shots, end, folder, aid)
            except TerminalFailure:
                self.ledger.update(aid, "terminal_failure", reason="explicit_server_task_failure")
                raise
            except InvalidResult:
                self.ledger.update(aid, "received_invalid", reason="Result or returned-circuit validation failed")
                raise
            except BaseException as exc:
                self.ledger.update(aid, "observation_unknown", error_type=type(exc).__name__,
                    reason="Observation ended without validated completion; backend may continue; no retry")
                if isinstance(exc, (KeyboardInterrupt, SystemExit)): raise
                raise UnknownSubmission("Observation ended; backend status unknown; new submissions blocked") from None
            atomic_json(folder / "validated_results.json", out)
            self.ledger.update(aid, "validated", evidence_file=str(folder / "validated_results.json"))
            return out

    probabilities = probability

    def _poll(self, ids, specs, shots, end, folder, aid):
        received, query_count = {}, 0
        while self.monotonic() < end:
            response = self._safe(self.transport.query(ids, end))
            query_count += 1
            atomic_json(folder / ("query_%05d.json" % query_count), response)
            if not isinstance(response, dict) or type(response.get("code")) is not int:
                raise InvalidResult("Malformed query envelope")
            if response["code"] != 0:
                msg = str(response.get("msg", response.get("message", "")))
                if response["code"] == 1 and len(ids) == 1 and ids[0] in msg and "运行失败" in msg:
                    raise TerminalFailure("Named task explicitly failed; cause not inferred")
                raise UnknownSubmission("Query returned non-success; no queue/failure inference")
            data = response.get("data")
            if not isinstance(data, dict) or not isinstance(data.get("experimentResultModelList"), list):
                raise InvalidResult("Malformed result-list envelope")
            items = data["experimentResultModelList"]
            # An empty list is only 'no result observed', never proof of queue state.
            seen = set()
            for item in items:
                if not isinstance(item, dict) or type(item.get("experimentTaskId")) not in (str, int):
                    raise InvalidResult("Missing task identity")
                jid = str(item["experimentTaskId"])
                if jid not in ids or jid in seen: raise InvalidResult("Unexpected/duplicate result task ID")
                seen.add(jid)
                if item.get("resultStatus") is None and type(item.get("status")) is int and item["status"] == 1:
                    continue
                validation = validate_raw_result(item, jid, specs[ids.index(jid)], shots)
                if jid in received and received[jid] != validation:
                    raise InvalidResult("Completed result changed between observations")
                received[jid] = validation
            if len(received) == len(ids):
                result = []
                for jid, spec in zip(ids, specs):
                    if self.monotonic() >= end: raise TimeoutError("No observation time remains for circuit validation")
                    returned = self._safe(self.transport.returned_circuit(jid, end))
                    atomic_json(folder / ("returned_circuit_" + str(ids.index(jid)) + ".json"), returned)
                    if not isinstance(returned, list) or len(returned) != 1 or not isinstance(returned[0], dict):
                        raise InvalidResult("Returned circuit missing or ambiguous")
                    if ("experimentTaskId" in returned[0]
                            and str(returned[0]["experimentTaskId"]) != jid):
                        raise InvalidResult("Returned circuit carries a different task ID")
                    try: circuit_hash = hashlib.sha256(normalize_qcis(returned[0].get("qcis")).encode()).hexdigest()
                    except PreflightBlocked as exc: raise InvalidResult("Returned QCIS unavailable") from exc
                    if circuit_hash != spec["sha256"]: raise InvalidResult("Returned QCIS hash differs from submission")
                    if self.monotonic() >= end: raise TimeoutError("Circuit retrieval exceeded observation deadline")
                    provenance = {"platform": "TianYan", "backend": self.backend,
                        "execution_location": "mock_only" if getattr(self.transport, "is_mock", False) else "TianYan cloud simulator",
                        "interface": INTERFACE, "sdk_version": getattr(self.transport, "sdk_version", "unverified_mock"),
                        "p0_mode": "empirical_raw_shot_frequency", "server_precision": None,
                        "server_compute_seconds": None, "server_peak_memory_bytes": None,
                        "account_scope_sha256": self.ledger.account_hash, "attempt_id": aid,
                        "circuit_sha256": circuit_hash, "returned_circuit_hash_match": True,
                        "compiled_circuit_available": bool(returned[0].get("computerQcis")),
                        "mapped_circuit_available": bool(returned[0].get("mapQcis")),
                        "server_compilation_semantics_certified": False,
                        "automatic_submission_retry": False, "observed_utc": utc(self.ledger.clock()),
                        "client_code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                               for name in ("cloud.py", "ledger.py", "preflight.py")}}
                    result.append({"id": spec["id"], "circuit_id": spec["id"], "job_id": jid,
                        **received[jid], "evidence": {"directory": str(folder), "query_count": query_count},
                        "execution_provenance": provenance})
                return result
            delay = self.poll_delays[min(query_count - 1, len(self.poll_delays) - 1)]
            self.sleep(min(delay, max(0, end-self.monotonic())))
        raise TimeoutError("Bounded observation expired; backend task may still be running")


def create_provider(*, platform, account_ledger_root, account_scope, evidence,
                    backend=BACKEND, observation_seconds=300.0, poll_seconds=5.0):
    """Production factory. Caller injects existing authentication, never a key file.

    Fee evidence is checked before SDK inspection, directory creation or network.
    All experiments for this account use the SAME account_ledger_root. Passing a
    fresh run directory conflicts with the frozen canonical-ledger binding.
    """
    account_hash = hashlib.sha256(account_scope.encode()).hexdigest()
    ledger_path = Path(account_ledger_root).expanduser().resolve() / ("account-" + account_hash) / "ledger.json"
    validate_evidence(evidence, account_hash, backend,
                      ledger_path_hash=hashlib.sha256(str(ledger_path).encode()).hexdigest())
    transport = Cqlib1311Transport(platform)
    ledger = PersistentLedger(account_ledger_root, account_scope)
    return TianyanCloudAdapter(transport, ledger, evidence, backend,
                              observation_seconds=observation_seconds, poll_seconds=poll_seconds)
