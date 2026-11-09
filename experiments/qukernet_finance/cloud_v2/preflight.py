"""Pure fail-closed gates. Evidence is supplied by an independent account audit.

No values default to approval. A fee record must bind account, backend and
interface and prove BOTH zero fees and zero real-device quota deduction. The
caller must provide a current capability record and independently reconcile
other jobs. Neither an isToll flag nor a successful submission proves this.
"""
from datetime import datetime, timezone
import hashlib
import math
import re
import uuid

INTERFACE = "cqlib-1.3.11:TianYanPlatform.submit_experiment"
BACKEND = "tianyan_sw"


class CloudError(RuntimeError): pass
class PreflightBlocked(CloudError): pass


def inspect_current_backend(transport, *, backend=BACKEND, account_scope_sha256,
                            read_only_authorized=False, deadline, output_root=None):
    """One authorized READ of the reviewed SDK resource-list endpoint.

    Accept an existing transport (which holds authentication only in memory).
    No credential files, login, submit, cancellation or quota-consuming probes.
    ``deadline`` is an absolute monotonic deadline. Optional records are written
    only to an explicitly chosen private local directory. This observation can
    prove current isToll/status, NEVER account quota isolation or shot limits.
    The caller still needs independent fee/quota evidence before any submission.
    """
    if read_only_authorized is not True:
        raise PreflightBlocked("Explicit account resource-read authorization is required")
    if backend != BACKEND or not re.fullmatch(r"[0-9a-f]{64}", str(account_scope_sha256)):
        raise PreflightBlocked("Read-only backend/account scope is invalid")
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise PreflightBlocked("Read-only inspection needs a finite monotonic deadline")
    from .ledger import atomic_json, digest, private_directory
    # Validate a requested output destination BEFORE the network read.
    output = private_directory(output_root) if output_root is not None else None
    record = {"backend": backend, "account_scope_sha256": account_scope_sha256,
        "read_only": True, "submitted_circuits": 0, "running_and_free": False,
        "quota_isolation_proven_by_this_read": False,
        "limits_inferred_from_this_read": False,
        "source": "Cqlib 1.3.11 MACHINE_LIST_PATH GET; raw isToll/status values"}
    try:
        response = transport.backend_snapshot(deadline)
        clean = transport.sanitized(response) if callable(getattr(transport, "sanitized", None)) else response
        record["response_sha256"] = digest(clean)
        if not isinstance(response, dict) or type(response.get("code")) is not int or response["code"] != 0:
            raise PreflightBlocked("Backend resource query was not successful")
        rows = response.get("data")
        if not isinstance(rows, list): raise PreflightBlocked("Resource data is not the verified list schema")
        selected = [r for r in rows if isinstance(r, dict) and r.get("code") == backend]
        if len(selected) != 1: raise PreflightBlocked("Named backend is missing or duplicated")
        row = selected[0]
        record.update(isToll=row.get("isToll"), status=row.get("status"))
        if (type(row.get("isToll")) is not int or row["isToll"] != 1
                or type(row.get("status")) is not int or row["status"] != 0):
            raise PreflightBlocked("Named backend is not currently confirmed free and running")
        record["running_and_free"] = True
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        # Do not persist external exception strings that could contain headers.
        record["check_error"] = str(exc) if isinstance(exc, PreflightBlocked) else "Read failed; no submission made"
    record["observed_utc"] = datetime.now(timezone.utc).isoformat()
    if output is not None:
        path = output / ("backend_readonly_" + uuid.uuid4().hex + ".json")
        record["evidence_file"] = str(path)
        atomic_json(path, record)
    return record


def _timestamp(value):
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if d.tzinfo is None: raise ValueError()
        return d.timestamp()
    except (ValueError, TypeError, AttributeError) as exc:
        raise PreflightBlocked("Evidence needs timezone-aware validity timestamps") from exc


def validate_evidence(evidence, account_hash, backend=BACKEND, now=None, mock=False, ledger_path_hash=None):
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    if backend != BACKEND:
        raise PreflightBlocked("Only tianyan_sw finite-shot mode is implemented; SA semantics remain unverified")
    if not isinstance(evidence, dict): raise PreflightBlocked("Independent account evidence missing")
    if ledger_path_hash is not None and evidence.get("canonical_ledger_path_sha256") != ledger_path_hash:
        raise PreflightBlocked("Approval binds one canonical account ledger; another output root cannot reset its budget")
    for key, expected in (("account_scope_sha256", account_hash), ("backend", backend),
                          ("interface", INTERFACE), ("authorized", True), ("free", True),
                          ("zero_real_device_quota", True), ("no_untracked_inflight_jobs", True)):
        if type(expected) is bool:
            if evidence.get(key) is not True: raise PreflightBlocked("Unconfirmed evidence field: " + key)
        elif evidence.get(key) != expected: raise PreflightBlocked("Evidence scope mismatch: " + key)
    kinds = {"official_account_terms", "official_written_confirmation", "official_nonexecuting_estimate"}
    if mock: kinds.add("test_fixture")
    if evidence.get("source_kind") not in kinds:
        raise PreflightBlocked("Affirmative official fee/quota evidence required")
    if (not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("source_sha256", "")))
            or not isinstance(evidence.get("source_locator"), str) or not evidence["source_locator"].strip()):
        raise PreflightBlocked("Fee evidence source hash and locator are required")
    observed, expiry = _timestamp(evidence.get("observed_utc")), _timestamp(evidence.get("valid_until_utc"))
    if not observed <= now < expiry:
        raise PreflightBlocked("Fee/capability evidence is not currently valid")
    caps = evidence.get("capabilities")
    if not isinstance(caps, dict) or caps.get("backend") != backend or caps.get("running") is not True:
        raise PreflightBlocked("Current named-backend capability evidence missing")
    for key in ("max_qubits", "client_max_circuit_characters", "approved_probe_shots_limit"):
        if type(caps.get(key)) is not int or caps[key] <= 0:
            raise PreflightBlocked("Explicit positive capability/client limit required: " + key)
    for key in ("server_max_shots", "server_max_circuit_characters", "approved_main_shots_limit"):
        if caps.get(key) is not None and (type(caps[key]) is not int or caps[key] <= 0):
            raise PreflightBlocked("Limit must be a positive integer or explicitly unknown: " + key)
    if "server_max_shots" not in caps or "server_max_circuit_characters" not in caps:
        raise PreflightBlocked("Server shot/text limits must be recorded, including null when unknown")
    if not isinstance(caps.get("supported_gates"), list) or not caps["supported_gates"]:
        raise PreflightBlocked("Reviewed gate list required")
    if caps.get("gate_validation_basis") not in ("official_backend_metadata", "installed_cqlib_1.3.11_offline_parser"):
        raise PreflightBlocked("Distinguish backend gate evidence from installed-SDK offline parsing")
    if not caps.get("source_locator") or not re.fullmatch(r"[0-9a-f]{64}", str(caps.get("source_sha256", ""))):
        raise PreflightBlocked("Capability evidence source required")
    if not _timestamp(caps.get("observed_utc")) <= now < _timestamp(caps.get("valid_until_utc")):
        raise PreflightBlocked("Backend capability evidence is not currently valid")
    return caps


def approved_shots_limit(caps, phase):
    """Approved client limit, optionally capped by a separately known server max."""
    if phase == "preflight":
        limit = caps.get("approved_probe_shots_limit")
    elif phase == "main":
        limit = caps.get("approved_main_shots_limit")
    else:
        raise PreflightBlocked("Unknown execution phase")
    if type(limit) is not int or limit <= 0:
        raise PreflightBlocked("Explicit client shot approval required for this phase")
    maximum = caps.get("server_max_shots")
    return min(limit, maximum) if maximum is not None else limit


def normalize_qcis(text):
    if not isinstance(text, str) or "\x00" in text:
        raise PreflightBlocked("QCIS must be a text circuit")
    value = "\n".join(line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip())
    if not value: raise PreflightBlocked("QCIS cannot be empty")
    return value


def circuit_spec(value, caps):
    if not isinstance(value, dict): raise PreflightBlocked("Circuit specification must be a dictionary")
    ident = value.get("circuit_id", value.get("id"))
    if not isinstance(ident, str) or not ident or len(ident) > 256:
        raise PreflightBlocked("Circuit needs a bounded non-secret ID")
    if value.get("id", ident) != ident: raise PreflightBlocked("Conflicting circuit IDs")
    q = value.get("qubits")
    if type(q) is not int or not 1 <= q <= min(36, caps["max_qubits"]):
        raise PreflightBlocked("Circuit qubit count exceeds verified sw capability")
    text = normalize_qcis(value.get("qcis"))
    text_limit = min(20_000, caps["client_max_circuit_characters"])
    if caps.get("server_max_circuit_characters") is not None:
        text_limit = min(text_limit, caps["server_max_circuit_characters"])
    if len(text) > text_limit:
        raise PreflightBlocked("Circuit text exceeds the explicit client/known server character limit")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if value.get("sha256", checksum) != checksum: raise PreflightBlocked("Input circuit SHA256 mismatch")
    measurements, used, reached_measurement = [], set(), False
    # This adapter supports the research route's numeric RX/RY/RZ, CX/CNOT,
    # CZ and elementary one-qubit gates. Other SDK gates need a reviewed route.
    one = {"I", "X", "Y", "Z", "H", "S", "SD", "T", "TD", "X2P", "X2M", "Y2P", "Y2M"}
    for line in text.splitlines():
        parts = line.split(); gate = parts[0]
        if gate not in caps["supported_gates"]: raise PreflightBlocked("Gate is outside the reviewed gate list")
        if gate == "M":
            reached_measurement = True
            if len(parts) < 2 or any(not re.fullmatch(r"Q\d+", x) for x in parts[1:]):
                raise PreflightBlocked("Malformed terminal measurement")
            labels = [int(x[1:]) for x in parts[1:]]
            measurements.extend(labels); used.update(labels); continue
        if reached_measurement: raise PreflightBlocked("Only terminal computational-basis measurements are allowed")
        width = 3 if gate in {"RX", "RY", "RZ", "CX", "CNOT", "CZ"} else 2 if gate in one else None
        if width is None or len(parts) != width or not re.fullmatch(r"Q\d+", parts[1]):
            raise PreflightBlocked("Unsupported or malformed QCIS instruction")
        labels = [int(parts[1][1:])]
        if gate in {"RX", "RY", "RZ"}:
            try: angle = float(parts[2])
            except ValueError as exc: raise PreflightBlocked("Rotation must be a numeric angle") from exc
            if not math.isfinite(angle): raise PreflightBlocked("Nonfinite rotation")
        elif gate in {"CX", "CNOT", "CZ"}:
            if not re.fullmatch(r"Q\d+", parts[2]): raise PreflightBlocked("Malformed two-qubit gate")
            labels.append(int(parts[2][1:]))
            if labels[0] == labels[1]: raise PreflightBlocked("Two-qubit gate repeats a label")
        used.update(labels)
    if len(measurements) != q or len(set(measurements)) != q or used != set(measurements):
        raise PreflightBlocked("Every used qubit must be measured exactly once; full p(0) only")
    if set(measurements) not in (set(range(q)), set(range(1, q + 1))):
        raise PreflightBlocked("Only explicit contiguous Q0..Q(q-1) or Q1..Qq labels are supported")
    supplied = value.get("measured_qubits", measurements)
    if supplied != measurements: raise PreflightBlocked("Measurement order differs from declared QCIS")
    if any(type(x) is not int or x < 0 for x in measurements): raise PreflightBlocked("Invalid qubit labels")
    return {"id": ident, "circuit_id": ident, "qcis": text, "qubits": q,
            "measured_qubits": measurements, "sha256": checksum}
