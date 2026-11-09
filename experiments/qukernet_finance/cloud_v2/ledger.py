"""Persistent account-scoped reservation ledger; no network or credentials.

All clients for one account MUST use the same private ``account_ledger_root``.
An OS lock serializes cooperating processes on that host. This cannot police an
unrelated application or another host; preflight requires reconciliation of
those jobs. Rejected, uncertain and failed attempts all consume reservations.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

HARD_LIMITS = {"seconds": 72 * 3600, "shots": 10_000_000_000,
               "circuits": 80_000_000, "batch": 20,
               "preflight_circuits": 50, "preflight_shots": 20_000}
TERMINAL = {"validated", "terminal_failure", "submission_rejected"}
_THREAD_LOCKS = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class LedgerError(RuntimeError): pass
class BudgetExceeded(LedgerError): pass
class UnresolvedTask(LedgerError): pass
class DuplicateSubmission(LedgerError): pass


def utc(ts=None):
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def public_source_root():
    """Course checkout uses experiments/; deployed snapshot uses code/."""
    source = Path(__file__).resolve().parents[2]
    return source.parent if source.name == "experiments" else source


def private_directory(root):
    """Reject network/public destinations BEFORE any mkdir or lock-file write."""
    raw = os.fspath(root)
    if raw.startswith(("\\\\", "//")):
        raise LedgerError("Private output must be a local directory")
    path = Path(root).expanduser().resolve()
    public = public_source_root()
    if path == public or public in path.parents:
        raise LedgerError("Private output must be outside the public course tree")
    if os.name == "nt":
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(path.anchor) == 4:
            raise LedgerError("Mapped network output is forbidden")
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False).encode("utf-8")
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("xb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            fd = os.open(str(path.parent), os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
    finally:
        if tmp.exists(): tmp.unlink()


@contextmanager
def file_lock(path):
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        guard = _THREAD_LOCKS.setdefault(key, threading.Lock())
    if not guard.acquire(blocking=False):
        raise UnresolvedTask("Another account operation holds the global lock")
    handle = None
    locked = False
    try:
        handle = path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except (OSError, BlockingIOError) as exc:
            raise UnresolvedTask("Another process holds the account lock") from exc
        yield
    finally:
        if handle is not None:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        guard.release()


class PersistentLedger:
    def __init__(self, account_ledger_root, account_scope, limits=None, clock=time.time):
        if not isinstance(account_scope, str) or not account_scope.strip():
            raise LedgerError("A stable non-secret account scope is required")
        self.account_hash = hashlib.sha256(account_scope.encode()).hexdigest()
        base = private_directory(account_ledger_root)
        self.root = base / ("account-" + self.account_hash)
        if self.root.is_symlink() or self.root.resolve().parent != base:
            raise LedgerError("Account directory may not redirect outside the private ledger root")
        self.root.mkdir(exist_ok=True)
        self.path = self.root / "ledger.json"
        self.clock = clock
        self.limits = dict(HARD_LIMITS)
        for key, value in (limits or {}).items():
            if key not in HARD_LIMITS or type(value) is not int or not 0 < value <= HARD_LIMITS[key]:
                raise LedgerError("Limits may only reduce the approved positive integer caps")
            self.limits[key] = value
        with file_lock(self.root / "ledger.lock"):
            if self.path.exists():
                self._read()
            else:
                now = clock()
                atomic_json(self.path, {"schema_version": 1, "account_scope_sha256": self.account_hash,
                    "started_epoch": now, "started_utc": utc(now),
                    "deadline_epoch": now + self.limits["seconds"], "limits": self.limits,
                    "entries": [], "scope": "all cooperating clients using this canonical account ledger"})

    def _read(self):
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if (value.get("schema_version") != 1 or value.get("account_scope_sha256") != self.account_hash
                or value.get("limits") != self.limits or not isinstance(value.get("entries"), list)):
            raise LedgerError("Ledger schema, account or frozen limits mismatch")
        if value.get("deadline_epoch") != value.get("started_epoch", 0) + self.limits["seconds"]:
            raise LedgerError("Ledger deadline was changed")
        ids = [e.get("attempt_id") for e in value["entries"]]
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids):
            raise LedgerError("Duplicate or malformed ledger attempts")
        for e in value["entries"]:
            if (type(e.get("circuits")) is not int or e["circuits"] <= 0
                    or type(e.get("shots_per_circuit")) is not int or e["shots_per_circuit"] <= 0
                    or e.get("requested_shots") != e["circuits"] * e["shots_per_circuit"]):
                raise LedgerError("Malformed reserved resource counters")
        return value

    @contextmanager
    def execution(self):
        """Hold across reservation, submission, polling and returned-circuit checks."""
        with file_lock(self.root / "execution.lock"):
            yield

    def snapshot(self):
        with file_lock(self.root / "ledger.lock"):
            return self._read()

    def reserve(self, circuits, shots, phase, fingerprint, metadata):
        if type(circuits) is not int or not 1 <= circuits <= self.limits["batch"]:
            raise BudgetExceeded("Empty batch or approved batch limit exceeded")
        if type(shots) is not int or shots <= 0 or phase not in ("main", "preflight"):
            raise LedgerError("Invalid shots or phase")
        with file_lock(self.root / "ledger.lock"):
            data = self._read()
            if self.clock() >= data["deadline_epoch"]:
                raise BudgetExceeded("72-hour account execution window expired")
            if any(e.get("state") not in TERMINAL for e in data["entries"]):
                raise UnresolvedTask("Unresolved or invalid prior attempt blocks account-wide submission")
            if any(e.get("fingerprint") == fingerprint for e in data["entries"]):
                raise DuplicateSubmission("Request already reserved; do not resubmit")
            all_c = sum(e["circuits"] for e in data["entries"]) + circuits
            all_s = sum(e["requested_shots"] for e in data["entries"]) + circuits * shots
            if all_c > self.limits["circuits"] or all_s > self.limits["shots"]:
                raise BudgetExceeded("Account-wide requested resource cap exceeded")
            if phase == "preflight":
                prev = [e for e in data["entries"] if e["phase"] == "preflight"]
                if (sum(e["circuits"] for e in prev) + circuits > self.limits["preflight_circuits"]
                        or sum(e["requested_shots"] for e in prev) + circuits * shots > self.limits["preflight_shots"]):
                    raise BudgetExceeded("Preflight cap exceeded (included in global totals)")
            entry = {"attempt_id": uuid.uuid4().hex, "state": "submission_pending",
                "submission_started_utc": utc(self.clock()), "phase": phase,
                "circuits": circuits, "shots_per_circuit": shots, "requested_shots": circuits * shots,
                "fingerprint": fingerprint, "metadata": metadata,
                "history": [{"state": "submission_pending", "utc": utc(self.clock())}]}
            data["entries"].append(entry)
            atomic_json(self.path, data)  # Durable reservation BEFORE any submission call.
            return entry

    def update(self, attempt_id, state, **fields):
        allowed = {"submission_pending": {"submitted", "submission_unknown", "submission_rejected"},
                   "submitted": {"submitted", "validated", "terminal_failure", "observation_unknown", "received_invalid"}}
        with file_lock(self.root / "ledger.lock"):
            data = self._read()
            found = [e for e in data["entries"] if e["attempt_id"] == attempt_id]
            if len(found) != 1:
                raise LedgerError("Attempt missing from account ledger")
            e = found[0]
            if state not in allowed.get(e["state"], set()):
                raise LedgerError("Forbidden state transition; unresolved jobs require independent reconciliation")
            protected = {"attempt_id", "fingerprint", "circuits", "shots_per_circuit", "requested_shots",
                         "submission_started_utc", "phase", "history", "metadata", "state"}
            if protected.intersection(fields):
                raise LedgerError("Reserved counters or provenance cannot be overwritten")
            if "job_ids" in fields:
                previous = {jid for other in data["entries"] if other is not e for jid in other.get("job_ids", [])}
                if previous.intersection(fields["job_ids"]):
                    raise LedgerError("Server reused a prior task ID; independent reconciliation required")
            e.update(fields); e["state"] = state
            e["history"].append({"state": state, "utc": utc(self.clock())})
            atomic_json(self.path, data)
