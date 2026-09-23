"""Finite RWX-PVC mailbox for one SGW simulator attempt.

No socket, credential, or retry protocol is used.  The receiver owns physics;
the client only reads immutable response evidence and caches it for the
ProductionAdapter's snapshot/render follow-ups.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from types import SimpleNamespace

import numpy as np

from .adapters import AdapterError


class MailboxError(AdapterError):
    pass


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(_bytes(value)); f.flush(); os.fsync(f.fileno())
    directory = os.open(path.parent, os.O_RDONLY); os.fsync(directory); os.close(directory)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MailboxError(f"invalid mailbox manifest: {path}") from exc
    if not isinstance(value, dict):
        raise MailboxError("mailbox manifest must be an object")
    return value


def _array(path: Path, value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype == object or not np.isfinite(array).all():
        raise MailboxError("mailbox array is nonnumeric or nonfinite")
    with path.open("xb") as f:
        np.save(f, array, allow_pickle=False); f.flush(); os.fsync(f.fileno())
    return {"path": path.name, "sha256": _digest(path), "shape": list(array.shape), "dtype": str(array.dtype)}


def _load_array(root: Path, record: Mapping[str, Any]) -> np.ndarray:
    path = (root / str(record.get("path", ""))).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or _digest(path) != record.get("sha256"):
        raise MailboxError("mailbox array path or hash mismatch")
    with path.open("rb") as f:
        value = np.load(f, allow_pickle=False)
    if list(value.shape) != record.get("shape") or str(value.dtype) != record.get("dtype"):
        raise MailboxError("mailbox array shape/dtype mismatch")
    return value


def _encode_tree(directory: Path, prefix: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        if not value:
            raise MailboxError("empty policy-observation mapping is not transportable")
        return {"mapping": {str(key): _encode_tree(directory, f"{prefix}.{key}", item)
                            for key, item in value.items()}}
    return {"array": _array(directory / f"{prefix}.npy", value)}


def _decode_tree(root: Path, value: Any) -> Any:
    if not isinstance(value, Mapping):
        raise MailboxError("invalid policy-observation tree")
    if set(value) == {"array"}:
        return _load_array(root, value["array"])
    if set(value) == {"mapping"} and isinstance(value["mapping"], Mapping):
        return {key: _decode_tree(root, item) for key, item in value["mapping"].items()}
    raise MailboxError("invalid policy-observation tree node")


class MailboxClient:
    def __init__(self, *, root: Path, identity: Mapping[str, str], timeout_s: float = 30) -> None:
        self.root, self.identity, self.timeout_s = Path(root), dict(identity), timeout_s
        self.command = 0; self._cache: dict[str, Any] | None = None; self._closed = False
        if not self.root.is_dir() or not all(isinstance(v, str) and v for v in self.identity.values()):
            raise MailboxError("mailbox root or immutable identity is invalid")

    def _call(self, operation: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._closed: raise MailboxError("mailbox session is closed")
        self.command += 1
        token = f"{self.command:04d}-{operation}"
        request = self.root / "requests" / f"{token}.json"
        body = {"schema": "sgw-01-simulator-mailbox-v1", "operation": operation, "command_id": self.command,
                "identity": self.identity, "channel_nonce": self.identity["channel_nonce"], "payload": dict(payload or {})}
        try:
            _write(request, body)
        except FileExistsError as exc:
            self._closed = True
            raise MailboxError("mailbox command ID already exists; session cannot be reused") from exc
        response = self.root / "responses" / f"{token}.json"
        end = time.monotonic() + self.timeout_s
        while not response.exists():
            if time.monotonic() >= end:
                self._closed = True
                raise MailboxError("mailbox action timed out; session is permanently closed")
            time.sleep(.01)
        result = _read(response)
        if result.get("command_id") != self.command or result.get("identity") != self.identity or result.get("status") != "ok":
            self._closed = True; raise MailboxError("mailbox response identity/status mismatch")
        return result

    def reset(self) -> Any:
        result = self._call("reset"); self._cache = _decode_response(self.root / "responses", result)
        reset = self._cache["reset"]
        return SimpleNamespace(snapshot=reset["snapshot"], receipt=reset["receipt"])

    def step(self, action: Any) -> Mapping[str, Any]:
        temp = self.root / "requests" / f"{self.command + 1:04d}-step.action.npy"
        record = _array(temp, np.asarray(action, dtype=np.float32))
        if record["shape"] != [8]: raise MailboxError("mailbox step action must be finite shape [8]")
        result = self._call("step", {"action": record}); self._cache = _decode_response(self.root / "responses", result)
        return self._cache["step_result"]

    def snapshot(self) -> Any:
        if self._cache is None: raise MailboxError("no mailbox response cached")
        return self._cache["snapshot"]

    def render_viewport(self) -> np.ndarray:
        if self._cache is None: raise MailboxError("no mailbox response cached")
        return self._cache["viewport"]

    def policy_observation(self) -> Mapping[str, Any]:
        if self._cache is None: raise MailboxError("no mailbox response cached")
        return self._cache["policy_observation"]

    def close(self) -> None:
        if not self._closed:
            try: self._call("close")
            finally: self._closed = True


def create_mailbox_environment(*, cell: Any, evidence_root: Path) -> MailboxClient:
    """Explicit B200 factory; direct in-process environments remain unchanged."""
    root = Path(os.environ.get("SGW01_SIMULATOR_MAILBOX_ROOT", "")).resolve()
    identity_path = Path(os.environ.get("SGW01_SIMULATOR_MAILBOX_IDENTITY", "")).resolve()
    expected = os.environ.get("SGW01_SIMULATOR_MAILBOX_IDENTITY_SHA256", "")
    if not root.is_dir() or not identity_path.is_file() or _digest(identity_path) != expected:
        raise MailboxError("mailbox factory requires a hash-bound prospective simulator identity")
    identity = _read(identity_path)
    row = getattr(cell, "row", cell)
    if not isinstance(row, Mapping) or identity.get("cell_id") != row.get("cell_id"):
        raise MailboxError("mailbox identity does not bind the released cell")
    required = ("release_id", "cell_id", "attempt_id", "channel_nonce", "candidate_sha256",
                "binding_sha256", "simulator_job_uid", "simulator_pod_uid")
    if any(not isinstance(identity.get(key), str) or not identity[key] for key in required):
        raise MailboxError("mailbox identity is incomplete")
    # Evidence root is intentionally not used as a remote authority. The
    # recorder retains B200-side artifacts while the immutable mailbox root
    # holds A40-side raw request/response evidence.
    if not Path(evidence_root).is_absolute():
        raise MailboxError("recorder evidence root must be absolute")
    return MailboxClient(root=root, identity={key: identity[key] for key in required},
                         timeout_s=float(os.environ.get("SGW01_SIMULATOR_MAILBOX_TIMEOUT_S", "30")))


def _decode_response(root: Path, response: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(response.get("data", {}))
    viewport = _load_array(root, data.pop("viewport"))
    policy = _decode_tree(root, data.pop("policy_observation"))
    return {**data, "viewport": viewport, "policy_observation": policy}


class MailboxReceiver:
    """One finite receiver; caller supplies an already-created native environment."""
    def __init__(self, *, root: Path, identity: Mapping[str, str], environment: Any) -> None:
        self.root, self.identity, self.environment = Path(root), dict(identity), environment
        self.last = 0; self.closed = False

    def serve_one(self, request_path: Path) -> None:
        try:
            request = _read(request_path); command = request.get("command_id")
            if self.closed or type(command) is not int or command != self.last + 1 or request.get("identity") != self.identity:
                raise MailboxError("stale, duplicate, or identity-mismatched mailbox command")
            operation = request.get("operation")
            if operation == "reset":
                reset = self.environment.reset()
                reset = {"snapshot": asdict(reset.snapshot) if is_dataclass(reset.snapshot) else reset.snapshot,
                         "receipt": dict(reset.receipt)}
                result = {"reset": reset, "step_result": None}
            elif operation == "step":
                action = _load_array(request_path.parent, request["payload"]["action"])
                result = {"reset": None, "step_result": self.environment.step(action)}
            elif operation == "close":
                self.environment.close(); self.closed = True; result = {"reset": None, "step_result": None}
            else: raise MailboxError("unsupported mailbox operation")
            snapshot, viewport, policy = self.environment.snapshot(), self.environment.render_viewport(), self.environment.policy_observation()
            out = self.root / "responses"; token = request_path.stem
            arrays = {"viewport": _array(out / f"{token}.viewport.npy", viewport)}
            payload = {"command_id": command, "identity": self.identity, "status": "ok",
                       "data": {**result, "snapshot": asdict(snapshot) if is_dataclass(snapshot) else snapshot,
                                "viewport": arrays["viewport"],
                                "policy_observation": _encode_tree(out, f"{token}.policy", policy)}}
            response_path = out / f"{token}.json"
            _write(response_path, payload); self.last = command
            if operation == "close":
                _write(self.root / "receiver_complete.json", {
                    "identity": self.identity,
                    "close_command_id": command,
                    "command_count": self.last,
                    "close_response_sha256": _digest(response_path),
                })
        except Exception as exc:
            fault = self.root / "faults" / f"{request_path.stem}.json"
            if not fault.exists():
                _write(fault, {"request": request_path.name, "error_type": type(exc).__name__, "error": str(exc),
                               "identity": self.identity})
            raise
