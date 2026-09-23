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


def _decode_response(root: Path, response: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(response.get("data", {}))
    viewport = _load_array(root, data.pop("viewport"))
    policy = {key: _load_array(root, value) for key, value in data.pop("policy_arrays").items()}
    return {**data, "viewport": viewport, "policy_observation": policy}


class MailboxReceiver:
    """One finite receiver; caller supplies an already-created native environment."""
    def __init__(self, *, root: Path, identity: Mapping[str, str], environment: Any) -> None:
        self.root, self.identity, self.environment = Path(root), dict(identity), environment
        self.last = 0; self.closed = False

    def serve_one(self, request_path: Path) -> None:
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
            reset = None
        elif operation == "close": self.environment.close(); self.closed = True; reset = None; result = {"reset": None, "step_result": None}
        else: raise MailboxError("unsupported mailbox operation")
        snapshot, viewport, policy = self.environment.snapshot(), self.environment.render_viewport(), self.environment.policy_observation()
        out = self.root / "responses"; token = request_path.stem
        arrays = {"viewport": _array(out / f"{token}.viewport.npy", viewport)}
        policy_arrays = {key: _array(out / f"{token}.policy.{key.replace('/', '_')}.npy", value) for key, value in policy.items()}
        payload = {"command_id": command, "identity": self.identity, "status": "ok",
                   "data": {**result, "snapshot": asdict(snapshot) if is_dataclass(snapshot) else snapshot,
                            "viewport": arrays["viewport"], "policy_arrays": policy_arrays}}
        _write(out / f"{token}.json", payload); self.last = command
