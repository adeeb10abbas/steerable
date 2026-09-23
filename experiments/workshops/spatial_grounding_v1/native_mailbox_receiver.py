"""A40-side entrypoint for one bounded SGW native simulator mailbox attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

from .adapters import AdapterError
from .simulator_mailbox import MailboxReceiver


def _failure(path: Path, error: BaseException, identity: dict[str, str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        payload = {"error_type": type(error).__name__, "error": str(error), "identity": identity,
                   "traceback": "".join(traceback.format_exception(error))}
        with path.open("x") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())


def _load_identity(path: Path, expected_sha256: str) -> dict[str, str]:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise AdapterError("receiver identity record is absent or hash-mismatched")
    value = json.loads(path.read_text())
    required = ("release_id", "cell_id", "attempt_id", "channel_nonce", "candidate_sha256",
                "binding_sha256", "simulator_job_uid", "simulator_pod_uid")
    if not isinstance(value, dict) or any(not isinstance(value.get(key), str) or not value[key] for key in required):
        raise AdapterError("receiver identity record is incomplete")
    return {key: value[key] for key in required}


def verify_receiver_completion(root: Path, identity: dict[str, str]) -> dict[str, Any]:
    """Validate a completed learned-policy receiver from a separate process."""
    root = Path(root)
    complete = root / "receiver_complete.json"
    failures = ("receiver_failure.json", "receiver_cleanup_failure.json", "receiver_app_cleanup_failure.json")
    if any((root / name).exists() for name in failures) or any((root / "faults").glob("*.json")) or not complete.is_file():
        raise AdapterError("receiver has no clean completion receipt")
    try:
        receipt = json.loads(complete.read_text())
        if (receipt.get("schema") != "sgw-01-mailbox-receiver-completion-v1"
                or receipt.get("attempt_scope") != "learned_policy_remote_simulator"
                or receipt.get("identity") != identity
                or type(receipt.get("close_command_id")) is not int
                or receipt["close_command_id"] < 1
                or receipt.get("close_command_id") != receipt.get("command_count")):
            raise ValueError("completion fields do not bind the learned attempt")
        response = root / "responses" / f"{receipt['close_command_id']:04d}-close.json"
        if not response.is_file() or hashlib.sha256(response.read_bytes()).hexdigest() != receipt.get("close_response_sha256"):
            raise ValueError("close response is absent or hash-mismatched")
        payload = json.loads(response.read_text())
        if (payload.get("status") != "ok" or payload.get("identity") != identity
                or payload.get("command_id") != receipt["close_command_id"]
                or payload.get("data") != {"reset": None, "step_result": None}):
            raise ValueError("close response is invalid")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AdapterError("receiver completion receipt is malformed") from exc
    return receipt


def _load_bound_cell(path: Path, expected_sha256: str, identity: dict[str, str]) -> dict[str, Any]:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise AdapterError("receiver release-cell binding is absent or hash-mismatched")
    try:
        cell = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError("receiver release-cell JSON is malformed") from exc
    if not isinstance(cell, dict) or any(cell.get(key) != identity[key] for key in
                                        ("cell_id", "candidate_sha256", "binding_sha256")):
        raise AdapterError("receiver identity does not bind the actual release-cell bytes")
    return cell


def _verify_native_authority(cell: dict[str, Any], identity: dict[str, str]) -> None:
    """Bind mailbox labels to the exact native binding before AppLauncher."""
    job_uid, pod_uid = os.environ.get("JOB_UID"), os.environ.get("POD_UID")
    if (not isinstance(job_uid, str) or not job_uid or not isinstance(pod_uid, str) or not pod_uid
            or identity["simulator_job_uid"] != job_uid or identity["simulator_pod_uid"] != pod_uid):
        raise AdapterError("receiver mailbox identity differs from Downward API Job/pod identity")
    from .robolab_jointpos_environment import JointPositionBinding, _sha256
    binding_path = Path(os.environ.get("SGW01_ENV_BINDING", "")).resolve()
    if not binding_path.is_file():
        raise AdapterError("receiver has no concrete joint-position binding path")
    binding = JointPositionBinding.load()
    if _sha256(binding_path) != identity["binding_sha256"]:
        raise AdapterError("mailbox identity binding hash differs from actual joint-position binding")
    record, _candidate = binding.cell(cell)
    if record.get("candidate_file_sha256") != identity["candidate_sha256"]:
        raise AdapterError("mailbox identity candidate hash differs from actual selected native candidate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mailbox-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--deadline-seconds", type=int, required=True)
    parser.add_argument("--release-cell-json", type=Path, required=True)
    parser.add_argument("--release-cell-sha256", required=True)
    args = parser.parse_args()
    if args.deadline_seconds <= 0 or args.mailbox_root.exists():
        raise AdapterError("receiver requires a new mailbox root and finite positive deadline")
    identity = _load_identity(args.identity, args.identity_sha256)
    cell = _load_bound_cell(args.release_cell_json, args.release_cell_sha256, identity)
    _verify_native_authority(cell, identity)
    args.mailbox_root.mkdir(mode=0o700)
    for name in ("requests", "responses", "faults"):
        (args.mailbox_root / name).mkdir()
    environment: Any = None
    receiver: MailboxReceiver | None = None
    app: Any = None
    failure = args.mailbox_root / "receiver_failure.json"
    try:
        # Imports and initialization remain inside the failure guard because
        # AppLauncher cleanup can terminate an otherwise propagating process.
        from isaaclab.app import AppLauncher
        from .robolab_jointpos_environment import create_environment
        app = AppLauncher({"headless": True, "enable_cameras": True}).app
        environment = create_environment(cell=cell, evidence_root=args.mailbox_root / "evidence")
        receiver = MailboxReceiver(root=args.mailbox_root, identity=identity, environment=environment)
        deadline = time.monotonic() + args.deadline_seconds
        while not receiver.closed and time.monotonic() < deadline:
            for request in sorted((args.mailbox_root / "requests").glob("*.json")):
                if int(request.name[:4]) > receiver.last:
                    receiver.serve_one(request)
            time.sleep(.01)
        if not receiver.closed:
            raise AdapterError("receiver deadline elapsed before close")
    except BaseException as exc:
        _failure(failure, exc, identity)
        raise
    finally:
        try:
            if receiver is not None:
                receiver.close_environment()
            elif environment is not None:
                environment.close()
        except BaseException as exc:
            _failure(args.mailbox_root / "receiver_cleanup_failure.json", exc, identity)
            raise
        finally:
            try:
                if app is not None:
                    app.close()
            except BaseException as exc:
                _failure(args.mailbox_root / "receiver_app_cleanup_failure.json", exc, identity)
                raise


if __name__ == "__main__":
    main()
