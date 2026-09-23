"""A40-side entrypoint for one bounded SGW native simulator mailbox attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .adapters import AdapterError
from .simulator_mailbox import MailboxReceiver


def _load_identity(path: Path, expected_sha256: str) -> dict[str, str]:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise AdapterError("receiver identity record is absent or hash-mismatched")
    value = json.loads(path.read_text())
    required = ("release_id", "cell_id", "attempt_id", "channel_nonce", "candidate_sha256",
                "binding_sha256", "simulator_job_uid", "simulator_pod_uid")
    if not isinstance(value, dict) or any(not isinstance(value.get(key), str) or not value[key] for key in required):
        raise AdapterError("receiver identity record is incomplete")
    return {key: value[key] for key in required}


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
    if not args.release_cell_json.is_file() or hashlib.sha256(args.release_cell_json.read_bytes()).hexdigest() != args.release_cell_sha256:
        raise AdapterError("receiver release-cell binding is absent or hash-mismatched")
    args.mailbox_root.mkdir(mode=0o700)
    for name in ("requests", "responses", "faults"):
        (args.mailbox_root / name).mkdir()
    # Imports after immutable argument validation: no Isaac app exists on bad input.
    from isaaclab.app import AppLauncher
    from .robolab_jointpos_environment import create_environment
    app = AppLauncher({"headless": True, "enable_cameras": True}).app
    environment: Any = None
    try:
        cell = json.loads(args.release_cell_json.read_text())
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
    finally:
        if environment is not None:
            environment.close()
        app.close()


if __name__ == "__main__":
    main()
