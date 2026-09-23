"""Bounded, read-only NFS metadata refresh for live cross-node mailboxes.

NFS negative lookup caching can hide a published response for 30 seconds.
STATX_FORCE_SYNC refreshes directory metadata without touching evidence files,
restarting either worker, or reissuing a simulator/model command.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


class DirectoryRefresher:
    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.statx = self.libc.statx
        self.statx.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint,
            ctypes.c_void_p,
        ]
        self.statx.restype = ctypes.c_int

    def __call__(self, directory: Path):
        # Linux's statx ABI reserves 256 bytes, including future extensions.
        result = ctypes.create_string_buffer(256)
        if self.statx(-100, os.fsencode(directory), 0x2000, 0x7FF, result) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(directory))


def refresh_mailboxes(root: Path, refresh: DirectoryRefresher) -> int:
    mailboxes = root / "mailboxes"
    refresh(mailboxes)
    count = 1
    for channel in sorted(mailboxes.iterdir()):
        if not channel.is_dir():
            continue
        refresh(channel)
        count += 1
        if (channel / "receiver_complete.json").is_file():
            continue
        for name in ("requests", "responses", "faults"):
            directory = channel / name
            if directory.is_dir():
                refresh(directory)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("policy", "simulator"))
    parser.add_argument("--seconds", required=True, type=int)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 14400:
        parser.error("--seconds must be in [1, 14400]")
    root = args.root.resolve(strict=True)
    plan = json.loads((root / "execution-plan.json").read_bytes())
    owner = json.loads((root / f"claims/{args.role}/owner.json").read_bytes())
    refresh = DirectoryRefresher()
    deadline = time.monotonic() + args.seconds
    count = 0
    print(json.dumps({
        "event": "read_only_metadata_refresh_started",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role, "owner": owner, "root": str(root),
        "native_source_commit": plan["source_commit"],
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pid": os.getpid(), "hostname": os.environ.get("HOSTNAME"),
        "maximum_seconds": args.seconds,
        "changes_evidence_or_commands": False,
    }), flush=True)
    reason = "deadline"
    while time.monotonic() < deadline:
        refresh(root)
        if list(root.glob("failure-*.json")):
            reason = "worker_failure"
            break
        if (root / f"{args.role}-complete.json").is_file():
            reason = "worker_complete"
            break
        count += refresh_mailboxes(root, refresh)
        time.sleep(0.1)
    print(json.dumps({
        "event": "read_only_metadata_refresh_stopped",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role, "reason": reason, "directory_refreshes": count,
    }), flush=True)


if __name__ == "__main__":
    main()
