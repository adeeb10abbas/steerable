"""Bounded cross-pod PVC lock probe; never imports a policy or simulator."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import time


def persist(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink()


def wait_for(path: Path, deadline: float) -> dict:
    while time.monotonic() < deadline:
        if path.is_file():
            return json.loads(path.read_text())
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {path}")


def hold_lock(root: Path, timeout: float) -> None:
    with (root / "kernel.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        persist(root / "owner_acquired.json", {
            "pid": os.getpid(), "hostname": socket.gethostname(),
            "probe_root": str(root), "acquired_at_unix": time.time(),
        })
        time.sleep(timeout)


def probe(root: Path, role: str, timeout: float) -> dict:
    root = root.resolve(strict=True)
    if root == Path("/") or timeout <= 0 or timeout > 600:
        raise ValueError("A specific existing probe root and timeout in (0, 600] are required")
    deadline = time.monotonic() + timeout
    if role == "owner":
        child = multiprocessing.Process(target=hold_lock, args=(root, timeout))
        child.start()
        try:
            acquired = wait_for(root / "owner_acquired.json", deadline)
            denied = wait_for(root / "contender_denied.json", deadline)
            if denied["owner_pid"] != acquired["pid"]:
                raise ValueError("Contender observed the wrong owner")
            if denied["hostname"] == acquired["hostname"]:
                raise ValueError("Lock qualification requires two different pods")
            if not child.is_alive():
                raise RuntimeError("Lock owner exited before the crash probe")
            os.kill(child.pid, signal.SIGKILL)
            child.join(timeout=max(0, deadline - time.monotonic()))
            if child.exitcode != -signal.SIGKILL:
                raise RuntimeError(f"Lock owner did not exit from SIGKILL: {child.exitcode}")
            result = {"status": "owner_killed_after_cross_pod_exclusion",
                      "owner_pid": acquired["pid"], "owner_exitcode": child.exitcode,
                      "hostname": socket.gethostname()}
            persist(root / "owner_crashed.json", result)
            return wait_for(root / "lock_receipt.json", deadline)
        finally:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
    acquired = wait_for(root / "owner_acquired.json", deadline)
    with (root / "kernel.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
            raise RuntimeError("Cross-pod exclusion failed: duplicate writer acquired live lock")
        persist(root / "contender_denied.json", {
            "owner_pid": acquired["pid"], "hostname": socket.gethostname(),
            "status": "live_owner_excludes_other_pod",
        })
        crashed = wait_for(root / "owner_crashed.json", deadline)
        if crashed["owner_pid"] != acquired["pid"] or crashed["owner_exitcode"] != -9:
            raise ValueError("Crash receipt does not bind the lock owner")
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("Kernel lock was not released after owner crash") from exc
                time.sleep(0.25)
        result = {
            "study_id": "SGW-01", "status": "passed",
            "scope": "cross_pod_flock_exclusion_and_sigkill_recovery_only",
            "probe_root": str(root), "owner_pod": acquired["hostname"],
            "contender_pod": socket.gethostname(), "owner_exitcode": -9,
            "persistent_write_fsync": True, "completed_at_unix": time.time(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_requests": 0, "behavioral_episodes": 0,
        }
        persist(root / "lock_receipt.json", result)
        fcntl.flock(stream, fcntl.LOCK_UN)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("owner", "contender"), required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    try:
        result = probe(args.root, args.role, args.timeout_seconds)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(json.dumps({"status": "failed", "role": args.role,
                          "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
