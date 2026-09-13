#!/usr/bin/env python3
"""Fail-closed, fully-offline recovery for the cluster queue release lock.

This utility is deliberately independent of cluster_queue.py.  ``probe`` is a
read-only eligibility report.  ``recover`` requires pinned identities and an
explicit offline assertion, holds the recovery guard and every extant queue
controller lifetime lock, and rotates release.lock only after two bounded
LOCK_NB attempts both return EWOULDBLOCK.  Quarantined inodes are never removed.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
import tempfile
import time
import uuid

NAMESPACE = "wmf_ablation_001_20260912"
TERMINAL = {"succeeded", "failed", "timed_out", "interrupted"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
WORKER_LOCK = re.compile(r"worker-[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.lock\Z")


class RecoveryError(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def identity(path: Path, *, digest: bool = False):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RecoveryError(f"cannot open regular file: {path.name}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RecoveryError(f"not a regular file: {path.name}")
        answer = {"device": before.st_dev, "inode": before.st_ino, "bytes": before.st_size}
        if digest:
            h = hashlib.sha256()
            while block := os.read(fd, 1024 * 1024):
                h.update(block)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ):
                raise RecoveryError(f"file changed while read: {path.name}")
            answer["sha256"] = h.hexdigest()
        return answer
    finally:
        os.close(fd)


def control_identity(state: Path):
    path = state / "control.json"
    value = identity(path, digest=True)
    try:
        control = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError("control.json is unreadable or invalid") from exc
    if not isinstance(control, dict) or control.get("namespace") != NAMESPACE:
        raise RecoveryError("control.json namespace mismatch")
    value.update(
        control_commit=control.get("control_commit"),
        control_generation=control.get("control_generation"),
        active_job_ids=control.get("active_job_ids"),
    )
    return value


def audit_claims(state: Path):
    jobs = state / "jobs"
    if not jobs.is_dir() or jobs.is_symlink():
        raise RecoveryError("jobs directory is missing or unsafe")
    rows = []
    for job in sorted(jobs.iterdir(), key=lambda p: p.name):
        if job.is_symlink() or not job.is_dir():
            raise RecoveryError(f"unsafe jobs entry: {job.name}")
        claim = job / "claim"
        if not claim.exists() and not claim.is_symlink():
            continue
        if claim.is_symlink() or not claim.is_dir():
            raise RecoveryError(f"unsafe claim entry: {job.name}")
        result_path = job / "result.json"
        result_ident = identity(result_path, digest=True)
        try:
            result = json.loads(result_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryError(f"claim lacks valid result: {job.name}") from exc
        if result.get("status") not in TERMINAL or result.get("child_reaped") is not True:
            raise RecoveryError(f"claim is not terminal with child_reaped=true: {job.name}")
        rows.append({"job_id": job.name, "status": result["status"], "result_sha256": result_ident["sha256"]})
    return {"count": len(rows), "rows": rows, "sha256": hashlib.sha256(canonical(rows)).hexdigest()}


def lifetime_locks(state: Path):
    rows = []
    for path in sorted(state.iterdir(), key=lambda p: p.name):
        if path.name != "coordinator.lock" and not WORKER_LOCK.fullmatch(path.name):
            continue
        item = identity(path)
        rows.append({"name": path.name, "device": item["device"], "inode": item["inode"]})
    if not any(row["name"] == "coordinator.lock" for row in rows):
        raise RecoveryError("coordinator.lock is absent")
    if not any(row["name"].startswith("worker-") for row in rows):
        raise RecoveryError("no worker lifetime lock exists")
    return {"rows": rows, "sha256": hashlib.sha256(canonical(rows)).hexdigest()}


def snapshot(state: Path):
    return {
        "control": control_identity(state),
        "release_lock": identity(state / "release.lock"),
        "lifetime_locks": lifetime_locks(state),
        "claims": audit_claims(state),
    }


def _lock_helper(paths):
    descriptors = []
    acquired = []
    try:
        for raw in paths:
            path = Path(raw)
            fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
            descriptors.append(fd)
            opened = os.fstat(fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                print(json.dumps({"status": "would_block", "errno": exc.errno, "path": str(path),
                                  "device": opened.st_dev, "inode": opened.st_ino}), flush=True)
                return 3
            acquired.append(fd)
        print(json.dumps({"status": "acquired", "locks": [
            {"path": str(Path(raw)), "device": os.fstat(fd).st_dev, "inode": os.fstat(fd).st_ino}
            for raw, fd in zip(paths, descriptors)
        ]}), flush=True)
        sys.stdin.buffer.read(1)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__}), flush=True)
        return 4
    finally:
        # Never issue LOCK_UN for a descriptor whose acquisition failed.  In
        # particular, an NFS/NLM client may turn that nominal cleanup into a
        # second blocking RPC after the bounded LOCK_NB result was emitted.
        for fd in reversed(acquired):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        for fd in reversed(descriptors):
            os.close(fd)


class LockAttempt:
    def __init__(self, proc, response, timeout):
        self.proc, self.response, self.timeout = proc, response, timeout

    def close(self):
        if self.proc is None:
            return
        proc, self.proc = self.proc, None
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            try:
                proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise RecoveryError("lock helper could not be reaped") from exc
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def bounded_lock_attempt(paths, timeout):
    command = [sys.executable, str(Path(__file__).resolve()), "__lock_helper", *map(str, paths)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    events = selector.select(timeout)
    selector.close()
    if not events:
        proc.kill()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RecoveryError("bounded lock attempt timed out and could not be reaped") from exc
        raise RecoveryError("bounded lock attempt timed out")
    line = proc.stdout.readline()
    try:
        response = json.loads(line)
    except json.JSONDecodeError as exc:
        proc.kill(); proc.wait(timeout=timeout)
        raise RecoveryError("lock helper returned invalid output") from exc
    attempt = LockAttempt(proc, response, timeout)
    if response.get("status") == "would_block":
        attempt.close()
        if response.get("errno") not in {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK}:
            raise RecoveryError("lock attempt did not return EWOULDBLOCK")
    elif response.get("status") != "acquired":
        attempt.close()
        raise RecoveryError("lock helper failed")
    return attempt


def assert_pinned(current, expected_control_sha256, expected_release_device,
                  expected_release_inode, expected_lifetime_sha256):
    if not SHA256.fullmatch(expected_control_sha256 or ""):
        raise RecoveryError("expected control SHA-256 is invalid")
    if not SHA256.fullmatch(expected_lifetime_sha256 or ""):
        raise RecoveryError("expected lifetime-lock set SHA-256 is invalid")
    if expected_release_device < 0 or expected_release_inode <= 0:
        raise RecoveryError("expected release device/inode is invalid")
    checks = (
        (current["control"]["sha256"], expected_control_sha256, "control SHA-256"),
        (current["release_lock"]["device"], expected_release_device, "release device"),
        (current["release_lock"]["inode"], expected_release_inode, "release inode"),
        (current["lifetime_locks"]["sha256"], expected_lifetime_sha256, "lifetime-lock set SHA-256"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise RecoveryError(f"pinned {label} mismatch")


def ensure_guard(state: Path):
    path = state / "recovery.guard"
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError:
        identity(path)
        return path
    with os.fdopen(fd, "wb") as stream:
        stream.flush(); os.fsync(stream.fileno())
    fsync_dir(state)
    return path


def fsync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_receipt(path: Path, value, state: Path):
    path = path.resolve()
    if not path.is_relative_to(state.resolve()):
        raise RecoveryError("receipt must remain inside state directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RecoveryError("receipt path already exists")
    fd, temporary = tempfile.mkstemp(prefix=".release-recovery-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        fsync_dir(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def recover(state: Path, receipt: Path, *, expected_control_sha256: str,
            expected_release_device: int, expected_release_inode: int,
            expected_lifetime_sha256: str, timeout: float, controllers_offline: bool):
    if not controllers_offline:
        raise RecoveryError("--confirm-controllers-offline is required")
    initial = snapshot(state)
    assert_pinned(initial, expected_control_sha256, expected_release_device,
                  expected_release_inode, expected_lifetime_sha256)
    guard = ensure_guard(state)
    lock_paths = [guard] + [state / row["name"] for row in initial["lifetime_locks"]["rows"]]
    with bounded_lock_attempt(lock_paths, timeout) as fence:
        if fence.response["status"] != "acquired":
            raise RecoveryError("a recovery/lifetime lock is still held")
        fenced = snapshot(state)
        assert_pinned(fenced, expected_control_sha256, expected_release_device,
                      expected_release_inode, expected_lifetime_sha256)
        if fenced["claims"]["sha256"] != initial["claims"]["sha256"]:
            raise RecoveryError("claim set changed while fencing")
        first = bounded_lock_attempt([state / "release.lock"], timeout)
        if first.response["status"] == "acquired":
            release_guard, probes, action = first, [first.response], "same_inode_acquired_no_rotation"
            quarantine = None
        else:
            first.close()
            middle = snapshot(state)
            assert_pinned(middle, expected_control_sha256, expected_release_device,
                          expected_release_inode, expected_lifetime_sha256)
            if middle["claims"]["sha256"] != initial["claims"]["sha256"]:
                raise RecoveryError("claim set changed before second probe")
            second = bounded_lock_attempt([state / "release.lock"], timeout)
            probes = [first.response, second.response]
            if second.response["status"] == "acquired":
                release_guard, action, quarantine = second, "same_inode_acquired_no_rotation", None
            else:
                second.close()
                final_check = snapshot(state)
                assert_pinned(final_check, expected_control_sha256, expected_release_device,
                              expected_release_inode, expected_lifetime_sha256)
                if final_check["claims"]["sha256"] != initial["claims"]["sha256"]:
                    raise RecoveryError("claim set changed before rotation")
                release = state / "release.lock"
                quarantine = state / (
                    f"release.lock.quarantine.dev{expected_release_device}.ino{expected_release_inode}."
                    f"{time.time_ns()}.{os.getpid()}.{uuid.uuid4().hex}"
                )
                os.rename(release, quarantine)
                moved = identity(quarantine)
                if (moved["device"], moved["inode"]) != (expected_release_device, expected_release_inode):
                    raise RecoveryError("renamed quarantine identity mismatch")
                fd = os.open(release, os.O_RDWR | os.O_CREAT | os.O_EXCL |
                             getattr(os, "O_NOFOLLOW", 0), 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.flush(); os.fsync(stream.fileno())
                fsync_dir(state)
                release_guard = bounded_lock_attempt([release], timeout)
                if release_guard.response["status"] != "acquired":
                    release_guard.close()
                    raise RecoveryError("replacement release lock is not independently acquirable")
                action = "rotated_stale_inode"
        with release_guard:
            after = snapshot(state)
            if after["control"]["sha256"] != expected_control_sha256:
                raise RecoveryError("control changed before receipt")
            if after["claims"]["sha256"] != initial["claims"]["sha256"]:
                raise RecoveryError("claim set changed before receipt")
            receipt_value = {
                "schema_version": "wmf-release-lock-recovery-v1",
                "namespace": NAMESPACE,
                "recorded_at_unix": time.time(),
                "mode": "recover",
                "action": action,
                "controllers_offline_asserted": True,
                "initial": initial,
                "release_lock_probes": probes,
                "replacement_release_lock": after["release_lock"],
                "quarantine": None if quarantine is None else {
                    "path": str(quarantine), **identity(quarantine)
                },
                "fence_locks": fence.response["locks"],
            }
            atomic_receipt(receipt, receipt_value, state)
            return receipt_value


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state-dir", type=Path, required=True)
    common.add_argument("--probe-timeout-seconds", type=float, default=3.0)
    root = argparse.ArgumentParser(description=__doc__)
    modes = root.add_subparsers(dest="mode", required=True)
    modes.add_parser("probe", parents=[common])
    run = modes.add_parser("recover", parents=[common])
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--expected-control-sha256", required=True)
    run.add_argument("--expected-release-device", type=int, required=True)
    run.add_argument("--expected-release-inode", type=int, required=True)
    run.add_argument("--expected-lifetime-lock-set-sha256", required=True)
    run.add_argument("--confirm-controllers-offline", action="store_true")
    return root


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "__lock_helper":
        return _lock_helper(argv[1:])
    args = parser().parse_args(argv)
    if not 0.05 <= args.probe_timeout_seconds <= 30:
        raise RecoveryError("probe timeout must be between 0.05 and 30 seconds")
    state = args.state_dir.resolve()
    if not state.is_dir() or state.is_symlink():
        raise RecoveryError("state directory is missing or unsafe")
    if args.mode == "probe":
        report = snapshot(state)
        attempt = bounded_lock_attempt([state / "release.lock"], args.probe_timeout_seconds)
        report["release_lock_probe"] = attempt.response
        attempt.close()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    report = recover(
        state, args.receipt,
        expected_control_sha256=args.expected_control_sha256,
        expected_release_device=args.expected_release_device,
        expected_release_inode=args.expected_release_inode,
        expected_lifetime_sha256=args.expected_lifetime_lock_set_sha256,
        timeout=args.probe_timeout_seconds,
        controllers_offline=args.confirm_controllers_offline,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecoveryError as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
