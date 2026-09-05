"""Atomic durable group leases and write-once attempt finalization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Optional


class LeaseConflict(Exception):
    """Raised when an exclusive group lease is already held."""


class WriteOnceViolation(Exception):
    """Raised when a finalized attempt directory would be overwritten."""


class StaleLeaseTakeoverDenied(Exception):
    """Raised when stale lease recovery lacks a verified dead-owner receipt."""


@dataclass(frozen=True)
class DeadOwnerVerificationReceipt:
    """Evidence that the prior lease holder is dead; TTL alone is insufficient."""

    group_id: str
    prior_owner_lane: str
    prior_attempt_id: str
    verified_by: str
    verification_method: str
    verified_at_unix: float
    process_exit_observed: bool
    heartbeat_absent: bool
    evidence_sha256: str

    def validates(self, existing: dict[str, Any]) -> bool:
        return (
            existing.get("group_id") == self.group_id
            and existing.get("owner_lane") == self.prior_owner_lane
            and existing.get("attempt_id") == self.prior_attempt_id
            and self.process_exit_observed
            and self.evidence_sha256
        )


@dataclass(frozen=True)
class GroupLease:
    group_id: str
    owner_lane: str
    attempt_id: str
    acquired_at_monotonic: float
    lease_path: Path
    manifest_sha256: str


@dataclass
class GroupLeaseStore:
    root: Path
    liveness_probe: Optional[Callable[[dict[str, Any]], bool]] = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "leases").mkdir(exist_ok=True)
        (self.root / "attempts").mkdir(exist_ok=True)
        (self.root / "receipts").mkdir(exist_ok=True)
        (self.root / "takeover_receipts").mkdir(exist_ok=True)

    def _lease_path(self, group_id: str) -> Path:
        safe = group_id.replace(":", "_").replace("/", "_")
        return self.root / "leases" / f"{safe}.lease"

    def _write_takeover_receipt(self, receipt: DeadOwnerVerificationReceipt) -> Path:
        path = self.root / "takeover_receipts" / f"{receipt.group_id.replace(':', '_')}_{receipt.prior_attempt_id}.json"
        payload = {
            "group_id": receipt.group_id,
            "prior_owner_lane": receipt.prior_owner_lane,
            "prior_attempt_id": receipt.prior_attempt_id,
            "verified_by": receipt.verified_by,
            "verification_method": receipt.verification_method,
            "verified_at_unix": receipt.verified_at_unix,
            "process_exit_observed": receipt.process_exit_observed,
            "heartbeat_absent": receipt.heartbeat_absent,
            "evidence_sha256": receipt.evidence_sha256,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(tmp, path)
        return path

    def acquire(
        self,
        *,
        group_id: str,
        owner_lane: str,
        attempt_id: str,
        manifest_sha256: str,
        dead_owner_receipt: Optional[DeadOwnerVerificationReceipt] = None,
    ) -> GroupLease:
        path = self._lease_path(group_id)
        payload = {
            "group_id": group_id,
            "owner_lane": owner_lane,
            "attempt_id": attempt_id,
            "manifest_sha256": manifest_sha256,
            "acquired_at_monotonic": time.monotonic(),
            "acquired_at_unix": time.time(),
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        while True:
            try:
                fd = os.open(path, flags, 0o644)
            except FileExistsError:
                existing = json.loads(path.read_text())
                if dead_owner_receipt is not None and dead_owner_receipt.validates(existing):
                    self._write_takeover_receipt(dead_owner_receipt)
                    path.unlink(missing_ok=True)
                    continue
                if self.liveness_probe is not None and not self.liveness_probe(existing):
                    raise StaleLeaseTakeoverDenied(
                        "prior owner failed liveness probe but no dead-owner receipt was supplied"
                    )
                raise LeaseConflict(
                    f"group {group_id!r} is leased by {existing.get('owner_lane')!r}"
                ) from None
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            return GroupLease(
                group_id=group_id,
                owner_lane=owner_lane,
                attempt_id=attempt_id,
                acquired_at_monotonic=payload["acquired_at_monotonic"],
                lease_path=path,
                manifest_sha256=manifest_sha256,
            )

    def release(self, lease: GroupLease) -> None:
        if not lease.lease_path.exists():
            return
        existing = json.loads(lease.lease_path.read_text())
        if existing.get("owner_lane") != lease.owner_lane or existing.get("attempt_id") != lease.attempt_id:
            raise LeaseConflict("lease ownership mismatch on release")
        lease.lease_path.unlink(missing_ok=True)

    def verify(self, lease: GroupLease) -> bool:
        if not lease.lease_path.exists():
            return False
        existing = json.loads(lease.lease_path.read_text())
        return (
            existing.get("group_id") == lease.group_id
            and existing.get("owner_lane") == lease.owner_lane
            and existing.get("attempt_id") == lease.attempt_id
        )


@dataclass
class AttemptFinalizer:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def attempt_dir(self, episode_id: str, attempt_id: str) -> Path:
        safe_episode = episode_id.replace("/", "_")
        return self.root / safe_episode / attempt_id

    def begin_attempt(self, *, episode_id: str, attempt_id: str, metadata: dict[str, Any]) -> Path:
        attempt_path = self.attempt_dir(episode_id, attempt_id)
        if attempt_path.exists():
            raise WriteOnceViolation(f"attempt directory already exists: {attempt_path}")
        attempt_path.mkdir(parents=True, exist_ok=False)
        meta_path = attempt_path / "attempt_metadata.json"
        meta_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
        return attempt_path

    def write_incremental(self, attempt_path: Path, name: str, payload: dict[str, Any]) -> None:
        if not attempt_path.exists():
            raise FileNotFoundError(str(attempt_path))
        target = attempt_path / name
        tmp = attempt_path / f".{name}.tmp"
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        os.replace(tmp, target)

    def finalize(self, attempt_path: Path, receipt: dict[str, Any]) -> Path:
        complete_marker = attempt_path / "COMPLETE.json"
        if complete_marker.exists():
            raise WriteOnceViolation(f"attempt already finalized: {attempt_path}")
        tmp = attempt_path / ".COMPLETE.json.tmp"
        tmp.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        os.replace(tmp, complete_marker)
        return complete_marker

    def is_finalized(self, attempt_path: Path) -> bool:
        return (attempt_path / "COMPLETE.json").exists()

    @staticmethod
    def with_temp_store() -> tuple["AttemptFinalizer", tempfile.TemporaryDirectory[str]]:
        tmp = tempfile.TemporaryDirectory()
        return AttemptFinalizer(Path(tmp.name)), tmp
