"""Durable append-only SGW-01 attempt recording and no-overwrite publication."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .contract import Cell, ContractError, Release, canonical_bytes, sha256_file, verify_completion_pointer


VALID_OUTCOMES = {"valid_success", "valid_model_failure", "technical_invalid", "censored"}
REQUIRED_VALID_ARTIFACTS = {"actions", "states", "observations", "videos"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


class AttemptRecorder:
    def __init__(self, release: Release, cell: Cell, attempt_id: str):
        self.release = release
        self.cell = cell
        self.attempt_id = attempt_id
        self.root = release.root.parent
        self.path = self.root / "attempts" / cell.cell_id / attempt_id

    def begin(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=False)
        intent = {
            "schema_version": "sgw-01-attempt-intent-v1", "release_id": self.release.release_id,
            "cell_id": self.cell.cell_id, "attempt_id": self.attempt_id, "created_at_utc": utc_now(),
            "cell": dict(self.cell.row),
        }
        atomic_json(self.path / "intent.json", intent)
        self.event("attempt_started")
        return self.path

    def event(self, kind: str, **details: Any) -> None:
        if not kind:
            raise ValueError("event kind is required")
        entry = {"timestamp_utc": utc_now(), "event": kind, **details}
        path = self.path / "events.jsonl"
        with path.open("ab", buffering=0) as stream:
            stream.write(canonical_bytes(entry))
            os.fsync(stream.fileno())

    def request(self, request: Mapping[str, Any]) -> None:
        required = {"request_id", "request_index", "started_at_utc", "prompt_sha256", "reset_sha256"}
        if not required.issubset(request) or request["prompt_sha256"] != self.cell.row["prompt_sha256"]:
            raise ContractError("request provenance is incomplete or prompt changed")
        path = self.path / "request_index.jsonl"
        with path.open("ab", buffering=0) as stream:
            stream.write(canonical_bytes(dict(request)))
            os.fsync(stream.fileno())

    def record_reset(self, reset: Mapping[str, Any]) -> None:
        required = {"full_reset", "reset_id", "camera_name", "camera_fingerprint", "reset_sha256"}
        if not required.issubset(reset) or reset.get("full_reset") is not True:
            raise ContractError("reset record lacks full physical/cache-reset identity")
        atomic_json(self.path / "states" / "reset.json", dict(reset))
        self.event("reset_attested", reset_id=reset["reset_id"], camera_name=reset["camera_name"])

    def prediction(self, prediction: Any) -> None:
        """Persist the adapter's raw request/response envelope without decoding it."""
        request = getattr(prediction, "raw_request", None)
        response = getattr(prediction, "raw_response", None)
        if not isinstance(request, Mapping) or not isinstance(response, Mapping):
            raise ContractError("prediction lacks raw request/response provenance")
        index = getattr(prediction, "request_index", None)
        if type(index) is not int or index < 0:
            raise ContractError("prediction lacks a valid request index")
        record = {
            "request_id": getattr(prediction, "request_id", None),
            "request_index": index,
            "action_step_start": getattr(prediction, "action_step_start", None),
            "target_action_step": getattr(prediction, "target_action_step", None),
            "reset_id": getattr(prediction, "reset_id", None),
            "camera_name": getattr(prediction, "camera_name", None),
            "future_status": getattr(prediction, "future_status", None),
            "raw_request": dict(request),
            "raw_response": dict(response),
        }
        path = self.path / "predictions" / f"request-{index:04d}.json"
        atomic_json(path, record)

    def artifact_manifest(self, result: Mapping[str, Any]) -> dict[str, Any]:
        records: dict[str, dict[str, Any]] = {}
        for path in sorted(self.path.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json"}:
                records[path.relative_to(self.path).as_posix()] = {
                    "bytes": path.stat().st_size, "sha256": sha256_file(path),
                }
        if not records:
            raise ContractError("attempt has no durable artifacts")
        if result["status"] != "technical_invalid":
            roots = {relative.split("/", 1)[0] for relative in records}
            missing = REQUIRED_VALID_ARTIFACTS - roots
            if missing:
                raise ContractError(f"valid/censored attempt lacks required raw artifacts: {', '.join(sorted(missing))}")
            actions = result.get("executed_action_count")
            safety = result.get("safety_terminated")
            if type(actions) is not int or actions < 1 or actions > 450:
                raise ContractError("valid/censored attempt needs finite executed_action_count within 1..450")
            if actions != 450 and safety is not True:
                raise ContractError("short valid attempt requires explicit safety truncation")
        manifest = {
            "schema_version": "sgw-01-attempt-manifest-v1",
            "complete": True,
            "release_id": self.release.release_id,
            "release_hashes": dict(self.release.hashes),
            "cell_id": self.cell.cell_id,
            "attempt_id": self.attempt_id,
            "artifacts": records,
            "result": dict(result),
        }
        atomic_json(self.path / "manifest.json", manifest)
        return manifest

    def complete(self, outcome: Mapping[str, Any]) -> bool:
        status = outcome.get("status")
        if status not in VALID_OUTCOMES:
            raise ContractError("attempt outcome must classify success, model failure, technical invalidity, or censoring")
        if status == "technical_invalid" and not isinstance(outcome.get("technical_cause"), str):
            raise ContractError("technical invalidity needs a durable technical cause")
        result = {"schema_version": "sgw-01-result-v1", "release_id": self.release.release_id,
                  "cell_id": self.cell.cell_id, "attempt_id": self.attempt_id, "completed_at_utc": utc_now(),
                  **dict(outcome)}
        atomic_json(self.path / "result.json", result)
        self.event("attempt_finished", status=status)
        manifest = self.artifact_manifest(result)
        if status == "technical_invalid":
            return False
        pointer = self.root / "cells" / f"{self.cell.cell_id}.complete.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        value = {"release_id": self.release.release_id, "cell_id": self.cell.cell_id,
                 "attempt_id": self.attempt_id, "result": {"path": str(self.path / "result.json"),
                 "sha256": sha256_file(self.path / "result.json")}, "manifest_path": str(self.path / "manifest.json"),
                 "manifest_sha256": sha256_file(self.path / "manifest.json")}
        with tempfile.NamedTemporaryFile("wb", dir=pointer.parent, prefix=".pointer.", delete=False) as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        try:
            os.link(temporary, pointer)
            _fsync_directory(pointer.parent)
            return True
        except FileExistsError:
            verify_completion_pointer(self.release, pointer)
            return False
        finally:
            temporary.unlink(missing_ok=True)


def next_attempt_number(release: Release, cell: Cell) -> int:
    attempts = release.root.parent / "attempts" / cell.cell_id
    if not attempts.exists():
        return 1
    return sum(1 for child in attempts.iterdir() if child.is_dir()) + 1
