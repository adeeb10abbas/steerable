"""Finite CPU-only partition runner for frozen remaining HEIGHT/DIST slots.

This module deliberately contains no Kubernetes or GPU integration.  It only
coordinates four deterministic callers around the existing single-slot runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Any, Callable, Mapping, Sequence

from .family_campaign_executor import CALIBRATION_SHA256, run_slot
from .family_campaign_verifier import verify_design

FREEZE_SHA256 = "55b2dbbce2376b1c297180630238b09a83e8f226697f8da6a1814d748eebefd3"
SCHEMA = "sgw-01-family-partition-worker-v1"
MIN_FREE_SPACE_FLOOR_BYTES = 100 * 1024**3
SMOKE_SLOTS = {
    ("HEIGHT", 0), ("HEIGHT", 1), ("DIST", 0), ("DIST", 3),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fsync_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _locked(root: Path):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root / ".partition.lock").open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _freeze(path: Path) -> dict[str, Any]:
    if _sha256(path) != FREEZE_SHA256:
        raise ValueError("family freeze receipt hash differs from the frozen authority")
    value = _json(path)
    if value.get("release_permitted") is not False or value.get("model_requests") != 0 or value.get("behavioral_episodes") != 0:
        raise ValueError("family freeze is not a zero-model, non-release authority")
    remaining = value.get("remaining_capture_eligible_slots")
    if not isinstance(remaining, list) or len(remaining) != 117:
        raise ValueError("family freeze does not contain exactly 117 remaining capture slots")
    seen = set()
    for row in remaining:
        key = (row.get("family"), row.get("slot_index"))
        if (not isinstance(row, dict) or row.get("family") not in {"HEIGHT", "DIST"} or type(row.get("slot_index")) is not int
                or not isinstance(row.get("design_id"), str) or key in seen or key in SMOKE_SLOTS):
            raise ValueError("family freeze remaining slot registry is malformed")
        seen.add(key)
    smoke = {(row.get("family"), row.get("slot_index")) for row in value.get("native_smoke_slots", []) if isinstance(row, dict)}
    if smoke != SMOKE_SLOTS:
        raise ValueError("family freeze native smoke registry differs")
    return value


def partition_slots(freeze: Mapping[str, Any], *, rank: int, workers: int) -> list[dict[str, Any]]:
    if type(rank) is not int or type(workers) is not int or workers != 4 or not 0 <= rank < workers:
        raise ValueError("partition requires one of exactly four ranks 0..3")
    rows = freeze["remaining_capture_eligible_slots"]
    return [dict(row) for offset, row in enumerate(rows) if offset % workers == rank]


def _campaigns(config: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Path]:
    raw = config.get("campaigns")
    if not isinstance(raw, Mapping):
        raise ValueError("partition config lacks family campaign paths")
    frozen = {row["family"]: row["campaign"] for row in freeze["families"]}
    result: dict[str, Path] = {}
    for family in ("HEIGHT", "DIST"):
        path = Path(raw.get(family, ""))
        expected = frozen[family]
        if not path.is_file() or _sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{family} campaign differs from frozen authority")
        result[family] = path
    return result


def _smoke(
    config: Mapping[str, Any], freeze: Mapping[str, Any], campaigns: Mapping[str, Path], output_root: Path,
    verifier: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    entries = config.get("smoke_evidence")
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("all four independently verified smoke evidence roots are required")
    expected = {(row["family"], row["slot_index"]): row["design_id"] for row in freeze["native_smoke_slots"]}
    verified = []
    for row in entries:
        if not isinstance(row, Mapping):
            raise ValueError("smoke evidence entry is malformed")
        family, index, design = row.get("family"), row.get("slot_index"), row.get("design_id")
        verification, evidence_root = Path(row.get("verification", "")), Path(row.get("root", ""))
        expected_file_sha, expected_digest = row.get("verification_file_sha256"), row.get("verification_sha256")
        if (expected.get((family, index)) != design or not verification.is_file() or not evidence_root.is_dir()
                or not isinstance(expected_file_sha, str) or not isinstance(expected_digest, str)
                or _sha256(verification) != expected_file_sha):
            raise ValueError("smoke evidence identity differs from frozen registry")
        result = _json(verification)
        if (result.get("status") != "verified_evidence_not_fixture_release"
                or result.get("family") != family or result.get("design_id") != design
                or result.get("release_permitted") is not False
                or result.get("model_request_count") != 0 or result.get("behavioral_episode_count") != 0
                or result.get("verification_sha256") != expected_digest):
            raise ValueError("smoke evidence is not independently verified zero-model evidence")
        recheck = output_root / "smoke-rechecks" / f"{family.lower()}-{index:03d}.json"
        fresh = verifier(campaign_path=campaigns[family], design_id=design, root=evidence_root, output=recheck)
        if (fresh.get("verification_sha256") != expected_digest
                or fresh.get("physical_geometry_rejection") is not None
                and not isinstance(fresh.get("physical_geometry_rejection"), Mapping)):
            raise ValueError("smoke raw evidence differs from authoritative verification")
        verified.append({
            "family": family, "slot_index": index, "design_id": design,
            "verification": _binding(verification), "recheck": _binding(recheck),
        })
    if {(row["family"], row["slot_index"]) for row in verified} != set(expected):
        raise ValueError("smoke evidence is incomplete or duplicated")
    return verified


def _config_bindings(config: Mapping[str, Any], freeze_path: Path, campaigns: Mapping[str, Path]) -> dict[str, Any]:
    source = Path(config.get("source_path", ""))
    expected_source = config.get("source_sha256")
    calibration = Path(config.get("controller_calibration", ""))
    commands = {"capture": config.get("capture_command"), "qualification": config.get("qualification_command")}
    if (not source.is_file() or not isinstance(expected_source, str) or _sha256(source) != expected_source
            or not calibration.is_file() or _sha256(calibration) != CALIBRATION_SHA256
            or any(not isinstance(command, list) or not all(isinstance(item, str) and item for item in command)
                   for command in commands.values())):
        raise ValueError("partition source, calibration, or child command binding differs")
    return {
        "freeze": _binding(freeze_path), "source": _binding(source), "calibration": _binding(calibration),
        "campaigns": {family: _binding(path) for family, path in campaigns.items()},
        "commands_sha256": _digest(commands),
    }


def _publish_stop(*, shared_root: Path, rank: int, bindings_sha256: str | None,
                  completed: Sequence[Mapping[str, Any]], error: str) -> None:
    """Publish a single durable stop record without replacing another worker's."""
    sentinel = shared_root / "infrastructure-stop.json"
    with _locked(shared_root):
        if not sentinel.exists():
            _fsync_json(sentinel, {
                "schema_version": SCHEMA, "origin_rank": rank, "completed_slots": list(completed),
                "bindings_sha256": bindings_sha256, "error": error, "status": "infrastructure_stop",
            })


def run_partition(
    *, config_path: Path, rank: int, root: Path, slot_runner: Callable[..., dict[str, Any]] = run_slot,
    verifier: Callable[..., dict[str, Any]] = verify_design,
) -> dict[str, Any]:
    """Run a disjoint finite slice; an infrastructure fault durably stops peers."""
    if root.exists():
        raise FileExistsError("refusing to reuse a partition worker root")
    root.mkdir(mode=0o700, parents=True)
    sentinel = root.parent / "infrastructure-stop.json"
    bindings_sha256: str | None = None
    completed: list[dict[str, Any]] = []
    try:
        config = _json(config_path)
        freeze_path = Path(config.get("freeze_receipt", ""))
        freeze = _freeze(freeze_path)
        campaigns = _campaigns(config, freeze)
        bindings = _config_bindings(config, freeze_path, campaigns)
        bindings_sha256 = _digest(bindings)
        workers = config.get("workers", 4)
        slots = partition_slots(freeze, rank=rank, workers=workers)
        floor, per_slot = config.get("free_space_floor_bytes"), config.get("declared_slot_bytes")
        if type(floor) is not int or type(per_slot) is not int or floor < MIN_FREE_SPACE_FLOOR_BYTES or per_slot <= 0:
            raise ValueError("partition requires a non-lowerable 100 GiB free-space floor and positive declared slot bytes")
        smoke = _smoke(config, freeze, campaigns, root, verifier)
        binding_path = root.parent / "partition-binding.json"
        binding = {
            "schema_version": SCHEMA, "freeze_sha256": bindings["freeze"]["sha256"],
            "bindings_sha256": bindings_sha256, "config_sha256": _sha256(config_path), "workers": workers,
        }
        rank_claim = root.parent / "rank-claims" / f"rank-{rank}.json"
        with _locked(root.parent):
            if binding_path.exists():
                if _json(binding_path) != binding:
                    raise ValueError("shared partition identity differs")
            else:
                _fsync_json(binding_path, binding)
            _fsync_json(rank_claim, {**binding, "rank": rank, "root": str(root.resolve())})
        receipt = root / "worker-receipt.json"
        _fsync_json(receipt, {
            "schema_version": SCHEMA, "rank": rank, "workers": workers, "slot_count": len(slots),
            "slot_order": slots, "bindings": bindings, "smoke_verifications": smoke,
            "free_space_floor_bytes": floor, "declared_slot_bytes": per_slot, "status": "running",
        })
        for slot in slots:
            slot_root = root / "slots" / f"{slot['family'].lower()}-{slot['slot_index']:03d}"
            claim = root.parent / "slot-claims" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json"
            with _locked(root.parent):
                if sentinel.exists():
                    break
                free = shutil.disk_usage(root).free
                required = floor + per_slot * len(freeze["remaining_capture_eligible_slots"])
                if free < required:
                    raise RuntimeError(f"storage allowance blocked: {free} < {required}")
                _fsync_json(claim, {
                    "schema_version": SCHEMA, "rank": rank, "slot": slot, "bindings_sha256": bindings_sha256,
                    "free_bytes": free, "required_bytes": required, "reserved_slots": 117, "status": "claimed",
                })
            result = slot_runner(
                campaign_path=campaigns[slot["family"]], index=slot["slot_index"], root=slot_root,
                controller_calibration=Path(config["controller_calibration"]),
                capture_command=config["capture_command"], qualification_command=config["qualification_command"],
                child_timeout_seconds=config.get("child_timeout_seconds", 1800),
            )
            status = result.get("status")
            if status not in {
                "externally_verified_candidate_slot_not_fixture_or_behavioral_release",
                "physical_geometry_rejection_accounted_slot_no_refill",
            }:
                raise RuntimeError("slot runner returned a nonterminal or infrastructure outcome")
            _fsync_json(root / "completed" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json", {
                "schema_version": SCHEMA, "slot": slot, "result": result, "slot_root": str(slot_root.resolve()),
            })
            completed.append(slot)
    except Exception:
        _publish_stop(
            shared_root=root.parent, rank=rank, bindings_sha256=bindings_sha256,
            completed=completed, error=traceback.format_exc(),
        )
        raise
    result = {
        "schema_version": SCHEMA, "rank": rank, "workers": workers, "completed_slots": completed,
        "stopped_by_peer": sentinel.exists(), "status": "stopped_before_next_slot" if sentinel.exists() else "complete",
    }
    _fsync_json(root / "worker-completion.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_partition(config_path=args.config, rank=args.rank, root=args.root), sort_keys=True))


if __name__ == "__main__":
    main()
