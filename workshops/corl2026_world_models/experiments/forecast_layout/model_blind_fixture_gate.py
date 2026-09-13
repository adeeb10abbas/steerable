#!/usr/bin/env python3
"""Run the fail-closed, model-blind live gate for forecast-layout candidates.

Simulator integrations implement :class:`FixtureGateAdapter` and return one
capture per arm/command/reset.  This module owns the scientific checks, the
append-only hash-chained decision ledger, deterministic spare advancement, and
the construction of a qualified-but-still-unreleased frozen pose manifest.
It imports no policy client and authorizes no model request.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from fixture_layouts import (
    COMMANDS,
    LAYOUT_ARMS,
    MOVABLE_OBJECTS,
    NAMESPACE,
    PLANNED_LAYOUT_IDS,
    POSE_MANIFEST_SCHEMA,
    POSE_MANIFEST_STATUS,
    LayoutContractError,
    candidate_index,
    canonical_json_bytes,
    load_json_file,
    require,
    sha256_bytes,
    sha256_file,
    validate_candidate,
    validate_candidate_pool,
    validate_frozen_pose_manifest,
    validate_source_contract,
    write_immutable,
)


CAPTURE_SCHEMA = "wmf-forecast-layout-live-fixture-capture-v1"
RECORD_SCHEMA = "wmf-forecast-layout-gate-ledger-record-v1"
DECISIONS = {"accepted", "physical_rejection", "technical_invalid"}
_SHA_LENGTH = 64


class GateEvidenceError(LayoutContractError):
    """Raised when a live adapter did not supply auditable gate evidence."""


class FixtureGateAdapter(Protocol):
    """Simulator boundary used before any policy client is constructed.

    Implementations must create a fresh environment for every call, set the
    candidate pose before construction, execute only reset/hold actions, and
    retain raw artifacts beneath ``attempt_dir``.  The returned mapping follows
    ``CAPTURE_SCHEMA``.  In particular, observation hashes exclude prompt bytes
    so LEFT/RIGHT non-language resets can be compared exactly.
    """

    def capture_condition(
        self,
        *,
        candidate: Mapping[str, Any],
        layout_arm: str,
        command: str,
        repeat_index: int,
        environment_seed: int,
        gate_contract: Mapping[str, Any],
        attempt_dir: Path,
    ) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == _SHA_LENGTH
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _finite(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _vector(value: Any, size: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == size, f"{label} must be a {size}-vector")
    return [_finite(item, label) for item in value]


def canonical_value_sha256(value: Any) -> str:
    """Hash a model-blind state/observation payload for adapter implementations."""

    return sha256_bytes(canonical_json_bytes(value))


def environment_seed(candidate: Mapping[str, Any], repeat_index: int) -> int:
    require(type(repeat_index) is int and repeat_index >= 0, "repeat index is invalid")
    digest = hashlib.sha256(
        f"{NAMESPACE}:fixture-gate-v1:{candidate['candidate_payload_sha256']}:repeat-{repeat_index}".encode("ascii")
    ).digest()
    return 1 + int.from_bytes(digest, "big") % 2_147_483_646


def expected_capture_keys(repeat_count: int) -> set[tuple[str, str, int]]:
    return {
        (arm, command, repeat)
        for arm in LAYOUT_ARMS
        for command in COMMANDS
        for repeat in range(repeat_count)
    }


def _validate_capture_identity(
    capture: Any,
    candidate: Mapping[str, Any],
    arm: str,
    command: str,
    repeat_index: int,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(capture, dict):
        raise GateEvidenceError("live adapter capture must be an object")
    checks = {
        "schema": capture.get("schema_version") == CAPTURE_SCHEMA,
        "namespace": capture.get("study_namespace") == NAMESPACE,
        "candidate_id": capture.get("candidate_id") == candidate["candidate_id"],
        "candidate_payload_sha256": capture.get("candidate_payload_sha256") == candidate["candidate_payload_sha256"],
        "layout_pair_id": capture.get("layout_pair_id") == candidate["layout_pair_id"],
        "layout_arm": capture.get("layout_arm") == arm,
        "command": capture.get("command") == command,
        "repeat_index": capture.get("repeat_index") == repeat_index,
        "environment_seed": capture.get("environment_seed") == seed,
        "zero_model_requests": capture.get("model_request_count") == 0,
        "zero_behavioral_actions": capture.get("behavioral_action_count") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise GateEvidenceError(f"capture identity/count evidence failed: {failed}")
    return capture


def _validate_one_capture(
    capture: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    arm = str(capture["layout_arm"])
    gate = source["live_gate"]
    failures: list[str] = []
    checks: dict[str, bool] = {}

    configured = capture.get("configured_poses")
    if not isinstance(configured, dict) or set(configured) != set(MOVABLE_OBJECTS):
        raise GateEvidenceError("configured pose inventory is incomplete")
    expected_positions = candidate["layouts"][arm]["positions_robot_base_m"]
    expected_quaternions = candidate["layouts"][arm]["quaternions_wxyz"]
    for name in MOVABLE_OBJECTS:
        row = configured[name]
        if not isinstance(row, dict):
            raise GateEvidenceError(f"configured pose row is invalid: {name}")
        if _vector(row.get("position_robot_base_m"), 3, f"configured {name} position") != expected_positions[name]:
            raise GateEvidenceError(f"configured {name} position differs from candidate")
        if _vector(row.get("quaternion_wxyz"), 4, f"configured {name} quaternion") != expected_quaternions[name]:
            raise GateEvidenceError(f"configured {name} quaternion differs from candidate source")

    settled = capture.get("settled_poses")
    if not isinstance(settled, dict) or set(settled) != set(MOVABLE_OBJECTS):
        raise GateEvidenceError("settled pose inventory is incomplete")
    tolerance = _finite(gate["pose_tolerance_m"], "pose tolerance")
    for name in MOVABLE_OBJECTS:
        row = settled[name]
        if not isinstance(row, dict):
            raise GateEvidenceError(f"settled pose row is invalid: {name}")
        observed = _vector(row.get("position_robot_base_m"), 3, f"settled {name} position")
        _vector(row.get("quaternion_wxyz"), 4, f"settled {name} quaternion")
        passed = max(abs(left - right) for left, right in zip(observed, expected_positions[name])) <= tolerance
        checks[f"{name}_settled_within_pose_tolerance"] = passed
        if not passed:
            failures.append(f"{name} settled pose differs from candidate by more than {tolerance} m")

    settle = capture.get("settle")
    if not isinstance(settle, dict):
        raise GateEvidenceError("settle evidence is missing")
    for key in ("settle_steps", "stability_window_steps"):
        if settle.get(key) != gate[key]:
            raise GateEvidenceError(f"settle contract changed: {key}")
    if type(settle.get("terminated_during_settle")) is not bool or type(settle.get("truncated_during_settle")) is not bool:
        raise GateEvidenceError("settle termination flags are missing")
    checks["no_termination_during_settle"] = not settle["terminated_during_settle"]
    checks["no_truncation_during_settle"] = not settle["truncated_during_settle"]
    if not checks["no_termination_during_settle"]:
        failures.append("environment terminated during model-blind settling")
    if not checks["no_truncation_during_settle"]:
        failures.append("environment truncated during model-blind settling")
    maxima = settle.get("maxima_by_object")
    if not isinstance(maxima, dict) or set(maxima) != set(MOVABLE_OBJECTS):
        raise GateEvidenceError("stability maxima inventory is incomplete")
    for name in MOVABLE_OBJECTS:
        row = maxima[name]
        if not isinstance(row, dict):
            raise GateEvidenceError(f"stability row is invalid: {name}")
        linear = _finite(row.get("max_linear_speed_m_s"), f"{name} linear speed")
        angular = _finite(row.get("max_angular_speed_rad_s"), f"{name} angular speed")
        linear_ok = 0.0 <= linear <= float(gate["linear_speed_tolerance_m_s"])
        angular_ok = 0.0 <= angular <= float(gate["angular_speed_tolerance_rad_s"])
        checks[f"{name}_linear_stability"] = linear_ok
        checks[f"{name}_angular_stability"] = angular_ok
        if not linear_ok:
            failures.append(f"{name} linear speed exceeded frozen tolerance")
        if not angular_ok:
            failures.append(f"{name} angular speed exceeded frozen tolerance")

    reset = capture.get("reset_fingerprints")
    if not isinstance(reset, dict):
        raise GateEvidenceError("reset fingerprints are missing")
    _sha(reset.get("reset_state_sha256"), "reset state fingerprint")
    _sha(reset.get("initial_observation_sha256"), "initial observation fingerprint")
    camera_hashes = reset.get("initial_camera_rgb_sha256")
    required_cameras = list(gate["required_cameras"])
    if not isinstance(camera_hashes, dict) or set(camera_hashes) != set(required_cameras):
        raise GateEvidenceError("initial camera fingerprint inventory changed")
    for name in required_cameras:
        _sha(camera_hashes[name], f"initial RGB fingerprint for {name}")

    cameras = capture.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != set(required_cameras):
        raise GateEvidenceError("live camera inventory differs from the frozen contract")
    for camera_name in required_cameras:
        row = cameras[camera_name]
        if not isinstance(row, dict):
            raise GateEvidenceError(f"camera row is invalid: {camera_name}")
        shape = row.get("shape_hwc")
        if not (
            isinstance(shape, list)
            and len(shape) == 3
            and all(type(item) is int and item > 0 for item in shape)
            and shape[2] == 3
        ):
            raise GateEvidenceError(f"camera shape is invalid: {camera_name}")
        if row.get("dtype") != "uint8":
            raise GateEvidenceError(f"camera dtype is not uint8: {camera_name}")
        _sha(row.get("rgb_sha256"), f"RGB digest for {camera_name}")
        _sha(row.get("visibility_source_sha256"), f"visibility digest for {camera_name}")
        pixel_range = row.get("pixel_range")
        if type(pixel_range) is not int or not 0 <= pixel_range <= 255:
            raise GateEvidenceError(f"camera pixel range is invalid: {camera_name}")
        nonblank = pixel_range >= int(gate["minimum_rgb_pixel_range"])
        checks[f"{camera_name}_rgb_nonblank"] = nonblank
        if not nonblank:
            failures.append(f"{camera_name} RGB is blank or too low-range")
        method = row.get("visibility_method")
        if method not in gate["visibility_methods"]:
            raise GateEvidenceError(f"visibility method is not frozen: {camera_name}")
        if method == "instance_segmentation":
            visible = row.get("visible_object_pixels")
            if not isinstance(visible, dict) or set(visible) != set(MOVABLE_OBJECTS):
                raise GateEvidenceError(f"visibility inventory is incomplete: {camera_name}")
            for name in MOVABLE_OBJECTS:
                if type(visible[name]) is not int or visible[name] < 0:
                    raise GateEvidenceError(f"visible pixel count is invalid: {camera_name}/{name}")
                if camera_name in gate["visibility_cameras"]:
                    passed = visible[name] >= int(gate["minimum_visible_pixels_by_object"][name])
                    checks[f"{camera_name}_{name}_visible"] = passed
                    if not passed:
                        failures.append(f"{name} is not visibly resolved in {camera_name}")
        else:
            visible = row.get("projected_unoccluded_by_object")
            projections = row.get("projected_object_centers_uv")
            if not isinstance(visible, dict) or set(visible) != set(MOVABLE_OBJECTS):
                raise GateEvidenceError(f"projected visibility inventory is incomplete: {camera_name}")
            if not isinstance(projections, dict) or set(projections) != set(MOVABLE_OBJECTS):
                raise GateEvidenceError(f"projection inventory is incomplete: {camera_name}")
            _sha(row.get("camera_geometry_source_sha256"), f"camera geometry digest for {camera_name}")
            for name in MOVABLE_OBJECTS:
                if type(visible[name]) is not bool:
                    raise GateEvidenceError(f"projected visibility flag is invalid: {camera_name}/{name}")
                _vector(projections[name], 2, f"projected pixel for {camera_name}/{name}")
                if camera_name in gate["visibility_cameras"]:
                    checks[f"{camera_name}_{name}_visible"] = visible[name]
                    if not visible[name]:
                        failures.append(f"{name} is outside calibrated view or OBB-occluded in {camera_name}")

    collisions = capture.get("collision_checks")
    if not isinstance(collisions, dict) or collisions.get("query_complete") is not True:
        raise GateEvidenceError("collision query is missing or incomplete")
    pairs = collisions.get("forbidden_pairs")
    expected_pairs = list(gate["forbidden_collision_pairs"])
    if not isinstance(pairs, dict) or set(pairs) != set(expected_pairs):
        raise GateEvidenceError("forbidden collision-pair inventory changed")
    for pair in expected_pairs:
        row = pairs[pair]
        if not isinstance(row, dict) or type(row.get("clear")) is not bool:
            raise GateEvidenceError(f"collision row is incomplete: {pair}")
        _sha(row.get("evidence_sha256"), f"collision evidence for {pair}")
        checks[f"collision_clear_{pair}"] = row["clear"]
        if not row["clear"]:
            failures.append(f"forbidden reset collision: {pair}")

    predicates = capture.get("success_predicates")
    if not isinstance(predicates, dict) or set(predicates) != {"left", "right"}:
        raise GateEvidenceError("both frozen success predicates must be measured")
    if any(type(value) is not bool for value in predicates.values()):
        raise GateEvidenceError("success predicates must be booleans")
    checks["left_success_false"] = predicates["left"] is False
    checks["right_success_false"] = predicates["right"] is False
    if predicates["left"]:
        failures.append("LEFT success predicate is already true after settling")
    if predicates["right"]:
        failures.append("RIGHT success predicate is already true after settling")
    return checks, failures


def evaluate_candidate_captures(
    candidate: Mapping[str, Any],
    captures: Sequence[Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_candidate(candidate, source_sha256=candidate["source_contract_sha256"])
    source = validate_source_contract(source)
    gate = source["live_gate"]
    repeat_count = int(gate["repeat_resets_per_condition"])
    wanted = expected_capture_keys(repeat_count)
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    all_checks: dict[str, dict[str, bool]] = {}
    failures: list[str] = []
    for raw in captures:
        if not isinstance(raw, dict):
            raise GateEvidenceError("capture list contains a non-object")
        key = (raw.get("layout_arm"), raw.get("command"), raw.get("repeat_index"))
        if key not in wanted or key in indexed:
            raise GateEvidenceError(f"unexpected or duplicate capture key: {key}")
        arm, command, repeat_index = key
        seed = environment_seed(candidate, int(repeat_index))
        capture = _validate_capture_identity(raw, candidate, str(arm), str(command), int(repeat_index), seed)
        checks, row_failures = _validate_one_capture(capture, candidate, source)
        label = f"{arm}-{command}-repeat{repeat_index}"
        indexed[(str(arm), str(command), int(repeat_index))] = capture
        all_checks[label] = checks
        failures.extend(f"{label}: {failure}" for failure in row_failures)
    if set(indexed) != wanted:
        missing = sorted(wanted - set(indexed))
        raise GateEvidenceError(f"live gate capture matrix is incomplete: {missing}")

    matched_reset_checks: dict[str, bool] = {}
    for arm in LAYOUT_ARMS:
        for repeat in range(repeat_count):
            left = indexed[(arm, "left", repeat)]["reset_fingerprints"]
            right = indexed[(arm, "right", repeat)]["reset_fingerprints"]
            prefix = f"{arm}-repeat{repeat}"
            state_equal = left["reset_state_sha256"] == right["reset_state_sha256"]
            observation_equal = left["initial_observation_sha256"] == right["initial_observation_sha256"]
            cameras_equal = left["initial_camera_rgb_sha256"] == right["initial_camera_rgb_sha256"]
            matched_reset_checks[f"{prefix}_reset_state_hash_equal"] = state_equal
            matched_reset_checks[f"{prefix}_initial_observation_hash_equal"] = observation_equal
            matched_reset_checks[f"{prefix}_initial_camera_hashes_equal"] = cameras_equal
            if not state_equal:
                failures.append(f"{prefix}: LEFT/RIGHT reset-state hashes differ")
            if not observation_equal:
                failures.append(f"{prefix}: LEFT/RIGHT initial-observation hashes differ")
            if not cameras_equal:
                failures.append(f"{prefix}: LEFT/RIGHT initial-camera hashes differ")

    return {
        "passed": not failures,
        "decision": "accepted" if not failures else "physical_rejection",
        "capture_count": len(indexed),
        "expected_capture_count": len(wanted),
        "condition_checks": all_checks,
        "matched_left_right_checks": matched_reset_checks,
        "failures": failures,
        "capture_evidence": [
            indexed[key]
            for key in sorted(indexed, key=lambda item: (LAYOUT_ARMS.index(item[0]), COMMANDS.index(item[1]), item[2]))
        ],
        "claim_boundary": "A pass establishes only model-blind reset/settle/camera/visibility/collision qualification for this numeric candidate. It is not a model request or inference release.",
    }


def read_ledger(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    previous: str | None = None
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise GateEvidenceError(f"cannot read gate ledger: {error}") from error
    for number, line in enumerate(lines, 1):
        require(bool(line.strip()), f"blank gate-ledger line: {number}")
        value = json.loads(line)
        require(isinstance(value, dict), f"gate-ledger line {number} is not an object")
        require(value.get("schema_version") == RECORD_SCHEMA, f"gate-ledger schema changed at line {number}")
        require(value.get("sequence") == number - 1, f"gate-ledger sequence changed at line {number}")
        require(value.get("previous_record_sha256") == previous, f"gate-ledger chain broke at line {number}")
        observed = _sha(value.get("record_sha256"), f"gate-ledger record {number}")
        core = dict(value)
        core.pop("record_sha256")
        require(observed == sha256_bytes(canonical_json_bytes(core)), f"gate-ledger digest mismatch at line {number}")
        require(value.get("decision") in DECISIONS, f"unknown gate decision at line {number}")
        previous = observed
        records.append(value)
    return records


def append_ledger_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o664)
    try:
        with os.fdopen(descriptor, "r+b", closefd=True) as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            payload = stream.read()
            temporary = path.with_name(path.name + ".validation")
            # Validate the exact locked bytes without exposing an unlocked read.
            records: list[dict[str, Any]] = []
            previous: str | None = None
            for number, line in enumerate(payload.splitlines(), 1):
                require(bool(line.strip()), f"blank gate-ledger line: {number}")
                value = json.loads(line)
                require(value.get("sequence") == number - 1, f"gate-ledger sequence changed at line {number}")
                require(value.get("previous_record_sha256") == previous, f"gate-ledger chain broke at line {number}")
                observed = _sha(value.get("record_sha256"), f"gate-ledger record {number}")
                core = dict(value)
                core.pop("record_sha256")
                require(observed == sha256_bytes(canonical_json_bytes(core)), f"gate-ledger digest mismatch at line {number}")
                previous = observed
                records.append(value)
            del temporary
            core = dict(record)
            for forbidden in ("sequence", "previous_record_sha256", "record_sha256", "recorded_at_utc"):
                require(forbidden not in core, f"caller cannot set ledger chain field: {forbidden}")
            core.update({
                "schema_version": RECORD_SCHEMA,
                "study_namespace": NAMESPACE,
                "sequence": len(records),
                "previous_record_sha256": previous,
                "recorded_at_utc": recorded_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            })
            require(core.get("decision") in DECISIONS, "unknown live-gate decision")
            core["record_sha256"] = sha256_bytes(canonical_json_bytes(core))
            line = json.dumps(core, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
            stream.seek(0, os.SEEK_END)
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
            return core
    except Exception:
        # fdopen owns and closes descriptor on every entered path.
        raise


def _attempt_number(records: Sequence[Mapping[str, Any]], candidate_id: str) -> int:
    return sum(record.get("candidate_id") == candidate_id for record in records)


def qualify_candidate(
    *,
    adapter: FixtureGateAdapter,
    candidate: Mapping[str, Any],
    source: Mapping[str, Any],
    candidate_pool_sha256: str,
    ledger_path: Path,
    attempt_root: Path,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    records = read_ledger(ledger_path)
    attempt_number = _attempt_number(records, candidate["candidate_id"])
    attempt_dir = Path(attempt_root) / candidate["layout_pair_id"] / candidate["candidate_id"] / f"attempt_{attempt_number:03d}"
    require(not attempt_dir.exists(), f"refusing to overwrite live gate attempt: {attempt_dir}")
    attempt_dir.mkdir(parents=True)
    captures: list[Mapping[str, Any]] = []
    try:
        repeats = int(source["live_gate"]["repeat_resets_per_condition"])
        for arm in LAYOUT_ARMS:
            for command in COMMANDS:
                for repeat in range(repeats):
                    seed = environment_seed(candidate, repeat)
                    capture = adapter.capture_condition(
                        candidate=candidate,
                        layout_arm=arm,
                        command=command,
                        repeat_index=repeat,
                        environment_seed=seed,
                        gate_contract=source["live_gate"],
                        attempt_dir=attempt_dir / f"{arm}_{command}_repeat{repeat:02d}",
                    )
                    captures.append(capture)
        evaluation = evaluate_candidate_captures(candidate, captures, source)
        decision = evaluation["decision"]
        passed = evaluation["passed"]
        error = None
    except Exception as caught:
        decision = "technical_invalid"
        passed = False
        error = {"type": type(caught).__name__, "message": str(caught)}
        evaluation = {
            "passed": False,
            "decision": decision,
            "capture_count_before_failure": len(captures),
            "capture_evidence": list(captures),
            "failures": [f"technical evidence failure: {type(caught).__name__}: {caught}"],
        }
    attempt_receipt = {
        "schema_version": "wmf-forecast-layout-gate-attempt-receipt-v1",
        "candidate_id": candidate["candidate_id"],
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "candidate_pool_sha256": candidate_pool_sha256,
        "attempt_number": attempt_number,
        "decision": decision,
        "passed": passed,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "evaluation": evaluation,
        "error": error,
    }
    receipt_path = attempt_dir / "gate_attempt_receipt.json"
    receipt_payload = canonical_json_bytes(attempt_receipt)
    receipt_path.write_bytes(receipt_payload)
    record = append_ledger_record(
        ledger_path,
        {
            "layout_pair_id": candidate["layout_pair_id"],
            "candidate_id": candidate["candidate_id"],
            "candidate_rank": candidate["candidate_rank"],
            "candidate_payload_sha256": candidate["candidate_payload_sha256"],
            "candidate_pool_sha256": candidate_pool_sha256,
            "decision": decision,
            "passed": passed,
            "model_request_count": 0,
            "behavioral_action_count": 0,
            "attempt_number": attempt_number,
            "attempt_receipt": {
                "path": str(receipt_path.resolve()),
                "sha256": sha256_bytes(receipt_payload),
                "bytes": len(receipt_payload),
            },
            "failure_count": len(evaluation.get("failures", [])),
            "failures": evaluation.get("failures", []),
            "release_boundary": "accepted means eligible for a physical pose manifest only; no model request is released by this ledger record",
        },
        recorded_at_utc=recorded_at_utc,
    )
    return {"record": record, "evaluation": evaluation, "attempt_dir": str(attempt_dir)}


def next_candidate_for_layout(
    pool: Mapping[str, Any], records: Sequence[Mapping[str, Any]], layout_pair_id: str
) -> dict[str, Any] | None:
    require(layout_pair_id in PLANNED_LAYOUT_IDS, "layout pair is not planned")
    accepted = [record for record in records if record.get("layout_pair_id") == layout_pair_id and record.get("decision") == "accepted"]
    require(len(accepted) <= 1, f"multiple accepted candidates exist for {layout_pair_id}")
    if accepted:
        return None
    rows = sorted(
        (row for row in pool["candidates"] if row["layout_pair_id"] == layout_pair_id),
        key=lambda row: row["candidate_rank"],
    )
    for candidate in rows:
        decisions = [record["decision"] for record in records if record.get("candidate_id") == candidate["candidate_id"]]
        if "physical_rejection" in decisions:
            continue
        return candidate
    return None


def build_frozen_pose_manifest(
    *,
    pool: Mapping[str, Any],
    candidate_pool_sha256: str,
    records: Sequence[Mapping[str, Any]],
    gate_ledger_sha256: str,
    layout_pair_ids: Sequence[str],
) -> dict[str, Any]:
    validate_candidate_pool(pool)
    _sha(candidate_pool_sha256, "candidate-pool digest")
    _sha(gate_ledger_sha256, "gate-ledger digest")
    requested = list(layout_pair_ids)
    require(bool(requested) and len(set(requested)) == len(requested), "pose-manifest layout IDs must be unique and nonempty")
    require(set(requested).issubset(set(PLANNED_LAYOUT_IDS)), "pose manifest requested an unplanned layout")
    candidates = candidate_index(pool)
    selected: dict[str, Any] = {}
    for layout_id in requested:
        accepted = [
            record for record in records
            if record.get("layout_pair_id") == layout_id and record.get("decision") == "accepted" and record.get("passed") is True
        ]
        require(len(accepted) == 1, f"{layout_id} needs exactly one accepted live-gate record")
        record = accepted[0]
        require(record.get("candidate_pool_sha256") == candidate_pool_sha256, f"{layout_id} gate used another candidate pool")
        candidate = candidates.get(record.get("candidate_id"))
        require(candidate is not None, f"{layout_id} accepted candidate is absent from the pool")
        require(record.get("candidate_payload_sha256") == candidate["candidate_payload_sha256"], f"{layout_id} candidate digest changed")
        selected[layout_id] = {
            "layout_pair_id": layout_id,
            "candidate_id": candidate["candidate_id"],
            "candidate_rank": candidate["candidate_rank"],
            "candidate_payload_sha256": candidate["candidate_payload_sha256"],
            "accepted_gate_record_sha256": record["record_sha256"],
            "accepted_gate_attempt_receipt": record["attempt_receipt"],
            "layouts": candidate["layouts"],
            "object_asset_provenance": candidate["object_asset_provenance"],
        }
    manifest = {
        "schema_version": POSE_MANIFEST_SCHEMA,
        "study_namespace": NAMESPACE,
        "status": POSE_MANIFEST_STATUS,
        "physical_layout_gate_passed": True,
        "released_for_model_inference": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_pool_sha256": candidate_pool_sha256,
        "gate_ledger_sha256": gate_ledger_sha256,
        "gate_ledger_last_record_sha256": records[-1]["record_sha256"] if records else None,
        "qualified_layout_count": len(selected),
        "layout_pair_ids": requested,
        "layout_pairs": selected,
        "task_contract": {
            "action_cap": 450,
            "termination_terms": ["time_out"],
            "success_is_measurement_only": True,
            "success_termination_present": False,
        },
        "release_boundary": "This manifest authorizes hash-pinned timeout-only scene construction. Runtime/model/recording/forecast gates and the queue release remain separate requirements.",
    }
    return validate_frozen_pose_manifest(manifest)


def _load_adapter(specification: str, config: Mapping[str, Any]) -> FixtureGateAdapter:
    require(":" in specification, "adapter must be MODULE:FACTORY")
    module_name, factory_name = specification.rsplit(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    require(callable(factory), f"fixture adapter factory is not callable: {specification}")
    adapter = factory(config)
    require(callable(getattr(adapter, "capture_condition", None)), "fixture adapter lacks capture_condition")
    require(callable(getattr(adapter, "close", None)), "fixture adapter lacks close")
    return adapter


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    qualify = subparsers.add_parser("qualify", help="attempt one exact candidate and append its decision")
    qualify.add_argument("--source-contract", type=Path, required=True)
    qualify.add_argument("--source-contract-sha256", required=True)
    qualify.add_argument("--candidate-pool", type=Path, required=True)
    qualify.add_argument("--candidate-pool-sha256", required=True)
    qualify.add_argument("--layout-pair-id", choices=PLANNED_LAYOUT_IDS, required=True)
    qualify.add_argument("--candidate-id")
    qualify.add_argument("--adapter", required=True, help="Python MODULE:FACTORY")
    qualify.add_argument("--adapter-config", type=Path)
    qualify.add_argument("--ledger", type=Path, required=True)
    qualify.add_argument("--attempt-root", type=Path, required=True)
    freeze = subparsers.add_parser("freeze", help="freeze accepted gate records into a pose manifest")
    freeze.add_argument("--candidate-pool", type=Path, required=True)
    freeze.add_argument("--candidate-pool-sha256", required=True)
    freeze.add_argument("--ledger", type=Path, required=True)
    freeze.add_argument("--ledger-sha256", required=True)
    freeze.add_argument("--layout-pair-id", choices=PLANNED_LAYOUT_IDS, action="append")
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "qualify":
        source, _ = load_json_file(args.source_contract, args.source_contract_sha256)
        source = validate_source_contract(source)
        pool, _ = load_json_file(args.candidate_pool, args.candidate_pool_sha256)
        pool = validate_candidate_pool(pool, source_sha256=args.source_contract_sha256)
        records = read_ledger(args.ledger)
        index = candidate_index(pool)
        if args.candidate_id:
            candidate = index.get(args.candidate_id)
            require(candidate is not None and candidate["layout_pair_id"] == args.layout_pair_id, "candidate ID does not belong to requested layout")
            require(not any(record.get("candidate_id") == args.candidate_id and record.get("decision") == "physical_rejection" for record in records), "refusing to rerun a physically rejected candidate")
        else:
            candidate = next_candidate_for_layout(pool, records, args.layout_pair_id)
            require(candidate is not None, "no eligible candidate remains or layout is already accepted")
        config: Mapping[str, Any] = {}
        if args.adapter_config:
            config, _ = load_json_file(args.adapter_config)
            require(isinstance(config, dict), "adapter config must be an object")
        config = dict(config)
        injected = {
            "source_contract_path": str(args.source_contract.resolve()),
            "source_contract_sha256": args.source_contract_sha256,
            "candidate_pool_path": str(args.candidate_pool.resolve()),
            "candidate_pool_sha256": args.candidate_pool_sha256,
            "candidate_id": candidate["candidate_id"],
        }
        for key, value in injected.items():
            require(key not in config or config[key] == value, f"adapter config conflicts with gate argument: {key}")
            config[key] = value
        adapter = _load_adapter(args.adapter, config)
        try:
            result = qualify_candidate(
                adapter=adapter,
                candidate=candidate,
                source=source,
                candidate_pool_sha256=args.candidate_pool_sha256,
                ledger_path=args.ledger,
                attempt_root=args.attempt_root,
            )
        finally:
            adapter.close()
        print(json.dumps({
            "candidate_id": candidate["candidate_id"],
            "decision": result["record"]["decision"],
            "record_sha256": result["record"]["record_sha256"],
            "attempt_dir": result["attempt_dir"],
        }, sort_keys=True))
        return {"accepted": 0, "physical_rejection": 2, "technical_invalid": 3}[result["record"]["decision"]]

    pool, _ = load_json_file(args.candidate_pool, args.candidate_pool_sha256)
    pool = validate_candidate_pool(pool)
    require(args.ledger.exists(), "gate ledger does not exist")
    require(sha256_file(args.ledger) == args.ledger_sha256, "gate-ledger digest mismatch")
    records = read_ledger(args.ledger)
    layout_ids = args.layout_pair_id or list(PLANNED_LAYOUT_IDS)
    manifest = build_frozen_pose_manifest(
        pool=pool,
        candidate_pool_sha256=args.candidate_pool_sha256,
        records=records,
        gate_ledger_sha256=args.ledger_sha256,
        layout_pair_ids=layout_ids,
    )
    payload = canonical_json_bytes(manifest)
    write_immutable(args.output, payload)
    print(json.dumps({
        "path": str(args.output.resolve()),
        "sha256": sha256_bytes(payload),
        "qualified_layout_count": manifest["qualified_layout_count"],
        "released_for_model_inference": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
