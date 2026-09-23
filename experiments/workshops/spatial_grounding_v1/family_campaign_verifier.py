"""Offline, fail-closed verifier for a finished HEIGHT/DIST campaign design."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .fixtures import ACTION_CAP, FixtureCandidate, FixtureError, Pose, validate_reset
from .family_campaign import SCHEMA
from .prospective_family_capture import verify_capture_artifacts
from .prospective_family_designs import (
    CANDIDATE_STATUS, _aabb_xy_clearance_m, _candidate_manifest, _digest, require_design_capture,
)
from .qualification_batch_verifier import VerificationError, verify_trial_evidence, verify_zero_action_trial_evidence
from .robolab_height_dist_qualification import CALIBRATION_SCHEMA, validate_candidate_inputs
from .lat_candidate_generator import workspace_digest
from .model_blind_qualification import QualificationError, validate_preaction_reset_geometry
from .simulator_bridge import SimulatorSnapshot


def verify_design(*, campaign_path: Path, design_id: str, root: Path, output: Path) -> dict[str, Any]:
    """Verify family binding, six trial identities, retained videos, and postprocess inputs."""

    if output.exists():
        raise FileExistsError("refusing to overwrite family verification")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != SCHEMA or campaign.get("campaign_sha256") != _digest(campaign, "campaign_sha256"):
        raise ValueError("campaign identity differs")
    job = next((row for row in campaign["jobs"] if row["design_id"] == design_id), None)
    if not isinstance(job, Mapping) or job.get("status") != "blocked_pending_candidate_overlay_and_fresh_zero_model_capture":
        raise ValueError("design is not a campaign qualification job")
    candidate_manifest = root / "candidate_manifest.json"
    candidate_capture = root / "candidate_capture.json"
    materialized_candidate = root / "candidate.json"
    qualification = root / "qualification.json"
    if not candidate_manifest.is_file() or not candidate_capture.is_file() or not materialized_candidate.is_file() or not qualification.is_file():
        raise ValueError("candidate manifest, capture, materialization, and qualification outputs are all required")
    manifest = _candidate_manifest(candidate_manifest)
    verify_capture_artifacts(candidate_capture)
    capture = require_design_capture(candidate_manifest_path=candidate_manifest, capture_path=candidate_capture)
    candidate_value = json.loads(materialized_candidate.read_text(encoding="utf-8"))
    candidate = FixtureCandidate.from_json(candidate_value)
    result = json.loads(qualification.read_text(encoding="utf-8"))
    _verify_design_chain(
        job, manifest, capture, _sha256(candidate_capture), candidate_value, candidate, result,
    )
    if capture.get("family") != campaign["family"] or result.get("family") != campaign["family"]:
        raise ValueError("family binding differs from campaign")
    if capture.get("model_request_count") != 0 or capture.get("behavioral_episode_count") != 0:
        raise ValueError("candidate capture is not zero-model")
    controller = _verified_calibration(root, candidate, result)
    if result.get("action_cap") != ACTION_CAP or result.get("controller_identity") != controller:
        raise ValueError("qualification lacks calibrated 450-action controller identity")
    early = result.get("physical_geometry_rejection_before_actions")
    if isinstance(early, Mapping):
        return _verify_early_rejection(
            campaign=campaign, design_id=design_id, root=root, output=output, candidate=candidate,
            candidate_capture=candidate_capture, candidate_manifest=candidate_manifest,
            materialized_candidate=materialized_candidate, qualification=qualification, result=result, early=early,
        )
    expected = [(sign, reset) for sign in (1, -1) for reset in range(3)]
    checks = result.get("checks")
    if not isinstance(checks, list) or [(row.get("goal_sign"), row.get("reset_index")) for row in checks] != expected:
        raise ValueError("qualification must retain both goals and three fresh resets per goal")
    trial_reports, reset_groups = [], {1: [], -1: []}
    for check in checks:
        trial = root / "trials" / f"goal-{check['goal_sign']:+d}" / f"reset-{check['reset_index']}" / "trial.json"
        if not trial.is_file():
            raise ValueError("qualification trial receipt is missing")
        _verify_preaction_guard(trial.parent, candidate, check)
        try:
            report, reset = verify_trial_evidence(
                evidence_root=root, trial=trial.parent, check=check, candidate=candidate,
            )
        except VerificationError as error:
            raise ValueError(f"raw trial integrity or physical scoring differs: {error}") from error
        _verify_reset_banana_geometry(trial.parent, candidate)
        reset_groups[check["goal_sign"]].append(reset)
        _verify_aggregate_trial(check, report)
        trial_sha256 = _sha256(trial)
        trial_reports.append({
            "goal_sign": check["goal_sign"], "reset_index": check["reset_index"],
            "trial_sha256": trial_sha256, "sha256": trial_sha256, "bytes": trial.stat().st_size,
            "recomputed_score": report["score"],
        })
    reset_errors = _verify_reset_groups(candidate, checks, reset_groups)
    expected_passes = {
        (check["goal_sign"], check["reset_index"]): bool(check["score"]["requested_success"])
        and reset_errors[check["goal_sign"]] is None
        for check in checks
    }
    if any(check.get("passed") is not expected_passes[(check["goal_sign"], check["reset_index"])] for check in checks):
        raise ValueError("aggregate pass differs from recomputed physical/reset gates")
    accepted = all(expected_passes.values())
    if result.get("status") != (
        "accepted_model_blind_fixture_candidate" if accepted else "rejected_model_blind_fixture_candidate"
    ):
        raise ValueError("aggregate qualification status differs from recomputed physical/reset gates")
    value = {
        "schema_version": "sgw-01-family-campaign-verification-v1",
        "campaign_sha256": campaign["campaign_sha256"],
        "design_id": design_id,
        "family": campaign["family"],
        "evidence_root": str(root.resolve()),
        "candidate_capture_sha256": _sha256(candidate_capture),
        "candidate_manifest_sha256": _sha256(candidate_manifest),
        "materialized_candidate_sha256": _sha256(materialized_candidate),
        "qualification_sha256": _sha256(qualification),
        "candidate_capture": _record(candidate_capture),
        "candidate_manifest": _record(candidate_manifest),
        "materialized_candidate": _record(materialized_candidate),
        "qualification": _record(qualification),
        "trials": [
            {**row, "path": str((root / "trials" / f"goal-{row['goal_sign']:+d}" / f"reset-{row['reset_index']}" / "trial.json").resolve())}
            for row in trial_reports
        ],
        "status": "verified_evidence_not_fixture_release",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "release_permitted": False,
    }
    value["verification_sha256"] = _digest(value, "verification_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _verify_early_rejection(
    *, campaign: Mapping[str, Any], design_id: str, root: Path, output: Path, candidate: FixtureCandidate,
    candidate_capture: Path, candidate_manifest: Path, materialized_candidate: Path, qualification: Path,
    result: Mapping[str, Any], early: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a real zero-action geometry rejection and its completed prefix."""
    goal, reset = early.get("goal_sign"), early.get("reset_index")
    if (goal not in {1, -1} or reset not in range(3) or early.get("rejection_scope") not in {"candidate", "reset"}
            or not isinstance(early.get("reason"), str) or not early["reason"]
            or early.get("controller_actions_executed") != 0
            or result.get("status") != "rejected_model_blind_fixture_candidate"):
        raise ValueError("early physical geometry rejection aggregate is malformed")
    identities = [(sign, index) for sign in (1, -1) for index in range(3)]
    terminal = identities.index((goal, reset))
    checks = result.get("checks")
    if not isinstance(checks, list) or [(row.get("goal_sign"), row.get("reset_index")) for row in checks] != identities[:terminal + 1]:
        raise ValueError("early physical geometry rejection has a non-prefix trial aggregate")
    reports, reset_groups = [], {1: [], -1: []}
    for check in checks[:terminal]:
        trial = root / "trials" / f"goal-{check['goal_sign']:+d}" / f"reset-{check['reset_index']}"
        _verify_preaction_guard(trial, candidate, check)
        try:
            report, reset_snapshot = verify_trial_evidence(evidence_root=root, trial=trial, check=check, candidate=candidate)
        except VerificationError as error:
            raise ValueError(f"completed prefix trial differs from raw evidence: {error}") from error
        _verify_reset_banana_geometry(trial, candidate)
        _verify_aggregate_trial(check, report)
        reset_groups[check["goal_sign"]].append(reset_snapshot)
        reports.append(_trial_record(trial, check))
    complete_groups = {sign: rows for sign, rows in reset_groups.items() if len(rows) == 3}
    reset_errors = _verify_reset_groups(candidate, checks[:-1], complete_groups)
    for check in checks[:-1]:
        sign = check["goal_sign"]
        if sign not in complete_groups and any(key in check for key in ("reset_validation", "reset_error")):
            raise ValueError("incomplete reset group claims a completed reset comparison")
        expected_pass = bool(check["score"]["requested_success"]) and reset_errors.get(sign) is None
        if check.get("passed") is not expected_pass:
            raise ValueError("completed prefix pass differs from physical/reset gates")
    rejected_check = checks[-1]
    trial = root / "trials" / f"goal-{goal:+d}" / f"reset-{reset}"
    _verify_rejected_trial(root, trial, candidate, rejected_check, early)
    reports.append(_trial_record(trial, rejected_check))
    for sign, index in identities[terminal + 1:]:
        future = root / "trials" / f"goal-{sign:+d}" / f"reset-{index}"
        if future.exists():
            raise ValueError("geometry rejection was followed by another trial")
    value = {
        "schema_version": "sgw-01-family-campaign-verification-v1",
        "campaign_sha256": campaign["campaign_sha256"], "design_id": design_id, "family": campaign["family"],
        "evidence_root": str(root.resolve()), "candidate_capture_sha256": _sha256(candidate_capture),
        "candidate_manifest_sha256": _sha256(candidate_manifest),
        "materialized_candidate_sha256": _sha256(materialized_candidate),
        "qualification_sha256": _sha256(qualification), "candidate_capture": _record(candidate_capture),
        "candidate_manifest": _record(candidate_manifest), "materialized_candidate": _record(materialized_candidate),
        "qualification": _record(qualification), "trials": reports,
        "physical_geometry_rejection": dict(early), "status": "verified_evidence_not_fixture_release",
        "model_request_count": 0, "behavioral_episode_count": 0, "release_permitted": False,
    }
    value["verification_sha256"] = _digest(value, "verification_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _trial_record(trial: Path, check: Mapping[str, Any]) -> dict[str, Any]:
    receipt = trial / "trial.json"
    return {
        "goal_sign": check["goal_sign"], "reset_index": check["reset_index"],
        "trial_sha256": _sha256(receipt), "sha256": _sha256(receipt), "bytes": receipt.stat().st_size,
        "path": str(receipt.resolve()),
    }


def _verify_rejected_trial(
    root: Path, trial: Path, candidate: FixtureCandidate, check: Mapping[str, Any], early: Mapping[str, Any],
) -> None:
    receipt_path = trial / "trial.json"
    guard_path = trial / "preaction-geometry-guard.json"
    if not receipt_path.is_file() or not guard_path.is_file():
        raise ValueError("early rejection lacks trial receipt or guard")
    _verify_preaction_guard(
        trial, candidate, check, expected_status="physical_geometry_rejection_before_actions",
    )
    guard = json.loads(guard_path.read_text())
    if check.get("physical_geometry_rejection") != {
        "rejection_scope": early["rejection_scope"], "reason": early["reason"],
    }:
        raise ValueError("early rejection aggregate reason differs from trial")
    try:
        raw = verify_zero_action_trial_evidence(
            evidence_root=root, trial=trial, check=check, candidate=candidate,
        )
    except VerificationError as error:
        raise ValueError(f"early rejection raw recording differs: {error}") from error
    try:
        recomputed = validate_preaction_reset_geometry(
            SimulatorSnapshot(objects={}, context_measurements=raw.get("context_measurements")), candidate,
        )
    except QualificationError as error:
        raise ValueError(f"early rejection reset geometry is malformed: {error}") from error
    if recomputed is None or recomputed.scope != early["rejection_scope"] or recomputed.reason != early["reason"]:
        raise ValueError("early rejection reason differs from recomputed measured geometry")
    if (
        guard.get("rejection_scope") != early["rejection_scope"] or guard.get("reason") != early["reason"]
    ):
        raise ValueError("early rejection guard differs from recomputed reset evidence")


def _verify_design_chain(
    job: Mapping[str, Any], manifest: Mapping[str, Any], capture: Mapping[str, Any], capture_file_sha256: str,
    candidate_value: Mapping[str, Any], candidate: FixtureCandidate, result: Mapping[str, Any],
) -> None:
    plan_path = Path(str(job.get("plan_path", "")))
    if not plan_path.is_file():
        raise ValueError("campaign job lacks its frozen plan path")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != _digest(plan, "plan_sha256"):
        raise ValueError("frozen plan bytes or digest differ from the campaign binding")
    if job.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("frozen plan no longer matches the campaign job binding")
    row = next((value for value in plan.get("designs", []) if value.get("design_id") == job["design_id"]), None)
    if (
        not isinstance(row, Mapping)
        or manifest.get("status") != CANDIDATE_STATUS
        or manifest.get("design_id") != job["design_id"]
        or manifest.get("plan_sha256") != plan.get("plan_sha256")
        or manifest.get("design_sha256") != _digest(row)
        or job.get("plan_sha256") != plan.get("plan_sha256")
        or job.get("design_sha256") != _digest(row)
    ):
        raise ValueError("candidate overlay does not bind the selected campaign design")
    if candidate.family != job["family"] or result.get("candidate_id") != candidate.candidate_id:
        raise ValueError("materialized candidate identity differs from campaign qualification")
    if result.get("candidate_sha256") != _candidate_sha256(candidate):
        raise ValueError("qualification candidate hash differs from materialized candidate")
    metadata = candidate_value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("materialized candidate lacks prospective chain metadata")
    if (
        metadata.get("prospective_design_id") != job["design_id"]
        or metadata.get("candidate_overlay_manifest_sha256") != manifest["manifest_sha256"]
        or metadata.get("candidate_capture_sha256") != capture_file_sha256
    ):
        raise ValueError("materialized candidate does not bind its design, overlay, and capture")
    validate_candidate_inputs(candidate)
    _verify_measured_candidate_geometry(capture, manifest, candidate)


def _verified_calibration(root: Path, candidate: FixtureCandidate, result: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "controller.json"
    if not path.is_file():
        raise ValueError("qualification controller identity is missing")
    controller = json.loads(path.read_text(encoding="utf-8"))
    calibration = controller.get("calibration")
    if (
        controller.get("recipe") != CALIBRATION_SCHEMA
        or not isinstance(calibration, Mapping)
        or calibration.get("schema_version") != CALIBRATION_SCHEMA
        or calibration.get("receipt_sha256") != workspace_digest(calibration)
    ):
        raise ValueError("controller calibration identity is malformed")
    record = candidate.metadata.get("controller_calibration")
    if not isinstance(record, Mapping):
        raise ValueError("materialized candidate lacks bound calibration file")
    calibration_path = Path(str(record.get("path", "")))
    try:
        bound_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("bound calibration file cannot be decoded") from error
    if (
        not calibration_path.is_file()
        or record.get("sha256") != _sha256(calibration_path)
        or record.get("bytes") != calibration_path.stat().st_size
        or controller.get("calibration_sha256") != _sha256(calibration_path)
        or candidate.metadata.get("controller_calibration_sha256") != _sha256(calibration_path)
        or controller["calibration"] != bound_calibration
    ):
        raise ValueError("materialized candidate calibration binding differs from qualification")
    return controller


def _verify_measured_candidate_geometry(
    capture: Mapping[str, Any], manifest: Mapping[str, Any], candidate: FixtureCandidate,
) -> None:
    objects = capture.get("objects")
    if not isinstance(objects, Mapping):
        raise ValueError("candidate capture lacks measured objects")
    required = set(candidate.object_poses) | {"banana", "table"}
    if not required.issubset(objects):
        raise ValueError("candidate capture lacks measured scored objects, banana, or table")
    for name, expected in candidate.object_poses.items():
        observed = _capture_pose(objects[name], name)
        position_error, angle_error = _pose_error(observed, expected)
        if position_error > 0.003 or angle_error > 2.0:
            raise ValueError(f"candidate capture {name} differs from materialized pose")
    banana = _capture_pose(objects["banana"], "banana")
    source_baseline = manifest.get("source_baseline")
    baseline_record = source_baseline.get("overlay_manifest") if isinstance(source_baseline, Mapping) else None
    if not isinstance(baseline_record, Mapping):
        raise ValueError("candidate overlay lacks its bound baseline manifest")
    baseline_path = Path(str(baseline_record.get("path", "")))
    if not baseline_path.is_file() or _sha256(baseline_path) != baseline_record.get("sha256"):
        raise ValueError("candidate overlay baseline manifest bytes differ from binding")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    authored = baseline["prospective_design"]["authored_actor_root_overrides_env_local_xyz_m"]["banana"]
    orientations = baseline["prospective_design"]["authored_actor_quaternion_overrides_wxyz"]["banana"]
    position_error, angle_error = _pose_error(banana, Pose.from_json({"position_m": authored, "quaternion_wxyz": orientations}))
    if position_error > 0.003 or angle_error > 2.0:
        raise ValueError("candidate capture banana differs from authored prospective pose")
    _verify_banana_clearance(objects, manifest["native_import_contract"]["kinematic_or_static_bodies"])


def _verify_banana_clearance(objects: Mapping[str, Any], support_names: list[str]) -> None:
    table_minimum, table_maximum = _bbox(objects["table"], "table")
    banana_minimum, banana_maximum = _bbox(objects["banana"], "banana")
    if (
        banana_minimum[0] < table_minimum[0] or banana_maximum[0] > table_maximum[0]
        or banana_minimum[1] < table_minimum[1] or banana_maximum[1] > table_maximum[1]
    ):
        raise ValueError("measured candidate banana leaves the measured table bounds")
    for name in support_names:
        minimum, maximum = _bbox(objects.get(name), name)
        if _aabb_xy_clearance_m(banana_minimum, banana_maximum, minimum[:2], maximum[:2]) < 0.02:
            raise ValueError(f"measured candidate banana clearance is below 20 mm for {name}")


def _verify_reset_banana_geometry(trial: Path, candidate: FixtureCandidate) -> None:
    """Require each physical reset to retain its own banana/table/support geometry."""

    raw = json.loads((trial / "state-0000.json").read_text(encoding="utf-8")).get("raw_snapshot")
    context = raw.get("context_measurements") if isinstance(raw, Mapping) else None
    if not isinstance(context, Mapping):
        raise ValueError("raw reset snapshot lacks measured banana/table/support geometry")
    try:
        rejection = validate_preaction_reset_geometry(SimulatorSnapshot(objects={}, context_measurements=context), candidate)
    except QualificationError as error:
        raise ValueError(f"raw reset geometry evidence is malformed: {error}") from error
    if rejection is not None:
        raise ValueError(f"raw reset is a physical geometry rejection: {rejection.reason}")


def _verify_preaction_guard(
    trial: Path, candidate: FixtureCandidate, check: Mapping[str, Any], *,
    expected_status: str = "measured_banana_geometry_valid_before_actions",
) -> None:
    path = trial / "preaction-geometry-guard.json"
    if not path.is_file():
        raise ValueError("trial lacks preaction geometry guard")
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = value.get("raw_reset")
    state = trial / "state-0000.json"
    if (
        value.get("schema_version") != "sgw-01-family-preaction-geometry-guard-v1"
        or value.get("design_id") != candidate.metadata.get("prospective_design_id")
        or value.get("candidate_sha256") != _candidate_sha256(candidate)
        or value.get("candidate_capture_sha256") != candidate.metadata.get("candidate_capture_sha256")
        or value.get("goal_sign") != check["goal_sign"] or value.get("reset_index") != check["reset_index"]
        or value.get("status") != expected_status
        or value.get("controller_actions_executed") != 0
        or not isinstance(raw, Mapping) or raw.get("path") != str(state.resolve())
        or raw.get("sha256") != _sha256(state) or raw.get("bytes") != state.stat().st_size
    ):
        raise ValueError("preaction geometry guard differs from retained reset state")


def _bbox(row: Any, name: str) -> tuple[list[float], list[float]]:
    if not isinstance(row, Mapping):
        raise ValueError(f"candidate capture lacks measured {name}")
    minimum, maximum = row.get("bbox_env_local_min_xyz_m"), row.get("bbox_env_local_max_xyz_m")
    if not _finite_vector(minimum, 3) or not _finite_vector(maximum, 3) or any(low > high for low, high in zip(minimum, maximum, strict=True)):
        raise ValueError(f"candidate capture has invalid measured {name} bounds")
    return list(minimum), list(maximum)


def _capture_pose(row: Any, name: str) -> Pose:
    if not isinstance(row, Mapping):
        raise ValueError(f"candidate capture lacks measured {name}")
    return Pose.from_json({
        "position_m": row.get("root_position_env_local_xyz_m"),
        "quaternion_wxyz": row.get("root_quaternion_world_wxyz"),
    })


def _pose_error(observed: Pose, expected: Pose) -> tuple[float, float]:
    from .fixtures import pose_error
    return pose_error(observed, expected)


def _verify_aggregate_trial(check: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    if check.get("score") != report["score"] or check.get("requested_margin_m") != report["score"]["terminal_margin_m"]:
        raise ValueError("aggregate score differs from recomputed physical score")


def _verify_reset_groups(
    candidate: FixtureCandidate, checks: list[Mapping[str, Any]], groups: Mapping[int, list[Any]],
) -> dict[int, str | None]:
    errors: dict[int, str | None] = {}
    for sign in groups:
        try:
            rows = validate_reset(candidate, groups[sign])
        except FixtureError as error:
            errors[sign] = str(error)
            for check in checks:
                if check["goal_sign"] == sign and check.get("reset_error") != str(error):
                    raise ValueError("aggregate reset rejection differs from recomputation") from error
        else:
            errors[sign] = None
            for check in checks:
                if check["goal_sign"] == sign:
                    expected = sorted(
                        (row for row in rows if row["repeat"] == check["reset_index"]),
                        key=lambda row: str(row["object"]),
                    )
                    recorded = check.get("reset_validation")
                    if isinstance(recorded, list):
                        recorded = sorted(recorded, key=lambda row: str(row.get("object", "")) if isinstance(row, Mapping) else "")
                    if (
                        _canonical_json(recorded) != _canonical_json(expected)
                        or check.get("reset_error") is not None
                    ):
                        raise ValueError("aggregate reset validation differs from recomputation")
    return errors


def _candidate_sha256(candidate: FixtureCandidate) -> str:
    from dataclasses import asdict
    return hashlib.sha256(json.dumps(asdict(candidate), sort_keys=True).encode()).hexdigest()


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(
        isinstance(item, (int, float)) and math.isfinite(item) for item in value
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
