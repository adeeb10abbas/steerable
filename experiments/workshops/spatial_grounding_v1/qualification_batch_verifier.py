"""Verify retained LAT qualification evidence without starting a simulator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import imageio.v2 as iio
import numpy as np

from .fixtures import ACTION_CAP, FixtureCandidate, FixtureError, Pose, ResetSnapshot, validate_reset
from .grasp_calibration import SCHEMA
from .lat_candidate_generator import workspace_digest
from .recorder import atomic_json
from .scoring import GoalSpec, OutcomeStatus, score_episode


class VerificationError(ValueError):
    """Retained evidence does not reproduce its recorded identity or result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scoped(root: Path, path: str | Path) -> Path:
    value = Path(path)
    value = (root / value).resolve() if not value.is_absolute() else value.resolve()
    _require(value.is_relative_to(root.resolve()), f"evidence escapes its root: {path}")
    return value


def _file(path: Path, record: Mapping[str, Any]) -> None:
    _require(path.is_file(), f"missing evidence: {path}")
    _require(path.stat().st_size == record["bytes"] and _digest(path) == record["sha256"],
             f"file hash/size mismatch: {path}")


def _frame(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    _require(isinstance(value, np.ndarray) and value.dtype == np.uint8
             and value.ndim == 3 and value.shape[-1] == 3 and bool(np.ptp(value)),
             f"invalid RGB frame: {path}")
    return value


def _video(root: Path, record: Mapping[str, Any], count: int, fps: float, shape: tuple[int, ...]) -> None:
    path = _scoped(root, record["path"])
    _file(path, record)
    _require(record["frame_count"] == count and math.isclose(record["fps"], fps, abs_tol=1e-6),
             f"video receipt timing mismatch: {path}")
    with iio.get_reader(path) as reader:
        _require(math.isclose(reader.get_meta_data()["fps"], fps, abs_tol=1e-3),
                 f"encoded video FPS mismatch: {path}")
        decoded = 0
        for frame in reader:
            _require(frame.shape == shape and frame.dtype == np.uint8, f"decoded video shape mismatch: {path}")
            decoded += 1
    _require(decoded == count, f"video did not decode all {count} frames: {path}")


@dataclass(frozen=True)
class Registration:
    plan: dict[str, Any]
    plan_sha256: str
    candidates: Mapping[str, FixtureCandidate]
    controller: dict[str, Any]


def load_registration(plan: Path, proposals: Path, calibration: Path, *, expected_plan_sha256: str) -> Registration:
    _require(_digest(plan) == expected_plan_sha256, "frozen batch plan hash mismatch")
    value = _json(plan)
    _require(value["schema_version"] == "sgw-01-finite-model-blind-qualification-batch-v1"
             and value["family"] == "LAT" and value["model_requests"] == value["behavioral_episodes"] == 0,
             "not a registered model-blind LAT batch")
    _require(value["controller_actions_per_trial"] == ACTION_CAP and value["scripted_trials_per_candidate"] == 6,
             "batch action/trial count differs from frozen contract")
    _require(_digest(proposals) == value["proposal_sha256"], "frozen proposal hash mismatch")
    _require(_digest(calibration) == value["controller_calibration_sha256"], "frozen calibration hash mismatch")
    proposal = _json(proposals)
    rows = proposal["candidates"]
    names = value["candidate_ids_in_frozen_hash_order"]
    _require(len(rows) == 100 and len({row["candidate_id"] for row in rows}) == 100,
             "frozen 100-candidate inventory differs")
    _require(len(names) == value["remaining_geometrically_eligible_candidates"] == len(set(names)),
             "batch candidate count/uniqueness differs")
    _require([name for name in proposal["qualification_order"] if name in names] == names,
             "batch is not in frozen hash order")
    selected = {row["candidate_id"]: row for row in rows if row["candidate_id"] in names}
    _require(set(selected) == set(names) and all(
        row["metadata"]["geometric_screen_status"] == "passed" for row in selected.values()
    ), "batch includes a missing/geometrically rejected candidate")
    measured = _json(calibration)
    _require(measured["schema_version"] == SCHEMA and measured["receipt_sha256"] == workspace_digest(measured),
             "measured controller receipt identity mismatch")
    return Registration(
        value, expected_plan_sha256,
        {name: FixtureCandidate.from_json(row) for name, row in selected.items()},
        {"recipe": SCHEMA, "calibration_sha256": value["controller_calibration_sha256"], "calibration": measured},
    )


def _raw_state(raw: Mapping[str, Any], index: int, candidate: FixtureCandidate) -> dict[str, Any]:
    objects = raw["objects"]
    _require(set(objects) == set(candidate.object_poses), "raw snapshot object inventory differs")
    for obj in objects.values():
        Pose.from_json(obj["pose"])
        for key in ("linear_speed_m_s", "angular_speed_rad_s"):
            _require(type(obj[key]) in (float, int) and math.isfinite(obj[key]) and obj[key] >= 0,
                     f"invalid raw {key}")
        _require(type(obj["supported"]) is bool and type(obj["attached_to_gripper"]) is bool,
                 "raw support/attachment must be measured booleans")
    cube = objects["rubiks_cube"]
    _require(index == ACTION_CAP or raw["termination_reason"] is None, "early native termination")
    return {
        "action_step": index, "sim_time_s": raw["simulated_time_s"],
        "cube_xyz_m": cube["pose"]["position_m"], "bowl_xyz_m": objects["bowl"]["pose"]["position_m"],
        "plate_xyz_m": objects["plate"]["pose"]["position_m"] if "plate" in objects else None,
        "supported": cube["supported"], "final_detached_release": not cube["attached_to_gripper"],
        "gripper_holding": cube["attached_to_gripper"],
        "linear_speed_m_s": cube["linear_speed_m_s"], "angular_speed_rad_s": cube["angular_speed_rad_s"],
    }


def _warmup(root: Path, receipt: Mapping[str, Any], shape: tuple[int, ...]) -> int:
    _require(receipt["physics_actions"] == 0 and receipt["render_frames"] == 120
             and receipt["simulation_time_unchanged"] is True, "reset warmup advanced physics or changed length")
    rows = receipt["snapshots"]
    _require(len(rows) == 121 and [row["render_frame"] for row in rows] == list(range(121)),
             "reset warmup frame inventory differs")
    time = rows[0]["sim_time_s"]
    _require(math.isfinite(time) and all(row["sim_time_s"] == time for row in rows),
             "reset warmup physical time differs")
    for row in rows:
        _require(set(row["views"]) == {"over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"},
                 "reset warmup camera inventory differs")
        for record in row["views"].values():
            path = _scoped(root, record["path"])
            _file(path, record)
            _require(_frame(path).shape == shape, "reset warmup RGB shape differs")
    _video(root, receipt["viewport_video"], 121, 30, shape)
    return 121 * 3 + 1


def _trial(root: Path, check: dict[str, Any], candidate: FixtureCandidate) -> tuple[dict[str, Any], ResetSnapshot]:
    sign, repeat = check["goal_sign"], check["reset_index"]
    trial = root / "result/trials" / f"goal-{sign:+d}" / f"reset-{repeat}"
    saved = _json(trial / "trial.json")
    # The producer adds reset_validation/reset_error and may change passed only
    # after all three resets. The on-disk trial is the pre-reset-gate record.
    base_check = {key: value for key, value in check.items() if key not in {"passed", "reset_validation", "reset_error"}}
    _require({key: value for key, value in saved.items() if key != "passed"} == base_check,
             f"trial receipt differs from aggregate: {trial}")
    _require(saved["observed_actions"] == saved["actions_executed"] == ACTION_CAP
             and saved["model_request_count"] == saved["behavioral_episode_count"] == 0,
             f"trial is not a complete zero-model trace: {trial}")
    expected = {"reset.json", "viewport.mp4"}
    expected.update(f"{prefix}-{index:04d}.{suffix}" for prefix, suffix, start in (
        ("state", "json", 0), ("frame", "npy", 0), ("action", "npy", 1), ("command", "json", 1)
    ) for index in range(start, ACTION_CAP + 1))
    _require(set(saved["files"]) == expected, f"trial manifest omits/adds files: {trial}")
    _require({path.name for path in trial.iterdir()} == expected | {"trial.json"},
             f"raw trial inventory differs: {trial}")
    for name in sorted(expected):
        _file(_scoped(trial, name), saved["files"][name])
    reset = saved["reset_receipt"]
    _require(_json(trial / "reset.json") == reset, "reset receipt differs from trial")
    ordinal = repeat + (1 if sign == 1 else 4)
    _require(reset["reset_id"] == f"{candidate.candidate_id}:{ordinal}"
             and reset["camera_id"] == reset["camera_name"] == "over_shoulder_left_camera"
             and reset["temporal_cache_reset"] is True, "native reset identity differs")
    dt = reset["control_step_dt_s"]
    _require(math.isfinite(dt) and dt > 0, "invalid physical control period")
    states = []
    initial = None
    shape = _frame(trial / "frame-0000.npy").shape
    for index in range(ACTION_CAP + 1):
        record = _json(trial / f"state-{index:04d}.json")
        state = _raw_state(record["raw_snapshot"], index, candidate)
        _require(state == record["state"] and math.isclose(state["sim_time_s"], index * dt, rel_tol=0, abs_tol=1e-9),
                 f"raw/scored state or physical time mismatch at {trial}:{index}")
        name = f"frame-{index:04d}.npy"
        _require(record["viewport_path"] == name and record["viewport_sha256"] == saved["files"][name]["sha256"],
                 f"state/RGB binding differs at {trial}:{index}")
        _require(_frame(trial / name).shape == shape, f"RGB dimensions changed at {trial}:{index}")
        states.append(state)
        if index == 0:
            initial = record["raw_snapshot"]
        else:
            action = np.load(trial / f"action-{index:04d}.npy", allow_pickle=False)
            _require(isinstance(action, np.ndarray) and action.shape == (1, 8)
                     and action.dtype == np.float32 and np.isfinite(action).all(),
                     f"invalid native Abs-IK action at {trial}:{index}")
            command = _json(trial / f"command-{index:04d}.json")
            _require(command == {"action_step": index, "status": "issued_not_yet_observed",
                                 "sha256": saved["files"][f"action-{index:04d}.npy"]["sha256"]},
                     f"command/action binding differs at {trial}:{index}")
    _require(states == saved["per_step_states"], "aggregate scored states differ from raw states")
    score = score_episode({"states": states, "termination_reason": "action_cap"}, GoalSpec(candidate.family, sign))
    _require(score.status is OutcomeStatus.VALID_MODEL, "scorer could not verify complete physical evidence")
    _require(saved["score"] == asdict(score) and saved["passed"] is score.requested_success
             and saved["requested_margin_m"] == score.terminal_margin_m
             and saved["status"] == ("passed_scripted_goal" if score.requested_success else "rejected_scripted_goal"),
             "recorded physical score differs from recomputation")
    _require(initial is not None, "missing initial snapshot")
    fingerprint = hashlib.sha256(json.dumps(
        {name: obj["pose"] for name, obj in initial["objects"].items()}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    _require(reset["fingerprint"] == fingerprint, "reset scoring-pose fingerprint differs")
    roots = ResetSnapshot.from_json({"poses": initial["reset_root_poses"]})
    _video(root, saved["viewport_video"], ACTION_CAP + 1, 1 / dt, shape)
    warmup_files = _warmup(root, reset["render_only_warmup"], shape)
    return {
        "goal_sign": sign, "reset_index": repeat, "score": asdict(score),
        "verified_trial_files": len(expected), "verified_warmup_files": warmup_files,
        "decoded_trial_frames": ACTION_CAP + 1, "decoded_warmup_frames": 121,
        "control_step_dt_s": dt, "trial_receipt_sha256": _digest(trial / "trial.json"),
    }, roots


def verify_candidate(registration: Registration, *, index: int, root: Path) -> dict[str, Any]:
    name = registration.plan["candidate_ids_in_frozen_hash_order"][index]
    candidate = registration.candidates[name]
    _require(root.name == f"{index}-{name}", "index/candidate output directory differs")
    storage = _json(root / "storage_preflight.json")
    _require(storage["plan_sha256"] == registration.plan_sha256 and storage["index"] == index
             and storage["candidate_id"] == name and storage["status"] == "passed_conservative_batch_space_check"
             and storage["model_requests"] == storage["behavioral_episodes"] == 0,
             "storage/launch plan identity differs")
    result = _json(root / "result/qualification.json")
    candidate_hash = hashlib.sha256(json.dumps(asdict(candidate), sort_keys=True).encode()).hexdigest()
    _require(result["candidate_id"] == name and result["candidate_sha256"] == candidate_hash
             and result["family"] == candidate.family and result["seed"] == candidate.seed
             and result["action_cap"] == ACTION_CAP
             and result["model_request_count"] == result["behavioral_episode_count"] == 0,
             "candidate/seed/asset/zero-model identity differs")
    _require(result["controller_identity"] == _json(root / "result/controller.json") == registration.controller,
             "executed controller identity differs from frozen calibration")
    checks = result["checks"]
    _require([(row["goal_sign"], row["reset_index"]) for row in checks] ==
             [(sign, repeat) for sign in (1, -1) for repeat in range(3)], "six-trial identity/order differs")
    reports = []
    for offset in (0, 3):
        group = checks[offset:offset + 3]
        verified = [_trial(root, check, candidate) for check in group]
        try:
            reset_rows = validate_reset(candidate, [reset for _, reset in verified])
        except FixtureError as error:
            reset_error = str(error)
            reset_rows = []
        else:
            reset_error = None
        for check, (report, _) in zip(group, verified, strict=True):
            passed = report["score"]["requested_success"] and reset_error is None
            _require(check["passed"] is passed, "aggregate pass disagrees with physical/reset gates")
            if reset_error is None:
                _require("reset_error" not in check and check["reset_validation"] ==
                         [row for row in reset_rows if row["repeat"] == check["reset_index"]],
                         "recorded reset comparison differs")
            else:
                _require(check.get("reset_error") == reset_error, "recorded reset rejection differs")
            reports.append({**report, "passed": passed, "reset_error": reset_error})
    passed_count = sum(row["passed"] for row in reports)
    _require(result["status"] == ("accepted_model_blind_fixture_candidate" if passed_count == 6
                                 else "rejected_model_blind_fixture_candidate"), "aggregate qualification status differs")
    return {
        "index": index, "candidate_id": name,
        "status": "verified_all_six_pass" if passed_count == 6 else "verified_physical_rejection",
        "passed_checks": passed_count, "checks": reports, "model_requests": 0, "behavioral_episodes": 0,
        "qualification_sha256": _digest(root / "result/qualification.json"),
        "storage_preflight_sha256": _digest(root / "storage_preflight.json"),
    }


def _candidate_report(registration: Registration, index: int, root: Path) -> dict[str, Any]:
    identity = {"index": index, "candidate_id": registration.plan["candidate_ids_in_frozen_hash_order"][index]}
    try:
        return verify_candidate(registration, index=index, root=root)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {**identity, "status": "technical_invalid_evidence",
                "error_type": type(error).__name__, "reason": str(error)}


def _summary(registration: Registration, reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "sgw-01-lat-batch-verification-v1",
        "plan_sha256": registration.plan_sha256, "source_commit_in_launch_plan": registration.plan["source_commit"],
        "controller_calibration_sha256": registration.plan["controller_calibration_sha256"],
        "candidates": reports,
        "counts": {status: sum(row["status"] == status for row in reports) for status in (
            "verified_all_six_pass", "verified_physical_rejection", "technical_invalid_evidence",
            "pending_or_partial", "not_started_or_missing",
        )},
        "model_requests": 0, "behavioral_episodes": 0, "release_permitted": False,
        "claim_boundary": "Raw evidence integrity and recomputed model-blind qualification only. "
                          "Launch/source attestation, historical deduplication and all model/family release gates remain separate.",
    }


def verify_batch(registration: Registration, raw_root: Path) -> dict[str, Any]:
    reports = []
    for index, name in enumerate(registration.plan["candidate_ids_in_frozen_hash_order"]):
        root = raw_root / f"{index}-{name}"
        identity = {"index": index, "candidate_id": name}
        if not (root / "result/qualification.json").is_file():
            reports.append({**identity, "status": "pending_or_partial" if root.exists() else "not_started_or_missing"})
            continue
        reports.append(_candidate_report(registration, index, root))
    return _summary(registration, reports)


def watch_batch(registration: Registration, raw_root: Path, output: Path, *, wait_until_utc: str) -> dict[str, Any]:
    """Persist each completed verification once, bounded by the producer deadline."""
    deadline = datetime.fromisoformat(wait_until_utc.replace("Z", "+00:00"))
    _require(deadline.tzinfo is not None, "verification deadline must include its timezone")
    reports_root = output.with_suffix(".candidates")
    reports_root.mkdir(parents=True, exist_ok=False)
    names = registration.plan["candidate_ids_in_frozen_hash_order"]
    completed: dict[int, dict[str, Any]] = {}
    while True:
        for index, name in enumerate(names):
            root = raw_root / f"{index}-{name}"
            if index in completed or not (root / "result/qualification.json").is_file():
                continue
            report = _candidate_report(registration, index, root)
            report["plan_sha256"] = registration.plan_sha256
            report["release_permitted"] = False
            atomic_json(reports_root / f"{index:02d}-{name}.json", report)
            completed[index] = report
            print(json.dumps({"index": index, "candidate_id": name, "status": report["status"]}), flush=True)
        remaining = deadline.timestamp() - time.time()
        if len(completed) == len(names) or remaining <= 0:
            break
        time.sleep(min(30, remaining))
    reports = []
    for index, name in enumerate(names):
        if index in completed:
            reports.append(completed[index])
        else:
            root = raw_root / f"{index}-{name}"
            reports.append({"index": index, "candidate_id": name,
                            "status": "pending_or_partial" if root.exists() else "not_started_or_missing"})
    return {**_summary(registration, reports), "wait_until_utc": wait_until_utc}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--controller-calibration", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-until-utc", help="Optionally verify new completed candidates until this bounded deadline.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verification: {args.output}")
    registration = load_registration(args.plan, args.proposals, args.controller_calibration,
                                     expected_plan_sha256=args.expected_plan_sha256)
    report = (watch_batch(registration, args.raw_root, args.output, wait_until_utc=args.wait_until_utc)
              if args.wait_until_utc else verify_batch(registration, args.raw_root))
    atomic_json(args.output, report)
    print(json.dumps(report["counts"], sort_keys=True))
    if report["counts"]["technical_invalid_evidence"] or (
        args.wait_until_utc and report["counts"]["pending_or_partial"] + report["counts"]["not_started_or_missing"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
