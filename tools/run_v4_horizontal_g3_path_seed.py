#!/usr/bin/env python3
"""Run one zero-policy V4 horizontal G3 path seed on RoboLab."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
FIXTURE_PROMPTS = {
    "horizontal": (
        "Place the cube so that the cube is left of the bowl. "
        "Use the robot's fixed viewpoint for left, right, front, and behind."
    ),
    "object_pair": (
        "Place the sponge so that the sponge is left of the tray. "
        "Use the robot's fixed viewpoint for left, right, front, and behind."
    ),
}


def _trajectory_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-path-trajectory-v1"


def _infra_failure_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-infrastructure-failure-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--campaign-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--reset-registry", type=Path, required=True)
    parser.add_argument("--reset-registry-sha256", required=True)
    parser.add_argument(
        "--fixture-id",
        choices=tuple(FIXTURE_PROMPTS),
        default="horizontal",
    )
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-study-commit", required=True)
    parser.add_argument("--expected-robolab-commit", default=ROBOLAB_COMMIT)
    parser.add_argument("--expected-driver-version", required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--pod")
    parser.add_argument("--pod-uid")
    parser.add_argument("--native-control-dt-s", type=float, required=True)
    return parser


def _git_identity(path: Path, *, expected_commit: str | None = None) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        text=True,
    )
    if tracked_diff:
        raise RuntimeError(f"gate requires a clean tracked checkout: {path}")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"checkout commit {commit} differs from required {expected_commit}: {path}"
        )
    return {"path": str(path.resolve()), "commit": commit, "tracked_diff_empty": True}


def _gpu_identity(*, expected_driver: str, gpu_uuid: str | None) -> dict[str, str]:
    lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    matching = [line for line in lines if gpu_uuid in line] if gpu_uuid else lines
    if len(matching) != 1:
        raise RuntimeError("exactly one assigned GPU must be visible")
    fields = [field.strip() for field in matching[0].split(",")]
    if len(fields) != 4 or fields[3] != expected_driver:
        raise RuntimeError(
            f"GPU driver differs from required {expected_driver}: {matching[0]}"
        )
    return {
        "index": fields[0],
        "uuid": fields[1],
        "name": fields[2],
        "driver_version": fields[3],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.online_correction_v4.model_blind_g3 import (
        canonical_json_bytes,
        sha256_bytes,
    )

    body = canonical_json_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(body),
        "bytes": len(body),
    }


def path_sample_times(planned_duration_s: float, sample_interval_s: float) -> list[float]:
    """Return planned sample times from zero through duration, always including the endpoint."""
    if (
        isinstance(planned_duration_s, bool)
        or not isinstance(planned_duration_s, (int, float))
        or not math.isfinite(float(planned_duration_s))
        or planned_duration_s <= 0
    ):
        raise ValueError("planned_duration_s must be positive and finite")
    if (
        isinstance(sample_interval_s, bool)
        or not isinstance(sample_interval_s, (int, float))
        or not math.isfinite(float(sample_interval_s))
        or sample_interval_s <= 0
    ):
        raise ValueError("sample_interval_s must be positive and finite")
    duration = float(planned_duration_s)
    interval = float(sample_interval_s)
    times: list[float] = []
    current = 0.0
    while current <= duration + 1e-12:
        times.append(round(current, 12))
        current += interval
    if times[-1] < duration - 1e-12:
        times.append(round(duration, 12))
    return times


def motion_onset_s_for_scenario(scenario: str) -> float | None:
    """Return the scheduled event onset for a G3 path scenario, if any."""
    if scenario == "destination_static":
        return None
    return 0.0


def configure_motion_controller(
    scenario: str,
    *,
    displacement_m: float,
    motion_config: Mapping[str, Any],
) -> Any:
    from experiments.online_correction_v4.motion import ReferenceMotionController

    controller = ReferenceMotionController.from_scenario(
        scenario,
        displacement_m=displacement_m,
        motion_config=dict(motion_config),
    )
    onset = motion_onset_s_for_scenario(scenario)
    if onset is not None:
        controller.schedule_event(onset)
    return controller


def expected_reference_world_position(
    *,
    baseline_world: Sequence[float],
    robot_quaternion_wxyz: Sequence[float],
    direction_task: Sequence[float],
    displacement_m: float,
) -> tuple[float, float, float]:
    from experiments.online_correction_v4.model_blind_g2 import quaternion_rotate_wxyz

    task_left = float(direction_task[0])
    task_front = float(direction_task[1])
    norm = math.hypot(task_left, task_front)
    if abs(norm - 1.0) > 1e-6:
        raise ValueError("direction_task must be a unit vector in task coordinates")
    offset_robot = (
        -task_front * float(displacement_m),
        task_left * float(displacement_m),
        0.0,
    )
    offset_world = quaternion_rotate_wxyz(robot_quaternion_wxyz, offset_robot)
    return (
        float(baseline_world[0]) + offset_world[0],
        float(baseline_world[1]) + offset_world[1],
        float(baseline_world[2]) + offset_world[2],
    )


def position_error_m(
    measured: Sequence[float],
    expected: Sequence[float],
) -> float:
    return float(
        math.sqrt(
            sum(
                (float(left) - float(right)) ** 2
                for left, right in zip(measured[:3], expected[:3])
            )
        )
    )


def stationary_drift_m(
    baseline_xyz: Sequence[float],
    current_xyz: Sequence[float],
) -> float:
    return position_error_m(baseline_xyz, current_xyz)


def aggregate_path_check_reasons(reasons: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(reason) for reason in reasons if reason))


def evaluate_path_sample(
    *,
    goal: str,
    geometry: Mapping[str, Any],
    geometry_contract: Mapping[str, Any],
    baseline_reference_world: Sequence[float],
    baseline_target_world: Sequence[float],
    baseline_distractor_world: Sequence[float] | None,
    direction_task: Sequence[float],
    planned_displacement_m: float,
    planned_reference_world: Sequence[float],
    measured_reference_world: Sequence[float],
    measured_target_world: Sequence[float],
    measured_distractor_world: Sequence[float] | None,
    contacts: Mapping[str, Any],
    target_object: str,
) -> tuple[dict[str, Any], list[str]]:
    from experiments.online_correction_v4.droid_g3 import (
        goal_set_for_reference,
        reference_is_supported,
    )

    frame = geometry["frame"]
    table_bounds = geometry["table_bounds_task"]
    reference_footprint = geometry["reference_footprint"]
    clearance_m = float(geometry_contract["relation_clearance_m"])
    edge_margin_m = float(geometry_contract["support_edge_margin_m"])
    pose_error_max_m = float(geometry_contract["reference_pose_error_max_m"])
    drift_max_m = float(geometry_contract["stationary_object_drift_max_m"])

    reasons: list[str] = []
    reference_pose_error_m = position_error_m(
        measured_reference_world, planned_reference_world
    )
    if reference_pose_error_m > pose_error_max_m + 1e-12:
        reasons.append("reference_pose_error_exceeds_contract")
    target_drift_m = stationary_drift_m(baseline_target_world, measured_target_world)
    if target_drift_m > drift_max_m + 1e-12:
        reasons.append("target_drift_exceeds_contract")
    distractor_drift_m = None
    if baseline_distractor_world is not None and measured_distractor_world is not None:
        distractor_drift_m = stationary_drift_m(
            baseline_distractor_world, measured_distractor_world
        )
        if distractor_drift_m > drift_max_m + 1e-12:
            reasons.append("distractor_drift_exceeds_contract")
    path_conformance = (
        reference_pose_error_m <= pose_error_max_m + 1e-12
        and target_drift_m <= drift_max_m + 1e-12
        and (
            distractor_drift_m is None
            or distractor_drift_m <= drift_max_m + 1e-12
        )
    )

    support_valid = bool(contacts.get("support_valid"))
    if not support_valid:
        reasons.append("support_invalid")
    reference_robot_contact = bool(contacts.get("reference_robot_contact"))
    if reference_robot_contact:
        reasons.append("reference_robot_contact")
    unmodeled_collision = bool(contacts.get("unmodeled_collision"))
    if unmodeled_collision:
        reasons.append("unmodeled_collision")

    reference_position = tuple(float(value) for value in measured_reference_world)
    reference_supported = reference_is_supported(
        frame=frame,
        reference_position_world=reference_position,
        table_bounds_task=table_bounds,
        reference_footprint=reference_footprint,
        edge_margin_m=edge_margin_m,
    )
    if not reference_supported:
        reasons.append("reference_not_supported")

    reachable_workspace = reference_supported
    if not reachable_workspace:
        reasons.append("reference_outside_supported_workspace")

    goal_set = goal_set_for_reference(
        geometry=geometry,
        relation=goal,
        reference_position_world=reference_position,
        clearance_m=clearance_m,
    )
    legal_goal_nonempty = not goal_set.empty
    if not legal_goal_nonempty:
        reasons.append("legal_goal_empty")

    sample = {
        "planned_displacement_m": planned_displacement_m,
        "planned_reference_world_xyz_m": [
            float(value) for value in planned_reference_world[:3]
        ],
        "measured_reference_world_xyz_m": [
            float(value) for value in measured_reference_world[:3]
        ],
        "reference_pose_error_m": reference_pose_error_m,
        f"{target_object}_drift_m": target_drift_m,
        "path_conformance": path_conformance,
        "support_valid": support_valid,
        "reference_robot_contact": reference_robot_contact,
        "unmodeled_collision": unmodeled_collision,
        "reference_supported": reference_supported,
        "reachable_workspace": reachable_workspace,
        "legal_goal_nonempty": legal_goal_nonempty,
        "direction_task_coefficients": [float(direction_task[0]), float(direction_task[1])],
    }
    return sample, reasons


def summarize_path_check(
    *,
    goal: str,
    scenario: str,
    planned_duration_s: float,
    sample_interval_s: float,
    sample_records: Sequence[Mapping[str, Any]],
    measured_trajectory_evidence: Mapping[str, Any],
    reference_trajectory_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = aggregate_path_check_reasons(
        reason
        for record in sample_records
        for reason in record.get("reasons", [])
    )
    path_conformance = all(record.get("path_conformance") for record in sample_records)
    collision_free = all(
        not record.get("unmodeled_collision")
        and not record.get("reference_robot_contact")
        for record in sample_records
    )
    support_valid = all(record.get("support_valid") for record in sample_records)
    reachable_workspace = all(record.get("reachable_workspace") for record in sample_records)
    legal_goal_nonempty = all(record.get("legal_goal_nonempty") for record in sample_records)
    reference_robot_contact = any(record.get("reference_robot_contact") for record in sample_records)
    unmodeled_collision = any(record.get("unmodeled_collision") for record in sample_records)
    return {
        "planned_duration_s": planned_duration_s,
        "sample_interval_s": sample_interval_s,
        "sample_count": len(sample_records),
        "measured_pose_evidence": dict(measured_trajectory_evidence),
        "reference_pose_evidence": dict(reference_trajectory_evidence),
        "path_conformance": path_conformance,
        "collision_free": collision_free,
        "support_valid": support_valid,
        "reachable_workspace": reachable_workspace,
        "legal_goal_nonempty": legal_goal_nonempty,
        "reference_robot_contact": reference_robot_contact,
        "unmodeled_collision": unmodeled_collision,
        "reasons": reasons,
    }


def validate_gate_inputs(
    *,
    plan: Mapping[str, Any],
    campaign: Mapping[str, Any],
    campaign_path: Path,
    campaign_sha256: str,
    plan_path: Path,
    plan_sha256: str,
    reset_registry_path: Path,
    reset_registry_sha256: str,
    environment_seed: int,
    scale: float,
    sha256_file: Any,
) -> None:
    from experiments.online_correction_v4.model_blind_g3 import validate_plan_payload

    validate_plan_payload(plan)
    if campaign.get("campaign_id") != plan.get("campaign_id"):
        raise RuntimeError("campaign identity differs from G3 plan")
    source = plan.get("source_identity")
    if not isinstance(source, Mapping):
        raise RuntimeError("G3 plan lacks source identity")
    if sha256_file(plan_path.resolve()) != plan_sha256:
        raise RuntimeError("plan SHA-256 mismatch")
    for label, path, digest in (
        ("campaign", campaign_path, campaign_sha256),
        ("reset_registry", reset_registry_path, reset_registry_sha256),
    ):
        record = source.get(label)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"G3 plan lacks source identity for {label}")
        if sha256_file(path.resolve()) != digest:
            raise RuntimeError(f"{label} SHA-256 mismatch")
        if record.get("sha256") != digest:
            raise RuntimeError(f"{label} digest differs from pinned G3 plan")
    if environment_seed not in {
        int(seed) for seed in plan.get("registered_env_seeds", [])
    }:
        raise RuntimeError("environment seed is absent from G3 plan")
    candidates = plan.get("scale_selection", {}).get("candidate_scales_descending", [])
    if float(scale) not in {float(item) for item in candidates}:
        raise RuntimeError("scale is absent from G3 plan candidates")


def build_goal_area_cases(
    *,
    geometry: Mapping[str, Any],
    baseline_reference_world: Sequence[float],
    direction_by_goal: Mapping[str, Sequence[float]],
    displacement_m: float,
    robot_quaternion_wxyz: Sequence[float],
    clearance_m: float,
    minimum_shrinking_area_fraction: float,
) -> list[dict[str, Any]]:
    from experiments.online_correction_v4.droid_g3 import goal_area_case
    from experiments.online_correction_v4.model_blind_g3 import HORIZONTAL_GOALS

    cases: list[dict[str, Any]] = []
    original = tuple(float(value) for value in baseline_reference_world[:3])
    for goal in HORIZONTAL_GOALS:
        direction = direction_by_goal[str(goal)]
        endpoint = expected_reference_world_position(
            baseline_world=original,
            robot_quaternion_wxyz=robot_quaternion_wxyz,
            direction_task=direction,
            displacement_m=displacement_m,
        )
        cases.append(
            goal_area_case(
                geometry=geometry,
                relation=goal,
                original_reference_world=original,
                endpoint_reference_world=endpoint,
                clearance_m=clearance_m,
                minimum_shrinking_area_fraction=minimum_shrinking_area_fraction,
            )
        )
    return cases


def _check_slug(goal: str, scenario: str) -> str:
    return f"{goal}__{scenario}"


def _run_path_check(
    *,
    env: Any,
    backend: Any,
    output_dir: Path,
    goal: str,
    scenario: str,
    displacement_m: float,
    direction_task: tuple[float, float],
    motion_config: Mapping[str, Any],
    geometry_contract: Mapping[str, Any],
    shared_reset: Mapping[str, Any],
    environment_seed: int,
    scale: float,
    fixture_id: str,
    target_object: str,
    reference_object: str,
    distractor_object: str | None,
    trajectory_schema: str,
    fixture_spec: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from experiments.online_correction_v4.droid_g3 import (
        classify_contacts,
        physics_sampling_stride,
        scenario_duration_s,
    )
    from experiments.online_correction_v4.droid_task_files.constants import ENV_ACTIVE_GOAL

    check_id = (
        f"online-correction-v4-g3-{fixture_id.replace('_', '-')}-{environment_seed}-"
        f"scale-{scale:g}-{goal}-{scenario}"
    )
    env.config.goal = goal
    env.config.episode_id = check_id
    os.environ[ENV_ACTIVE_GOAL] = goal
    backend.restore_g3_state(shared_reset["restore_state"])
    physical_reset = shared_reset["physical_reset"]
    geometry = shared_reset["geometry"]
    baseline_reference = tuple(
        float(value)
        for value in physical_reset["objects"][reference_object]["position_world_xyz_m"]
    )
    baseline_target = tuple(
        float(value)
        for value in physical_reset["objects"][target_object]["position_world_xyz_m"]
    )
    baseline_distractor = (
        tuple(
            float(value)
            for value in physical_reset["objects"][distractor_object][
                "position_world_xyz_m"
            ]
        )
        if distractor_object is not None
        else None
    )
    robot_quaternion = tuple(
        float(value) for value in physical_reset["robot_quaternion_world_wxyz"]
    )

    motion = configure_motion_controller(
        scenario,
        displacement_m=displacement_m,
        motion_config=motion_config,
    )
    planned_duration_s = scenario_duration_s(scenario, motion_config)
    stride, sample_interval_s = physics_sampling_stride(
        backend.physics_dt_s,
        maximum_interval_s=float(geometry_contract["path_sample_max_interval_s"]),
    )
    sample_times = path_sample_times(planned_duration_s, sample_interval_s)
    backend.begin_g3_physics_sampling()

    reference_samples: list[dict[str, Any]] = []
    measured_samples: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    force_threshold = float(geometry_contract["active_contact_force_threshold_n"])

    for index, planned_time_s in enumerate(sample_times):
        state = motion.pose_at(planned_time_s)
        env.set_reference_kinematic_offset(state.displacement_m, direction_task)
        planned_reference = expected_reference_world_position(
            baseline_world=baseline_reference,
            robot_quaternion_wxyz=robot_quaternion,
            direction_task=direction_task,
            displacement_m=state.displacement_m,
        )
        backend.step_g3_physics_substeps(1 if index == 0 else stride)
        scene = backend.g3_scene_state()
        measured_reference = tuple(
            float(value)
            for value in scene["objects"][reference_object]["position_world_xyz_m"]
        )
        measured_target = tuple(
            float(value)
            for value in scene["objects"][target_object]["position_world_xyz_m"]
        )
        measured_distractor = (
            tuple(
                float(value)
                for value in scene["objects"][distractor_object]["position_world_xyz_m"]
            )
            if distractor_object is not None
            else None
        )
        contacts = classify_contacts(
            scene.get("contact_force_n_by_pair", {}),
            active_force_threshold_n=force_threshold,
            fixture_spec=fixture_spec,
        )
        sample_summary, sample_reasons = evaluate_path_sample(
            goal=goal,
            geometry=geometry,
            geometry_contract=geometry_contract,
            baseline_reference_world=baseline_reference,
            baseline_target_world=baseline_target,
            baseline_distractor_world=baseline_distractor,
            direction_task=direction_task,
            planned_displacement_m=state.displacement_m,
            planned_reference_world=planned_reference,
            measured_reference_world=measured_reference,
            measured_target_world=measured_target,
            measured_distractor_world=measured_distractor,
            contacts=contacts,
            target_object=target_object,
        )
        sample_record = {
            "planned_time_s": planned_time_s,
            "physics_time_s": float(scene.get("physics_time_s", 0.0)),
            **sample_summary,
            "reasons": sample_reasons,
            **contacts,
        }
        reference_samples.append(
            {
                "planned_time_s": planned_time_s,
                "planned_displacement_m": state.displacement_m,
                "planned_reference_world_xyz_m": sample_summary[
                    "planned_reference_world_xyz_m"
                ],
            }
        )
        measured_samples.append(sample_record)
        summary_records.append(sample_record)

    check_dir = output_dir / "checks" / _check_slug(goal, scenario)
    reference_record = _write_json(
        check_dir / "reference_trajectory.json",
        {
            "schema_version": trajectory_schema,
            "goal": goal,
            "scenario": scenario,
            "planned_duration_s": planned_duration_s,
            "sample_interval_s": sample_interval_s,
            "sample_count": len(reference_samples),
            "samples": reference_samples,
        },
    )
    measured_record = _write_json(
        check_dir / "measured_trajectory.json",
        {
            "schema_version": trajectory_schema,
            "goal": goal,
            "scenario": scenario,
            "planned_duration_s": planned_duration_s,
            "sample_interval_s": sample_interval_s,
            "sample_count": len(measured_samples),
            "samples": measured_samples,
        },
    )
    observation = summarize_path_check(
        goal=goal,
        scenario=scenario,
        planned_duration_s=planned_duration_s,
        sample_interval_s=sample_interval_s,
        sample_records=summary_records,
        measured_trajectory_evidence=measured_record,
        reference_trajectory_evidence=reference_record,
    )
    return observation, {
        "reset_attestation": shared_reset["reset_attestation_artifact"],
        "physical_reset": shared_reset["physical_reset_artifact"],
        "reference_trajectory": reference_record,
        "measured_trajectory": measured_record,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_root = args.study_root.resolve()
    robolab_root = args.robolab_root.resolve()
    campaign_path = args.campaign.resolve()
    plan_path = args.plan.resolve()
    registry_path = args.reset_registry.resolve()
    if str(study_root) not in sys.path:
        sys.path.insert(0, str(study_root))
    output_raw = args.output_dir or (
        Path(os.environ["EPISODE_OUTPUT_DIR"])
        if os.environ.get("EPISODE_OUTPUT_DIR")
        else None
    )
    if output_raw is None:
        raise RuntimeError("--output-dir or EPISODE_OUTPUT_DIR is required")
    output_dir = output_raw.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite gate output: {output_dir}")
    output_dir.mkdir(parents=True)

    env = None
    try:
        from experiments.online_correction_v4.droid_contract import sha256_bytes
        from experiments.online_correction_v4.droid_robolab import (
            ResetFixtureBinding,
            RoboLabSession,
            build_live_robolab_env,
            close_live_droid_stack,
            write_queue_row,
        )
        from experiments.online_correction_v4.droid_task_files.binding import sha256_file
        from experiments.online_correction_v4.droid_task_files.reset_registry import (
            MODEL_BLIND_CANDIDATE_STATUS,
            load_reset_registry,
        )
        from experiments.online_correction_v4.droid_task_files.constants import (
            fixture_object_spec,
        )
        from experiments.online_correction_v4.droid_g3 import (
            geometry_from_scene_for_fixture,
        )
        from experiments.online_correction_v4.model_blind_g2 import (
            task_frame_evidence,
        )
        from experiments.online_correction_v4.model_blind_g3 import (
            canonical_json_bytes,
            compile_path_seed_receipt,
            expected_path_check_keys,
            validate_path_seed_receipt,
        )

        plan_bytes = plan_path.read_bytes()
        campaign_bytes = campaign_path.read_bytes()
        plan = json.loads(plan_bytes)
        campaign = json.loads(campaign_bytes)
        if canonical_json_bytes(plan) != plan_bytes:
            raise RuntimeError("G3 plan is not canonical JSON")
        validate_gate_inputs(
            plan=plan,
            campaign=campaign,
            campaign_path=campaign_path,
            campaign_sha256=args.campaign_sha256,
            plan_path=plan_path,
            plan_sha256=args.plan_sha256,
            reset_registry_path=registry_path,
            reset_registry_sha256=args.reset_registry_sha256,
            environment_seed=args.environment_seed,
            scale=args.scale,
            sha256_file=sha256_file,
        )
        if sha256_file(registry_path) != args.reset_registry_sha256:
            raise RuntimeError("reset registry SHA-256 mismatch")
        study_identity = _git_identity(
            study_root, expected_commit=args.expected_study_commit
        )
        robolab_identity = _git_identity(
            robolab_root, expected_commit=args.expected_robolab_commit
        )
        gpu_identity = _gpu_identity(
            expected_driver=args.expected_driver_version,
            gpu_uuid=args.gpu_uuid,
        )
        pod_name = args.pod or os.environ.get("POD_NAME")
        pod_uid = args.pod_uid or os.environ.get("POD_UID")
        if not pod_name or not pod_uid:
            raise RuntimeError("--pod/--pod-uid or POD_NAME/POD_UID are required")

        registry = load_reset_registry(
            registry_path=str(registry_path),
            registry_sha256=args.reset_registry_sha256,
            required_status=MODEL_BLIND_CANDIDATE_STATUS,
        )
        if args.environment_seed not in registry.positions_by_env_seed:
            raise RuntimeError("environment seed is absent from reset registry")

        nominal = float(plan["scale_selection"]["nominal_displacement_m"])
        displacement_m = nominal * float(args.scale)
        geometry_contract = dict(plan["geometry_contract"])
        motion_config = dict(campaign["motion"])
        directions = plan["direction_task_coefficients_by_env_seed"][
            str(args.environment_seed)
        ]
        minimum_fraction = float(
            plan["scale_selection"]["minimum_shrinking_area_fraction"]
        )
        clearance_m = float(geometry_contract["relation_clearance_m"])

        fixture_id = args.fixture_id
        fixture_spec = fixture_object_spec(fixture_id)
        prompt = FIXTURE_PROMPTS[fixture_id]
        trajectory_schema = _trajectory_schema(fixture_id)
        infra_failure_schema = _infra_failure_schema(fixture_id)
        target_object = fixture_spec.target_object
        reference_object = fixture_spec.reference_object
        distractor_object = fixture_spec.distractor_object

        episode_id = (
            f"online-correction-v4-g3-{fixture_id.replace('_', '-')}-"
            f"{args.environment_seed}-"
            f"scale-{args.scale:g}"
        )
        prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
        queue_row, queue_row_sha256 = write_queue_row(
            output_dir=output_dir,
            episode_id=episode_id,
            fixture_id=fixture_id,
            prompt_text=prompt,
            prompt_sha256=prompt_sha256,
            env_seed=args.environment_seed,
            goal="left",
        )
        runtime_identity = {
            "study_checkout": study_identity,
            "robolab_checkout": robolab_identity,
            "gpu": gpu_identity,
            "pod": pod_name,
            "pod_uid": pod_uid,
            "gate_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "gate_core_sha256": sha256_file(
                study_root / "experiments/online_correction_v4/model_blind_g3.py"
            ),
            "droid_robolab_sha256": sha256_file(
                study_root / "experiments/online_correction_v4/droid_robolab.py"
            ),
            "droid_g3_sha256": sha256_file(
                study_root / "experiments/online_correction_v4/droid_g3.py"
            ),
            "plan_sha256": args.plan_sha256,
            "campaign_sha256": args.campaign_sha256,
            "reset_registry_sha256": args.reset_registry_sha256,
            "native_control_dt_s": args.native_control_dt_s,
            "scale": float(args.scale),
        }
        runtime_identity_sha256 = sha256_bytes(
            canonical_json_bytes(runtime_identity)
        )
        fixture = ResetFixtureBinding(
            fixture_id=fixture_id,
            reset_registry_sha256=args.reset_registry_sha256,
            reset_registry_uri=f"file://{registry_path}",
        )
        os.environ["V4_DROID_RENDERER"] = "realtime"
        os.environ["V4_DROID_RENDERING_MODE"] = "balanced"
        os.environ["ONLINE_CORRECTION_V4_OUTPUT_DIR"] = str(
            (output_dir / "robolab_native").resolve()
        )
        RoboLabSession.begin_episode(episode_id)
        env = build_live_robolab_env(
            fixture=fixture,
            env_seed=args.environment_seed,
            episode_id=episode_id,
            goal="left",
            prompt_text=prompt,
            prompt_sha256=prompt_sha256,
            policy_id="model_blind_no_policy",
            queue_row_path=queue_row,
            queue_row_sha256=queue_row_sha256,
            output_dir=output_dir / "robolab_native",
            locked_native_control_dt_s=args.native_control_dt_s,
            g3_contact_probe=True,
        )

        reset_check_id = f"{episode_id}-registered-reset"
        reset_attestation, physical_reset = env.reset_for_model_blind_g3(
            check_id=reset_check_id,
            prompt_sha256=prompt_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
        )
        task_frame = task_frame_evidence(physical_reset)
        initial_scene = env.backend.g3_scene_state()
        geometry = geometry_from_scene_for_fixture(
            fixture_id=fixture_id,
            task_frame_evidence=task_frame,
            scene_state=initial_scene,
            support_edge_margin_m=float(
                geometry_contract["support_edge_margin_m"]
            ),
        )
        shared_dir = output_dir / "registered_reset"
        reset_attestation_artifact = _write_json(
            shared_dir / "reset_attestation.json", reset_attestation
        )
        physical_reset_artifact = _write_json(
            shared_dir / "physical_reset.json", physical_reset
        )
        initial_scene_artifact = _write_json(
            shared_dir / "initial_scene.json", initial_scene
        )
        shared_reset = {
            "reset_attestation_artifact": reset_attestation_artifact,
            "physical_reset_artifact": physical_reset_artifact,
            "initial_scene_artifact": initial_scene_artifact,
            "physical_reset": physical_reset,
            "task_frame": task_frame,
            "geometry": geometry,
            "restore_state": env.backend.capture_g3_restore_state(),
        }
        baseline_reference = physical_reset["objects"][reference_object][
            "position_world_xyz_m"
        ]
        robot_quaternion = physical_reset["robot_quaternion_world_wxyz"]
        goal_area_inputs = {
            "geometry": geometry,
            "baseline_reference_world": baseline_reference,
            "robot_quaternion_wxyz": robot_quaternion,
        }

        check_observations: list[dict[str, Any]] = []
        check_artifacts: dict[str, Any] = {}

        for goal, scenario in expected_path_check_keys():
            observation, artifacts = _run_path_check(
                env=env,
                backend=env.backend,
                output_dir=output_dir,
                goal=goal,
                scenario=scenario,
                displacement_m=displacement_m,
                direction_task=(
                    float(directions[goal][0]),
                    float(directions[goal][1]),
                ),
                motion_config=motion_config,
                geometry_contract=geometry_contract,
                shared_reset=shared_reset,
                environment_seed=args.environment_seed,
                scale=float(args.scale),
                fixture_id=fixture_id,
                target_object=target_object,
                reference_object=reference_object,
                distractor_object=distractor_object,
                trajectory_schema=trajectory_schema,
                fixture_spec=fixture_spec,
            )
            check_observations.append(observation)
            check_artifacts[_check_slug(goal, scenario)] = artifacts

        goal_area_cases = build_goal_area_cases(
            geometry=goal_area_inputs["geometry"],
            baseline_reference_world=goal_area_inputs["baseline_reference_world"],
            direction_by_goal={
                goal: directions[goal] for goal in directions
            },
            displacement_m=displacement_m,
            robot_quaternion_wxyz=goal_area_inputs["robot_quaternion_wxyz"],
            clearance_m=clearance_m,
            minimum_shrinking_area_fraction=minimum_fraction,
        )
        plan_receipt = {"path": str(plan_path.resolve()), "sha256": args.plan_sha256}
        receipt = compile_path_seed_receipt(
            plan=plan,
            plan_receipt=plan_receipt,
            environment_seed=args.environment_seed,
            scale=float(args.scale),
            check_observations=check_observations,
            goal_area_cases=goal_area_cases,
        )
        validate_path_seed_receipt(receipt, plan=plan)
        receipt["runtime_identity"] = runtime_identity
        receipt["artifacts"] = {
            "queue_row": {
                "path": str(queue_row.resolve()),
                "sha256": queue_row_sha256,
                "bytes": queue_row.stat().st_size,
            },
            "checks": check_artifacts,
            "registered_reset": {
                "reset_attestation": reset_attestation_artifact,
                "physical_reset": physical_reset_artifact,
                "initial_scene": initial_scene_artifact,
            },
        }
        receipt_record = _write_json(output_dir / "g3_path_seed_receipt.json", receipt)
        print(json.dumps({"passed": receipt["passed"], "receipt": receipt_record}, indent=2))
        return 0
    except Exception as exc:
        failure_fixture = args.fixture_id
        failure = {
            "schema_version": _infra_failure_schema(failure_fixture),
            "campaign_id": "online_correction_v4",
            "fixture_id": failure_fixture,
            "environment_seed": args.environment_seed,
            "scale": args.scale,
            "status": "infrastructure_invalid",
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_dir / "infrastructure_failure.json", failure)
        print(
            f"[V4 {failure_fixture} G3 path] infrastructure failure: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            from experiments.online_correction_v4.droid_robolab import (
                close_live_droid_stack,
            )

            close_live_droid_stack()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
