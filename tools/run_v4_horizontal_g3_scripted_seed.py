#!/usr/bin/env python3
"""Run one zero-policy V4 horizontal G3 scripted seed on RoboLab."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence

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
SCRIPTED_MODES = ("stationary", "moving")
STATIONARY_CHECK_COUNT = 12
MOVING_CHECK_COUNT = 4


def _seed_receipt_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-scripted-seed-receipt-v1"


def _infra_failure_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-infrastructure-failure-v1"


SEED_RECEIPT_SCHEMA = _seed_receipt_schema("horizontal")
INFRA_FAILURE_SCHEMA = _infra_failure_schema("horizontal")

_PATH_SEED_MODULE: Any | None = None


def _path_seed_module() -> Any:
    global _PATH_SEED_MODULE
    if _PATH_SEED_MODULE is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_v4_horizontal_g3_path_seed",
            ROOT / "tools/run_v4_horizontal_g3_path_seed.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to load horizontal G3 path-seed helpers")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PATH_SEED_MODULE = module
    return _PATH_SEED_MODULE


FROZEN_SCRIPTED_CONTROLLER_CONFIG: dict[str, Any] = {
    "phase_ticks": {
        "approach": 30,
        "descend": 30,
        "close_dwell": 30,
        "lift": 30,
        "transport": 30,
        "place_descend": 30,
        "open_dwell": 30,
        "retreat": 30,
        "settle": 15,
    },
    "geometry_offsets": {
        "approach_height_m": 0.12,
        "descend_offset_m": 0.025,
        "lift_height_m": 0.12,
        "transport_height_m": 0.12,
        "place_descend_offset_m": 0.04,
        "retreat_height_m": 0.10,
        "target_inset_m": 0.015,
    },
    "gripper_close": 0.785398,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--robolab-root", type=Path)
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
    parser.add_argument("--mode", choices=SCRIPTED_MODES, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-study-commit")
    parser.add_argument("--expected-robolab-commit", default=ROBOLAB_COMMIT)
    parser.add_argument("--expected-driver-version")
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


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
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


def reference_displacement_m(reference_position: str, displacement_m: float) -> float:
    """Return the kinematic reference offset for a scripted check position."""
    if reference_position == "original":
        return 0.0
    if reference_position == "midpoint":
        return float(displacement_m) / 2.0
    if reference_position == "endpoint":
        return float(displacement_m)
    raise ValueError(f"unsupported reference position {reference_position!r}")


def check_slug(goal: str, reference_position: str) -> str:
    return f"{goal}__{reference_position}"


def expected_scripted_seed_checks(
    *,
    mode: str,
    plan: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return goal/reference pairs for one scripted seed job in declared order."""
    from experiments.online_correction_v4.model_blind_g3 import (
        HORIZONTAL_GOALS,
        REFERENCE_POSITIONS,
    )

    if mode not in SCRIPTED_MODES:
        raise ValueError(f"unsupported scripted mode {mode!r}")
    scripted = plan.get("scripted_controller")
    if not isinstance(scripted, Mapping):
        raise RuntimeError("G3 plan lacks scripted checks")
    if mode == "stationary":
        positions = scripted.get("stationary", {}).get("reference_positions")
        if list(positions) != list(REFERENCE_POSITIONS):
            raise RuntimeError("G3 stationary scripted reference positions differ")
        return tuple(
            (goal, position)
            for goal in HORIZONTAL_GOALS
            for position in REFERENCE_POSITIONS
        )
    moving = scripted.get("moving")
    if not isinstance(moving, Mapping):
        raise RuntimeError("G3 plan lacks moving scripted checks")
    if moving.get("scenario") != "move_stop":
        raise RuntimeError("G3 moving scripted scenario differs")
    if moving.get("goals") != list(HORIZONTAL_GOALS):
        raise RuntimeError("G3 moving scripted goals differ")
    return tuple((goal, "endpoint") for goal in HORIZONTAL_GOALS)


def validate_scripted_seed_gate_inputs(
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
    mode: str,
    expected_fixture_id: str = "horizontal",
    sha256_file: Any,
) -> None:
    if plan.get("fixture_id") != expected_fixture_id:
        raise RuntimeError("G3 plan fixture differs from expected fixture")
    _path_seed_module().validate_gate_inputs(
        plan=plan,
        campaign=campaign,
        campaign_path=campaign_path,
        campaign_sha256=campaign_sha256,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        reset_registry_path=reset_registry_path,
        reset_registry_sha256=reset_registry_sha256,
        environment_seed=environment_seed,
        scale=scale,
        sha256_file=sha256_file,
    )
    if mode not in SCRIPTED_MODES:
        raise RuntimeError(f"unsupported scripted mode {mode!r}")
    scripted = plan.get("scripted_controller")
    if not isinstance(scripted, Mapping):
        raise RuntimeError("G3 plan lacks scripted checks")
    reset_seeds = scripted.get("reset_env_seeds")
    if not isinstance(reset_seeds, list):
        raise RuntimeError("G3 scripted reset selection differs")
    canonical_seed = scripted.get("moving", {}).get("canonical_env_seed")
    if mode == "stationary":
        if environment_seed not in {int(seed) for seed in reset_seeds}:
            raise RuntimeError(
                "stationary scripted mode requires one of the selected reset env seeds"
            )
    elif environment_seed != int(canonical_seed):
        raise RuntimeError("moving scripted mode requires the canonical env seed")


def frozen_scripted_controller_config(fixture_id: str = "horizontal") -> dict[str, Any]:
    config = json.loads(json.dumps(FROZEN_SCRIPTED_CONTROLLER_CONFIG))
    if fixture_id == "object_pair":
        # RoboLab's absolute IK controls the Robotiq base flange, while the
        # object-pair waypoints describe the center of its finger-pad grasp
        # region (14 cm below the flange), not the 16.28 cm fingertip edge.
        config["eef_tool_length_m"] = 0.14
        # The measured C7 placement bias is 24.2 mm toward the goal boundary;
        # retain 40 mm of legal-region inset for the qualification rerun.
        config["geometry_offsets"]["target_inset_m"] = 0.04
        # Correct horizontal IK drift during descent using the privileged
        # object pose; this signal remains confined to the G3 controller.
        config["place_xy_feedback_gain"] = 1.0
        config["place_xy_feedback_max_m"] = 0.08
    return config


def build_moving_reference_motion_callback(
    env: Any,
    *,
    displacement_m: float,
    direction_task: tuple[float, float],
    motion_config: Mapping[str, Any],
    object_grabbed_probe: Callable[[], bool],
    set_reference_offset: Callable[[float, tuple[float, float]], None],
) -> tuple[Callable[[int, float], None], dict[str, Any]]:
    """Build a callback that starts move_stop motion once the object is grabbed."""
    motion = _path_seed_module().configure_motion_controller(
        "move_stop",
        displacement_m=displacement_m,
        motion_config=motion_config,
    )
    state = {"motion_started": False, "motion_origin_sim_time_s": None}

    def callback(_tick: int, sim_time_s: float) -> None:
        if not state["motion_started"] and object_grabbed_probe():
            state["motion_started"] = True
            state["motion_origin_sim_time_s"] = float(sim_time_s)
            motion.schedule_event(0.0)
        if state["motion_started"] and state["motion_origin_sim_time_s"] is not None:
            local_time = float(sim_time_s) - float(state["motion_origin_sim_time_s"])
            pose = motion.pose_at(local_time)
            set_reference_offset(pose.displacement_m, direction_task)
        else:
            set_reference_offset(0.0, direction_task)

    return callback, state


def compile_scripted_seed_summary(
    *,
    mode: str,
    environment_seed: int,
    scale: float,
    displacement_m: float,
    plan_sha256: str,
    campaign_sha256: str,
    reset_registry_sha256: str,
    runtime_identity: Mapping[str, Any],
    controller_config: Mapping[str, Any],
    check_records: Sequence[Mapping[str, Any]],
    registered_reset: Mapping[str, Any],
    fixture_id: str = "horizontal",
) -> dict[str, Any]:
    expected_count = STATIONARY_CHECK_COUNT if mode == "stationary" else MOVING_CHECK_COUNT
    if len(check_records) != expected_count:
        raise RuntimeError(
            f"scripted seed summary requires exactly {expected_count} check records"
        )
    passed_count = sum(1 for record in check_records if record.get("passed") is True)
    failed_count = len(check_records) - passed_count
    return {
        "schema_version": _seed_receipt_schema(fixture_id),
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
        "mode": mode,
        "environment_seed": environment_seed,
        "scale": float(scale),
        "displacement_m": float(displacement_m),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "controller_config": dict(controller_config),
        "plan_sha256": plan_sha256,
        "campaign_sha256": campaign_sha256,
        "reset_registry_sha256": reset_registry_sha256,
        "runtime_identity": dict(runtime_identity),
        "check_order": ["goal_declared_order", "reference_position_declared_order"],
        "check_count": expected_count,
        "passed_check_count": passed_count,
        "failed_check_count": failed_count,
        "passed": failed_count == 0,
        "checks": list(check_records),
        "registered_reset": dict(registered_reset),
    }


def _direction_task(
    plan: Mapping[str, Any], *, environment_seed: int, goal: str
) -> tuple[float, float]:
    directions = plan["direction_task_coefficients_by_env_seed"][str(environment_seed)]
    raw = directions[goal]
    return (float(raw[0]), float(raw[1]))


def _run_stationary_check(
    *,
    env: Any,
    backend: Any,
    output_dir: Path,
    goal: str,
    reference_position: str,
    displacement_m: float,
    direction_task: tuple[float, float],
    geometry: Mapping[str, Any],
    geometry_contract: Mapping[str, Any],
    shared_reset: Mapping[str, Any],
    environment_seed: int,
    scale: float,
    controller_config: Mapping[str, Any],
    fixture_id: str = "horizontal",
    target_object: str = "rubiks_cube",
    reference_object: str = "bowl",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from experiments.online_correction_v4.droid_g3 import goal_set_for_reference
    from experiments.online_correction_v4.droid_g3_scripted import run_scripted_check
    from experiments.online_correction_v4.droid_task_files.constants import ENV_ACTIVE_GOAL
    from experiments.online_correction_v4.model_blind_g3 import (
        compile_scripted_check_receipt,
        validate_scripted_check_receipt,
    )

    slug = check_slug(goal, reference_position)
    check_id = (
        f"online-correction-v4-g3-scripted-stationary-"
        f"{environment_seed}-scale-{scale:g}-{slug}"
    )
    env.config.goal = goal
    env.config.episode_id = check_id
    os.environ[ENV_ACTIVE_GOAL] = goal
    backend.restore_g3_state(shared_reset["restore_state"])
    offset_m = reference_displacement_m(reference_position, displacement_m)
    env.set_reference_kinematic_offset(offset_m, direction_task)
    scene = backend.g3_scene_state()
    reference_position_world = tuple(
        float(value)
        for value in scene["objects"][reference_object]["position_world_xyz_m"]
    )
    clearance_m = float(geometry_contract["relation_clearance_m"])
    goal_set = goal_set_for_reference(
        geometry=geometry,
        relation=goal,
        reference_position_world=reference_position_world,
        clearance_m=clearance_m,
    )
    table_bounds = geometry["table_bounds_task"]
    target_footprint = geometry["target_footprint"]
    trajectory_result = run_scripted_check(
        env,
        target_object=target_object,
        reference_object=reference_object,
        relation=goal,
        goal=goal_set,
        frame=geometry["frame"],
        config=controller_config,
        table_top_z_task=float(table_bounds.z_max),
        object_half_up=float(target_footprint.half_up),
        fixture_id=fixture_id,
    )
    trajectory_result["controller_config"] = dict(controller_config)
    trajectory_result["check_kind"] = "stationary"
    trajectory_result["environment_seed"] = environment_seed
    trajectory_result["reference_position"] = reference_position
    trajectory_result["scale"] = float(scale)
    trajectory_result["displacement_m"] = float(displacement_m)
    trajectory_result["reference_offset_m"] = offset_m
    trajectory_result["direction_task_coefficients"] = [
        float(direction_task[0]),
        float(direction_task[1]),
    ]
    trajectory_record = _write_json_exclusive(
        output_dir / "trajectories" / f"{slug}.json",
        trajectory_result,
    )
    receipt = compile_scripted_check_receipt(
        check_kind="stationary",
        environment_seed=environment_seed,
        goal=goal,
        reference_position=reference_position,
        scale=float(scale),
        displacement_m=float(displacement_m),
        observation={
            **trajectory_result["stages"],
            "evidence": trajectory_record,
            "reasons": list(trajectory_result.get("reasons") or []),
            "passed": trajectory_result.get("passed"),
        },
        fixture_id=fixture_id,
    )
    validate_scripted_check_receipt(receipt)
    receipt_record = _write_json_exclusive(
        output_dir / "receipts" / f"{slug}.json",
        receipt,
    )
    check_record = {
        "goal": goal,
        "reference_position": reference_position,
        "passed": receipt["passed"],
        "trajectory": trajectory_record,
        "receipt": receipt_record,
    }
    return receipt, check_record, trajectory_result


def _run_moving_check(
    *,
    env: Any,
    backend: Any,
    output_dir: Path,
    goal: str,
    displacement_m: float,
    direction_task: tuple[float, float],
    geometry: Mapping[str, Any],
    geometry_contract: Mapping[str, Any],
    shared_reset: Mapping[str, Any],
    environment_seed: int,
    scale: float,
    controller_config: Mapping[str, Any],
    motion_config: Mapping[str, Any],
    fixture_id: str = "horizontal",
    target_object: str = "rubiks_cube",
    reference_object: str = "bowl",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from experiments.online_correction_v4.droid_g3 import goal_set_for_reference
    from experiments.online_correction_v4.droid_g3_scripted import run_scripted_check
    from experiments.online_correction_v4.droid_task_files.constants import ENV_ACTIVE_GOAL
    from experiments.online_correction_v4.model_blind_g3 import (
        compile_scripted_check_receipt,
        validate_scripted_check_receipt,
    )
    reference_position = "endpoint"
    slug = check_slug(goal, reference_position)
    check_id = (
        f"online-correction-v4-g3-scripted-moving-"
        f"{environment_seed}-scale-{scale:g}-{slug}"
    )
    env.config.goal = goal
    env.config.episode_id = check_id
    os.environ[ENV_ACTIVE_GOAL] = goal
    backend.restore_g3_state(shared_reset["restore_state"])
    env.set_reference_kinematic_offset(0.0, direction_task)
    physical_reset = shared_reset["physical_reset"]
    baseline_reference = tuple(
        float(value)
        for value in physical_reset["objects"][reference_object]["position_world_xyz_m"]
    )
    robot_quaternion = tuple(
        float(value) for value in physical_reset["robot_quaternion_world_wxyz"]
    )
    endpoint_reference = _path_seed_module().expected_reference_world_position(
        baseline_world=baseline_reference,
        robot_quaternion_wxyz=robot_quaternion,
        direction_task=direction_task,
        displacement_m=displacement_m,
    )
    clearance_m = float(geometry_contract["relation_clearance_m"])
    goal_set = goal_set_for_reference(
        geometry=geometry,
        relation=goal,
        reference_position_world=endpoint_reference,
        clearance_m=clearance_m,
    )
    probe = env.backend.modules["object_grabbed"]

    def object_grabbed_probe() -> bool:
        return bool(probe(env.backend.env, object=target_object, env_id=0))

    callback, motion_state = build_moving_reference_motion_callback(
        env,
        displacement_m=displacement_m,
        direction_task=direction_task,
        motion_config=motion_config,
        object_grabbed_probe=object_grabbed_probe,
        set_reference_offset=env.set_reference_kinematic_offset,
    )
    table_bounds = geometry["table_bounds_task"]
    target_footprint = geometry["target_footprint"]
    trajectory_result = run_scripted_check(
        env,
        target_object=target_object,
        reference_object=reference_object,
        relation=goal,
        goal=goal_set,
        frame=geometry["frame"],
        config=controller_config,
        table_top_z_task=float(table_bounds.z_max),
        object_half_up=float(target_footprint.half_up),
        reference_motion_callback=callback,
        fixture_id=fixture_id,
    )
    trajectory_result["controller_config"] = dict(controller_config)
    trajectory_result["check_kind"] = "moving"
    trajectory_result["environment_seed"] = environment_seed
    trajectory_result["reference_position"] = reference_position
    trajectory_result["scale"] = float(scale)
    trajectory_result["displacement_m"] = float(displacement_m)
    trajectory_result["direction_task_coefficients"] = [
        float(direction_task[0]),
        float(direction_task[1]),
    ]
    trajectory_result["moving_motion_state"] = dict(motion_state)
    trajectory_record = _write_json_exclusive(
        output_dir / "trajectories" / f"{slug}.json",
        trajectory_result,
    )
    receipt = compile_scripted_check_receipt(
        check_kind="moving",
        environment_seed=environment_seed,
        goal=goal,
        reference_position=reference_position,
        scale=float(scale),
        displacement_m=float(displacement_m),
        observation={
            **trajectory_result["stages"],
            "evidence": trajectory_record,
            "reasons": list(trajectory_result.get("reasons") or []),
            "passed": trajectory_result.get("passed"),
        },
        fixture_id=fixture_id,
    )
    validate_scripted_check_receipt(receipt)
    receipt_record = _write_json_exclusive(
        output_dir / "receipts" / f"{slug}.json",
        receipt,
    )
    check_record = {
        "goal": goal,
        "reference_position": reference_position,
        "passed": receipt["passed"],
        "trajectory": trajectory_record,
        "receipt": receipt_record,
    }
    return receipt, check_record, trajectory_result


def _write_infrastructure_failure(
    output_dir: Path, failure: Mapping[str, Any]
) -> None:
    infra_path = output_dir / "infrastructure_failure.json"
    if infra_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing infrastructure failure: {infra_path}"
        )
    _write_json_exclusive(infra_path, failure)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_root = args.study_root.resolve()
    robolab_root = (args.robolab_root or Path(os.environ["ROBOLAB_ROOT"])).resolve()
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
        from experiments.online_correction_v4.droid_g3 import (
            fixture_object_spec,
            geometry_from_scene_for_fixture,
            horizontal_geometry_from_scene,
            task_frame_from_evidence,
        )
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
        from experiments.online_correction_v4.model_blind_g2 import task_frame_evidence
        from experiments.online_correction_v4.model_blind_g3 import canonical_json_bytes

        plan_bytes = plan_path.read_bytes()
        campaign_bytes = campaign_path.read_bytes()
        plan = json.loads(plan_bytes)
        campaign = json.loads(campaign_bytes)
        if canonical_json_bytes(plan) != plan_bytes:
            raise RuntimeError("G3 plan is not canonical JSON")
        validate_scripted_seed_gate_inputs(
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
            mode=args.mode,
            expected_fixture_id=args.fixture_id,
            sha256_file=sha256_file,
        )
        if sha256_file(registry_path) != args.reset_registry_sha256:
            raise RuntimeError("reset registry SHA-256 mismatch")
        expected_checks = expected_scripted_seed_checks(plan=plan, mode=args.mode)
        if args.expected_study_commit:
            study_identity = _git_identity(
                study_root, expected_commit=args.expected_study_commit
            )
        else:
            study_identity = _git_identity(study_root)
        if args.expected_robolab_commit:
            robolab_identity = _git_identity(
                robolab_root, expected_commit=args.expected_robolab_commit
            )
        else:
            robolab_identity = _git_identity(robolab_root)
        if not args.expected_driver_version:
            raise RuntimeError("--expected-driver-version is required for live execution")
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
            expected_fixture_id=args.fixture_id,
        )
        if args.environment_seed not in registry.positions_by_env_seed:
            raise RuntimeError("environment seed is absent from reset registry")

        nominal = float(plan["scale_selection"]["nominal_displacement_m"])
        displacement_m = nominal * float(args.scale)
        geometry_contract = dict(plan["geometry_contract"])
        motion_config = dict(campaign["motion"])
        fixture_id = args.fixture_id
        controller_config = frozen_scripted_controller_config(fixture_id)
        fixture_spec = fixture_object_spec(fixture_id)
        prompt = FIXTURE_PROMPTS[fixture_id]
        target_object = fixture_spec.target_object
        reference_object = fixture_spec.reference_object

        episode_id = (
            f"online-correction-v4-g3-scripted-{fixture_id.replace('_', '-')}-"
            f"{args.mode}-{args.environment_seed}-scale-{args.scale:g}"
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
            "droid_g3_scripted_sha256": sha256_file(
                study_root / "experiments/online_correction_v4/droid_g3_scripted.py"
            ),
            "plan_sha256": args.plan_sha256,
            "campaign_sha256": args.campaign_sha256,
            "reset_registry_sha256": args.reset_registry_sha256,
            "native_control_dt_s": args.native_control_dt_s,
            "scale": float(args.scale),
            "mode": args.mode,
            "fixture_id": fixture_id,
        }
        runtime_identity_sha256 = sha256_bytes(canonical_json_bytes(runtime_identity))
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
            action_mode="absolute_ik",
        )

        reset_check_id = f"{episode_id}-registered-reset"
        reset_attestation, physical_reset = env.reset_for_model_blind_g3(
            check_id=reset_check_id,
            prompt_sha256=prompt_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
        )
        task_frame_dict = task_frame_evidence(physical_reset)
        task_frame_from_evidence(task_frame_dict)
        initial_scene = env.backend.g3_scene_state()
        if fixture_id == "horizontal":
            geometry = horizontal_geometry_from_scene(
                task_frame_evidence=task_frame_dict,
                scene_state=initial_scene,
                support_edge_margin_m=float(geometry_contract["support_edge_margin_m"]),
            )
        else:
            geometry = geometry_from_scene_for_fixture(
                fixture_id=fixture_id,
                task_frame_evidence=task_frame_dict,
                scene_state=initial_scene,
                support_edge_margin_m=float(geometry_contract["support_edge_margin_m"]),
            )
        shared_dir = output_dir / "registered_reset"
        reset_attestation_artifact = _write_json_exclusive(
            shared_dir / "reset_attestation.json", reset_attestation
        )
        physical_reset_artifact = _write_json_exclusive(
            shared_dir / "physical_reset.json", physical_reset
        )
        initial_scene_artifact = _write_json_exclusive(
            shared_dir / "initial_scene.json", initial_scene
        )
        shared_reset = {
            "reset_attestation_artifact": reset_attestation_artifact,
            "physical_reset_artifact": physical_reset_artifact,
            "initial_scene_artifact": initial_scene_artifact,
            "physical_reset": physical_reset,
            "geometry": geometry,
            "restore_state": env.backend.capture_g3_restore_state(),
        }

        check_records: list[dict[str, Any]] = []
        for goal, reference_position in expected_checks:
            direction_task = _direction_task(
                plan, environment_seed=args.environment_seed, goal=goal
            )
            if args.mode == "stationary":
                _receipt, check_record, _trajectory = _run_stationary_check(
                    env=env,
                    backend=env.backend,
                    output_dir=output_dir,
                    goal=goal,
                    reference_position=reference_position,
                    displacement_m=displacement_m,
                    direction_task=direction_task,
                    geometry=geometry,
                    geometry_contract=geometry_contract,
                    shared_reset=shared_reset,
                    environment_seed=args.environment_seed,
                    scale=float(args.scale),
                    controller_config=controller_config,
                    fixture_id=fixture_id,
                    target_object=target_object,
                    reference_object=reference_object,
                )
            else:
                _receipt, check_record, _trajectory = _run_moving_check(
                    env=env,
                    backend=env.backend,
                    output_dir=output_dir,
                    goal=goal,
                    displacement_m=displacement_m,
                    direction_task=direction_task,
                    geometry=geometry,
                    geometry_contract=geometry_contract,
                    shared_reset=shared_reset,
                    environment_seed=args.environment_seed,
                    scale=float(args.scale),
                    controller_config=controller_config,
                    motion_config=motion_config,
                    fixture_id=fixture_id,
                    target_object=target_object,
                    reference_object=reference_object,
                )
            check_records.append(check_record)

        summary = compile_scripted_seed_summary(
            mode=args.mode,
            environment_seed=args.environment_seed,
            scale=float(args.scale),
            displacement_m=displacement_m,
            plan_sha256=args.plan_sha256,
            campaign_sha256=args.campaign_sha256,
            reset_registry_sha256=args.reset_registry_sha256,
            runtime_identity=runtime_identity,
            controller_config=controller_config,
            check_records=check_records,
            registered_reset={
                "reset_attestation": reset_attestation_artifact,
                "physical_reset": physical_reset_artifact,
                "initial_scene": initial_scene_artifact,
            },
            fixture_id=fixture_id,
        )
        summary_record = _write_json_exclusive(
            output_dir / "g3_scripted_seed_receipt.json", summary
        )
        print(
            json.dumps(
                {"passed": summary["passed"], "receipt": summary_record},
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        failure_fixture = args.fixture_id
        failure = {
            "schema_version": _infra_failure_schema(failure_fixture),
            "campaign_id": "online_correction_v4",
            "fixture_id": failure_fixture,
            "mode": args.mode,
            "environment_seed": args.environment_seed,
            "scale": args.scale,
            "status": "infrastructure_invalid",
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_infrastructure_failure(output_dir, failure)
        print(
            f"[V4 {failure_fixture} G3 scripted] infrastructure failure: {exc}",
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
