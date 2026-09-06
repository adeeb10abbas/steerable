#!/usr/bin/env python3
"""Run privileged grasp/transport/release qualification for C8."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.second_stack import (  # noqa: E402
    CUBE_HALF_EXTENT_M,
    ENV_NAME,
    REFERENCE_OBJECT,
    RELATION_AXES_SCENE_XY,
    SOURCE_OBJECT,
    active_contact_pairs,
    apply_registered_reset,
    ensure_registered_support,
    set_reference_xy,
    unwrap_simpler_env,
)
from tools.run_v4_second_stack_g2 import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    sha256_file,
    verify_external_stack,
)
from tools.run_v4_second_stack_g3_path import minimum_jerk_fraction  # noqa: E402


class SecondStackG3ScriptedError(RuntimeError):
    pass


EXTREME_CRITERIA = (
    ("minimum_x", lambda row: row[1]),
    ("maximum_x", lambda row: -row[1]),
    ("minimum_y", lambda row: row[2]),
    ("maximum_y", lambda row: -row[2]),
    ("minimum_x_plus_y", lambda row: row[1] + row[2]),
    ("maximum_x_plus_y", lambda row: -(row[1] + row[2])),
    ("minimum_x_minus_y", lambda row: row[1] - row[2]),
    ("maximum_x_minus_y", lambda row: -(row[1] - row[2])),
)

POSITION_FRACTIONS = {
    "original": 0.0,
    "midpoint": 0.5,
    "endpoint": 1.0,
}
GOAL_CENTER_OFFSET_M = 0.09
RELATION_CLEARANCE_M = 0.01
PLACEMENT_XY_TOLERANCE_M = 0.015
REFERENCE_POSITION_TOLERANCE_M = 0.003
STABILITY_TOLERANCE_M = 0.005
SUPPORT_Z_TOLERANCE_M = 0.003


def select_extreme_seeds(
    resets: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = sorted(
        (
            int(seed),
            float(row["jitter_scene_xy_m"][0]),
            float(row["jitter_scene_xy_m"][1]),
        )
        for seed, row in resets.items()
    )
    if len(rows) < 9:
        raise SecondStackG3ScriptedError("C8 requires at least nine reset rows")
    selected = [{"label": "canonical", "environment_seed": rows[0][0]}]
    used = {rows[0][0]}
    for label, key in EXTREME_CRITERIA:
        candidate = next(row for row in sorted(rows, key=key) if row[0] not in used)
        used.add(candidate[0])
        selected.append({"label": label, "environment_seed": candidate[0]})
    if len(selected) != 9:
        raise SecondStackG3ScriptedError("C8 extreme reset selection is incomplete")
    return selected


def _distance(first: Any, second: Any) -> float:
    return math.sqrt(
        sum((float(left) - float(right)) ** 2 for left, right in zip(first, second))
    )


def _position_at_fraction(
    plan_row: Mapping[str, Any],
    fraction: float,
) -> list[float]:
    start = plan_row["initial_reference_scene_xy_m"]
    end = plan_row["endpoint_reference_scene_xy_m"]
    return [
        float(start[index]) + fraction * (float(end[index]) - float(start[index]))
        for index in range(2)
    ]


def _robot_contact(
    records: list[dict[str, Any]],
    object_name: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if object_name in {row["actor0"], row["actor1"]}
        and any(
            token in {row["actor0"], row["actor1"]}
            for token in ("left_finger_link", "right_finger_link", "fingers_link")
        )
    ]


def _run_check(
    *,
    env: Any,
    env_seed: int,
    reset_row: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    position_label: str,
    moving_reference: bool,
) -> dict[str, Any]:
    import numpy as np
    import sapien.core as sapien

    env.reset(seed=env_seed)
    apply_registered_reset(env, reset_row, settle_steps=30)
    raw = unwrap_simpler_env(env)
    ensure_registered_support(env)
    arm = raw.agent.controller.controllers["arm"]
    gripper = raw.agent.controller.controllers["gripper"]
    orientation = raw.agent.ee_pose.q.copy()
    links = {link.name: link for link in raw.agent.robot.get_links()}
    left_finger = links["left_finger_link"]
    right_finger = links["right_finger_link"]
    forbidden_reference_contacts: list[dict[str, Any]] = []
    ik_failures: list[list[float]] = []

    def record_contacts() -> list[dict[str, Any]]:
        contacts = active_contact_pairs(env)
        for contact in contacts:
            actors = {contact["actor0"], contact["actor1"]}
            if REFERENCE_OBJECT in actors and any(
                "link" in actor and actor != REFERENCE_OBJECT for actor in actors
            ):
                forbidden_reference_contacts.append(contact)
        return contacts

    def simulate(steps: int) -> None:
        for _ in range(steps):
            raw.agent.before_simulation_step()
            raw._scene.step()
            raw._after_simulation_step()
            record_contacts()

    def move_once(command: list[float], steps: int = 140) -> bool:
        target = sapien.Pose(p=command, q=orientation)
        target_at_base = raw.agent.robot.pose.inv().transform(target)
        joint_target = arm.compute_ik(target_at_base)
        if joint_target is None:
            ik_failures.append(list(command))
            return False
        arm.set_drive_targets(joint_target)
        simulate(steps)
        return True

    def align(
        desired: list[float],
        initial_command: list[float],
        *,
        iterations: int = 6,
    ) -> float | None:
        command = list(initial_command)
        for _ in range(iterations):
            if not move_once(command):
                return None
            actual = raw.agent.ee_pose.p
            command = [
                command[index] + desired[index] - float(actual[index])
                for index in range(3)
            ]
        return _distance(raw.agent.ee_pose.p, desired)

    def slow_move(
        desired: list[float],
        *,
        segments: int = 8,
    ) -> float | None:
        start = [float(value) for value in raw.agent.ee_pose.p]
        for segment in range(1, segments + 1):
            fraction = segment / segments
            waypoint = [
                start[index] + fraction * (desired[index] - start[index])
                for index in range(3)
            ]
            if not move_once(waypoint, steps=100):
                return None
        return align(desired, desired, iterations=3)

    def align_gripper_to_source(
        source_position: list[float],
        *,
        finger_x_offset_m: float,
        finger_y_offset_m: float,
        finger_z_offset_m: float,
    ) -> float | None:
        command = [
            source_position[0],
            source_position[1] - 0.013,
            0.89,
        ]
        for _ in range(10):
            if not move_once(command, steps=160):
                return None
            live_source = raw.episode_source_obj.pose.p
            desired_midpoint = [
                float(live_source[0]) + finger_x_offset_m,
                float(live_source[1]) + finger_y_offset_m,
                float(live_source[2]) + finger_z_offset_m,
            ]
            midpoint = [
                0.5
                * (
                    float(left_finger.pose.p[index])
                    + float(right_finger.pose.p[index])
                )
                for index in range(3)
            ]
            command = [
                command[index] + desired_midpoint[index] - midpoint[index]
                for index in range(3)
            ]
        live_source = raw.episode_source_obj.pose.p
        desired_midpoint = [
            float(live_source[0]) + finger_x_offset_m,
            float(live_source[1]) + finger_y_offset_m,
            float(live_source[2]) + finger_z_offset_m,
        ]
        midpoint = [
            0.5
            * (
                float(left_finger.pose.p[index])
                + float(right_finger.pose.p[index])
            )
            for index in range(3)
        ]
        return _distance(midpoint, desired_midpoint)

    def set_gripper(value: float, steps: int) -> None:
        gripper.set_action(np.asarray([value], dtype=np.float32))
        simulate(steps)

    fraction = POSITION_FRACTIONS[position_label]
    initial_reference_xy = _position_at_fraction(plan_row, fraction)
    endpoint_reference_xy = _position_at_fraction(plan_row, 1.0)
    set_reference_xy(env, initial_reference_xy)
    simulate(30)
    source_initial = [float(value) for value in raw.episode_source_obj.pose.p]
    reference_initial = [float(value) for value in raw.episode_target_obj.pose.p]
    final_reference_xy = (
        endpoint_reference_xy if moving_reference else initial_reference_xy
    )
    axis = RELATION_AXES_SCENE_XY[str(plan_row["relation"])]
    destination_xy = [
        final_reference_xy[index] + GOAL_CENTER_OFFSET_M * axis[index]
        for index in range(2)
    ]

    grasp_attempts: list[dict[str, Any]] = []
    grasp_contacts: list[dict[str, Any]] = []
    grasp_alignment_error: float | None = None
    lift_alignment_error: float | None = None
    source_lifted = list(source_initial)
    lift_height = 0.0
    grasp_offsets = (
        (0.0143, -0.004, 0.0585),
        (0.0143, 0.0, 0.0585),
        (0.0143, -0.008, 0.0585),
        (0.0143, -0.004, 0.0555),
        (0.0143, -0.004, 0.0615),
        (0.0103, -0.004, 0.0585),
        (0.0183, -0.004, 0.0585),
        (0.0143, -0.010, 0.0555),
        (0.0143, 0.002, 0.0555),
        (0.0063, -0.004, 0.0585),
        (0.0223, -0.004, 0.0585),
    )
    for attempt, (
        finger_x_offset,
        finger_y_offset,
        finger_z_offset,
    ) in enumerate(grasp_offsets):
        set_gripper(0.8, 100 if attempt else 60)
        live_source = [float(value) for value in raw.episode_source_obj.pose.p]
        grasp_alignment_error = align_gripper_to_source(
            live_source,
            finger_x_offset_m=finger_x_offset,
            finger_y_offset_m=finger_y_offset,
            finger_z_offset_m=finger_z_offset,
        )
        set_gripper(-1.0, 180)
        seated_ee = [float(value) for value in raw.agent.ee_pose.p]
        move_once(
            [seated_ee[0], seated_ee[1], seated_ee[2] - 0.004],
            steps=80,
        )
        simulate(80)
        grasp_contacts = _robot_contact(record_contacts(), SOURCE_OBJECT)
        current_ee = [float(value) for value in raw.agent.ee_pose.p]
        lift_alignment_error = slow_move(
            [current_ee[0], current_ee[1], 1.01],
            segments=8,
        )
        source_lifted = [float(value) for value in raw.episode_source_obj.pose.p]
        lift_height = source_lifted[2] - source_initial[2]
        retention_errors: list[float | None] = []
        if lift_height >= 0.04:
            retained_ee = [float(value) for value in raw.agent.ee_pose.p]
            retention_errors.append(
                slow_move(
                    [
                        retained_ee[0] + 0.02 * axis[0],
                        retained_ee[1] + 0.02 * axis[1],
                        retained_ee[2],
                    ],
                    segments=4,
                )
            )
            retention_errors.append(slow_move(retained_ee, segments=4))
            simulate(50)
            source_lifted = [
                float(value) for value in raw.episode_source_obj.pose.p
            ]
            lift_height = source_lifted[2] - source_initial[2]
        grasp_attempts.append(
            {
                "attempt_index": attempt,
                "finger_x_offset_m": finger_x_offset,
                "finger_y_offset_m": finger_y_offset,
                "finger_z_offset_m": finger_z_offset,
                "alignment_error_m": grasp_alignment_error,
                "contact_count": len(grasp_contacts),
                "lift_height_m": lift_height,
                "retention_alignment_errors_m": retention_errors,
            }
        )
        if lift_height >= 0.04:
            break

    if moving_reference:
        start = list(initial_reference_xy)
        for index in range(51):
            progress = minimum_jerk_fraction(index / 50)
            set_reference_xy(
                env,
                [
                    start[axis]
                    + progress * (endpoint_reference_xy[axis] - start[axis])
                    for axis in range(2)
                ],
            )
            simulate(5)
    held_offset = [
        source_lifted[index] - float(raw.agent.ee_pose.p[index])
        for index in range(3)
    ]
    desired_release_ee = [
        destination_xy[0] - held_offset[0],
        destination_xy[1] - held_offset[1],
        0.905 - held_offset[2],
    ]
    transport_alignment_error = slow_move(
        [desired_release_ee[0], desired_release_ee[1], 1.01],
        segments=12,
    )
    release_alignment_error = slow_move(
        desired_release_ee,
        segments=8,
    )
    placement_correction_errors: list[float | None] = []
    for _ in range(3):
        live_source = [float(value) for value in raw.episode_source_obj.pose.p]
        if math.hypot(
            live_source[0] - destination_xy[0],
            live_source[1] - destination_xy[1],
        ) <= 0.005:
            break
        live_ee = [float(value) for value in raw.agent.ee_pose.p]
        placement_correction_errors.append(
            align(
                [
                    live_ee[0] + destination_xy[0] - live_source[0],
                    live_ee[1] + destination_xy[1] - live_source[1],
                    live_ee[2],
                ],
                live_ee,
                iterations=3,
            )
        )
    source_before_release = [float(value) for value in raw.episode_source_obj.pose.p]
    set_gripper(1.0, 180)
    align(
        [desired_release_ee[0], desired_release_ee[1], 0.98],
        [desired_release_ee[0], desired_release_ee[1], 0.97],
    )
    simulate(100)
    source_settled = [float(value) for value in raw.episode_source_obj.pose.p]
    reference_settled = [float(value) for value in raw.episode_target_obj.pose.p]
    stability_start = list(source_settled)
    simulate(100)
    source_final = [float(value) for value in raw.episode_source_obj.pose.p]
    final_contacts = record_contacts()

    projected_half_extent = CUBE_HALF_EXTENT_M * (
        abs(axis[0]) + abs(axis[1])
    )
    signed_relation_clearance = (
        (source_final[0] - reference_settled[0]) * axis[0]
        + (source_final[1] - reference_settled[1]) * axis[1]
        - 2.0 * projected_half_extent
    )
    placement_xy_error = math.hypot(
        source_final[0] - destination_xy[0],
        source_final[1] - destination_xy[1],
    )
    reference_position_error = math.hypot(
        reference_settled[0] - final_reference_xy[0],
        reference_settled[1] - final_reference_xy[1],
    )
    stability_drift = _distance(stability_start, source_final)
    support_height_error = abs(
        source_final[2] - (float(raw.scene_table_height) + CUBE_HALF_EXTENT_M)
    )
    released_robot_contacts = _robot_contact(final_contacts, SOURCE_OBJECT)
    alignment_errors = (
        grasp_alignment_error,
        lift_alignment_error,
        transport_alignment_error,
        release_alignment_error,
    )
    passed = (
        not ik_failures
        and grasp_alignment_error is not None
        and grasp_alignment_error <= 0.01
        and all(
            error is not None and error <= 0.04
            for error in alignment_errors[1:]
        )
        and all(
            error is not None and error <= 0.04
            for error in retention_errors
        )
        and all(
            error is not None and error <= 0.04
            for error in placement_correction_errors
        )
        and lift_height >= 0.04
        and placement_xy_error <= PLACEMENT_XY_TOLERANCE_M
        and signed_relation_clearance >= RELATION_CLEARANCE_M
        and reference_position_error <= REFERENCE_POSITION_TOLERANCE_M
        and stability_drift <= STABILITY_TOLERANCE_M
        and support_height_error <= SUPPORT_Z_TOLERANCE_M
        and not released_robot_contacts
        and not forbidden_reference_contacts
    )
    return {
        "environment_seed": env_seed,
        "relation": plan_row["relation"],
        "physical_translation_sign": plan_row["physical_translation_sign"],
        "reference_position_label": position_label,
        "moving_reference": moving_reference,
        "passed": passed,
        "initial_source_scene_m": source_initial,
        "initial_reference_scene_m": reference_initial,
        "final_reference_target_scene_xy_m": final_reference_xy,
        "goal_center_scene_xy_m": destination_xy,
        "lifted_source_scene_m": source_lifted,
        "source_before_release_scene_m": source_before_release,
        "final_source_scene_m": source_final,
        "final_reference_scene_m": reference_settled,
        "lift_height_m": lift_height,
        "placement_xy_error_m": placement_xy_error,
        "signed_relation_clearance_m": signed_relation_clearance,
        "reference_position_error_m": reference_position_error,
        "post_release_stability_drift_m": stability_drift,
        "support_height_error_m": support_height_error,
        "alignment_errors_m": {
            "grasp": grasp_alignment_error,
            "lift": lift_alignment_error,
            "transport": transport_alignment_error,
            "release": release_alignment_error,
        },
        "grasp_contact_count": len(grasp_contacts),
        "grasp_attempts": grasp_attempts,
        "placement_correction_errors_m": placement_correction_errors,
        "released_robot_contacts": released_robot_contacts,
        "forbidden_reference_contacts": forbidden_reference_contacts,
        "ik_failures": ik_failures,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def run_scripted_gate(
    *,
    registry_path: Path,
    plan_path: Path,
    path_receipt_path: Path,
    integration_root: Path,
    max_checks: int | None = None,
    retry_failures_path: Path | None = None,
) -> dict[str, Any]:
    registry = load_json(registry_path)
    plan = load_json(plan_path)
    path_receipt = load_json(path_receipt_path)
    analytical_scale = float(plan["selected_analytical_scale"])
    selected_scale = float(path_receipt.get("scale", float("nan")))
    if (
        path_receipt.get("passed") is not True
        or not math.isfinite(selected_scale)
        or selected_scale > analytical_scale
        or path_receipt.get("reset_registry", {}).get("sha256")
        != sha256_file(registry_path)
        or path_receipt.get("geometry_plan", {}).get("sha256")
        != sha256_file(plan_path)
    ):
        raise SecondStackG3ScriptedError("C8 passing path receipt binding differs")
    stack_receipt = verify_external_stack(
        integration_root=integration_root,
        registry=registry,
    )
    sys.path.insert(0, str(integration_root))
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

    register_simpler_envs()
    import gymnasium as gym
    import numpy
    import sapien

    expected_runtime = registry["external_stack_identity"]["runtime_dependencies"]
    observed_runtime = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "gymnasium": gym.__version__,
        "sapien": sapien.__version__,
    }
    if (
        observed_runtime["numpy"] != expected_runtime["numpy"]
        or observed_runtime["sapien"] != expected_runtime["sapien"]
        or not observed_runtime["python"].startswith(expected_runtime["python"])
    ):
        raise SecondStackG3ScriptedError("C8 scripted runtime identity differs")

    selected = next(
        row for row in plan["scales"] if float(row["scale"]) == selected_scale
    )
    checks_by_key = {
        (int(row["environment_seed"]), str(row["relation"])): row
        for row in selected["checks"]
    }
    resets = registry["resets_by_env_seed"]
    extreme_resets = select_extreme_seeds(resets)
    specifications: list[tuple[int, str, str, bool]] = []
    for selected_reset in extreme_resets:
        env_seed = int(selected_reset["environment_seed"])
        for relation in RELATION_AXES_SCENE_XY:
            for position_label in POSITION_FRACTIONS:
                specifications.append((env_seed, relation, position_label, False))
    canonical_seed = int(extreme_resets[0]["environment_seed"])
    for relation in RELATION_AXES_SCENE_XY:
        specifications.append((canonical_seed, relation, "original", True))
    if len(specifications) != 112:
        raise SecondStackG3ScriptedError("C8 scripted plan must contain 112 checks")
    if max_checks is not None and retry_failures_path is not None:
        raise SecondStackG3ScriptedError(
            "max_checks and retry_failures_path are mutually exclusive"
        )
    if retry_failures_path is not None:
        prior = load_json(retry_failures_path)
        if (
            float(prior.get("selected_scale", float("nan"))) != selected_scale
            or prior.get("model_request_count") != 0
            or prior.get("geometry_plan", {}).get("sha256")
            != sha256_file(plan_path)
            or prior.get("path_receipt", {}).get("sha256")
            != sha256_file(path_receipt_path)
        ):
            raise SecondStackG3ScriptedError(
                "C8 failed-check retry receipt binding differs"
            )
        failed_specifications = {
            (
                int(row["environment_seed"]),
                str(row["relation"]),
                str(row["reference_position_label"]),
                bool(row["moving_reference"]),
            )
            for row in prior.get("records", [])
            if row.get("passed") is False
        }
        if not failed_specifications:
            raise SecondStackG3ScriptedError(
                "C8 failed-check retry receipt has no failed checks"
            )
        specifications = [
            row for row in specifications if row in failed_specifications
        ]
        if len(specifications) != len(failed_specifications):
            raise SecondStackG3ScriptedError(
                "C8 failed-check retry receipt contains unknown checks"
            )
    if max_checks is not None:
        specifications = specifications[:max_checks]

    env = gym.make(ENV_NAME)
    records: list[dict[str, Any]] = []
    try:
        env.reset(seed=canonical_seed)
        ensure_registered_support(env)
        for env_seed, relation, position_label, moving in specifications:
            records.append(
                _run_check(
                    env=env,
                    env_seed=env_seed,
                    reset_row=resets[str(env_seed)],
                    plan_row=checks_by_key[(env_seed, relation)],
                    position_label=position_label,
                    moving_reference=moving,
                )
            )
    finally:
        env.close()
    complete = max_checks is None and retry_failures_path is None
    passed = complete and len(records) == 112 and all(row["passed"] for row in records)
    return {
        "schema_version": "v4-second-stack-g3-scripted-aggregate-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": "second_stack",
        "family_id": "C8",
        "status": "passed" if passed else "smoke_only" if not complete else "failed",
        "passed": passed,
        "qualification_scope": "confirmatory",
        "selected_scale": selected_scale,
        "analytical_selected_scale": analytical_scale,
        "selected_displacement_m": float(selected["displacement_m"]),
        "expected_check_count": 112,
        "observed_check_count": len(records),
        "passed_check_count": sum(row["passed"] for row in records),
        "stationary_check_count": sum(not row["moving_reference"] for row in records),
        "moving_check_count": sum(row["moving_reference"] for row in records),
        "extreme_reset_selection": extreme_resets,
        "runtime_dependencies": observed_runtime,
        "stack_receipt": stack_receipt,
        "reset_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "geometry_plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "path_receipt": {
            "path": str(path_receipt_path),
            "sha256": sha256_file(path_receipt_path),
        },
        "thresholds": {
            "goal_center_offset_m": GOAL_CENTER_OFFSET_M,
            "relation_clearance_m": RELATION_CLEARANCE_M,
            "placement_xy_tolerance_m": PLACEMENT_XY_TOLERANCE_M,
            "reference_position_tolerance_m": REFERENCE_POSITION_TOLERANCE_M,
            "stability_tolerance_m": STABILITY_TOLERANCE_M,
            "support_z_tolerance_m": SUPPORT_Z_TOLERANCE_M,
        },
        "records": records,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "debug_selection": {
            "max_checks": max_checks,
            "retry_failures_path": (
                str(retry_failures_path)
                if retry_failures_path is not None
                else None
            ),
        },
        "release_boundary": (
            "A passing receipt completes C8 G3 physical feasibility only. It does "
            "not authorize policy inference until G4-G6 and release-lock closure."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--path-receipt", type=Path, required=True)
    parser.add_argument("--integration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-checks", type=int)
    parser.add_argument("--retry-failures-from", type=Path)
    args = parser.parse_args()
    payload = run_scripted_gate(
        registry_path=args.registry,
        plan_path=args.plan,
        path_receipt_path=args.path_receipt,
        integration_root=args.integration_root,
        max_checks=args.max_checks,
        retry_failures_path=args.retry_failures_from,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "path": str(args.output),
                "status": payload["status"],
                "observed_check_count": payload["observed_check_count"],
                "passed_check_count": payload["passed_check_count"],
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return (
        0
        if payload["passed"]
        or args.max_checks is not None
        or args.retry_failures_from is not None
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
