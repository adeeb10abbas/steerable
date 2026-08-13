from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.v3.phase_e.canonical_stage_localization_v3e006.ood_reference import (
    _quat_normalize_wxyz,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r009.pinch_geometry import (
    collision_center_env_local,
    pinch_alignment_command,
    pinch_geometry,
    reconstruct_collision_bounds_env_local,
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r009.predecessor_contract import (
    validate_r008_exhaustion_closure,
)
from tools.validate_v3e006_r009 import (
    ValidationError,
    canonical_sha,
    slerp,
    validate_candidate_state,
    validate_construction_lifecycle,
    validate_static,
    validate_terminal_diagnostic_pattern,
)


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009"
R008 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name
    )
    return ast.dump(node, include_attributes=False)


def test_real_r008_closure_and_mutations() -> None:
    value = load(R008 / "results/results.json")
    validate_r008_exhaustion_closure(value)
    for key, replacement in (
        ("candidate_pair_evaluation_count", 3),
        ("accepted_candidate_rank", 1),
        ("model_request_count", 1),
    ):
        bad = deepcopy(value)
        bad[key] = replacement
        with pytest.raises(ValueError):
            validate_r008_exhaustion_closure(bad)
    bad = deepcopy(value)
    bad["raw_result"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_r008_exhaustion_closure(bad)


def test_registration_is_finite_prospective_and_preserves_r008_targets() -> None:
    registration = load(ART / "repair_registration.json")
    schedule = load(ART / "gates/candidate_schedule.json")
    predecessor = load(R008 / "gates/candidate_schedule.json")
    assert registration["predecessor_repair_amendment_id"] == "V3-E006-R008"
    assert registration["counts_at_registration"] == {
        "r009_live_diagnostics": 0,
        "r009_live_candidate_evaluations": 0,
        "model_requests": 0,
        "behavioral_episodes": 0,
    }
    assert schedule["candidate_budget"] == schedule["diagnostic_budget"] == 4
    assert [row["candidate_rank"] for row in schedule["candidate_pairs"]] == [1, 2, 3, 4]
    assert schedule["known_reachable_diagnostics"] == predecessor["known_reachable_diagnostics"]
    for actual, old in zip(schedule["candidate_pairs"], predecessor["candidate_pairs"], strict=True):
        assert actual["construction_method"] == (
            "exact_reset_uniform_collision_pinch_acquire_lift_transport_q_handoff"
        )
        for stage in ("canonical_grasp", "canonical_carry"):
            assert actual[stage]["r009_target_cube_pose"] == old[stage]["r008_target_cube_pose"]
            assert actual[stage]["r009_target_cube_pose"]["position_world_m"][1] == 0.0
            stripped = deepcopy(actual[stage])
            for key in (
                "r009_target_cube_pose",
                "r009_acquisition_base_quaternion_world_wxyz",
                "r009_final_base_quaternion_world_wxyz",
            ):
                stripped.pop(key)
            assert stripped == old[stage]


def test_uniform_pinch_math_is_capped_symmetric_and_sign_invariant() -> None:
    geometry = pinch_geometry(
        left_center_env_local=[0.30, -0.04, 0.10],
        right_center_env_local=[0.30, 0.04, 0.10],
    )
    assert geometry["pinch_midpoint_env_local_m"] == [0.3, 0.0, 0.1]
    assert np.allclose(geometry["pinch_axis_left_to_right_env_local"], [0.0, 1.0, 0.0])
    kwargs = dict(
        live_base_position_env_local=[0.0, 0.0, 0.0],
        live_base_quaternion=[1.0, 0.0, 0.0, 0.0],
        live_left_center_env_local=[0.30, -0.04, 0.10],
        live_right_center_env_local=[0.30, 0.04, 0.10],
        target_pinch_midpoint_env_local=[0.31, 0.0, 0.12],
        target_base_quaternion=[0.0, 0.0, 1.0, 0.0],
        translation_gain=0.2,
        rotation_gain=0.2,
        translation_cap_m_per_step=0.002,
        rotation_cap_deg_per_step=2.0,
    )
    value = pinch_alignment_command(**kwargs)
    flipped = pinch_alignment_command(
        **{**kwargs, "target_base_quaternion": [0.0, 0.0, -1.0, 0.0]}
    )
    assert np.linalg.norm(value["applied_translation_correction_env_local_m"]) <= 0.002 + 1e-15
    assert np.linalg.norm(value["applied_rotation_correction_env_local_rotvec_rad"]) <= np.deg2rad(2.0) + 1e-15
    assert np.allclose(
        value["command_base_position_env_local_m"],
        flipped["command_base_position_env_local_m"],
    )
    assert abs(
        np.dot(
            value["command_base_quaternion_world_wxyz"],
            flipped["command_base_quaternion_world_wxyz"],
        )
    ) >= 1.0 - 1e-12
    assert np.allclose(
        collision_center_env_local(
            body_position_env_local=[1.0, 2.0, 3.0],
            body_quaternion_world_wxyz=[1.0, 0.0, 0.0, 0.0],
            collision_center_body=[0.1, -0.2, 0.3],
        ),
        [1.1, 1.8, 3.3],
    )


def test_pinch_contract_and_mutations() -> None:
    contract = load(ART / "gates/candidate_schedule.json")["pinch_geometry_contract"]
    validate_contract(contract)
    for key, value in (
        ("translation_gain", 0.3),
        ("open_approach_steps", 179),
        ("contact_or_grab_conditioned_branch", True),
        ("early_stop", True),
    ):
        bad = deepcopy(contract)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_contract(bad)


def _box_corners(center: list[float], half: list[float]) -> list[list[float]]:
    center_array = np.asarray(center, dtype=np.float64)
    half_array = np.asarray(half, dtype=np.float64)
    return [
        (center_array + np.asarray([x, y, z]) * half_array).tolist()
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ]


def _static_geometry(body: str, collision: str, center: list[float], half: list[float]) -> dict:
    corners = _box_corners(center, half)
    minimum = (np.asarray(center) - np.asarray(half)).tolist()
    maximum = (np.asarray(center) + np.asarray(half)).tolist()
    value = {
        "body_prim_path": body,
        "body_has_rigid_body_api": True,
        "all_collision_prims_owned_by_exact_body_without_nested_boundary": True,
        "collision_prim_paths": [collision],
        "shape_local_geometry": [{
            "collision_prim_path": collision,
            "collision_prim_local_minimum_m": minimum,
            "collision_prim_local_maximum_m": maximum,
            "collision_corners_body_m": corners,
        }],
        "collision_corners_body_m": corners,
        "minimum_body_m": minimum,
        "maximum_body_m": maximum,
        "center_body_m": center,
        "half_extents_body_m": half,
        "derivation": "one-time USD local collision bounds transformed into owning rigid-body coordinates before candidate actions; never polled dynamically",
    }
    value["canonical_sha256"] = canonical_sha(value)
    return value


def _live_geometry(position: list[float], quaternion: list[float], origin: list[float], static: dict) -> dict:
    bounds = reconstruct_collision_bounds_env_local(
        body_position_env_local=position,
        body_quaternion_world_wxyz=quaternion,
        collision_corners_body=static["collision_corners_body_m"],
        collision_center_body=static["center_body_m"],
    )
    return {
        "live_tensor_pose": {
            "position_tensor_world_m": (np.asarray(position) + np.asarray(origin)).tolist(),
            "scene_env_origin_world_m": origin,
            "position_env_local_m": position,
            "quaternion_world_wxyz": quaternion,
            "position_semantics": "env-local world-axis = tensor world position - scene env origin",
        },
        "static_body_local_geometry_sha256": static["canonical_sha256"],
        "reconstructed_bounds_env_local": bounds,
    }


def _frame() -> dict:
    return {
        "passed": True,
        "position_composition_residual_m": 0.0,
        "orientation_composition_residual_deg": 0.0,
    }


def _contact() -> dict:
    return {"contact_force_n": {"gripper__rubiks_cube": 0.0}, "object_grabbed": False}


def _row(
    *,
    phase: str,
    step: int,
    action: list[float] | None,
    base_p: list[float],
    base_q: list[float],
    cube_p: list[float],
    cube_q: list[float],
    joint_q: list[float],
    env_origin: list[float],
) -> dict:
    value = {
        "phase": phase,
        "phase_step_one_based": step,
        "eef_position_world_m": (np.asarray(base_p) + np.asarray(env_origin)).tolist(),
        "eef_quaternion_world_wxyz": base_q,
        "base_link_position_world_m": base_p,
        "base_link_quaternion_world_wxyz": base_q,
        "base_link_to_eef_frame_identity": _frame(),
        "joint_position_rad": joint_q,
        "joint_velocity_rad_s": [0.0] * 13,
        "cube_position_world_m": (np.asarray(cube_p) + np.asarray(env_origin)).tolist(),
        "cube_position_env_local_m": cube_p,
        "cube_quaternion_world_wxyz": cube_q,
        "cube_linear_velocity_m_s": [0.0] * 3,
        "cube_angular_velocity_rad_s": [0.0] * 3,
    }
    if action is not None:
        value["command_action_8d"] = action
    return value


def _candidate_fixture() -> tuple[dict, dict, dict, dict]:
    schedule = load(ART / "gates/candidate_schedule.json")
    stage = schedule["candidate_pairs"][0]["canonical_grasp"]
    contract = schedule["pinch_geometry_contract"]
    reset_p = [0.303, 0.0, 0.081]
    reset_q = [1.0, 0.0, 0.0, 0.0]
    env_origin = [10.0, 20.0, 30.0]
    reset_center = np.asarray(reset_p, dtype=np.float64)
    half = np.asarray([0.03, 0.03, 0.03], dtype=np.float64)
    local_center = [0.0, 0.0, 0.0]
    target_pose = [
        *stage["r009_target_cube_pose"]["position_world_m"],
        *stage["r009_target_cube_pose"]["quaternion_world_wxyz"],
    ]
    target_center = collision_center_env_local(
        body_position_env_local=target_pose[:3],
        body_quaternion_world_wxyz=target_pose[3:],
        collision_center_body=local_center,
    )
    lift_center = reset_center.copy()
    lift_center[2] = target_center[2]
    acquisition_q = stage["r009_acquisition_base_quaternion_world_wxyz"]
    final_q = stage["r009_final_base_quaternion_world_wxyz"]
    base_p = [0.30, 0.0, 0.20]
    base_q = [1.0, 0.0, 0.0, 0.0]
    cube_p, cube_q = reset_p.copy(), reset_q.copy()
    joint_q = [0.0] * 13
    left_body = "/World/envs/env_0/robot/left_inner_finger"
    right_body = "/World/envs/env_0/robot/right_inner_finger"
    cube_body = "/World/envs/env_0/rubiks_cube"
    left_collision = left_body + "/collision"
    right_collision = right_body + "/collision"
    cube_collision = cube_body + "/collision"
    static = {
        "left": _static_geometry(left_body, left_collision, [0.0, -0.04, 0.0], [0.01] * 3),
        "right": _static_geometry(right_body, right_collision, [0.0, 0.04, 0.0], [0.01] * 3),
        "cube": _static_geometry(cube_body, cube_collision, local_center, half.tolist()),
    }
    trace: list[dict] = []
    contacts: list[dict] = []
    phases = (
        ("open_approach", 180, 0.0),
        ("open_descent", 180, 0.0),
        ("normal_close", 120, 1.0),
        ("closed_vertical_lift", 240, 1.0),
        ("closed_stage_transport", 300, 1.0),
    )
    for phase, steps, grip in phases:
        for step in range(1, steps + 1):
            fraction = step / steps
            live_geometry = {
                "left": _live_geometry(base_p, base_q, env_origin, static["left"]),
                "right": _live_geometry(base_p, base_q, env_origin, static["right"]),
                "cube": _live_geometry(cube_p, cube_q, env_origin, static["cube"]),
            }
            left_center = live_geometry["left"]["reconstructed_bounds_env_local"]["collision_center_env_local_m"]
            right_center = live_geometry["right"]["reconstructed_bounds_env_local"]["collision_center_env_local_m"]
            cube_bounds = live_geometry["cube"]["reconstructed_bounds_env_local"]
            if phase == "open_approach":
                target_mid = np.asarray(cube_bounds["collision_center_env_local_m"]) + [0.0, 0.0, 2 * cube_bounds["aabb_half_extents_env_local_m"][2]]
                target_q = acquisition_q
                live = True
            elif phase in {"open_descent", "normal_close"}:
                target_mid = np.asarray(cube_bounds["collision_center_env_local_m"])
                target_q = acquisition_q
                live = True
            elif phase == "closed_vertical_lift":
                target_mid = (1 - fraction) * reset_center + fraction * lift_center
                target_q = acquisition_q
                live = False
            else:
                target_mid = (1 - fraction) * lift_center + fraction * target_center
                target_q = slerp(acquisition_q, final_q, fraction)
                live = False
            command = pinch_alignment_command(
                live_base_position_env_local=base_p,
                live_base_quaternion=base_q,
                live_left_center_env_local=left_center,
                live_right_center_env_local=right_center,
                target_pinch_midpoint_env_local=target_mid,
                target_base_quaternion=target_q,
                translation_gain=contract["translation_gain"],
                rotation_gain=contract["rotation_gain"],
                translation_cap_m_per_step=contract["translation_cap_m_per_step"],
                rotation_cap_deg_per_step=contract["rotation_cap_deg_per_step"],
            )
            action = np.concatenate(
                (
                    command["command_base_position_env_local_m"],
                    _quat_normalize_wxyz(command["command_base_quaternion_world_wxyz"]),
                    [grip],
                )
            ).astype(np.float32).tolist()
            pre = {
                "live_base_position_env_local_m": base_p,
                "live_base_quaternion_world_wxyz": base_q,
                "live_cube_position_env_local_m": cube_p,
                "live_cube_quaternion_world_wxyz": cube_q,
                "live_tensor_collision_geometry": live_geometry,
                "coordinate_semantics": "all controller positions are env-local world-axis coordinates; each retained tensor pose binds the subtracted scene env origin",
                "target_pinch_midpoint_env_local_m": target_mid.tolist(),
                "target_base_quaternion_world_wxyz": np.asarray(target_q).tolist(),
                "command_uses_live_cube_collision_center": live,
                "pinch_alignment_command": command,
                "gripper_command": grip,
                "contact_and_grab_diagnostic_before_action": _contact(),
            }
            base_p = action[:3]
            base_q = action[3:7]
            row = _row(
                phase=phase,
                step=step,
                action=action,
                base_p=base_p,
                base_q=base_q,
                cube_p=cube_p,
                cube_q=cube_q,
                joint_q=joint_q,
                env_origin=env_origin,
            )
            row["pre_action_pinch_geometry"] = pre
            row["contact_and_grab_diagnostic_after_action"] = _contact()
            trace.append(row)
            contacts.append(
                deepcopy({
                    "phase": phase,
                    "phase_step_one_based": step,
                    "before": pre["contact_and_grab_diagnostic_before_action"],
                    "after": row["contact_and_grab_diagnostic_after_action"],
                })
            )
    captured_q = joint_q.copy()
    for step in range(1, 601):
        row = _row(
            phase="captured_q_normal_joint_settle",
            step=step,
            action=None,
            base_p=base_p,
            base_q=base_q,
            cube_p=cube_p,
            cube_q=cube_q,
            joint_q=joint_q,
            env_origin=env_origin,
        )
        row["normal_joint_position_target_rad"] = captured_q
        row["cartesian_action_manager_applied"] = False
        row["contact_and_grab_diagnostic_after_step"] = _contact()
        trace.append(row)
        contacts.append(
            deepcopy({
                "phase": "captured_q_normal_joint_settle",
                "phase_step_one_based": step,
                "after": row["contact_and_grab_diagnostic_after_step"],
            })
        )
    samples = []
    for row in trace[-10:]:
        samples.append(
            {
                "cube_position_world_m": row["cube_position_world_m"],
                "cube_linear_velocity_m_s": row["cube_linear_velocity_m_s"],
                "cube_angular_velocity_rad_s": row["cube_angular_velocity_rad_s"],
                "eef_position_world_m": row["base_link_position_world_m"],
                "base_link_quaternion_world_wxyz": row["base_link_quaternion_world_wxyz"],
                "live_eef_frame_position_world_m": row["eef_position_world_m"],
                "live_eef_frame_quaternion_world_wxyz": row["eef_quaternion_world_wxyz"],
                "base_link_to_eef_frame_identity": row["base_link_to_eef_frame_identity"],
                "arm_joint_velocity_rad_s": row["joint_velocity_rad_s"][:7],
                **_contact(),
            }
        )
    inventory = {
        "robot_root_prim_path": "/World/envs/env_0/robot",
        "cube_root_prim_path": cube_body,
        "left_inner_finger_body_prim_path": left_body,
        "right_inner_finger_body_prim_path": right_body,
        "left_collision_prim_paths": [left_collision],
        "right_collision_prim_paths": [right_collision],
        "cube_collision_prim_paths": [cube_collision],
        "left_robot_body_tensor_name": "left_inner_finger",
        "left_robot_body_tensor_index": 8,
        "right_robot_body_tensor_name": "right_inner_finger",
        "right_robot_body_tensor_index": 9,
        "cube_tensor_source": "rubiks_cube.data.root_pos_w/root_quat_w",
    }
    registered = deepcopy(stage)
    registered["candidate_rank"] = 1
    construction = {
        "method": "exact_reset_uniform_collision_pinch_acquire_lift_transport_q_handoff",
        "stage": "canonical_grasp",
        "candidate_rank": 1,
        "registered_stage_schedule": registered,
        "pinch_geometry_contract": contract,
        "joint_handoff_contract": schedule["joint_handoff_contract"],
        "construction_lifecycle_contract": schedule["construction_lifecycle_contract"],
        "phase_steps": {
            "open_approach": 180,
            "open_descent": 180,
            "normal_close": 120,
            "closed_vertical_lift": 240,
            "closed_stage_transport": 300,
            "captured_q_normal_joint_settle": 600,
        },
        "collision_geometry_resolution": {
            "inventory": inventory,
            "static_body_local_collision_geometry": static,
            "scene_env_origin_world_m_at_resolution": env_origin,
            "dynamic_geometry_source": "IsaacLab tensor rigid-body/root poses minus explicit scene env origin",
            "dynamic_usd_world_bounds_used": False,
        },
        "reset_cube_pose_env_local_wxyz": [*reset_p, *reset_q],
        "reset_cube_collision_center_env_local_m": reset_p,
        "reset_cube_collision_half_extents_env_local_m": half.tolist(),
        "cube_collision_center_in_cube_m": local_center,
        "target_cube_pose_env_local_wxyz": target_pose,
        "target_cube_collision_center_env_local_m": target_center.tolist(),
        "lift_collision_center_env_local_m": lift_center.tolist(),
        "registered_approach_clearance_m": 2 * half[2],
        "episode_length_buf_before_candidate_actions": [75],
        "episode_length_buf_before_handoff": [1095],
        "episode_length_buf_after_candidate_actions": [1695],
        "post_reset_joint_state_write_count": 0,
        "post_reset_object_state_write_count": 0,
        "joint_or_object_state_write_count": 0,
        "contact_or_grab_conditioned_branch_count": 0,
        "all_registered_phases_executed_unconditionally": True,
        "captured_joint_target_write_count": 1,
        "cartesian_action_manager_apply_count_during_joint_settle": 0,
        "construction_action_trace": trace,
        "acquisition_lift_transport_trace": trace[:1020],
        "contact_and_grab_trace_diagnostic_only": contacts,
        "captured_joint_position_target_rad": captured_q,
        "settled_gate_samples": samples,
    }
    state = {
        "construction": construction,
        "robot": {"joint_position_rad": joint_q, "joint_velocity_rad_s": [0.0] * 13},
        "objects": {
            "rubiks_cube": {
                "position_world_m": (np.asarray(cube_p) + np.asarray(env_origin)).tolist(),
                "quaternion_world_wxyz": cube_q,
                "linear_velocity_m_s": [0.0] * 3,
                "angular_velocity_rad_s": [0.0] * 3,
            }
        },
        "eef": {"position_world_m": base_p, "quaternion_world_wxyz": base_q},
        "contact_evidence": {
            "settled_force_snapshots_n": [row["contact_force_n"] for row in samples],
            "object_grabbed_by_step": [row["object_grabbed"] for row in samples],
        },
        "physics_gate": {"settled_window_steps": 10, "passed": False},
        "ood_gate": {"passed": False},
        "camera_evidence": {"passed": True},
        "companion_pose_gate": {"passed": True},
        "passed": False,
    }
    fresh_reset = {
        "eef": {"position_world_m": [0.30, 0.0, 0.20], "quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "objects": {"rubiks_cube": {"position_world_m": (np.asarray(reset_p) + np.asarray(env_origin)).tolist(), "quaternion_world_wxyz": reset_q}},
        "base_link_to_eef_frame_identity": {**_frame(), "scene_env_origin_world_m": env_origin},
    }
    return state, stage, schedule, fresh_reset


def test_raw_recomputation_and_mutation_negatives() -> None:
    state, stage, schedule, fresh_reset = _candidate_fixture()
    validate_candidate_state(state, stage, 1, schedule, fresh_reset)
    mutations = (
        (state["construction"]["construction_action_trace"][0]["command_action_8d"], 0),
        (state["construction"]["construction_action_trace"][500]["pre_action_pinch_geometry"]["target_pinch_midpoint_env_local_m"], 2),
        (state["construction"]["construction_action_trace"][0]["pre_action_pinch_geometry"]["live_tensor_collision_geometry"]["cube"]["live_tensor_pose"]["scene_env_origin_world_m"], 0),
        (state["construction"]["construction_action_trace"][0]["pre_action_pinch_geometry"]["live_tensor_collision_geometry"]["left"]["live_tensor_pose"]["position_env_local_m"], 0),
        (state["construction"]["collision_geometry_resolution"]["static_body_local_collision_geometry"]["left"]["collision_corners_body_m"][0], 0),
        (state["construction"]["collision_geometry_resolution"]["static_body_local_collision_geometry"]["cube"]["center_body_m"], 0),
        (state["construction"]["construction_action_trace"][0]["pre_action_pinch_geometry"]["live_tensor_collision_geometry"]["cube"]["reconstructed_bounds_env_local"]["collision_center_env_local_m"], 0),
        (state["construction"]["construction_action_trace"][0]["pre_action_pinch_geometry"]["live_tensor_collision_geometry"]["cube"]["reconstructed_bounds_env_local"]["aabb_half_extents_env_local_m"], 0),
        (state["construction"]["contact_and_grab_trace_diagnostic_only"][3]["after"]["contact_force_n"], "gripper__rubiks_cube"),
        (state["objects"]["rubiks_cube"]["position_world_m"], 0),
        (state["construction"]["captured_joint_position_target_rad"], 0),
    )
    for target, key in mutations:
        old = target[key]
        target[key] = old + 0.01
        with pytest.raises(ValidationError):
            validate_candidate_state(state, stage, 1, schedule, fresh_reset)
        target[key] = old


def test_authoritative_1800_step_lifecycle_and_mutations() -> None:
    activation = {
        "status": "registered_construction_timeout_extended_before_first_reset_or_step",
        "only_mutated_field": "env.cfg.episode_length_s",
        "original_max_episode_length_steps": 450,
        "registered_max_episode_length_steps": 1800,
        "original_episode_length_s": 30.0,
        "registered_episode_length_s": 120.0,
        "step_dt_s": 1.0 / 15.0,
        "common_step_counter_before_and_after": 0,
        "episode_length_buf_before_and_after": [0.0],
        "termination_config_byte_equal": True,
        "termination_contract_before": {"active_terms": ["success", "time_out"]},
        "termination_contract_after": {"active_terms": ["success", "time_out"]},
        "registered_worst_case_steps": 1695,
        "registered_margin_steps": 105,
        "behavioral_horizon_mutated": False,
    }
    validate_construction_lifecycle({"construction_horizon_activation": activation}, "fixture")
    for key, value in (("registered_max_episode_length_steps", 1799), ("registered_worst_case_steps", 1694), ("registered_margin_steps", 106)):
        bad = deepcopy(activation)
        bad[key] = value
        with pytest.raises(ValidationError):
            validate_construction_lifecycle({"construction_horizon_activation": bad}, "bad")


def test_registered_diagnostic_failure_terminal_and_mutations() -> None:
    schedule = load(ART / "gates/candidate_schedule.json")
    diagnostics = []
    for index, registered in enumerate(
        schedule["known_reachable_diagnostics"][:3], start=1
    ):
        diagnostics.append(
            {
                "diagnostic_index_one_based": index,
                "stage": registered["stage"],
                "source_side": registered["source_side"],
                "registered_diagnostic": registered,
                "passed": index < 3,
            }
        )
    report = {
        "status": "r009_known_reachable_diagnostic_failed_candidates_not_evaluated",
        "passed": False,
        "r009_live_diagnostic_count": 3,
        "known_reachable_diagnostics": diagnostics,
        "attempts": [],
        "repair_candidate_evaluation_count": 0,
        "state_candidate_count": 0,
        "accepted_candidate_rank": None,
        "accepted_states": None,
        "candidate_budget": 4,
        "diagnostic_budget": 4,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    assert validate_terminal_diagnostic_pattern(report, schedule) is True
    mutations = (
        ("r009_live_diagnostic_count", 4),
        ("attempts", [{"candidate_rank": 1}]),
        ("repair_candidate_evaluation_count", 1),
        ("state_candidate_count", 1),
        ("accepted_candidate_rank", 1),
        ("accepted_states", {}),
        ("model_request_count", 1),
    )
    for key, value in mutations:
        bad = deepcopy(report)
        bad[key] = value
        with pytest.raises(ValidationError):
            validate_terminal_diagnostic_pattern(bad, schedule)
    bad = deepcopy(report)
    bad["known_reachable_diagnostics"][0]["passed"] = False
    with pytest.raises(ValidationError):
        validate_terminal_diagnostic_pattern(bad, schedule)
    bad = deepcopy(report)
    bad["known_reachable_diagnostics"][-1]["passed"] = True
    with pytest.raises(ValidationError):
        validate_terminal_diagnostic_pattern(bad, schedule)
    bad = deepcopy(report)
    bad["known_reachable_diagnostics"][-1]["registered_diagnostic"] = deepcopy(
        bad["known_reachable_diagnostics"][-1]["registered_diagnostic"]
    )
    bad["known_reachable_diagnostics"][-1]["registered_diagnostic"]["hold_steps"] = 31
    with pytest.raises(ValidationError):
        validate_terminal_diagnostic_pattern(bad, schedule)


def test_scientific_gate_ast_identity_and_materializer_prohibitions() -> None:
    old = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/state_repair_gate.py"
    new = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/state_repair_gate.py"
    for name in (
        "_contact_forces",
        "_contact_coverage",
        "_reference_bounds",
        "_save_camera_evidence",
        "_companion_gate",
        "_fresh_reset_and_gate",
        "_finalize_unchanged_gates",
    ):
        assert function_ast(old, name) == function_ast(new, name)
    material = function_ast(new, "_pinch_geometry_materialize_and_gate")
    assert "pinch_alignment_command" in material
    assert "_normal_joint_equilibrium_step" in material
    for forbidden in (
        "write_joint_state_to_sim",
        "write_root_pose_to_sim",
        "write_root_velocity_to_sim",
        "object_space_servo_command",
    ):
        assert forbidden not in material


def test_static_package_and_outer_contract() -> None:
    result = validate_static(ROOT, source_gate_required=False)
    assert result["passed"] is True
    assert result["candidate_pair_count"] == result["diagnostic_count"] == 4
    outer = (ROOT / "tools/run_v3e006_r009_state_repair.py").read_text(encoding="utf-8")
    assert 'REMOTE_BRANCH = "experiment/v3e006-r009-pinch-geometry-repair"' in outer
    assert "passed_r009_state_repair_not_released_for_behavior" in outer
    assert "r009_candidate_budget_exhausted_no_valid_state_pair" in outer
