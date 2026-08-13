#!/usr/bin/env python3
"""Static and target-raw validator for V3-E006-R009."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.v3.phase_e.canonical_stage_localization_v3e006.ood_reference import (
    _quat_normalize_wxyz,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r009.pinch_geometry import (
    collision_center_env_local,
    pinch_alignment_command,
    reconstruct_collision_bounds_env_local,
    validate_contract as validate_pinch_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r009.predecessor_contract import (
    R008_CLOSURE_COMMIT,
    R008_RESULTS_SHA256,
    validate_r008_exhaustion_closure,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r009.source_gate_contract import (
    SCHEMA as SOURCE_SCHEMA,
    STATUS as SOURCE_STATUS,
    validate_source_gate,
)


ROOT = Path(__file__).resolve().parents[1]
ART = Path("artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009")
BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
COLLISION_ASSETS = {
    "robot_usd": {
        "path": "/data/users/ali/vla_wam/external/RoboLab-11142d4/assets/robots/franka_robotiq_2f_85_flattened.usd",
        "bytes": 14156362,
        "sha256": "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41",
    },
    "cube_usd": {
        "path": "/data/users/ali/vla_wam/external/RoboLab-11142d4/assets/objects/hot3d/rubiks_cube.usd",
        "bytes": 682045,
        "sha256": "d9497c0a01c51df76d8c69e595ab91637fa028140f2656628549283267e65024",
    },
}


class ValidationError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ValidationError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify(path: Path, row: Mapping[str, Any], label: str) -> Path:
    require(path.is_file() and path.stat().st_size == row.get("bytes") and sha(path) == row.get("sha256"), f"{label} binding differs")
    return path


def resolve(root: Path, row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = root / path
    return verify(path, row, label)


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name)
    return ast.dump(node, include_attributes=False)


def quat_normalize(value: Sequence[float]) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    require(q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 0, "quaternion malformed")
    return q / np.linalg.norm(q)


def slerp(left: Sequence[float], right: Sequence[float], fraction: float) -> np.ndarray:
    a, b = quat_normalize(left), quat_normalize(right)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 1 - 1e-12:
        return quat_normalize((1 - fraction) * a + fraction * b)
    theta = math.acos(dot)
    return quat_normalize(
        math.sin((1 - fraction) * theta) / math.sin(theta) * a
        + math.sin(fraction * theta) / math.sin(theta) * b
    )


def same_float32(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(np.array_equal(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))


def expected_precontact(construction: Mapping[str, Any]) -> list[tuple[str, int, list[float]]]:
    targets = construction["registered_precontact_targets"]
    start = construction["start_base_link_pose_world_wxyz"]
    points = {
        "start": (np.asarray(start[:3]), quat_normalize(start[3:])),
        "approach": (
            np.asarray(targets["approach_base_link_pose"]["position_world_m"]),
            quat_normalize(targets["approach_base_link_pose"]["quaternion_world_wxyz"]),
        ),
        "contact": (
            np.asarray(targets["contact_base_link_pose_at_exact_reset_cube"]["position_world_m"]),
            quat_normalize(targets["contact_base_link_pose_at_exact_reset_cube"]["quaternion_world_wxyz"]),
        ),
    }
    phases = (
        ("open_approach", 120, "start", "approach", 0.0),
        ("open_descent", 120, "approach", "contact", 0.0),
        ("normal_close", 90, "contact", "contact", 1.0),
    )
    rows = []
    for phase, count, left, right, grip in phases:
        p0, q0 = points[left]; p1, q1 = points[right]
        for index in range(1, count + 1):
            f = index / count
            action = np.concatenate(((1-f)*p0 + f*p1, slerp(q0, q1, f), [grip])).astype(np.float32)
            rows.append((phase, index, action.tolist()))
    return rows


def validate_construction_lifecycle(lifecycle: Mapping[str, Any], label: str) -> None:
    activation = lifecycle.get("construction_horizon_activation")
    require(isinstance(activation, Mapping), f"{label} horizon activation absent")
    require(
        activation.get("status")
        == "registered_construction_timeout_extended_before_first_reset_or_step",
        f"{label} horizon activation status differs",
    )
    require(activation.get("only_mutated_field") == "env.cfg.episode_length_s", f"{label} mutation differs")
    require(activation.get("original_max_episode_length_steps") == 450, f"{label} original horizon differs")
    require(activation.get("registered_max_episode_length_steps") == 1800, f"{label} registered horizon differs")
    require(activation.get("original_episode_length_s") == 30.0, f"{label} original seconds differ")
    require(activation.get("registered_episode_length_s") == 120.0, f"{label} registered seconds differs")
    require(
        math.isclose(float(activation.get("step_dt_s")), 1.0 / 15.0, rel_tol=0.0, abs_tol=1e-12),
        f"{label} step dt differs",
    )
    require(activation.get("common_step_counter_before_and_after") == 0, f"{label} stepped before activation")
    require(activation.get("episode_length_buf_before_and_after") in ([0], [0.0]), f"{label} episode counter differs")
    require(activation.get("termination_config_byte_equal") is True, f"{label} termination config changed")
    require(
        activation.get("termination_contract_before") == activation.get("termination_contract_after"),
        f"{label} termination snapshots differ",
    )
    require(
        activation.get("termination_contract_before", {}).get("active_terms") == ["success", "time_out"],
        f"{label} termination terms differ",
    )
    require(activation.get("registered_worst_case_steps") == 1695, f"{label} worst case differs")
    require(activation.get("registered_margin_steps") == 105, f"{label} margin differs")
    require(activation.get("behavioral_horizon_mutated") is False, f"{label} behavioral horizon changed")


def quaternion_equivalent(left: Sequence[float], right: Sequence[float], *, atol: float = 1e-12) -> bool:
    a, b = quat_normalize(left), quat_normalize(right)
    return bool(abs(float(np.dot(a, b))) >= 1.0 - atol)


def _archived_validate_candidate_state_r008(
    state: Mapping[str, Any],
    expected_stage: Mapping[str, Any],
    rank: int,
    schedule: Mapping[str, Any],
) -> None:
    construction = state.get("construction")
    require(isinstance(construction, Mapping), "candidate construction absent")
    require(construction.get("method") == "exact_reset_open_close_uniform_object_servo_q_handoff", "construction method differs")
    retained = deepcopy(expected_stage); retained["candidate_rank"] = rank
    require(construction.get("registered_stage_schedule") == retained, "retained stage schedule differs")
    require(construction.get("registered_target_cube_pose") == expected_stage["r008_target_cube_pose"], "cube target differs")
    for retained_key, schedule_key in (
        ("object_space_servo_contract", "object_space_servo_contract"),
        ("joint_handoff_contract", "joint_handoff_contract"),
        ("construction_lifecycle_contract", "construction_lifecycle_contract"),
    ):
        require(
            construction.get(retained_key) == schedule.get(schedule_key),
            f"retained {retained_key} differs from global schedule",
        )
    require(construction.get("phase_steps") == {
        "open_approach": 120, "open_descent": 120, "normal_close": 90,
        "closed_object_space_servo": 360, "captured_q_normal_joint_settle": 600,
    }, "phase counts differ")
    require(construction.get("episode_length_buf_before_candidate_actions") == [75], "start counter differs")
    require(construction.get("episode_length_buf_before_handoff") == [765], "handoff counter differs")
    require(construction.get("episode_length_buf_after_candidate_actions") == [1365], "final counter differs")
    require(construction.get("post_reset_joint_state_write_count") == construction.get("post_reset_object_state_write_count") == 0, "post-reset state write differs")
    require(construction.get("captured_joint_target_write_count") == 1, "q target write count differs")
    require(construction.get("cartesian_action_manager_apply_count_during_joint_settle") == 0, "Cartesian settle action differs")
    require(construction.get("joint_or_object_state_write_count") == 0, "settle state write differs")
    trace = construction.get("construction_action_trace")
    servo_trace = construction.get("object_space_servo_trace")
    require(isinstance(trace, list) and len(trace) == 1290, "trace length differs")
    require(isinstance(servo_trace, list) and servo_trace == trace[330:690], "servo subtrace differs")
    vector_fields = {
        "eef_position_world_m": 3,
        "eef_quaternion_world_wxyz": 4,
        "base_link_position_world_m": 3,
        "base_link_quaternion_world_wxyz": 4,
        "joint_position_rad": 13,
        "joint_velocity_rad_s": 13,
        "cube_position_world_m": 3,
        "cube_quaternion_world_wxyz": 4,
        "cube_linear_velocity_m_s": 3,
        "cube_angular_velocity_rad_s": 3,
    }
    for index, row in enumerate(trace, start=1):
        for field, width in vector_fields.items():
            value = row.get(field)
            require(
                isinstance(value, list)
                and len(value) == width
                and np.isfinite(np.asarray(value, dtype=np.float64)).all(),
                f"trace {index} {field} differs",
            )
        frame = row.get("base_link_to_eef_frame_identity")
        require(isinstance(frame, Mapping) and frame.get("passed") is True, f"trace {index} frame identity failed")
        require(
            float(frame.get("position_composition_residual_m", math.inf)) <= 1e-6
            and float(frame.get("orientation_composition_residual_deg", math.inf)) <= 1e-4,
            f"trace {index} frame residual differs",
        )
    for row, (phase, step, action) in zip(trace[:330], expected_precontact(construction), strict=True):
        require(row.get("phase") == phase and row.get("phase_step_one_based") == step, "precontact phase differs")
        require(same_float32(row.get("command_action_8d", []), action), "precontact command differs")
    contract = construction["object_space_servo_contract"]
    for index, row in enumerate(servo_trace, start=1):
        evidence = row.get("pre_action_object_space_servo")
        require(isinstance(evidence, Mapping), "servo pre-action evidence absent")
        previous = trace[329 + index - 1]
        require(
            np.allclose(
                evidence.get("live_base_position_world_m"),
                previous.get("base_link_position_world_m"), atol=1e-12, rtol=0,
            ),
            f"servo {index} base pre-state is not chained",
        )
        require(
            quaternion_equivalent(
                evidence.get("live_base_quaternion_world_wxyz", []),
                previous.get("base_link_quaternion_world_wxyz", []),
            ),
            f"servo {index} base quaternion pre-state is not chained",
        )
        require(
            np.allclose(
                evidence.get("live_cube_position_world_m"),
                previous.get("cube_position_world_m"), atol=1e-12, rtol=0,
            ),
            f"servo {index} cube pre-state is not chained",
        )
        require(
            quaternion_equivalent(
                evidence.get("live_cube_quaternion_world_wxyz", []),
                previous.get("cube_quaternion_world_wxyz", []),
            ),
            f"servo {index} cube quaternion pre-state is not chained",
        )
        require(
            evidence.get("target_cube_position_world_m")
            == construction["registered_target_cube_pose"]["position_world_m"],
            f"servo {index} target position differs from frozen target",
        )
        require(
            quaternion_equivalent(
                evidence.get("target_cube_quaternion_world_wxyz", []),
                construction["registered_target_cube_pose"]["quaternion_world_wxyz"],
            ),
            f"servo {index} target quaternion differs from frozen target",
        )
        recomputed = object_space_servo_command(
            live_base_position=evidence["live_base_position_world_m"],
            live_base_quaternion=evidence["live_base_quaternion_world_wxyz"],
            live_cube_position=evidence["live_cube_position_world_m"],
            live_cube_quaternion=evidence["live_cube_quaternion_world_wxyz"],
            target_cube_position=evidence["target_cube_position_world_m"],
            target_cube_quaternion=evidence["target_cube_quaternion_world_wxyz"],
            translation_gain=contract["translation_gain"],
            rotation_gain=contract["rotation_gain"],
            translation_cap_m_per_step=contract["translation_cap_m_per_step"],
            rotation_cap_deg_per_step=contract["rotation_cap_deg_per_step"],
        )
        for key, expected in recomputed.items():
            actual = evidence.get(key)
            if isinstance(expected, list):
                require(np.allclose(actual, expected, atol=1e-12, rtol=0), f"servo {index} {key} differs")
            else:
                require(actual == expected, f"servo {index} {key} differs")
        expected_action = [*recomputed["command_base_position_world_m"], *recomputed["command_base_quaternion_world_wxyz"], 1.0]
        require(same_float32(row.get("command_action_8d", []), expected_action), f"servo {index} command differs")
    target = construction.get("captured_joint_position_target_rad")
    require(isinstance(target, list) and len(target) == 13 and np.isfinite(target).all(), "captured q differs")
    require(
        same_float32(target, servo_trace[-1].get("joint_position_rad", [])),
        "captured q is not the exact servo-step-360 observed q",
    )
    for index, row in enumerate(trace[690:], start=1):
        require(row.get("phase") == "captured_q_normal_joint_settle", "settle phase differs")
        require(row.get("phase_step_one_based") == index, "settle step differs")
        require(row.get("normal_joint_position_target_rad") == target, "settle q target changed")
        require(row.get("cartesian_action_manager_applied") is False, "Cartesian action applied during settle")
        for key in (
            "joint_position_rad", "joint_velocity_rad_s", "cube_position_world_m",
            "cube_quaternion_world_wxyz", "cube_linear_velocity_m_s", "cube_angular_velocity_rad_s",
        ):
            require(np.isfinite(np.asarray(row.get(key), dtype=np.float64)).all(), f"settle {key} nonfinite")
        require(row.get("base_link_to_eef_frame_identity", {}).get("passed") is True, "frame identity failed")
    samples = construction.get("settled_gate_samples")
    require(isinstance(samples, list) and len(samples) == 10, "final gate window differs")
    for sample, row in zip(samples, trace[-10:], strict=True):
        for sample_key, trace_key in (
            ("cube_position_world_m", "cube_position_world_m"),
            ("cube_linear_velocity_m_s", "cube_linear_velocity_m_s"),
            ("cube_angular_velocity_rad_s", "cube_angular_velocity_rad_s"),
            ("eef_position_world_m", "base_link_position_world_m"),
            ("base_link_quaternion_world_wxyz", "base_link_quaternion_world_wxyz"),
            ("live_eef_frame_position_world_m", "eef_position_world_m"),
            ("live_eef_frame_quaternion_world_wxyz", "eef_quaternion_world_wxyz"),
        ):
            require(sample.get(sample_key) == row.get(trace_key), f"final sample {sample_key} differs")
        require(sample.get("arm_joint_velocity_rad_s") == row.get("joint_velocity_rad_s")[:7], "final arm velocity differs")
        require(sample.get("base_link_to_eef_frame_identity") == row.get("base_link_to_eef_frame_identity"), "final frame sample differs")
        require(isinstance(sample.get("object_grabbed"), bool), "final grasp evidence differs")
        require(isinstance(sample.get("contact_force_n"), Mapping), "final contact evidence differs")
        require(
            all(math.isfinite(float(value)) for value in sample["contact_force_n"].values()),
            "final contact force is nonfinite",
        )
    last = trace[-1]
    require(state.get("robot", {}).get("joint_position_rad") == last.get("joint_position_rad"), "final robot q differs")
    require(state.get("robot", {}).get("joint_velocity_rad_s") == last.get("joint_velocity_rad_s"), "final robot qd differs")
    cube = state.get("objects", {}).get("rubiks_cube", {})
    for state_key, trace_key in (
        ("position_world_m", "cube_position_world_m"),
        ("quaternion_world_wxyz", "cube_quaternion_world_wxyz"),
        ("linear_velocity_m_s", "cube_linear_velocity_m_s"),
        ("angular_velocity_rad_s", "cube_angular_velocity_rad_s"),
    ):
        require(cube.get(state_key) == last.get(trace_key), f"final cube {state_key} differs")
    require(state.get("eef", {}).get("position_world_m") == last.get("base_link_position_world_m"), "final base position differs")
    require(state.get("eef", {}).get("quaternion_world_wxyz") == last.get("base_link_quaternion_world_wxyz"), "final base quaternion differs")
    contact = state.get("contact_evidence", {})
    require(contact.get("settled_force_snapshots_n") == [row["contact_force_n"] for row in samples], "contact samples differ")
    require(contact.get("object_grabbed_by_step") == [row["object_grabbed"] for row in samples], "grasp samples differ")
    require(state.get("physics_gate", {}).get("settled_window_steps") == 10, "physics window differs")
    require(state.get("passed") is all((
        state.get("physics_gate", {}).get("passed"),
        state.get("ood_gate", {}).get("passed"),
        state.get("camera_evidence", {}).get("passed"),
        state.get("companion_pose_gate", {}).get("passed"),
    )), "candidate pass recomputation differs")


def _finite_vector(value: Any, width: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    require(vector.shape == (width,) and np.isfinite(vector).all(), f"{label} differs")
    return vector


def _validate_local_collision_geometry(
    value: Mapping[str, Any],
    *,
    label: str,
    body_path: str,
    collision_paths: Sequence[str],
) -> None:
    require(value.get("body_prim_path") == body_path, f"{label} owner differs")
    require(
        value.get("body_has_rigid_body_api") is True
        and value.get("all_collision_prims_owned_by_exact_body_without_nested_boundary")
        is True,
        f"{label} rigid-body ownership differs",
    )
    require(value.get("collision_prim_paths") == list(collision_paths), f"{label} inventory differs")
    corners = np.asarray(value.get("collision_corners_body_m"), dtype=np.float64)
    require(
        corners.ndim == 2
        and corners.shape[0] >= 8
        and corners.shape[1] == 3
        and np.isfinite(corners).all(),
        f"{label} corners differ",
    )
    shape_rows = value.get("shape_local_geometry")
    require(isinstance(shape_rows, list) and len(shape_rows) == len(collision_paths), f"{label} shapes differ")
    flattened: list[list[float]] = []
    for row, path in zip(shape_rows, collision_paths, strict=True):
        require(row.get("collision_prim_path") == path, f"{label} shape path differs")
        minimum = _finite_vector(row.get("collision_prim_local_minimum_m"), 3, f"{label} local minimum")
        maximum = _finite_vector(row.get("collision_prim_local_maximum_m"), 3, f"{label} local maximum")
        require(np.all(maximum > minimum), f"{label} local range differs")
        shape_corners = np.asarray(row.get("collision_corners_body_m"), dtype=np.float64)
        require(shape_corners.shape == (8, 3) and np.isfinite(shape_corners).all(), f"{label} shape corners differ")
        flattened.extend(shape_corners.tolist())
    require(np.array_equal(corners, np.asarray(flattened, dtype=np.float64)), f"{label} aggregate corners differ")
    minimum = np.min(corners, axis=0); maximum = np.max(corners, axis=0)
    require(np.array_equal(minimum, _finite_vector(value.get("minimum_body_m"), 3, f"{label} minimum")), f"{label} minimum math differs")
    require(np.array_equal(maximum, _finite_vector(value.get("maximum_body_m"), 3, f"{label} maximum")), f"{label} maximum math differs")
    require(np.allclose(value.get("center_body_m"), 0.5 * (minimum + maximum), atol=1e-12, rtol=0), f"{label} center math differs")
    require(np.allclose(value.get("half_extents_body_m"), 0.5 * (maximum - minimum), atol=1e-12, rtol=0), f"{label} half math differs")
    unsigned = dict(value); claimed = unsigned.pop("canonical_sha256", None)
    require(claimed == canonical_sha(unsigned), f"{label} canonical digest differs")


def _validate_live_tensor_geometry(
    value: Mapping[str, Any], static: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    pose = value.get("live_tensor_pose")
    require(isinstance(pose, Mapping), f"{label} tensor pose absent")
    raw = _finite_vector(pose.get("position_tensor_world_m"), 3, f"{label} tensor position")
    origin = _finite_vector(pose.get("scene_env_origin_world_m"), 3, f"{label} env origin")
    local = _finite_vector(pose.get("position_env_local_m"), 3, f"{label} env-local position")
    quaternion = _finite_vector(pose.get("quaternion_world_wxyz"), 4, f"{label} quaternion")
    require(np.allclose(local, raw - origin, atol=1e-12, rtol=0), f"{label} origin subtraction differs")
    require(value.get("static_body_local_geometry_sha256") == static.get("canonical_sha256"), f"{label} static-geometry link differs")
    recomputed = reconstruct_collision_bounds_env_local(
        body_position_env_local=local,
        body_quaternion_world_wxyz=quaternion,
        collision_corners_body=static["collision_corners_body_m"],
        collision_center_body=static["center_body_m"],
    )
    actual = value.get("reconstructed_bounds_env_local")
    require(isinstance(actual, Mapping), f"{label} reconstructed bounds absent")
    for key, expected in recomputed.items():
        require(np.allclose(actual.get(key), expected, atol=1e-12, rtol=0), f"{label} {key} differs")
    return recomputed


def _canonical_action(
    position: Sequence[float], quaternion: Sequence[float], grip: float
) -> list[float]:
    return np.concatenate(
        (
            np.asarray(position, dtype=np.float64),
            _quat_normalize_wxyz(quaternion),
            [float(grip)],
        )
    ).astype(np.float32).tolist()


def validate_candidate_state(
    state: Mapping[str, Any],
    expected_stage: Mapping[str, Any],
    rank: int,
    schedule: Mapping[str, Any],
    fresh_reset: Mapping[str, Any],
) -> None:
    """Independently regenerate R009 geometry commands and unchanged selection."""

    construction = state.get("construction")
    require(isinstance(construction, Mapping), "candidate construction absent")
    require(
        construction.get("method")
        == "exact_reset_uniform_collision_pinch_acquire_lift_transport_q_handoff",
        "construction method differs",
    )
    retained = deepcopy(expected_stage)
    retained["candidate_rank"] = rank
    require(
        construction.get("registered_stage_schedule") == retained,
        "retained stage schedule differs",
    )
    require(
        construction.get("pinch_geometry_contract")
        == schedule.get("pinch_geometry_contract"),
        "retained pinch contract differs",
    )
    require(
        construction.get("joint_handoff_contract")
        == schedule.get("joint_handoff_contract"),
        "retained handoff contract differs",
    )
    require(
        construction.get("construction_lifecycle_contract")
        == schedule.get("construction_lifecycle_contract"),
        "retained lifecycle contract differs",
    )
    require(
        construction.get("phase_steps")
        == {
            "open_approach": 180,
            "open_descent": 180,
            "normal_close": 120,
            "closed_vertical_lift": 240,
            "closed_stage_transport": 300,
            "captured_q_normal_joint_settle": 600,
        },
        "phase counts differ",
    )
    require(
        construction.get("episode_length_buf_before_candidate_actions") == [75]
        and construction.get("episode_length_buf_before_handoff") == [1095]
        and construction.get("episode_length_buf_after_candidate_actions") == [1695],
        "construction counters differ",
    )
    require(
        construction.get("post_reset_joint_state_write_count")
        == construction.get("post_reset_object_state_write_count")
        == construction.get("joint_or_object_state_write_count")
        == construction.get("contact_or_grab_conditioned_branch_count")
        == 0,
        "R009 prohibited state write or branch differs",
    )
    require(
        construction.get("all_registered_phases_executed_unconditionally") is True
        and construction.get("captured_joint_target_write_count") == 1
        and construction.get("cartesian_action_manager_apply_count_during_joint_settle") == 0,
        "R009 unconditional execution or handoff differs",
    )

    resolution = construction.get("collision_geometry_resolution")
    require(isinstance(resolution, Mapping), "collision geometry resolution absent")
    inventory = resolution.get("inventory")
    require(isinstance(inventory, Mapping), "collision inventory absent")
    left_body = str(inventory.get("left_inner_finger_body_prim_path", ""))
    right_body = str(inventory.get("right_inner_finger_body_prim_path", ""))
    cube_root = str(inventory.get("cube_root_prim_path", ""))
    require(
        left_body.endswith("/left_inner_finger")
        and right_body.endswith("/right_inner_finger")
        and left_body != right_body,
        "inner-finger body uniqueness differs",
    )
    for key, root_path in (
        ("left_collision_prim_paths", left_body),
        ("right_collision_prim_paths", right_body),
        ("cube_collision_prim_paths", cube_root),
    ):
        paths = inventory.get(key)
        require(
            isinstance(paths, list)
            and paths
            and len(paths) == len(set(paths))
            and paths == sorted(paths)
            and all(path == root_path or path.startswith(root_path + "/") for path in paths),
            f"{key} differs",
        )
    require(
        resolution.get("dynamic_usd_world_bounds_used") is False
        and resolution.get("dynamic_geometry_source")
        == "IsaacLab tensor rigid-body/root poses minus explicit scene env origin",
        "dynamic geometry source differs",
    )
    resolution_origin = _finite_vector(
        resolution.get("scene_env_origin_world_m_at_resolution"),
        3,
        "resolution env origin",
    )
    require(
        np.array_equal(
            fresh_reset.get("base_link_to_eef_frame_identity", {}).get(
                "scene_env_origin_world_m"
            ),
            resolution_origin,
        ),
        "fresh-reset/resolution env origin differs",
    )
    static_geometry = resolution.get("static_body_local_collision_geometry")
    require(isinstance(static_geometry, Mapping), "static local collision geometry absent")
    for key, body_path, paths_key in (
        ("left", left_body, "left_collision_prim_paths"),
        ("right", right_body, "right_collision_prim_paths"),
        ("cube", cube_root, "cube_collision_prim_paths"),
    ):
        require(isinstance(static_geometry.get(key), Mapping), f"{key} local geometry absent")
        _validate_local_collision_geometry(
            static_geometry[key],
            label=f"static {key}",
            body_path=body_path,
            collision_paths=inventory[paths_key],
        )
    require(
        inventory.get("left_robot_body_tensor_name") == "left_inner_finger"
        and inventory.get("right_robot_body_tensor_name") == "right_inner_finger"
        and isinstance(inventory.get("left_robot_body_tensor_index"), int)
        and isinstance(inventory.get("right_robot_body_tensor_index"), int)
        and inventory["left_robot_body_tensor_index"]
        != inventory["right_robot_body_tensor_index"]
        and inventory.get("cube_tensor_source")
        == "rubiks_cube.data.root_pos_w/root_quat_w",
        "tensor body/root inventory differs",
    )

    reset_pose = _finite_vector(
        construction.get("reset_cube_pose_env_local_wxyz"), 7, "reset cube pose"
    )
    reset_center = _finite_vector(
        construction.get("reset_cube_collision_center_env_local_m"),
        3,
        "reset cube collision center",
    )
    reset_half = _finite_vector(
        construction.get("reset_cube_collision_half_extents_env_local_m"),
        3,
        "reset cube half extents",
    )
    local_center = _finite_vector(
        construction.get("cube_collision_center_in_cube_m"),
        3,
        "cube collision center in cube",
    )
    target_pose = _finite_vector(
        construction.get("target_cube_pose_env_local_wxyz"), 7, "target cube pose"
    )
    require(
        target_pose[:3].tolist()
        == expected_stage["r009_target_cube_pose"]["position_world_m"]
        and quaternion_equivalent(
            target_pose[3:],
            expected_stage["r009_target_cube_pose"]["quaternion_world_wxyz"],
        ),
        "frozen target cube pose differs",
    )
    require(
        np.allclose(
        collision_center_env_local(
            body_position_env_local=reset_pose[:3],
                body_quaternion_world_wxyz=reset_pose[3:],
                collision_center_body=local_center,
            ),
            reset_center,
            atol=1e-12,
            rtol=0,
        ),
        "reset collision-center reconstruction differs",
    )
    target_center = collision_center_env_local(
        body_position_env_local=target_pose[:3],
        body_quaternion_world_wxyz=target_pose[3:],
        collision_center_body=local_center,
    )
    require(
        np.allclose(
            target_center,
            construction.get("target_cube_collision_center_env_local_m"),
            atol=1e-12,
            rtol=0,
        ),
        "target collision-center reconstruction differs",
    )
    lift_center = reset_center.copy()
    lift_center[2] = target_center[2]
    require(
        np.allclose(
            lift_center,
            construction.get("lift_collision_center_env_local_m"),
            atol=1e-12,
            rtol=0,
        ),
        "lift collision-center target differs",
    )
    require(
        math.isclose(
            float(construction.get("registered_approach_clearance_m")),
            2.0 * float(reset_half[2]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "registered reset approach-clearance evidence differs",
    )

    trace = construction.get("construction_action_trace")
    acquisition = construction.get("acquisition_lift_transport_trace")
    contacts = construction.get("contact_and_grab_trace_diagnostic_only")
    require(isinstance(trace, list) and len(trace) == 1620, "trace length differs")
    require(
        isinstance(acquisition, list)
        and len(acquisition) == 1020
        and acquisition == trace[:1020],
        "acquisition subtrace differs",
    )
    require(isinstance(contacts, list) and len(contacts) == 1620, "contact trace differs")
    vector_fields = {
        "eef_position_world_m": 3,
        "eef_quaternion_world_wxyz": 4,
        "base_link_position_world_m": 3,
        "base_link_quaternion_world_wxyz": 4,
        "joint_position_rad": 13,
        "joint_velocity_rad_s": 13,
        "cube_position_world_m": 3,
        "cube_position_env_local_m": 3,
        "cube_quaternion_world_wxyz": 4,
        "cube_linear_velocity_m_s": 3,
        "cube_angular_velocity_rad_s": 3,
    }
    for index, row in enumerate(trace, start=1):
        for field, width in vector_fields.items():
            _finite_vector(row.get(field), width, f"trace {index} {field}")
        frame = row.get("base_link_to_eef_frame_identity")
        require(isinstance(frame, Mapping) and frame.get("passed") is True, f"trace {index} frame identity failed")
        require(
            float(frame.get("position_composition_residual_m", math.inf)) <= 1e-6
            and float(frame.get("orientation_composition_residual_deg", math.inf)) <= 1e-4,
            f"trace {index} frame residual differs",
        )
        require(
            np.allclose(
                np.asarray(row["cube_position_world_m"], dtype=np.float64)
                - resolution_origin,
                np.asarray(row["cube_position_env_local_m"], dtype=np.float64),
                atol=1e-12,
                rtol=0,
            ),
            f"trace {index} cube origin subtraction differs",
        )

    contract = schedule["pinch_geometry_contract"]
    acquisition_q = expected_stage["r009_acquisition_base_quaternion_world_wxyz"]
    final_q = expected_stage["r009_final_base_quaternion_world_wxyz"]
    phases = (
        ("open_approach", 0, 180, 0.0),
        ("open_descent", 180, 360, 0.0),
        ("normal_close", 360, 480, 1.0),
        ("closed_vertical_lift", 480, 720, 1.0),
        ("closed_stage_transport", 720, 1020, 1.0),
    )
    for phase, begin, end, grip in phases:
        steps = end - begin
        for offset, row in enumerate(trace[begin:end], start=1):
            index = begin + offset - 1
            require(
                row.get("phase") == phase
                and row.get("phase_step_one_based") == offset,
                f"{phase} phase trace differs",
            )
            evidence = row.get("pre_action_pinch_geometry")
            require(isinstance(evidence, Mapping), f"{phase} pinch evidence absent")
            live_geometry = evidence.get("live_tensor_collision_geometry")
            require(isinstance(live_geometry, Mapping), f"{phase} live geometry absent")
            reconstructed: dict[str, Mapping[str, Any]] = {}
            for key in ("left", "right", "cube"):
                require(isinstance(live_geometry.get(key), Mapping), f"{phase} {key} geometry absent")
                reconstructed[key] = _validate_live_tensor_geometry(
                    live_geometry[key],
                    static_geometry[key],
                    label=f"{phase} {offset} {key}",
                )
                require(
                    np.array_equal(
                        live_geometry[key]["live_tensor_pose"]["scene_env_origin_world_m"],
                        resolution_origin,
                    ),
                    f"{phase} {offset} {key} env origin drifted",
                )
            if index == 0:
                expected_pre_base_p = fresh_reset["eef"]["position_world_m"]
                expected_pre_base_q = fresh_reset["eef"]["quaternion_world_wxyz"]
                reset_cube = fresh_reset["objects"]["rubiks_cube"]
                expected_pre_cube_p = (
                    np.asarray(reset_cube["position_world_m"], dtype=np.float64)
                    - resolution_origin
                ).tolist()
                expected_pre_cube_q = reset_cube["quaternion_world_wxyz"]
            else:
                previous = trace[index - 1]
                expected_pre_base_p = previous["base_link_position_world_m"]
                expected_pre_base_q = previous["base_link_quaternion_world_wxyz"]
                expected_pre_cube_p = previous["cube_position_env_local_m"]
                expected_pre_cube_q = previous["cube_quaternion_world_wxyz"]
            require(
                np.allclose(
                    evidence.get("live_base_position_env_local_m"),
                    expected_pre_base_p,
                    atol=1e-12,
                    rtol=0,
                )
                and quaternion_equivalent(
                    evidence.get("live_base_quaternion_world_wxyz", []),
                    expected_pre_base_q,
                )
                and np.allclose(
                    evidence.get("live_cube_position_env_local_m"),
                    expected_pre_cube_p,
                    atol=1e-12,
                    rtol=0,
                )
                and quaternion_equivalent(
                    evidence.get("live_cube_quaternion_world_wxyz", []),
                    expected_pre_cube_q,
                ),
                f"{phase} {offset} pre-state chain differs",
            )
            cube_pose = live_geometry["cube"]["live_tensor_pose"]
            require(
                np.array_equal(
                    cube_pose["position_env_local_m"],
                    evidence["live_cube_position_env_local_m"],
                )
                and quaternion_equivalent(
                    cube_pose["quaternion_world_wxyz"],
                    evidence["live_cube_quaternion_world_wxyz"],
                ),
                f"{phase} {offset} cube tensor/evidence pose differs",
            )
            fraction = offset / steps
            live_cube_center = np.asarray(
                reconstructed["cube"]["collision_center_env_local_m"], dtype=np.float64
            )
            if phase == "open_approach":
                expected_midpoint = live_cube_center + np.asarray(
                    [0.0, 0.0, 2.0 * float(reconstructed["cube"]["aabb_half_extents_env_local_m"][2])]
                )
                expected_q = acquisition_q
                uses_live_cube = True
            elif phase in {"open_descent", "normal_close"}:
                expected_midpoint = live_cube_center
                expected_q = acquisition_q
                uses_live_cube = True
            elif phase == "closed_vertical_lift":
                expected_midpoint = (1.0 - fraction) * reset_center + fraction * lift_center
                expected_q = acquisition_q
                uses_live_cube = False
            else:
                expected_midpoint = (1.0 - fraction) * lift_center + fraction * target_center
                expected_q = slerp(acquisition_q, final_q, fraction)
                uses_live_cube = False
            require(
                np.allclose(
                    evidence.get("target_pinch_midpoint_env_local_m"),
                    expected_midpoint,
                    atol=1e-12,
                    rtol=0,
                )
                and quaternion_equivalent(
                    evidence.get("target_base_quaternion_world_wxyz", []), expected_q
                )
                and evidence.get("command_uses_live_cube_collision_center")
                is uses_live_cube
                and evidence.get("gripper_command") == grip,
                f"{phase} {offset} frozen target differs",
            )
            recomputed = pinch_alignment_command(
                live_base_position_env_local=evidence["live_base_position_env_local_m"],
                live_base_quaternion=evidence["live_base_quaternion_world_wxyz"],
                live_left_center_env_local=reconstructed["left"]["collision_center_env_local_m"],
                live_right_center_env_local=reconstructed["right"]["collision_center_env_local_m"],
                target_pinch_midpoint_env_local=expected_midpoint,
                target_base_quaternion=expected_q,
                translation_gain=contract["translation_gain"],
                rotation_gain=contract["rotation_gain"],
                translation_cap_m_per_step=contract[
                    "translation_cap_m_per_step"
                ],
                rotation_cap_deg_per_step=contract["rotation_cap_deg_per_step"],
            )
            actual_command = evidence.get("pinch_alignment_command")
            require(isinstance(actual_command, Mapping), f"{phase} command evidence absent")
            for key, expected in recomputed.items():
                actual = actual_command.get(key)
                if isinstance(expected, list):
                    require(
                        np.allclose(actual, expected, atol=1e-12, rtol=0),
                        f"{phase} {offset} command {key} differs",
                    )
                else:
                    require(actual == expected, f"{phase} {offset} command {key} differs")
            require(
                same_float32(
                    row.get("command_action_8d", []),
                    _canonical_action(
                        recomputed["command_base_position_env_local_m"],
                        recomputed["command_base_quaternion_world_wxyz"],
                        grip,
                    ),
                ),
                f"{phase} {offset} executed action differs",
            )
            for contact_key in (
                "contact_and_grab_diagnostic_before_action",
                "contact_and_grab_diagnostic_after_action",
            ):
                contact = (
                    evidence.get(contact_key)
                    if contact_key.endswith("before_action")
                    else row.get(contact_key)
                )
                require(
                    isinstance(contact, Mapping)
                    and isinstance(contact.get("object_grabbed"), bool)
                    and isinstance(contact.get("contact_force_n"), Mapping)
                    and all(
                        math.isfinite(float(value))
                        for value in contact["contact_force_n"].values()
                    ),
                    f"{phase} {offset} contact diagnostic differs",
                )
            expected_contact_row = {
                "phase": phase,
                "phase_step_one_based": offset,
                "before": evidence["contact_and_grab_diagnostic_before_action"],
                "after": row["contact_and_grab_diagnostic_after_action"],
            }
            require(contacts[index] == expected_contact_row, f"{phase} contact trace differs")

    target = construction.get("captured_joint_position_target_rad")
    require(
        isinstance(target, list)
        and len(target) == 13
        and np.isfinite(np.asarray(target, dtype=np.float64)).all()
        and same_float32(target, trace[1019].get("joint_position_rad", [])),
        "captured q is not exact transport-step-300 observed q",
    )
    for offset, row in enumerate(trace[1020:], start=1):
        require(
            row.get("phase") == "captured_q_normal_joint_settle"
            and row.get("phase_step_one_based") == offset
            and row.get("normal_joint_position_target_rad") == target
            and row.get("cartesian_action_manager_applied") is False,
            f"settle row {offset} differs",
        )
        contact = row.get("contact_and_grab_diagnostic_after_step")
        require(
            isinstance(contact, Mapping)
            and isinstance(contact.get("object_grabbed"), bool)
            and all(
                math.isfinite(float(value))
                for value in contact.get("contact_force_n", {}).values()
            ),
            f"settle contact row {offset} differs",
        )
        require(
            contacts[1019 + offset]
            == {
                "phase": "captured_q_normal_joint_settle",
                "phase_step_one_based": offset,
                "after": contact,
            },
            f"settle contact trace {offset} differs",
        )

    samples = construction.get("settled_gate_samples")
    require(isinstance(samples, list) and len(samples) == 10, "final gate window differs")
    for sample, row in zip(samples, trace[-10:], strict=True):
        for sample_key, trace_key in (
            ("cube_position_world_m", "cube_position_world_m"),
            ("cube_linear_velocity_m_s", "cube_linear_velocity_m_s"),
            ("cube_angular_velocity_rad_s", "cube_angular_velocity_rad_s"),
            ("eef_position_world_m", "base_link_position_world_m"),
            ("base_link_quaternion_world_wxyz", "base_link_quaternion_world_wxyz"),
            ("live_eef_frame_position_world_m", "eef_position_world_m"),
            ("live_eef_frame_quaternion_world_wxyz", "eef_quaternion_world_wxyz"),
        ):
            require(sample.get(sample_key) == row.get(trace_key), f"final sample {sample_key} differs")
        require(sample.get("arm_joint_velocity_rad_s") == row.get("joint_velocity_rad_s")[:7], "final arm velocity differs")
        require(sample.get("base_link_to_eef_frame_identity") == row.get("base_link_to_eef_frame_identity"), "final frame sample differs")
        require(
            sample.get("contact_force_n")
            == row["contact_and_grab_diagnostic_after_step"]["contact_force_n"]
            and sample.get("object_grabbed")
            is row["contact_and_grab_diagnostic_after_step"]["object_grabbed"],
            "final contact/grab sample differs",
        )
    last = trace[-1]
    require(state.get("robot", {}).get("joint_position_rad") == last.get("joint_position_rad"), "final robot q differs")
    require(state.get("robot", {}).get("joint_velocity_rad_s") == last.get("joint_velocity_rad_s"), "final robot qd differs")
    cube = state.get("objects", {}).get("rubiks_cube", {})
    for state_key, trace_key in (
        ("position_world_m", "cube_position_world_m"),
        ("quaternion_world_wxyz", "cube_quaternion_world_wxyz"),
        ("linear_velocity_m_s", "cube_linear_velocity_m_s"),
        ("angular_velocity_rad_s", "cube_angular_velocity_rad_s"),
    ):
        require(cube.get(state_key) == last.get(trace_key), f"final cube {state_key} differs")
    require(state.get("eef", {}).get("position_world_m") == last.get("base_link_position_world_m"), "final base position differs")
    require(state.get("eef", {}).get("quaternion_world_wxyz") == last.get("base_link_quaternion_world_wxyz"), "final base quaternion differs")
    contact = state.get("contact_evidence", {})
    require(contact.get("settled_force_snapshots_n") == [row["contact_force_n"] for row in samples], "contact samples differ")
    require(contact.get("object_grabbed_by_step") == [row["object_grabbed"] for row in samples], "grasp samples differ")
    require(state.get("physics_gate", {}).get("settled_window_steps") == 10, "physics window differs")
    require(
        state.get("passed")
        is all(
            (
                state.get("physics_gate", {}).get("passed"),
                state.get("ood_gate", {}).get("passed"),
                state.get("camera_evidence", {}).get("passed"),
                state.get("companion_pose_gate", {}).get("passed"),
            )
        ),
        "candidate pass recomputation differs",
    )


def _archived_validate_static_r008(root: Path, *, source_gate_required: bool) -> dict[str, Any]:
    root = root.resolve(); artifact = root / ART
    registration = load(artifact / "repair_registration.json")
    schedule = load(artifact / "gates/candidate_schedule.json")
    require(registration.get("repair_amendment_id") == "V3-E006-R009", "registration ID differs")
    require(registration.get("predecessor_repair_amendment_id") == "V3-E006-R007", "predecessor differs")
    require(registration.get("counts_at_registration") == {
        "r008_live_diagnostics": 0, "r008_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }, "registration counts differ")
    for commit in (BASE, R007_CLOSURE_COMMIT):
        require(subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode == 0, f"missing commit {commit}")
    predecessor_path = resolve(root, registration["r007_predecessor"]["results"], "R007 predecessor")
    require(sha(predecessor_path) == R007_RESULTS_SHA256, "R007 results digest differs")
    validate_r007_exhaustion_closure(load(predecessor_path))
    require(schedule.get("candidate_budget") == schedule.get("diagnostic_budget") == 4, "budgets differ")
    require([row.get("candidate_rank") for row in schedule["candidate_pairs"]] == [1,2,3,4], "rank order differs")
    require(schedule.get("r007_predecessor") == registration.get("r007_predecessor"), "predecessor schedule differs")
    for name, contract in (
        ("object_space_servo", schedule["object_space_servo_contract"]),
        ("joint_handoff", schedule["joint_handoff_contract"]),
        ("construction_lifecycle", schedule["construction_lifecycle_contract"]),
    ):
        require(canonical_sha(contract) == schedule[f"{name}_contract_sha256"], f"{name} digest differs")
    validate_servo_contract(schedule["object_space_servo_contract"])
    require(schedule["joint_handoff_contract"]["settle_steps"] == 600, "handoff settle differs")
    require(schedule["construction_lifecycle_contract"]["worst_case_steps"] == 1365, "horizon derivation differs")
    require(schedule["construction_lifecycle_contract"]["registered_max_episode_length_steps"] == 1500, "construction horizon differs")
    r007_schedule = load(resolve(root, schedule["r007_candidate_schedule"], "R007 schedule"))
    require("construction_horizon_contract" not in schedule, "inherited 900-step horizon remains active")
    require("open_contact_construction_contract" not in schedule, "inherited 900-step open-contact contract remains active")
    archived = schedule.get("archived_predecessor_contracts", {})
    require(
        archived.get("status") == "archived_lineage_only_not_active_r008_runtime_evidence",
        "predecessor contracts are not explicitly archived",
    )
    require(
        archived.get("r005_construction_horizon_contract")
        == r007_schedule.get("construction_horizon_contract")
        and archived.get("r005_construction_horizon_contract_sha256")
        == r007_schedule.get("construction_horizon_contract_sha256"),
        "archived R005 horizon differs",
    )
    require(
        archived.get("r007_open_contact_construction_contract")
        == r007_schedule.get("open_contact_construction_contract")
        and archived.get("r007_open_contact_construction_contract_sha256")
        == r007_schedule.get("open_contact_construction_contract_sha256"),
        "archived R007 construction contract differs",
    )
    stripped = deepcopy(schedule["candidate_pairs"])
    for pair in stripped:
        pair["construction_method"] = "exact_reset_open_approach_normal_close_lift"
        for stage in ("canonical_grasp", "canonical_carry"):
            pair[stage].pop("r008_target_cube_pose")
            pair[stage].pop("r008_precontact_targets")
    require(stripped == r007_schedule["candidate_pairs"], "candidate sources/targets/order changed")
    require(schedule["known_reachable_diagnostics"] == r007_schedule["known_reachable_diagnostics"], "diagnostics changed")
    r007_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r007/state_repair_gate.py"
    r008_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence",
        "_companion_gate", "_fresh_reset_and_gate", "_finalize_unchanged_gates",
    ):
        require(function_ast(r007_source, name) == function_ast(r008_source, name), f"scientific helper differs: {name}")
    source = r008_source.read_text(encoding="utf-8")
    require(
        "from experiments.v3.phase_e.canonical_stage_localization_v3e006.state_contract import" in source
        and "from experiments.v3.phase_e.canonical_stage_localization_v3e006.ood_reference import" in source,
        "frozen scientific gate imports differ",
    )
    require("requests.post" not in source and "policy_server" not in source, "model endpoint exists")
    material = function_ast(r008_source, "_open_contact_materialize_and_gate")
    for prohibited in ("write_joint_state_to_sim", "write_root_pose_to_sim", "write_root_velocity_to_sim"):
        require(prohibited not in material, f"state write entered materializer: {prohibited}")
    require("object_space_servo_command" in material and "_normal_joint_equilibrium_step" in material, "R009 controller topology absent")
    gate_summary = None
    if source_gate_required:
        gate_path = artifact / "source_push_gate.json"
        require(gate_path.is_file(), "source gate absent")
        gate = load(gate_path)
        require(gate.get("schema_version") == SOURCE_SCHEMA and gate.get("status") == SOURCE_STATUS, "source gate differs")
        try:
            validate_source_gate(gate, study_root=root, verify_raw_history=True)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        gate_summary = binding(gate_path)
    return {
        "passed": True,
        "registration": binding(artifact / "repair_registration.json"),
        "candidate_schedule": binding(artifact / "gates/candidate_schedule.json"),
        "source_push_gate": gate_summary,
        "candidate_pair_count": 4,
        "diagnostic_count": 4,
        "object_space_servo_contract_sha256": schedule["object_space_servo_contract_sha256"],
        "joint_handoff_contract_sha256": schedule["joint_handoff_contract_sha256"],
        "construction_lifecycle_contract_sha256": schedule["construction_lifecycle_contract_sha256"],
    }


def validate_static(root: Path, *, source_gate_required: bool) -> dict[str, Any]:
    root = root.resolve()
    artifact = root / ART
    registration = load(artifact / "repair_registration.json")
    schedule = load(artifact / "gates/candidate_schedule.json")
    require(registration.get("repair_amendment_id") == "V3-E006-R009", "registration ID differs")
    require(
        registration.get("status")
        == "prospectively_registered_before_any_r009_live_diagnostic_candidate_or_model_request",
        "registration status differs",
    )
    require(registration.get("predecessor_repair_amendment_id") == "V3-E006-R008", "predecessor differs")
    require(
        registration.get("counts_at_registration")
        == {
            "r009_live_diagnostics": 0,
            "r009_live_candidate_evaluations": 0,
            "model_requests": 0,
            "behavioral_episodes": 0,
        },
        "registration counts differ",
    )
    require(
        schedule.get("status")
        == "frozen_before_any_r009_live_diagnostic_candidate_or_model_request"
        and schedule.get("r009_live_diagnostic_count") == 0
        and schedule.get("r009_live_candidate_evaluation_count") == 0
        and schedule.get("model_request_count") == 0
        and schedule.get("behavioral_episode_count") == 0,
        "schedule prospective zero-count status differs",
    )
    for commit in (BASE, R008_CLOSURE_COMMIT):
        require(
            subprocess.run(
                ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                check=False,
            ).returncode
            == 0,
            f"missing commit {commit}",
        )
    predecessor_path = resolve(
        root, registration["r008_predecessor"]["results"], "R008 predecessor"
    )
    require(sha(predecessor_path) == R008_RESULTS_SHA256, "R008 results digest differs")
    validate_r008_exhaustion_closure(load(predecessor_path))
    require(
        schedule.get("r008_predecessor") == registration.get("r008_predecessor"),
        "predecessor schedule differs",
    )
    require(
        schedule.get("candidate_budget") == schedule.get("diagnostic_budget") == 4
        and [row.get("candidate_rank") for row in schedule["candidate_pairs"]]
        == [1, 2, 3, 4]
        and [
            row.get("diagnostic_index_one_based")
            for row in schedule["known_reachable_diagnostics"]
        ]
        == [1, 2, 3, 4],
        "finite schedule budget/order differs",
    )
    for name, contract in (
        ("pinch_geometry", schedule["pinch_geometry_contract"]),
        ("joint_handoff", schedule["joint_handoff_contract"]),
        ("construction_lifecycle", schedule["construction_lifecycle_contract"]),
        ("residual_correction", schedule["residual_correction_contract"]),
    ):
        require(
            canonical_sha(contract) == schedule[f"{name}_contract_sha256"],
            f"{name} digest differs",
        )
    validate_pinch_contract(schedule["pinch_geometry_contract"])
    require(
        schedule.get("selection_rule", {}).get("algorithm_version")
        == "r009-uniform-tensor-collision-pinch-first-passing-pair-v1"
        and schedule["pinch_geometry_contract"].get("dynamic_usd_world_bounds_used")
        is False,
        "R009 selection/geometry algorithm identifier differs",
    )
    require(
        schedule.get("construction_asset_bindings") == COLLISION_ASSETS
        and registration.get("frozen_inputs", {}).get("r009_robot_collision_usd")
        == COLLISION_ASSETS["robot_usd"]
        and registration.get("frozen_inputs", {}).get("r009_cube_collision_usd")
        == COLLISION_ASSETS["cube_usd"],
        "collision-geometry USD bindings differ",
    )
    require(
        schedule["pinch_geometry_contract"]["contact_or_grab_conditioned_branch"]
        is False
        and schedule["pinch_geometry_contract"]["early_stop"] is False,
        "contact-conditioned construction entered registration",
    )
    require(
        schedule["joint_handoff_contract"]["settle_steps"] == 600
        and schedule["joint_handoff_contract"]["joint_target_write_count"] == 1,
        "handoff contract differs",
    )
    require(
        schedule["construction_lifecycle_contract"]
        == {
            "fresh_reset_steps": 75,
            "open_approach_steps": 180,
            "open_descent_steps": 180,
            "normal_close_steps": 120,
            "closed_vertical_lift_steps": 240,
            "closed_stage_transport_steps": 300,
            "joint_equilibrium_settle_steps": 600,
            "worst_case_steps": 1695,
            "registered_max_episode_length_steps": 1800,
            "registered_margin_steps": 105,
            "required_step_dt_s": 1.0 / 15.0,
            "registered_episode_length_s": 120.0,
            "only_construction_environment_timeout_changes": True,
            "behavioral_horizon_unchanged": True,
        },
        "construction lifecycle differs",
    )
    r008_schedule = load(
        resolve(root, schedule["r008_candidate_schedule"], "R008 schedule")
    )
    stripped = deepcopy(schedule["candidate_pairs"])
    for index, pair in enumerate(stripped):
        pair["construction_method"] = r008_schedule["candidate_pairs"][index][
            "construction_method"
        ]
        for stage in ("canonical_grasp", "canonical_carry"):
            for key in (
                "r009_target_cube_pose",
                "r009_acquisition_base_quaternion_world_wxyz",
                "r009_final_base_quaternion_world_wxyz",
            ):
                pair[stage].pop(key)
    require(stripped == r008_schedule["candidate_pairs"], "candidate targets/sources/order changed")
    require(
        schedule["known_reachable_diagnostics"]
        == r008_schedule["known_reachable_diagnostics"],
        "diagnostics changed",
    )
    archived = schedule.get("archived_predecessor_contracts", {})
    require(
        archived.get("status")
        == "archived_lineage_only_not_active_r009_runtime_evidence"
        and archived.get("r008_object_space_servo_contract")
        == r008_schedule.get("object_space_servo_contract")
        and archived.get("r008_object_space_servo_contract_sha256")
        == r008_schedule.get("object_space_servo_contract_sha256")
        and archived.get("r008_construction_lifecycle_contract")
        == r008_schedule.get("construction_lifecycle_contract")
        and archived.get("r008_construction_lifecycle_contract_sha256")
        == r008_schedule.get("construction_lifecycle_contract_sha256"),
        "R008 construction contracts are not archived byte-identically",
    )
    require(
        "object_space_servo_contract" not in schedule
        and "open_contact_construction_contract" not in schedule
        and "construction_horizon_contract" not in schedule,
        "predecessor construction contract remains active",
    )
    r008_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/state_repair_gate.py"
    r009_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/state_repair_gate.py"
    for name in (
        "_contact_forces",
        "_contact_coverage",
        "_reference_bounds",
        "_save_camera_evidence",
        "_companion_gate",
        "_fresh_reset_and_gate",
        "_finalize_unchanged_gates",
    ):
        require(
            function_ast(r008_source, name) == function_ast(r009_source, name),
            f"scientific helper differs: {name}",
        )
    source = r009_source.read_text(encoding="utf-8")
    require(
        "from experiments.v3.phase_e.canonical_stage_localization_v3e006.state_contract import"
        in source
        and "from experiments.v3.phase_e.canonical_stage_localization_v3e006.ood_reference import"
        in source,
        "frozen scientific gate imports differ",
    )
    require("requests.post" not in source and "policy_server" not in source, "model endpoint exists")
    material = function_ast(r009_source, "_pinch_geometry_materialize_and_gate")
    for prohibited in (
        "write_joint_state_to_sim",
        "write_root_pose_to_sim",
        "write_root_velocity_to_sim",
        "object_space_servo_command",
    ):
        require(prohibited not in material, f"prohibited construction operation entered materializer: {prohibited}")
    require(
        "pinch_alignment_command" in material
        and "_normal_joint_equilibrium_step" in material
        and "contact_or_grab_conditioned_branch_count" in material
        and "all_registered_phases_executed_unconditionally" in material,
        "R009 construction topology absent",
    )
    require(
        "def _collision_world_bounds" not in source
        and "ComputeWorldBound(" not in source
        and "_live_pinch_bounds" in material,
        "dynamic USD geometry polling entered R009",
    )
    main_ast = function_ast(r009_source, "main")
    require(
        "_pinch_geometry_materialize_and_gate" in main_ast
        and "_archived_r008_open_contact_materialize_and_gate_never_called" not in main_ast,
        "R009 dispatch reaches an unregistered topology",
    )
    gate_summary = None
    if source_gate_required:
        gate_path = artifact / "source_push_gate.json"
        require(gate_path.is_file(), "source gate absent")
        gate = load(gate_path)
        require(
            gate.get("schema_version") == SOURCE_SCHEMA
            and gate.get("status") == SOURCE_STATUS,
            "source gate differs",
        )
        try:
            validate_source_gate(gate, study_root=root, verify_raw_history=True)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        gate_summary = binding(gate_path)
    return {
        "passed": True,
        "registration": binding(artifact / "repair_registration.json"),
        "candidate_schedule": binding(artifact / "gates/candidate_schedule.json"),
        "source_push_gate": gate_summary,
        "candidate_pair_count": 4,
        "diagnostic_count": 4,
        "pinch_geometry_contract_sha256": schedule[
            "pinch_geometry_contract_sha256"
        ],
        "joint_handoff_contract_sha256": schedule["joint_handoff_contract_sha256"],
        "construction_lifecycle_contract_sha256": schedule[
            "construction_lifecycle_contract_sha256"
        ],
    }


def validate_terminal_diagnostic_pattern(
    report: Mapping[str, Any], schedule: Mapping[str, Any]
) -> bool:
    """Validate the registered diagnostic prefix and its exact blocking terminal."""

    diagnostics = report.get("known_reachable_diagnostics")
    failed = (
        report.get("status")
        == "r009_known_reachable_diagnostic_failed_candidates_not_evaluated"
    )
    require(isinstance(diagnostics, list), "diagnostic rows absent")
    require(
        report.get("r009_live_diagnostic_count") == len(diagnostics)
        and (1 <= len(diagnostics) <= 4 if failed else len(diagnostics) == 4),
        "diagnostic terminal count differs",
    )
    require(
        [row.get("diagnostic_index_one_based") for row in diagnostics]
        == list(range(1, len(diagnostics) + 1)),
        "diagnostic terminal order differs",
    )
    for index, diagnostic in enumerate(diagnostics, start=1):
        expected = schedule["known_reachable_diagnostics"][index - 1]
        require(
            diagnostic.get("registered_diagnostic") == expected
            and diagnostic.get("stage") == expected["stage"]
            and diagnostic.get("source_side") == expected["source_side"],
            "diagnostic terminal registration differs",
        )
        require(
            diagnostic.get("passed") is (not failed or index < len(diagnostics)),
            "diagnostic terminal pass/fail prefix differs",
        )
    if failed:
        require(
            report.get("passed") is False
            and report.get("attempts") == []
            and report.get("repair_candidate_evaluation_count") == 0
            and report.get("state_candidate_count") == 0
            and report.get("accepted_candidate_rank") is None
            and report.get("accepted_states") is None
            and report.get("candidate_budget") == report.get("diagnostic_budget") == 4
            and report.get("model_request_count")
            == report.get("behavioral_episode_count")
            == 0,
            "diagnostic-failure terminal zero-count boundary differs",
        )
    else:
        require(all(row.get("passed") is True for row in diagnostics), "diagnostic did not pass")
    return failed


def validate_candidate_root(root: Path, candidate_root: Path) -> dict[str, Any]:
    launch_path = candidate_root / "launch.json"; harness_path = candidate_root / "harness_result.json"; runtime_path = candidate_root / "runtime.log"
    launch, harness = load(launch_path), load(harness_path)
    verify(launch_path, harness["launch"], "harness launch"); verify(runtime_path, harness["runtime_log"], "runtime log")
    require(
        harness.get("model_request_count")
        == harness.get("behavioral_episode_count")
        == 0,
        "harness counts differ",
    )
    child_path = verify(Path(harness["child_report"]["path"]), harness["child_report"], "child report")
    report = load(child_path)
    schedule = load(root / ART / "gates/candidate_schedule.json")
    attempts = report.get("attempts")
    require(
        harness.get("status") == "completed_r009_candidate_search"
        and harness.get("process_completed") is True
        and harness.get("child_status") == report.get("status"),
        "harness terminal status differs",
    )
    diagnostics = report.get("known_reachable_diagnostics")
    diagnostic_failure = validate_terminal_diagnostic_pattern(report, schedule)
    require(
        report.get("model_request_count")
        == report.get("behavioral_episode_count")
        == 0
        and isinstance(diagnostics, list),
        "child zero counts or diagnostic count differ",
    )
    for index, diagnostic in enumerate(diagnostics, start=1):
        expected_diagnostic = schedule["known_reachable_diagnostics"][index - 1]
        require(
            diagnostic.get("registered_diagnostic") == expected_diagnostic
            and diagnostic.get("stage") == expected_diagnostic["stage"]
            and diagnostic.get("source_side") == expected_diagnostic["source_side"],
            "registered diagnostic evidence differs",
        )
        validate_construction_lifecycle(diagnostic.get("environment_lifecycle", {}), "diagnostic environment")
        require(diagnostic.get("environment_lifecycle", {}).get("closed_before_next_environment") is True, "diagnostic env not closed")
        require(diagnostic.get("fresh_reset", {}).get("passed") is True, "diagnostic reset failed")
        for camera in diagnostic.get("fresh_reset", {}).get("camera_evidence", {}).get("bindings", {}).values():
            verify(Path(camera["rgb"]["path"]), camera["rgb"], "diagnostic camera")
    require(isinstance(attempts, list) and [row.get("candidate_rank") for row in attempts] == list(range(1,len(attempts)+1)), "attempt order differs")
    for attempt in attempts:
        rank = attempt["candidate_rank"]; expected_pair = schedule["candidate_pairs"][rank-1]
        for stage in ("canonical_grasp", "canonical_carry"):
            state = attempt["stages"][stage]["candidate_state"]
            materialization = attempt["stages"][stage]["materialization_environment"]
            validate_candidate_state(
                state,
                expected_pair[stage],
                rank,
                schedule,
                materialization["fresh_reset"],
            )
            validate_construction_lifecycle(materialization["environment_lifecycle"], "materialization environment")
            require(materialization["environment_lifecycle"]["closed_before_next_environment"] is True, "materialization env not closed")
            require(materialization["fresh_reset"].get("passed") is True, "materialization reset failed")
            for camera in state.get("camera_evidence", {}).get("bindings", {}).values():
                verify(Path(camera["rgb"]["path"]), camera["rgb"], "candidate camera")
        require(attempt.get("passed") is all(attempt["stages"][stage]["candidate_state"]["passed"] for stage in ("canonical_grasp","canonical_carry")), "pair pass differs")
    passed = [row for row in attempts if row.get("passed")]
    if diagnostic_failure:
        require(not passed, "diagnostic failure retained a passed candidate")
    elif report.get("passed"):
        require(
            report.get("status")
            == "passed_r009_state_repair_not_released_for_behavior",
            "pass status differs",
        )
        require(len(passed) == 1 and passed[0] is attempts[-1], "first pass differs")
        require(report.get("accepted_candidate_rank") == attempts[-1]["candidate_rank"], "accepted rank differs")
        require(report.get("accepted_states") == attempts[-1]["stages"], "accepted states differ")
    else:
        require(
            report.get("status")
            == "r009_candidate_budget_exhausted_no_valid_state_pair",
            "exhaustion status differs",
        )
        require(not passed and len(attempts) == 4 and report.get("accepted_candidate_rank") is None, "exhaustion differs")
        require(report.get("accepted_states") is None, "exhaustion retained accepted states")
    if not diagnostic_failure:
        require(
            report.get("repair_candidate_evaluation_count") == len(attempts)
            and report.get("first_passing_rule_obeyed") is True
            and report.get("candidate_budget") == report.get("diagnostic_budget") == 4,
            "terminal selection/count fields differ",
        )
    for group in (launch.get("input_bindings", {}), launch.get("formal_health_preflight", {})):
        for row in group.values():
            verify(Path(row["path"]), row, "launch evidence")
    for key in (
        "repair_registration", "candidate_schedule", "source_push_gate",
        "original_v3e006_closure_binding", "r008_predecessor_results", "ood_freeze",
        "e004_full_reset_reference", "e004_candidate", "construction_source", "video",
    ):
        verify(Path(report[key]["path"]), report[key], f"report {key}")
    verify(Path(report["frozen_e004_runtime_bindings"]["path"]), report["frozen_e004_runtime_bindings"], "runtime binding")
    for row in report["scene_assets"].values():
        verify(Path(row["path"]), row, "scene asset")
    execution = report.get("execution_evidence")
    require(isinstance(execution, Mapping), "execution evidence absent")
    for key, row in execution.get("input_bindings", {}).items():
        verify(Path(row["path"]), row, f"execution input {key}")
    require(
        report["repair_registration"] == launch["input_bindings"]["repair_registration"]
        and report["candidate_schedule"] == launch["input_bindings"]["candidate_schedule"]
        and report["source_push_gate"] == launch["input_bindings"]["source_push_gate"]
        and report["original_v3e006_closure_binding"]
        == launch["input_bindings"]["original_closure_binding"]
        and report["r008_predecessor_results"]
        == launch["input_bindings"]["predecessor_closure_binding"],
        "launch/child frozen-input cross-binding differs",
    )
    return {"passed": True, "launch": binding(launch_path), "harness": binding(harness_path), "child_report": binding(child_path), "model_request_count": 0, "behavioral_episode_count": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--pre-source-gate", action="store_true")
    parser.add_argument("--verify-raw", action="store_true")
    parser.add_argument("--candidate-root", type=Path)
    args = parser.parse_args()
    result = validate_static(args.study_root, source_gate_required=not args.pre_source_gate)
    if args.candidate_root:
        require(args.verify_raw, "candidate root requires --verify-raw")
        result["candidate_evidence"] = validate_candidate_root(args.study_root.resolve(), args.candidate_root.resolve())
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
