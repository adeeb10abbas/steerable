#!/usr/bin/env python3
"""Static and target-raw validator for V3-E006-R012."""

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
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r012.pinch_geometry import (
    collision_center_env_local,
    pinch_alignment_command,
    reconstruct_collision_bounds_env_local,
    transform_collision_corners_env_local,
    validate_attachment_preflight_contract,
    validate_contract as validate_pinch_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r012.predecessor_contract import (
    R011_CLOSURE_COMMIT,
    R011_RESULTS_SHA256,
    validate_r011_scene_sync_failure_closure,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r012.source_gate_contract import (
    SCHEMA as SOURCE_SCHEMA,
    STATUS as SOURCE_STATUS,
    validate_source_gate,
)


ROOT = Path(__file__).resolve().parents[1]
ART = Path("artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012")
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


def function_call_count(path: Path, function_name: str, call_name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        row
        for row in tree.body
        if isinstance(row, ast.FunctionDef) and row.name == function_name
    )
    return sum(
        1
        for row in ast.walk(node)
        if isinstance(row, ast.Call)
        and (
            (isinstance(row.func, ast.Name) and row.func.id == call_name)
            or (isinstance(row.func, ast.Attribute) and row.func.attr == call_name)
        )
    )


def _dotted_call_name(value: ast.expr) -> str:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        prefix = _dotted_call_name(value.value)
        return f"{prefix}.{value.attr}" if prefix else value.attr
    return ""


def validate_preflight_source_topology_text(source: str) -> bool:
    """Prove the one-shot tensor/sync/cache order from Python AST alone."""

    tree = ast.parse(source)
    function = next(
        (
            row
            for row in tree.body
            if isinstance(row, ast.FunctionDef)
            and row.name == "_run_geometry_attachment_preflight"
        ),
        None,
    )
    require(function is not None, "R012 scene-sync preflight function absent")
    calls = [row for row in ast.walk(function) if isinstance(row, ast.Call)]
    named = [(_dotted_call_name(row.func), row) for row in calls]
    snapshots = [row for name, row in named if name == "_live_pinch_bounds"]
    scene_calls = [
        row for name, row in named if name.endswith("update_transformations_scene")
    ]
    setting_reads = [row for name, row in named if name == "settings.get_as_bool"]
    bbox_caches = [row for name, row in named if name == "UsdGeom.BBoxCache"]
    xform_caches = [row for name, row in named if name == "UsdGeom.XformCache"]
    scene_integer_assignments = [
        row
        for row in ast.walk(function)
        if isinstance(row, ast.Assign)
        and len(row.targets) == 1
        and isinstance(row.targets[0], ast.Name)
        and row.targets[0].id == "scene_path_int"
    ]
    require(
        len(snapshots) == 1
        and len(scene_calls) == 1
        and len(setting_reads) == 2
        and len(bbox_caches) == 1
        and len(xform_caches) == 1,
        "R012 tensor/sync/setting/cache call multiplicity differs",
    )
    require(
        len(scene_integer_assignments) == 1,
        "R012 scene path integer assignment multiplicity differs",
    )
    scene_integer_assignment = scene_integer_assignments[0]
    outer = scene_integer_assignment.value
    require(
        isinstance(outer, ast.Call)
        and isinstance(outer.func, ast.Name)
        and outer.func.id == "int"
        and len(outer.args) == 1
        and not outer.keywords
        and isinstance(outer.args[0], ast.Call)
        and _dotted_call_name(outer.args[0].func)
        == "PhysicsSchemaTools.sdfPathToInt"
        and len(outer.args[0].args) == 1
        and not outer.args[0].keywords
        and isinstance(outer.args[0].args[0], ast.Name)
        and outer.args[0].args[0].id == "configured_scene_path",
        "R012 scene path-to-integer dataflow differs",
    )
    scene_call = scene_calls[0]
    require(
        len(scene_call.args) == 3
        and not scene_call.keywords
        and isinstance(scene_call.args[0], ast.Name)
        and scene_call.args[0].id == "scene_path_int"
        and isinstance(scene_call.args[1], ast.Constant)
        and scene_call.args[1].value is True
        and isinstance(scene_call.args[2], ast.Constant)
        and scene_call.args[2].value is False,
        "R012 scene-specific synchronization arguments differ",
    )
    before_read, after_read = sorted(setting_reads, key=lambda row: row.lineno)
    require(
        scene_integer_assignment.lineno
        < snapshots[0].lineno
        < before_read.lineno
        < scene_call.lineno
        < after_read.lineno
        < bbox_caches[0].lineno
        and after_read.lineno < xform_caches[0].lineno,
        "R012 tensor snapshot / scene sync / post-setting / fresh-cache order differs",
    )
    forbidden_motion = {
        name
        for name, _row in named
        if name.rsplit(".", 1)[-1] in {"step", "forward", "render"}
    }
    require(not forbidden_motion, "R012 preflight contains a step/forward/render call")
    require(
        not any(name.rsplit(".", 1)[-1] == "update_transformations" for name, _ in named),
        "R012 preflight contains the global PhysX synchronization call",
    )
    require(
        not any(
            name.startswith("settings.") and name != "settings.get_as_bool"
            for name, _row in named
        ),
        "R012 preflight mutates or otherwise accesses the setting outside exact reads",
    )
    return True


def validate_preflight_source_topology(path: Path) -> bool:
    return validate_preflight_source_topology_text(path.read_text(encoding="utf-8"))


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
        require(
            row.get("collision_prim_path") == path
            and row.get("compute_relative_bound_ancestor_prim_path") == body_path
            and row.get("additional_prim_or_world_transform_after_relative_bound")
            is False,
            f"{label} shape path/relative-bound semantics differ",
        )
        minimum = _finite_vector(
            row.get("collision_prim_body_relative_aligned_minimum_m"),
            3,
            f"{label} body-relative minimum",
        )
        maximum = _finite_vector(
            row.get("collision_prim_body_relative_aligned_maximum_m"),
            3,
            f"{label} body-relative maximum",
        )
        require(np.all(maximum > minimum), f"{label} local range differs")
        shape_corners = np.asarray(row.get("collision_corners_body_m"), dtype=np.float64)
        require(shape_corners.shape == (8, 3) and np.isfinite(shape_corners).all(), f"{label} shape corners differ")
        expected_corners = np.asarray(
            [
                [x, y, z]
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
                for z in (minimum[2], maximum[2])
            ],
            dtype=np.float64,
        )
        require(
            np.array_equal(shape_corners, expected_corners),
            f"{label} body-relative aligned corners differ",
        )
        flattened.extend(shape_corners.tolist())
    require(np.array_equal(corners, np.asarray(flattened, dtype=np.float64)), f"{label} aggregate corners differ")
    minimum = np.min(corners, axis=0); maximum = np.max(corners, axis=0)
    require(np.array_equal(minimum, _finite_vector(value.get("minimum_body_m"), 3, f"{label} minimum")), f"{label} minimum math differs")
    require(np.array_equal(maximum, _finite_vector(value.get("maximum_body_m"), 3, f"{label} maximum")), f"{label} maximum math differs")
    require(np.allclose(value.get("center_body_m"), 0.5 * (minimum + maximum), atol=1e-12, rtol=0), f"{label} center math differs")
    require(np.allclose(value.get("half_extents_body_m"), 0.5 * (maximum - minimum), atol=1e-12, rtol=0), f"{label} half math differs")
    require(
        value.get("extraction_api")
        == "UsdGeom.BBoxCache.ComputeRelativeBound(collision_prim, owning_rigid_body).ComputeAlignedRange"
        and value.get("additional_transform_after_compute_relative_bound") is False,
        f"{label} extraction API differs",
    )
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


def _archived_validate_geometry_attachment_preflight_r011(
    value: Mapping[str, Any], schedule: Mapping[str, Any]
) -> bool:
    """Independently recompute the retained one-shot tensor/USD oracle."""

    contract = schedule.get("geometry_attachment_preflight_contract")
    require(isinstance(contract, Mapping), "geometry preflight contract absent")
    try:
        validate_attachment_preflight_contract(contract)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    require(
        value.get("schema_version")
        == "vla-wam-shared-v3e006-r012-geometry-attachment-preflight-v1"
        and value.get("model_request_count")
        == value.get("behavioral_episode_count")
        == 0
        and value.get("geometry_attachment_preflight_contract") == contract
        and value.get("geometry_attachment_preflight_contract_sha256")
        == schedule.get("geometry_attachment_preflight_contract_sha256")
        == canonical_sha(contract),
        "geometry preflight identity/counts differ",
    )
    require(
        value.get("physics_to_usd_setting_path")
        == contract.get("physics_to_usd_setting_path")
        == "/physics/updateToUsd"
        and value.get("physics_to_usd_setting_before_one_shot_sync")
        == value.get("physics_to_usd_setting_after_one_shot_sync")
        and value.get("physics_to_usd_setting_unchanged") is True
        and value.get("physics_to_usd_synchronization_call")
        == "omni.physx.get_physx_interface().update_transformations_scene(scene_path_int, True, False)"
        and value.get("physics_to_usd_synchronization_call_count") == 1
        and value.get("simulation_forward_before_tensor_snapshot") is False
        and value.get("physics_to_usd_synchronized_reset_steps") == 75
        and value.get("physics_or_action_steps_between_tensor_snapshot_and_oracle")
        == 0
        and value.get("usd_world_bounds_validation_only_not_controller_input")
        is True,
        "geometry preflight synchronization evidence differs",
    )
    stage_identity = value.get("stage_and_scene_identity")
    require(isinstance(stage_identity, Mapping), "stage/scene identity evidence absent")
    stage_id_before = stage_identity.get("stage_cache_id_before")
    stage_id_after = stage_identity.get("stage_cache_id_after")
    expected_scene_path = contract["expected_configured_physics_scene_path"]
    require(
        stage_identity.get("simulation_context_api")
        == "isaaclab.sim.SimulationContext.instance()"
        and stage_identity.get("environment_sim_is_simulation_context_singleton")
        is True
        and stage_identity.get("initial_stage_is_current_stage_before_sync") is True
        and stage_identity.get("initial_stage_is_current_stage_after_sync") is True
        and isinstance(stage_id_before, int)
        and not isinstance(stage_id_before, bool)
        and stage_id_before != 0
        and stage_id_after == stage_id_before
        and stage_identity.get("stage_cache_id_valid_nonzero_and_unchanged") is True
        and isinstance(stage_identity.get("root_layer_identifier_before"), str)
        and bool(stage_identity.get("root_layer_identifier_before"))
        and stage_identity.get("root_layer_identifier_after")
        == stage_identity.get("root_layer_identifier_before")
        and stage_identity.get("root_layer_identifier_unchanged") is True
        and stage_identity.get("configured_physics_scene_path")
        == expected_scene_path
        and stage_identity.get("resolved_physics_scene_paths")
        == [expected_scene_path]
        and stage_identity.get("unique_configured_physics_scene") is True
        and isinstance(stage_identity.get("physics_scene_path_integer"), int)
        and not isinstance(stage_identity.get("physics_scene_path_integer"), bool)
        and stage_identity.get("physics_scene_path_integer") != 0
        and stage_identity.get("physics_scene_path_integer_api")
        == "pxr.PhysicsSchemaTools.sdfPathToInt(physics_scene_prim_path)",
        "stage/scene mapping evidence differs",
    )
    require(
        value.get("physics_to_usd_synchronization_call_arguments")
        == {
            "physics_scene_path": expected_scene_path,
            "physics_scene_path_integer": stage_identity.get(
                "physics_scene_path_integer"
            ),
            "update_to_usd": True,
            "update_velocities_to_usd": False,
        },
        "scene path-to-integer call-argument binding differs",
    )
    lifecycle = value.get("environment_lifecycle")
    require(isinstance(lifecycle, Mapping), "geometry preflight lifecycle absent")
    validate_construction_lifecycle(lifecycle, "geometry preflight environment")
    require(
        lifecycle.get("candidate_rank") == 0
        and lifecycle.get("stage") == "attachment_geometry"
        and lifecycle.get("role")
        == "scene_specific_relative_bound_tensor_world_oracle_preflight"
        and lifecycle.get("closed_before_next_environment") is True,
        "geometry preflight lifecycle identity differs",
    )
    fresh_reset = value.get("fresh_reset")
    require(
        isinstance(fresh_reset, Mapping) and fresh_reset.get("passed") is True,
        "geometry preflight fresh reset failed",
    )
    resolution = value.get("collision_geometry_resolution")
    require(isinstance(resolution, Mapping), "geometry preflight resolution absent")
    inventory = resolution.get("inventory")
    static = resolution.get("static_body_local_collision_geometry")
    require(
        isinstance(inventory, Mapping) and isinstance(static, Mapping),
        "geometry preflight inventory/static geometry absent",
    )
    role_specs = (
        (
            "left",
            inventory.get("left_inner_finger_body_prim_path"),
            inventory.get("left_collision_prim_paths"),
        ),
        (
            "right",
            inventory.get("right_inner_finger_body_prim_path"),
            inventory.get("right_collision_prim_paths"),
        ),
        ("cube", inventory.get("cube_root_prim_path"), inventory.get("cube_collision_prim_paths")),
    )
    for role, body, paths in role_specs:
        require(
            isinstance(body, str)
            and isinstance(paths, list)
            and paths
            and isinstance(static.get(role), Mapping),
            f"geometry preflight {role} inventory differs",
        )
        _validate_local_collision_geometry(
            static[role],
            label=f"geometry preflight {role}",
            body_path=body,
            collision_paths=paths,
        )
    origin = _finite_vector(
        value.get("scene_env_origin_world_m"), 3, "geometry preflight env origin"
    )
    require(
        np.array_equal(
            origin,
            _finite_vector(
                resolution.get("scene_env_origin_world_m_at_resolution"),
                3,
                "geometry resolution env origin",
            ),
        ),
        "geometry preflight env origin link differs",
    )
    identity = {
        "inventory": inventory,
        "static_body_local_collision_geometry": static,
        "scene_env_origin_world_m_at_resolution": origin.tolist(),
    }
    require(
        value.get("geometry_identity_sha256")
        == resolution.get("geometry_identity_sha256")
        == canonical_sha(identity),
        "geometry preflight identity digest differs",
    )

    owner_rows = value.get("owner_pose_oracle_rows")
    require(
        isinstance(owner_rows, list)
        and [row.get("role") for row in owner_rows] == ["left", "right", "cube"],
        "geometry preflight owner rows differ",
    )
    owner_passes: list[bool] = []
    role_pose: dict[str, Mapping[str, Any]] = {}
    for row, (role, body, _paths) in zip(owner_rows, role_specs, strict=True):
        require(
            row.get("owning_rigid_body_prim_path") == body,
            f"geometry preflight {role} owner path differs",
        )
        pose = row.get("live_tensor_body_pose")
        require(isinstance(pose, Mapping), f"geometry preflight {role} tensor pose absent")
        raw = _finite_vector(
            pose.get("position_tensor_world_m"), 3, f"geometry preflight {role} tensor origin"
        )
        pose_origin = _finite_vector(
            pose.get("scene_env_origin_world_m"), 3, f"geometry preflight {role} pose origin"
        )
        local = _finite_vector(
            pose.get("position_env_local_m"), 3, f"geometry preflight {role} local origin"
        )
        require(
            np.array_equal(pose_origin, origin)
            and np.allclose(local, raw - origin, atol=1e-12, rtol=0),
            f"geometry preflight {role} tensor origin semantics differ",
        )
        _finite_vector(
            pose.get("quaternion_world_wxyz"), 4, f"geometry preflight {role} quaternion"
        )
        quaternion = pose["quaternion_world_wxyz"]
        usd_origin = _finite_vector(
            row.get("usd_world_origin_m"), 3, f"geometry preflight {role} USD origin"
        )
        position_error = float(np.max(np.abs(usd_origin - raw)))
        usd_axes = np.asarray(row.get("usd_world_axes"), dtype=np.float64)
        tensor_axes = np.asarray(row.get("tensor_world_axes"), dtype=np.float64)
        require(
            usd_axes.shape == tensor_axes.shape == (3, 3)
            and np.isfinite(usd_axes).all()
            and np.isfinite(tensor_axes).all(),
            f"geometry preflight {role} axes differ",
        )
        recomputed_tensor_axes = np.asarray(
            [
                collision_center_env_local(
                    body_position_env_local=[0.0, 0.0, 0.0],
                    body_quaternion_world_wxyz=quaternion,
                    collision_center_body=axis,
                )
                for axis in np.eye(3, dtype=np.float64)
            ],
            dtype=np.float64,
        )
        require(
            np.allclose(
                tensor_axes, recomputed_tensor_axes, atol=1e-12, rtol=0
            ),
            f"geometry preflight {role} tensor axes/quaternion link differs",
        )
        axis_errors = [
            math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(usd_axis / np.linalg.norm(usd_axis), tensor_axis / np.linalg.norm(tensor_axis)),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            for usd_axis, tensor_axis in zip(usd_axes, tensor_axes, strict=True)
        ]
        orientation_error = max(axis_errors)
        row_passed = (
            position_error <= float(contract["owner_pose_tolerance_m_inclusive"])
            and orientation_error
            <= float(contract["owner_orientation_tolerance_deg_inclusive"])
        )
        require(
            math.isclose(
                float(row.get("maximum_absolute_position_error_m")),
                position_error,
                rel_tol=0,
                abs_tol=1e-15,
            )
            and math.isclose(
                float(row.get("maximum_axis_orientation_error_deg")),
                orientation_error,
                rel_tol=0,
                abs_tol=1e-12,
            )
            and row.get("position_tolerance_m_inclusive")
            == contract["owner_pose_tolerance_m_inclusive"]
            and row.get("orientation_tolerance_deg_inclusive")
            == contract["owner_orientation_tolerance_deg_inclusive"]
            and row.get("passed") is row_passed,
            f"geometry preflight {role} owner-pose decision differs",
        )
        role_pose[role] = pose
        owner_passes.append(row_passed)

    rows = value.get("oracle_rows")
    expected_paths = [
        path for _role, _body, paths in role_specs for path in paths
    ]
    require(
        isinstance(rows, list)
        and value.get("collision_prim_count") == len(rows) == len(expected_paths)
        and value.get("expected_collision_prim_paths") == expected_paths,
        "geometry preflight collision-row inventory differs",
    )
    shape_by_path = {
        row["collision_prim_path"]: (role, static[role]["body_prim_path"], row)
        for role in ("left", "right", "cube")
        for row in static[role]["shape_local_geometry"]
    }
    row_passes: list[bool] = []
    observed_paths: list[str] = []
    for row in rows:
        path = str(row.get("collision_prim_path", ""))
        require(path in shape_by_path, f"geometry preflight unknown collision path: {path}")
        role, body, shape = shape_by_path[path]
        require(
            row.get("role") == role
            and row.get("owning_rigid_body_prim_path") == body
            and row.get("body_relative_geometry") == shape
            and row.get("live_tensor_body_pose") == role_pose[role],
            f"geometry preflight {path} attachment link differs",
        )
        pose = role_pose[role]
        reconstructed_corners = transform_collision_corners_env_local(
            body_position_env_local=pose["position_env_local_m"],
            body_quaternion_world_wxyz=pose["quaternion_world_wxyz"],
            collision_corners_body=shape["collision_corners_body_m"],
        )
        require(
            np.allclose(
                row.get("tensor_reconstructed_corners_env_local_m"),
                reconstructed_corners,
                atol=1e-12,
                rtol=0,
            ),
            f"geometry preflight {path} reconstructed corners differ",
        )
        tensor_min = np.min(reconstructed_corners, axis=0) + origin
        tensor_max = np.max(reconstructed_corners, axis=0) + origin
        oracle_min = _finite_vector(
            row.get("usd_compute_world_bound_minimum_world_m"),
            3,
            f"geometry preflight {path} oracle minimum",
        )
        oracle_max = _finite_vector(
            row.get("usd_compute_world_bound_maximum_world_m"),
            3,
            f"geometry preflight {path} oracle maximum",
        )
        minimum_error = float(np.max(np.abs(tensor_min - oracle_min)))
        maximum_error = float(np.max(np.abs(tensor_max - oracle_max)))
        row_passed = (
            minimum_error <= float(contract["oracle_tolerance_m_inclusive"])
            and maximum_error <= float(contract["oracle_tolerance_m_inclusive"])
        )
        require(
            np.allclose(
                row.get("tensor_reconstructed_minimum_world_m"),
                tensor_min,
                atol=1e-12,
                rtol=0,
            )
            and np.allclose(
                row.get("tensor_reconstructed_maximum_world_m"),
                tensor_max,
                atol=1e-12,
                rtol=0,
            )
            and math.isclose(
                float(row.get("maximum_absolute_minimum_error_m")),
                minimum_error,
                rel_tol=0,
                abs_tol=1e-15,
            )
            and math.isclose(
                float(row.get("maximum_absolute_maximum_error_m")),
                maximum_error,
                rel_tol=0,
                abs_tol=1e-15,
            )
            and row.get("tolerance_m_inclusive")
            == contract["oracle_tolerance_m_inclusive"]
            and row.get("passed") is row_passed,
            f"geometry preflight {path} oracle decision differs",
        )
        observed_paths.append(path)
        row_passes.append(row_passed)
    require(
        observed_paths == expected_paths,
        "geometry preflight oracle row order/inventory differs",
    )
    overall = (
        observed_paths == expected_paths
        and value.get("all_inventory_paths_evaluated_once") is True
        and all(owner_passes)
        and all(row_passes)
    )
    require(
        value.get("passed") is overall
        and value.get("status")
        == (
            "passed_r012_scene_specific_relative_bound_tensor_world_oracle_preflight"
            if overall
            else "failed_r012_scene_specific_relative_bound_tensor_world_oracle_preflight"
        ),
        "geometry preflight terminal decision differs",
    )
    return overall


def validate_geometry_attachment_preflight(
    value: Mapping[str, Any], schedule: Mapping[str, Any]
) -> bool:
    """Independently recompute the R012 live-tensor-only sanity receipt."""

    contract = schedule.get("geometry_attachment_preflight_contract")
    require(isinstance(contract, Mapping), "geometry preflight contract absent")
    try:
        validate_attachment_preflight_contract(contract)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    require(
        value.get("schema_version")
        == "vla-wam-shared-v3e006-r012-live-tensor-geometry-sanity-preflight-v1"
        and value.get("model_request_count") == value.get("behavioral_episode_count") == 0
        and value.get("geometry_attachment_preflight_contract") == contract
        and value.get("geometry_attachment_preflight_contract_sha256")
        == schedule.get("geometry_attachment_preflight_contract_sha256")
        == canonical_sha(contract),
        "live-tensor preflight identity/counts differ",
    )
    require(
        value.get("dynamic_state_source") == contract["dynamic_state_source"]
        and value.get("dynamic_usd_world_state_used") is False
        and value.get("physics_to_usd_sync_call_count") == 0
        and value.get("dynamic_usd_world_bound_or_xform_query_count") == 0,
        "dynamic USD exclusion differs",
    )
    lifecycle = value.get("environment_lifecycle")
    require(isinstance(lifecycle, Mapping), "tensor preflight lifecycle absent")
    validate_construction_lifecycle(lifecycle, "tensor preflight environment")
    require(
        lifecycle.get("candidate_rank") == 0
        and lifecycle.get("stage") == "attachment_geometry"
        and lifecycle.get("role") == "live_physics_tensor_geometry_sanity_preflight"
        and lifecycle.get("closed_before_next_environment") is True,
        "tensor preflight lifecycle identity differs",
    )
    fresh_reset = value.get("fresh_reset")
    require(isinstance(fresh_reset, Mapping) and fresh_reset.get("passed") is True, "tensor preflight fresh reset failed")
    resolution = value.get("collision_geometry_resolution")
    require(isinstance(resolution, Mapping), "tensor preflight geometry absent")
    inventory = resolution.get("inventory")
    static = resolution.get("static_body_local_collision_geometry")
    require(isinstance(inventory, Mapping) and isinstance(static, Mapping), "tensor inventory/static geometry absent")
    role_specs = (
        ("left", inventory.get("left_inner_finger_body_prim_path"), inventory.get("left_collision_prim_paths")),
        ("right", inventory.get("right_inner_finger_body_prim_path"), inventory.get("right_collision_prim_paths")),
        ("cube", inventory.get("cube_root_prim_path"), inventory.get("cube_collision_prim_paths")),
    )
    for role, body, paths in role_specs:
        require(isinstance(body, str) and isinstance(paths, list) and paths and isinstance(static.get(role), Mapping), f"tensor preflight {role} inventory differs")
        _validate_local_collision_geometry(static[role], label=f"tensor preflight {role}", body_path=body, collision_paths=paths)
    origin = _finite_vector(value.get("scene_env_origin_world_m"), 3, "tensor preflight origin")
    require(
        np.array_equal(origin, _finite_vector(resolution.get("scene_env_origin_world_m_at_resolution"), 3, "geometry origin"))
        and resolution.get("dynamic_usd_world_bounds_used") is False,
        "tensor preflight origin/dynamic-source differs",
    )
    identity = {
        "inventory": inventory,
        "static_body_local_collision_geometry": static,
        "scene_env_origin_world_m_at_resolution": origin.tolist(),
    }
    require(
        value.get("geometry_identity_sha256") == resolution.get("geometry_identity_sha256") == canonical_sha(identity),
        "tensor geometry identity differs",
    )
    require(
        value.get("static_body_local_geometry_sha256")
        == {role: static[role]["canonical_sha256"] for role in ("left", "right", "cube")},
        "tensor static geometry hash map differs",
    )
    checks = value.get("tensor_body_index_name_ownership_checks")
    require(
        checks == {
            "left_index_in_range": True,
            "right_index_in_range": True,
            "indices_distinct": True,
            "left_name_exact": True,
            "right_name_exact": True,
        }
        and inventory.get("left_robot_body_tensor_index") != inventory.get("right_robot_body_tensor_index")
        and inventory.get("left_robot_body_tensor_name") == str(inventory["left_inner_finger_body_prim_path"]).rsplit("/", 1)[-1]
        and inventory.get("right_robot_body_tensor_name") == str(inventory["right_inner_finger_body_prim_path"]).rsplit("/", 1)[-1],
        "tensor index/name ownership differs",
    )
    live = value.get("live_tensor_geometry")
    require(isinstance(live, Mapping), "live tensor geometry absent")
    recomputed: dict[str, Mapping[str, Any]] = {}
    for role in ("left", "right", "cube"):
        require(isinstance(live.get(role), Mapping), f"live tensor {role} absent")
        recomputed[role] = _validate_live_tensor_geometry(
            live[role], static[role], origin=origin, label=f"tensor preflight {role}"
        )
    left = np.asarray(recomputed["left"]["collision_center_env_local_m"], dtype=np.float64)
    right = np.asarray(recomputed["right"]["collision_center_env_local_m"], dtype=np.float64)
    separation = float(np.linalg.norm(right - left))
    cube_dimensions = 2.0 * np.asarray(recomputed["cube"]["aabb_half_extents_env_local_m"], dtype=np.float64)
    sep_bounds = contract["pad_collision_center_separation_m_inclusive"]
    dim_bounds = contract["cube_aabb_dimension_m_each_inclusive"]
    require(
        math.isclose(float(value.get("pad_collision_center_separation_m")), separation, abs_tol=1e-12, rel_tol=0)
        and value.get("pad_collision_center_separation_m_inclusive") == sep_bounds
        and sep_bounds[0] <= separation <= sep_bounds[1]
        and np.allclose(value.get("cube_aabb_dimensions_m"), cube_dimensions, atol=1e-12, rtol=0)
        and value.get("cube_aabb_dimension_m_each_inclusive") == dim_bounds
        and bool(np.all(cube_dimensions >= dim_bounds[0]))
        and bool(np.all(cube_dimensions <= dim_bounds[1])),
        "tensor geometry dimension sanity differs",
    )
    reset_identity = value.get("fresh_reset_tensor_identity")
    require(isinstance(reset_identity, Mapping), "fresh-reset tensor identity absent")
    live_cube_p = _finite_vector(reset_identity.get("live_cube_position_tensor_world_m"), 3, "live reset cube position")
    reset_cube_p = _finite_vector(reset_identity.get("fresh_reset_cube_position_world_m"), 3, "saved reset cube position")
    live_cube_q = _finite_vector(reset_identity.get("live_cube_quaternion_world_wxyz"), 4, "live reset cube quaternion")
    reset_cube_q = _finite_vector(reset_identity.get("fresh_reset_cube_quaternion_world_wxyz"), 4, "saved reset cube quaternion")
    live_q = np.asarray(reset_identity.get("live_robot_joint_position_rad"), dtype=np.float64)
    reset_q = np.asarray(reset_identity.get("fresh_reset_robot_joint_position_rad"), dtype=np.float64)
    require(
        live_q.shape == reset_q.shape == (13,)
        and np.isfinite(live_q).all() and np.isfinite(reset_q).all()
        and np.array_equal(live_cube_p, reset_cube_p)
        and (np.array_equal(live_cube_q, reset_cube_q) or np.array_equal(live_cube_q, -reset_cube_q))
        and np.array_equal(live_q, reset_q)
        and reset_identity.get("cube_position_bitwise_equal") is True
        and reset_identity.get("cube_quaternion_sign_invariant_bitwise_equal") is True
        and reset_identity.get("robot_joint_position_bitwise_equal") is True
        and reset_identity.get("passed") is True,
        "fresh-reset tensor identity differs",
    )
    expected_pass = bool(value.get("all_live_tensor_values_finite") is True)
    require(value.get("passed") is expected_pass and value.get("status") == (
        "passed_r012_live_tensor_geometry_sanity_preflight" if expected_pass else "failed_r012_live_tensor_geometry_sanity_preflight"
    ), "tensor preflight pass/status differs")
    return expected_pass


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
    preflight: Mapping[str, Any],
) -> None:
    """Independently regenerate R012 geometry commands and unchanged selection."""

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
        "R012 prohibited state write or branch differs",
    )
    require(
        construction.get("all_registered_phases_executed_unconditionally") is True
        and construction.get("captured_joint_target_write_count") == 1
        and construction.get("cartesian_action_manager_apply_count_during_joint_settle") == 0,
        "R012 unconditional execution or handoff differs",
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
    identity = {
        "inventory": inventory,
        "static_body_local_collision_geometry": static_geometry,
        "scene_env_origin_world_m_at_resolution": resolution_origin.tolist(),
    }
    identity_sha = canonical_sha(identity)
    require(
        resolution.get("geometry_identity_sha256") == identity_sha
        and preflight.get("geometry_identity_sha256") == identity_sha
        and construction.get("geometry_attachment_preflight_identity")
        == {
            "preflight_geometry_identity_sha256": identity_sha,
            "rank_stage_geometry_identity_sha256": identity_sha,
            "passed": True,
        },
        "candidate/preflight geometry identity differs",
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
        == expected_stage["r010_target_cube_pose"]["position_world_m"]
        and quaternion_equivalent(
            target_pose[3:],
            expected_stage["r010_target_cube_pose"]["quaternion_world_wxyz"],
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
    acquisition_q = expected_stage["r010_acquisition_base_quaternion_world_wxyz"]
    final_q = expected_stage["r010_final_base_quaternion_world_wxyz"]
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
    require(registration.get("repair_amendment_id") == "V3-E006-R012", "registration ID differs")
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
    r008_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r012/state_repair_gate.py"
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
    require("object_space_servo_command" in material and "_normal_joint_equilibrium_step" in material, "R012 controller topology absent")
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


def validate_r012_runtime_aliases(
    candidate_pairs: list[Mapping[str, Any]],
    r010_candidate_pairs: list[Mapping[str, Any]],
) -> None:
    """Require every R010 runtime-driving alias and pair byte/data mapping unchanged."""

    require(
        candidate_pairs == r010_candidate_pairs
        and len(candidate_pairs) == 4
        and [row.get("candidate_rank") for row in candidate_pairs] == [1, 2, 3, 4],
        "R012 changed R010 candidate pairs or runtime-driving aliases",
    )
    for index, pair in enumerate(candidate_pairs, start=1):
        for stage_name in ("canonical_grasp", "canonical_carry"):
            stage = pair.get(stage_name)
            require(isinstance(stage, Mapping), f"rank {index} {stage_name} absent")
            require(
                stage.get("r010_target_cube_pose") == stage.get("r009_target_cube_pose")
                and stage.get("r010_acquisition_base_quaternion_world_wxyz")
                == stage.get("r009_acquisition_base_quaternion_world_wxyz")
                and stage.get("r010_final_base_quaternion_world_wxyz")
                == stage.get("r009_final_base_quaternion_world_wxyz"),
                f"rank {index} {stage_name} retained alias differs",
            )


def _archived_validate_static_r011(root: Path, *, source_gate_required: bool) -> dict[str, Any]:
    root = root.resolve()
    artifact = root / ART
    registration = load(artifact / "repair_registration.json")
    schedule = load(artifact / "gates/candidate_schedule.json")
    require(registration.get("repair_amendment_id") == "V3-E006-R012", "registration ID differs")
    require(
        registration.get("status")
        == "prospectively_registered_before_any_r012_live_preflight_diagnostic_candidate_or_model_request"
        and registration.get("predecessor_repair_amendment_id") == "V3-E006-R010",
        "registration status/predecessor differs",
    )
    require(
        registration.get("counts_at_registration")
        == {
            "r012_geometry_attachment_preflights": 0,
            "r012_live_diagnostics": 0,
            "r012_live_candidate_evaluations": 0,
            "model_requests": 0,
            "behavioral_episodes": 0,
        },
        "registration counts differ",
    )
    require(
        schedule.get("status")
        == "frozen_before_any_r012_live_preflight_diagnostic_candidate_or_model_request"
        and schedule.get("r012_geometry_attachment_preflight_count") == 0
        and schedule.get("r012_live_diagnostic_count") == 0
        and schedule.get("r012_live_candidate_evaluation_count") == 0
        and schedule.get("model_request_count") == 0
        and schedule.get("behavioral_episode_count") == 0,
        "schedule prospective zero-count status differs",
    )
    for commit in (BASE, R010_CLOSURE_COMMIT):
        require(
            subprocess.run(
                ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                check=False,
            ).returncode
            == 0,
            f"missing commit {commit}",
        )
    predecessor_path = resolve(
        root, registration["r010_predecessor"]["results"], "R010 predecessor"
    )
    require(sha(predecessor_path) == R010_RESULTS_SHA256, "R010 results digest differs")
    validate_r010_oracle_failure_closure(load(predecessor_path))
    require(
        schedule.get("r010_predecessor") == registration.get("r010_predecessor"),
        "R010 predecessor schedule differs",
    )
    r010_schedule = load(resolve(root, schedule["r010_candidate_schedule"], "R010 schedule"))
    require(
        schedule.get("candidate_budget") == schedule.get("diagnostic_budget") == 4
        and [row.get("candidate_rank") for row in schedule["candidate_pairs"]] == [1, 2, 3, 4]
        and [row.get("diagnostic_index_one_based") for row in schedule["known_reachable_diagnostics"]]
        == [1, 2, 3, 4],
        "finite schedule budget/order differs",
    )
    for name, contract in (
        ("pinch_geometry", schedule["pinch_geometry_contract"]),
        ("geometry_attachment_preflight", schedule["geometry_attachment_preflight_contract"]),
        ("joint_handoff", schedule["joint_handoff_contract"]),
        ("construction_lifecycle", schedule["construction_lifecycle_contract"]),
        ("residual_correction", schedule["residual_correction_contract"]),
    ):
        require(canonical_sha(contract) == schedule[f"{name}_contract_sha256"], f"{name} digest differs")
    validate_pinch_contract(schedule["pinch_geometry_contract"])
    validate_attachment_preflight_contract(schedule["geometry_attachment_preflight_contract"])
    require(
        schedule["candidate_pairs"] == r010_schedule["candidate_pairs"]
        and schedule["known_reachable_diagnostics"] == r010_schedule["known_reachable_diagnostics"]
        and schedule["pinch_geometry_contract"] == r010_schedule["pinch_geometry_contract"]
        and schedule["pinch_geometry_contract_sha256"] == r010_schedule["pinch_geometry_contract_sha256"]
        and schedule["joint_handoff_contract"] == r010_schedule["joint_handoff_contract"]
        and schedule["joint_handoff_contract_sha256"] == r010_schedule["joint_handoff_contract_sha256"]
        and schedule["construction_lifecycle_contract"] == r010_schedule["construction_lifecycle_contract"]
        and schedule["construction_lifecycle_contract_sha256"] == r010_schedule["construction_lifecycle_contract_sha256"],
        "R010 candidate/controller/geometry/lifecycle bytes changed",
    )
    validate_r012_runtime_aliases(schedule["candidate_pairs"], r010_schedule["candidate_pairs"])
    require(
        schedule.get("selection_rule", {}).get("algorithm_version")
        == "r012-scene-sync-validated-relative-bound-collision-pinch-first-passing-pair-v1"
        and schedule["pinch_geometry_contract"].get("dynamic_usd_world_bounds_used") is False,
        "R012 selection/geometry algorithm identifier differs",
    )
    require(
        schedule.get("scene_sync_source_bindings")
        == registration.get("scene_sync_source_bindings")
        == registration.get("frozen_inputs", {}).get("r012_scene_sync_source_bindings")
        == SCENE_SYNC_SOURCES,
        "scene-sync source bindings differ",
    )
    require(
        schedule.get("construction_asset_bindings") == r010_schedule.get("construction_asset_bindings") == COLLISION_ASSETS,
        "collision-geometry USD bindings changed",
    )
    archived = schedule.get("archived_predecessor_contracts", {})
    require(
        archived.get("status") == "archived_lineage_only_not_active_r012_runtime_evidence"
        and archived.get("r010_failed_global_sync_preflight_contract")
        == r010_schedule["geometry_attachment_preflight_contract"]
        and archived.get("r010_failed_global_sync_preflight_contract_sha256")
        == r010_schedule["geometry_attachment_preflight_contract_sha256"],
        "R010 failed global-sync contract is not archived byte-identically",
    )
    require(
        schedule["construction_lifecycle_contract"]["worst_case_steps"] == 1695
        and schedule["construction_lifecycle_contract"]["registered_max_episode_length_steps"] == 1800
        and schedule["joint_handoff_contract"]["settle_steps"] == 600
        and schedule["pinch_geometry_contract"]["contact_or_grab_conditioned_branch"] is False
        and schedule["pinch_geometry_contract"]["early_stop"] is False,
        "retained lifecycle/nonadaptive controller differs",
    )

    r010_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r010/state_repair_gate.py"
    r012_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r012/state_repair_gate.py"
    for name in (
        "_contact_forces",
        "_contact_coverage",
        "_reference_bounds",
        "_save_camera_evidence",
        "_companion_gate",
        "_fresh_reset_and_gate",
        "_finalize_unchanged_gates",
    ):
        require(function_ast(r010_source, name) == function_ast(r012_source, name), f"scientific helper differs: {name}")
    for name in (
        "_collision_geometry_body_local",
        "_resolve_pinch_scene_geometry",
        "_live_pinch_bounds",
        "_pinch_geometry_materialize_and_gate",
    ):
        require(
            function_ast(r010_source, name)
            == function_ast(r012_source, name).replace("R012", "R010"),
            f"R010 geometry/controller helper changed: {name}",
        )
    pure_function_inventory = {
        "object_servo.py": (
            "normalize_quaternion",
            "multiply",
            "inverse",
            "rotate",
            "canonical_shortest_arc",
            "quaternion_to_rotvec",
            "rotvec_to_quaternion",
            "clip_norm",
            "object_space_servo_command",
            "validate_contract",
        ),
        "residual_correction.py": (
            "normalize_quaternion",
            "multiply",
            "inverse",
            "canonical_shortest_arc",
            "quaternion_power",
            "corrected_command",
            "validate_contract",
        ),
        "pinch_geometry.py": (
            "_vector3",
            "collision_center_env_local",
            "transform_collision_corners_env_local",
            "reconstruct_collision_bounds_env_local",
            "pinch_geometry",
            "pinch_alignment_command",
            "relative_pose",
            "parent_pose_for_child_target",
            "validate_contract",
        ),
    }
    for filename, names in pure_function_inventory.items():
        old_module = r010_source.parent / filename
        new_module = r012_source.parent / filename
        for name in names:
            require(
                function_ast(old_module, name)
                == function_ast(new_module, name).replace("R012", "R010"),
                f"R010 pure controller math changed: {filename}:{name}",
            )
    source = r012_source.read_text(encoding="utf-8")
    require("requests.post" not in source and "policy_server" not in source, "model endpoint exists")
    require(source.count("ComputeWorldBound(") == 1 and "def _collision_world_bounds" not in source, "dynamic USD polling entered controller")
    validate_preflight_source_topology(r012_source)
    preflight_ast = function_ast(r012_source, "_run_geometry_attachment_preflight")
    require(
        "update_transformations_scene" in preflight_ast
        and "update_transformations'" not in preflight_ast
        and "PhysicsSchemaTools" in preflight_ast
        and "sdfPathToInt" in preflight_ast
        and "SimulationContext" in preflight_ast
        and "get_initial_stage" in preflight_ast
        and "StageCache" in preflight_ast
        and "GetId" in preflight_ast
        and "ToLongInt" in preflight_ast
        and "UsdPhysics" in preflight_ast
        and "Scene" in preflight_ast
        and "env.step" not in preflight_ast,
        "R012 scene-specific preflight topology differs",
    )
    require(
        function_call_count(
            r012_source,
            "_run_geometry_attachment_preflight",
            "update_transformations_scene",
        )
        == 1,
        "scene-specific call path count differs",
    )
    main_ast = function_ast(r012_source, "main")
    require(
        "_run_geometry_attachment_preflight" in main_ast
        and "_pinch_geometry_materialize_and_gate" in main_ast
        and main_ast.index("_run_geometry_attachment_preflight") < main_ast.index("known_reachable_diagnostics"),
        "R012 dispatch reaches an unregistered topology",
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
        "geometry_attachment_preflight_contract_sha256": schedule[
            "geometry_attachment_preflight_contract_sha256"
        ],
        "joint_handoff_contract_sha256": schedule["joint_handoff_contract_sha256"],
        "construction_lifecycle_contract_sha256": schedule[
            "construction_lifecycle_contract_sha256"
        ],
    }


def validate_static(root: Path, *, source_gate_required: bool) -> dict[str, Any]:
    """Validate the immutable R012 package and its narrow live-tensor delta."""

    root = root.resolve()
    artifact = root / ART
    registration = load(artifact / "repair_registration.json")
    schedule = load(artifact / "gates/candidate_schedule.json")
    require(
        registration.get("repair_amendment_id") == "V3-E006-R012"
        and registration.get("predecessor_repair_amendment_id") == "V3-E006-R011"
        and registration.get("status")
        == "prospectively_registered_before_any_r012_live_preflight_diagnostic_candidate_or_model_request",
        "registration identity/status differs",
    )
    require(registration.get("counts_at_registration") == {
        "r012_geometry_attachment_preflights": 0,
        "r012_live_diagnostics": 0,
        "r012_live_candidate_evaluations": 0,
        "model_requests": 0,
        "behavioral_episodes": 0,
    }, "registration counts differ")
    require(
        schedule.get("status") == "frozen_before_any_r012_live_preflight_diagnostic_candidate_or_model_request"
        and schedule.get("r012_geometry_attachment_preflight_count") == 0
        and schedule.get("r012_live_diagnostic_count") == 0
        and schedule.get("r012_live_candidate_evaluation_count") == 0
        and schedule.get("model_request_count") == schedule.get("behavioral_episode_count") == 0,
        "schedule prospective status/counts differ",
    )
    for commit in (BASE, R011_CLOSURE_COMMIT):
        require(subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode == 0, f"missing commit {commit}")
    predecessor_path = resolve(root, registration["r011_predecessor"]["results"], "R011 predecessor")
    require(sha(predecessor_path) == R011_RESULTS_SHA256, "R011 results digest differs")
    validate_r011_scene_sync_failure_closure(load(predecessor_path))
    require(schedule.get("r011_predecessor") == registration.get("r011_predecessor"), "R011 predecessor schedule differs")
    r011_schedule = load(resolve(root, schedule["r011_candidate_schedule"], "R011 schedule"))
    require(
        schedule.get("candidate_budget") == schedule.get("diagnostic_budget") == 4
        and [row.get("candidate_rank") for row in schedule["candidate_pairs"]] == [1, 2, 3, 4]
        and [row.get("diagnostic_index_one_based") for row in schedule["known_reachable_diagnostics"]] == [1, 2, 3, 4],
        "finite schedule budget/order differs",
    )
    for name, contract in (
        ("pinch_geometry", schedule["pinch_geometry_contract"]),
        ("geometry_attachment_preflight", schedule["geometry_attachment_preflight_contract"]),
        ("joint_handoff", schedule["joint_handoff_contract"]),
        ("construction_lifecycle", schedule["construction_lifecycle_contract"]),
        ("residual_correction", schedule["residual_correction_contract"]),
    ):
        require(canonical_sha(contract) == schedule[f"{name}_contract_sha256"], f"{name} digest differs")
    validate_pinch_contract(schedule["pinch_geometry_contract"])
    validate_attachment_preflight_contract(schedule["geometry_attachment_preflight_contract"])
    for key in (
        "candidate_pairs", "known_reachable_diagnostics", "pinch_geometry_contract",
        "pinch_geometry_contract_sha256", "joint_handoff_contract",
        "joint_handoff_contract_sha256", "construction_lifecycle_contract",
        "construction_lifecycle_contract_sha256", "residual_correction_contract",
        "residual_correction_contract_sha256", "construction_asset_bindings",
    ):
        require(schedule.get(key) == r011_schedule.get(key), f"R011 retained schedule differs: {key}")
    validate_r012_runtime_aliases(schedule["candidate_pairs"], r011_schedule["candidate_pairs"])
    require(
        schedule.get("selection_rule", {}).get("algorithm_version")
        == "r012-live-tensor-relative-bound-collision-pinch-first-passing-pair-v1"
        and "scene_sync_source_bindings" not in schedule
        and "scene_sync_source_bindings" not in registration
        and schedule.get("construction_asset_bindings") == COLLISION_ASSETS,
        "R012 live-tensor selection/source boundary differs",
    )
    archived = schedule.get("archived_predecessor_contracts", {})
    require(
        archived.get("status") == "archived_lineage_only_not_active_r012_runtime_evidence"
        and archived.get("r011_failed_scene_sync_preflight_contract") == r011_schedule["geometry_attachment_preflight_contract"]
        and archived.get("r011_failed_scene_sync_preflight_contract_sha256") == r011_schedule["geometry_attachment_preflight_contract_sha256"],
        "R011 failed preflight is not archived exactly",
    )
    require(
        schedule["construction_lifecycle_contract"]["worst_case_steps"] == 1695
        and schedule["construction_lifecycle_contract"]["registered_max_episode_length_steps"] == 1800
        and schedule["joint_handoff_contract"]["settle_steps"] == 600
        and schedule["pinch_geometry_contract"]["contact_or_grab_conditioned_branch"] is False
        and schedule["pinch_geometry_contract"]["early_stop"] is False,
        "retained nonadaptive controller/lifecycle differs",
    )
    old_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r011/state_repair_gate.py"
    new_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r012/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence",
        "_companion_gate", "_fresh_reset_and_gate", "_finalize_unchanged_gates",
        "_collision_geometry_body_local", "_resolve_pinch_scene_geometry",
        "_live_pinch_bounds", "_pinch_geometry_materialize_and_gate",
    ):
        require(function_ast(old_source, name) == function_ast(new_source, name).replace("R012", "R011"), f"R011 helper changed: {name}")
    for filename, names in {
        "object_servo.py": ("normalize_quaternion", "multiply", "inverse", "rotate", "canonical_shortest_arc", "quaternion_to_rotvec", "rotvec_to_quaternion", "clip_norm", "object_space_servo_command", "validate_contract"),
        "residual_correction.py": ("normalize_quaternion", "multiply", "inverse", "canonical_shortest_arc", "quaternion_power", "corrected_command", "validate_contract"),
        "pinch_geometry.py": ("_vector3", "collision_center_env_local", "transform_collision_corners_env_local", "reconstruct_collision_bounds_env_local", "pinch_geometry", "pinch_alignment_command", "relative_pose", "parent_pose_for_child_target", "validate_contract"),
    }.items():
        for name in names:
            require(function_ast(old_source.parent / filename, name) == function_ast(new_source.parent / filename, name).replace("R012", "R011"), f"R011 pure controller math changed: {filename}:{name}")
    source = new_source.read_text(encoding="utf-8")
    preflight_ast = function_ast(new_source, "_run_geometry_attachment_preflight")
    for forbidden in (
        "ComputeWorldBound", "XformCache", "update_transformations", "omni.physx",
        "PhysicsSchemaTools", "UsdUtils", "SimulationContext", "carb.settings", "env.step",
    ):
        require(forbidden not in preflight_ast, f"dynamic USD/sync entered active preflight: {forbidden}")
    require(
        "_live_pinch_bounds" in preflight_ast
        and "requests.post" not in source
        and "policy_server" not in source,
        "live tensor preflight or zero-model boundary differs",
    )
    main_ast = function_ast(new_source, "main")
    require(
        "_run_geometry_attachment_preflight" in main_ast
        and "_pinch_geometry_materialize_and_gate" in main_ast
        and main_ast.index("_run_geometry_attachment_preflight") < main_ast.index("known_reachable_diagnostics"),
        "R012 dispatch topology differs",
    )
    gate_summary = None
    if source_gate_required:
        gate_path = artifact / "source_push_gate.json"
        require(gate_path.is_file(), "source gate absent")
        gate = load(gate_path)
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
        "pinch_geometry_contract_sha256": schedule["pinch_geometry_contract_sha256"],
        "geometry_attachment_preflight_contract_sha256": schedule["geometry_attachment_preflight_contract_sha256"],
        "joint_handoff_contract_sha256": schedule["joint_handoff_contract_sha256"],
        "construction_lifecycle_contract_sha256": schedule["construction_lifecycle_contract_sha256"],
    }


def validate_terminal_diagnostic_pattern(
    report: Mapping[str, Any], schedule: Mapping[str, Any]
) -> bool:
    """Validate the registered diagnostic prefix and its exact blocking terminal."""

    diagnostics = report.get("known_reachable_diagnostics")
    failed = (
        report.get("status")
        == "r012_known_reachable_diagnostic_failed_candidates_not_evaluated"
    )
    require(isinstance(diagnostics, list), "diagnostic rows absent")
    require(
        report.get("r012_live_diagnostic_count") == len(diagnostics)
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


def validate_terminal_geometry_preflight_pattern(
    report: Mapping[str, Any], *, preflight_passed: bool
) -> bool:
    """Validate the registered zero-candidate attachment-preflight terminal."""

    failed = (
        report.get("status")
        == "r012_geometry_attachment_preflight_failed_candidates_not_evaluated"
    )
    if failed:
        require(
            preflight_passed is False
            and report.get("passed") is False
            and report.get("geometry_attachment_preflight_count") == 1
            and report.get("known_reachable_diagnostics") == []
            and report.get("r012_live_diagnostic_count") == 0
            and report.get("attempts") == []
            and report.get("repair_candidate_evaluation_count") == 0
            and report.get("state_candidate_count") == 0
            and report.get("accepted_candidate_rank") is None
            and report.get("accepted_states") is None
            and report.get("candidate_budget")
            == report.get("diagnostic_budget")
            == 4
            and report.get("model_request_count")
            == report.get("behavioral_episode_count")
            == 0,
            "geometry preflight-failure terminal boundary differs",
        )
    else:
        require(preflight_passed is True, "candidate evaluation followed failed preflight")
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
        harness.get("status") == "completed_r012_candidate_search"
        and harness.get("process_completed") is True
        and harness.get("child_status") == report.get("status"),
        "harness terminal status differs",
    )
    preflight = report.get("geometry_attachment_preflight")
    require(isinstance(preflight, Mapping), "geometry attachment preflight absent")
    preflight_path = verify(
        Path(report["geometry_attachment_preflight_receipt"]["path"]),
        report["geometry_attachment_preflight_receipt"],
        "geometry attachment preflight receipt",
    )
    require(
        load(preflight_path) == preflight
        and report.get("geometry_attachment_preflight_count") == 1
        and harness.get("geometry_attachment_preflight_count") == 1,
        "geometry attachment preflight receipt/count differs",
    )
    preflight_passed = validate_geometry_attachment_preflight(preflight, schedule)
    for camera in (
        preflight.get("fresh_reset", {})
        .get("camera_evidence", {})
        .get("bindings", {})
        .values()
    ):
        verify(Path(camera["rgb"]["path"]), camera["rgb"], "preflight camera")
    preflight_failure = validate_terminal_geometry_preflight_pattern(
        report, preflight_passed=preflight_passed
    )
    diagnostics = report.get("known_reachable_diagnostics")
    diagnostic_failure = (
        False
        if preflight_failure
        else validate_terminal_diagnostic_pattern(report, schedule)
    )
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
                preflight,
            )
            validate_construction_lifecycle(materialization["environment_lifecycle"], "materialization environment")
            require(materialization["environment_lifecycle"]["closed_before_next_environment"] is True, "materialization env not closed")
            require(materialization["fresh_reset"].get("passed") is True, "materialization reset failed")
            for camera in state.get("camera_evidence", {}).get("bindings", {}).values():
                verify(Path(camera["rgb"]["path"]), camera["rgb"], "candidate camera")
        require(attempt.get("passed") is all(attempt["stages"][stage]["candidate_state"]["passed"] for stage in ("canonical_grasp","canonical_carry")), "pair pass differs")
    passed = [row for row in attempts if row.get("passed")]
    if preflight_failure:
        require(not passed, "geometry preflight failure retained a passed candidate")
    elif diagnostic_failure:
        require(not passed, "diagnostic failure retained a passed candidate")
    elif report.get("passed"):
        require(
            report.get("status")
            == "passed_r012_state_repair_not_released_for_behavior",
            "pass status differs",
        )
        require(len(passed) == 1 and passed[0] is attempts[-1], "first pass differs")
        require(report.get("accepted_candidate_rank") == attempts[-1]["candidate_rank"], "accepted rank differs")
        require(report.get("accepted_states") == attempts[-1]["stages"], "accepted states differ")
    else:
        require(
            report.get("status")
            == "r012_candidate_budget_exhausted_no_valid_state_pair",
            "exhaustion status differs",
        )
        require(not passed and len(attempts) == 4 and report.get("accepted_candidate_rank") is None, "exhaustion differs")
        require(report.get("accepted_states") is None, "exhaustion retained accepted states")
    if not preflight_failure and not diagnostic_failure:
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
        "original_v3e006_closure_binding", "r011_predecessor_results", "ood_freeze",
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
        execution.get("geometry_attachment_preflight") == preflight
        and execution.get("geometry_attachment_preflight_count") == 1,
        "execution/preflight evidence cross-link differs",
    )
    require(
        report["repair_registration"] == launch["input_bindings"]["repair_registration"]
        and report["candidate_schedule"] == launch["input_bindings"]["candidate_schedule"]
        and report["source_push_gate"] == launch["input_bindings"]["source_push_gate"]
        and report["original_v3e006_closure_binding"]
        == launch["input_bindings"]["original_closure_binding"]
        and report["r011_predecessor_results"]
        == launch["input_bindings"]["predecessor_closure_binding"],
        "launch/child frozen-input cross-binding differs",
    )
    return {
        "passed": True,
        "launch": binding(launch_path),
        "harness": binding(harness_path),
        "child_report": binding(child_path),
        "geometry_attachment_preflight": binding(preflight_path),
        "geometry_attachment_preflight_passed": preflight_passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


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
