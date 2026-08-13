"""Pure geometric pinch alignment and fixed transport math for V3-E006-R012."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r012.object_servo import (
    canonical_shortest_arc,
    clip_norm,
    inverse,
    multiply,
    normalize_quaternion,
    quaternion_to_rotvec,
    rotate,
    rotvec_to_quaternion,
)


def _vector3(value: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} is malformed")
    return vector


def collision_center_env_local(
    *,
    body_position_env_local: Sequence[float],
    body_quaternion_world_wxyz: Sequence[float],
    collision_center_body: Sequence[float],
) -> np.ndarray:
    position = _vector3(body_position_env_local, "body position")
    center = _vector3(collision_center_body, "collision center")
    return position + rotate(body_quaternion_world_wxyz, center)


def transform_collision_corners_env_local(
    *,
    body_position_env_local: Sequence[float],
    body_quaternion_world_wxyz: Sequence[float],
    collision_corners_body: Sequence[Sequence[float]],
) -> np.ndarray:
    """Transform a frozen body-local collision inventory with one live tensor pose."""

    position = _vector3(body_position_env_local, "body env-local position")
    quaternion = normalize_quaternion(body_quaternion_world_wxyz)
    corners = np.asarray(collision_corners_body, dtype=np.float64)
    if (
        corners.ndim != 2
        or corners.shape[1:] != (3,)
        or corners.shape[0] < 8
        or not np.isfinite(corners).all()
    ):
        raise ValueError("body-local collision corners are malformed")
    return np.stack([position + rotate(quaternion, corner) for corner in corners])


def reconstruct_collision_bounds_env_local(
    *,
    body_position_env_local: Sequence[float],
    body_quaternion_world_wxyz: Sequence[float],
    collision_corners_body: Sequence[Sequence[float]],
    collision_center_body: Sequence[float],
) -> dict[str, Any]:
    """Reconstruct an env-local AABB solely from live tensors and static geometry."""

    corners = transform_collision_corners_env_local(
        body_position_env_local=body_position_env_local,
        body_quaternion_world_wxyz=body_quaternion_world_wxyz,
        collision_corners_body=collision_corners_body,
    )
    minimum = np.min(corners, axis=0)
    maximum = np.max(corners, axis=0)
    aabb_center = 0.5 * (minimum + maximum)
    half = 0.5 * (maximum - minimum)
    collision_center = collision_center_env_local(
        body_position_env_local=body_position_env_local,
        body_quaternion_world_wxyz=body_quaternion_world_wxyz,
        collision_center_body=collision_center_body,
    )
    if not np.isfinite(
        np.concatenate((minimum, maximum, aabb_center, half, collision_center))
    ).all():
        raise ValueError("reconstructed collision bounds are nonfinite")
    if np.any(half <= 0.0):
        raise ValueError("reconstructed collision bounds are empty")
    return {
        "aabb_minimum_env_local_m": minimum.tolist(),
        "aabb_maximum_env_local_m": maximum.tolist(),
        "aabb_center_env_local_m": aabb_center.tolist(),
        "aabb_half_extents_env_local_m": half.tolist(),
        "collision_center_env_local_m": collision_center.tolist(),
        "transformed_corners_env_local_m": corners.tolist(),
    }


def pinch_geometry(
    *,
    left_center_env_local: Sequence[float],
    right_center_env_local: Sequence[float],
) -> dict[str, Any]:
    left = _vector3(left_center_env_local, "left collision center")
    right = _vector3(right_center_env_local, "right collision center")
    separation = right - left
    norm = float(np.linalg.norm(separation))
    if not math.isfinite(norm) or norm <= 1e-6:
        raise ValueError("inner-finger collision centers are not unique")
    return {
        "left_inner_finger_collision_center_env_local_m": left.tolist(),
        "right_inner_finger_collision_center_env_local_m": right.tolist(),
        "pinch_midpoint_env_local_m": ((left + right) * 0.5).tolist(),
        "pinch_axis_left_to_right_env_local": (separation / norm).tolist(),
        "inner_finger_collision_center_separation_m": norm,
    }


def pinch_alignment_command(
    *,
    live_base_position_env_local: Sequence[float],
    live_base_quaternion: Sequence[float],
    live_left_center_env_local: Sequence[float],
    live_right_center_env_local: Sequence[float],
    target_pinch_midpoint_env_local: Sequence[float],
    target_base_quaternion: Sequence[float],
    translation_gain: float,
    rotation_gain: float,
    translation_cap_m_per_step: float,
    rotation_cap_deg_per_step: float,
) -> dict[str, Any]:
    """Compute one uniform command aligning live pad midpoint with a target point."""

    base_p = _vector3(live_base_position_env_local, "live base position")
    base_q = normalize_quaternion(live_base_quaternion)
    target_midpoint = _vector3(target_pinch_midpoint_env_local, "target pinch midpoint")
    target_q = normalize_quaternion(target_base_quaternion)
    geometry = pinch_geometry(
        left_center_env_local=live_left_center_env_local,
        right_center_env_local=live_right_center_env_local,
    )
    midpoint = np.asarray(geometry["pinch_midpoint_env_local_m"], dtype=np.float64)
    axis_world = np.asarray(geometry["pinch_axis_left_to_right_env_local"], dtype=np.float64)
    local_midpoint = rotate(inverse(base_q), midpoint - base_p)
    local_axis = rotate(inverse(base_q), axis_world)
    ideal_p = target_midpoint - rotate(target_q, local_midpoint)
    translation_error = ideal_p - base_p
    translation_correction = clip_norm(
        float(translation_gain) * translation_error,
        float(translation_cap_m_per_step),
    )
    rotation_error_q = canonical_shortest_arc(multiply(target_q, inverse(base_q)))
    rotation_error = quaternion_to_rotvec(rotation_error_q)
    rotation_correction = clip_norm(
        float(rotation_gain) * rotation_error,
        math.radians(float(rotation_cap_deg_per_step)),
    )
    command_p = base_p + translation_correction
    command_q = multiply(rotvec_to_quaternion(rotation_correction), base_q)
    predicted_midpoint = command_p + rotate(command_q, local_midpoint)
    predicted_axis = rotate(command_q, local_axis)
    return {
        **geometry,
        "live_base_position_env_local_m": base_p.tolist(),
        "live_base_quaternion_world_wxyz": base_q.tolist(),
        "pinch_midpoint_in_base_m": local_midpoint.tolist(),
        "pinch_axis_in_base": local_axis.tolist(),
        "target_pinch_midpoint_env_local_m": target_midpoint.tolist(),
        "target_base_quaternion_world_wxyz": target_q.tolist(),
        "ideal_base_position_env_local_m": ideal_p.tolist(),
        "translation_error_env_local_m": translation_error.tolist(),
        "rotation_error_env_local_rotvec_rad": rotation_error.tolist(),
        "translation_gain": float(translation_gain),
        "rotation_gain": float(rotation_gain),
        "translation_cap_m_per_step": float(translation_cap_m_per_step),
        "rotation_cap_deg_per_step": float(rotation_cap_deg_per_step),
        "applied_translation_correction_env_local_m": translation_correction.tolist(),
        "applied_rotation_correction_env_local_rotvec_rad": rotation_correction.tolist(),
        "command_base_position_env_local_m": command_p.tolist(),
        "command_base_quaternion_world_wxyz": command_q.tolist(),
        "command_predicted_pinch_midpoint_env_local_m": predicted_midpoint.tolist(),
        "command_predicted_pinch_axis_env_local": predicted_axis.tolist(),
    }


def relative_pose(
    *,
    parent_position: Sequence[float],
    parent_quaternion: Sequence[float],
    child_position: Sequence[float],
    child_quaternion: Sequence[float],
) -> dict[str, list[float]]:
    parent_p = _vector3(parent_position, "parent position")
    parent_q = normalize_quaternion(parent_quaternion)
    child_p = _vector3(child_position, "child position")
    child_q = normalize_quaternion(child_quaternion)
    parent_inv = inverse(parent_q)
    return {
        "position_parent_m": rotate(parent_inv, child_p - parent_p).tolist(),
        "quaternion_parent_wxyz": multiply(parent_inv, child_q).tolist(),
    }


def parent_pose_for_child_target(
    *,
    child_target_position: Sequence[float],
    child_target_quaternion: Sequence[float],
    child_in_parent_position: Sequence[float],
    child_in_parent_quaternion: Sequence[float],
) -> dict[str, list[float]]:
    child_p = _vector3(child_target_position, "child target position")
    child_q = normalize_quaternion(child_target_quaternion)
    relative_p = _vector3(child_in_parent_position, "child-in-parent position")
    relative_q = normalize_quaternion(child_in_parent_quaternion)
    parent_q = multiply(child_q, inverse(relative_q))
    parent_p = child_p - rotate(parent_q, relative_p)
    return {
        "position_world_m": parent_p.tolist(),
        "quaternion_world_wxyz": parent_q.tolist(),
    }


def validate_contract(value: Mapping[str, Any]) -> None:
    expected = {
        "algorithm_version": "uniform-relative-bound-tensor-collision-pinch-acquisition-v1",
        "applies_identically_to": "all four ranks and canonical_grasp/canonical_carry",
        "robot_body_resolution": "under the unique env_0 robot prim, require exactly one tensor rigid body whose path ends left_inner_finger and exactly one ending right_inner_finger; each selected owner must have UsdPhysics.RigidBodyAPI",
        "collision_resolution": "once before candidate actions, resolve each enabled UsdPhysics.CollisionAPI prim with ComputeRelativeBound(collisionPrim, owningRigidBody).ComputeAlignedRange; retain those corners directly as body-local and reject nested or different rigid-body boundaries",
        "relative_bound_api": "UsdGeom.BBoxCache.ComputeRelativeBound(collision_prim, owning_rigid_body).ComputeAlignedRange",
        "additional_transform_after_relative_bound": False,
        "dynamic_geometry_source": "at every action step reconstruct pad/cube centers and AABBs only from IsaacLab tensor rigid-body/root poses minus the explicitly retained scene env origin plus the frozen body-local corners",
        "left_inner_finger_body_suffix": "left_inner_finger",
        "right_inner_finger_body_suffix": "right_inner_finger",
        "target_midpoint_rule": "live tensor-reconstructed cube collision center in env-local world-axis coordinates",
        "approach_clearance_rule": "two times live tensor-reconstructed cube AABB half-extent z",
        "dynamic_usd_world_bounds_used": False,
        "controller_coordinate_semantics": "env-local world-axis positions; tensor world position minus scene env origin; quaternions remain world-axis WXYZ",
        "open_approach_target": "live cube collision center plus env-local world-z approach clearance, with the frozen rank-stage acquisition quaternion",
        "open_descent_target": "live cube collision center, with the same frozen acquisition quaternion",
        "normal_close_target": "live cube collision center, with unchanged normal binary close action",
        "closed_vertical_lift_target": "reset cube collision center with z replaced by target cube collision-center z; acquisition quaternion remains fixed",
        "closed_stage_transport_target": "target cube collision center; quaternion shortest-arc interpolates from acquisition quaternion to the unchanged frozen stage base quaternion",
        "target_cube_collision_center": "transform the frozen cube-body-local collision-center offset by the unchanged frozen target cube env-local world-axis pose",
        "translation_gain": 0.2,
        "rotation_gain": 0.2,
        "translation_cap_m_per_step": 0.002,
        "rotation_cap_deg_per_step": 2.0,
        "open_approach_steps": 180,
        "open_descent_steps": 180,
        "normal_close_steps": 120,
        "closed_vertical_lift_steps": 240,
        "closed_stage_transport_steps": 300,
        "gripper_open_command": 0.0,
        "gripper_closed_command": 1.0,
        "contact_or_grab_conditioned_branch": False,
        "early_stop": False,
        "contact_and_grab_trace_semantics": "diagnostic-only; never changes, skips, rejects, stops, or branches construction",
        "parameter_basis": "R009 numeric controller, phases, ranks, and targets are retained exactly; R010 changes only the collision-bound attachment extraction from double-transformed ComputeLocalBound to direct body-relative ComputeRelativeBound",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R012 pinch geometry contract differs: {key}")


def validate_attachment_preflight_contract(value: Mapping[str, Any]) -> None:
    """Validate the prospective live-physics-tensor sanity preflight."""

    expected = {
        "algorithm_version": "r012-live-physics-tensor-geometry-sanity-preflight-v1",
        "preflight_budget": 1,
        "execution_order": "exactly one dedicated fresh environment, completed and closed before the first known-reachable diagnostic or candidate environment",
        "fresh_reset_steps": 75,
        "dynamic_state_source": "finite IsaacLab live robot body_pos_w/body_quat_w and cube root_pos_w/root_quat_w tensors minus the exact retained scene env origin",
        "static_attachment_source": "the frozen R010 ComputeRelativeBound collision inventory and owning-rigid-body-local aligned-range corners",
        "exact_tensor_index_name_and_body_ownership_required": True,
        "left_right_tensor_indices_must_be_distinct": True,
        "pad_collision_center_separation_m_inclusive": [0.05, 0.2],
        "cube_aabb_dimension_m_each_inclusive": [0.03, 0.1],
        "fresh_reset_tensor_identity_required": True,
        "dynamic_usd_world_state_used": False,
        "physics_to_usd_sync_call_count": 0,
        "dynamic_usd_world_bound_or_xform_query_count": 0,
        "candidate_environment_identity_rule": "before its first construction action, every rank-stage environment must reproduce the exact preflight collision inventory and unchanged R010 body-relative geometry canonical SHA",
        "failure_policy": "any nonfinite tensor, tensor index/name/ownership mismatch, static inventory/hash mismatch, degenerate pad separation/cube dimension, or fresh-reset tensor identity mismatch produces one zero-candidate terminal preflight failure",
        "dynamic_usd_world_bounds_used_by_controller": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    if dict(value) != expected:
        differing = sorted(set(value) | set(expected))
        key = next((name for name in differing if value.get(name) != expected.get(name)), "mapping")
        raise ValueError(f"R012 geometry attachment preflight contract differs: {key}")
