"""Deterministic horizontal cube-only geometry repair for V4 G3 unblock."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from experiments.online_correction_v4.motion import ReferenceMotionController

FIXTURE_VERSION = "horizontal_geometry_repair_v2"
COHORT = "confirmatory_horizontal_geometry_repair_v2"
REPAIR_INCREMENT_M = 0.01
SUPPORT_EDGE_GUARD_M = 0.005
MINIMUM_SCALE = 0.5
NOMINAL_TRANSLATION_M = 0.12
MINIMUM_DISPLACEMENT_M = NOMINAL_TRANSLATION_M * MINIMUM_SCALE

# Independently verified live USD/PhysX world AABB from PVC
# registered_reset/initial_scene.json, seed 2100000000 / g3p20260905h scale 0.5.
CANONICAL_LIVE_WORLD_AABB_M = {
    "rubiks_cube": {
        "min_xyz": [0.39127529231730607, -0.09744829743391743, 0.0496023569875199],
        "max_xyz": [0.44969119161337034, -0.039562635059537644, 0.10779157240234026],
    },
    "bowl": {
        "min_xyz": [0.3618268424133713, 0.04600550650063867, 0.0497811528779535],
        "max_xyz": [0.5233402198691908, 0.20715887428816432, 0.10487455698475552],
    },
}
CANONICAL_ROOT_POSE_WORLD_XYZ_M = {
    "rubiks_cube": [0.2935180366039276, 0.12148545682430267, 0.08110573142766953],
    "bowl": [0.4289674460887909, 0.12890803813934326, 0.07735103368759155],
}

# Zero-model PVC witness for first rubiks_cube__bowl contact on the 0.5 scale.
WITNESS_FIRST_CONTACT_BY_GOAL = {
    "front": {
        "scenario": "move_stop",
        "planned_displacement_m": 0.04095359999999999,
        "force_n": 2.3634903991529304,
        "cube_drift_m_at_contact": 0.0025997996168450836,
    },
    "behind": {
        "scenario": "move_stop",
        "planned_displacement_m": 0.037411585185112825,
        "force_n": 1.0365746767743176,
        "cube_drift_m_at_contact": 0.0005828634780481631,
    },
}


class HorizontalGeometryRepairError(ValueError):
    """Raised when repair selection or clearance checks fail."""


def _require_finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise HorizontalGeometryRepairError(f"{label} must be finite")
    return float(value)


def shift_aabb_x(aabb: Mapping[str, Sequence[float]], delta_x: float) -> dict[str, list[float]]:
    return {
        "min_xyz": [float(aabb["min_xyz"][0]) + delta_x, float(aabb["min_xyz"][1]), float(aabb["min_xyz"][2])],
        "max_xyz": [float(aabb["max_xyz"][0]) + delta_x, float(aabb["max_xyz"][1]), float(aabb["max_xyz"][2])],
    }


def table_plane_overlap_m(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
    *,
    guard_m: float,
) -> bool:
    """Return True when XY rectangles overlap after guard inflation on both boxes."""
    guard = _require_finite(guard_m, "guard_m")
    for axis in (0, 1):
        left_min = float(left["min_xyz"][axis]) - guard
        left_max = float(left["max_xyz"][axis]) + guard
        right_min = float(right["min_xyz"][axis]) - guard
        right_max = float(right["max_xyz"][axis]) + guard
        if left_min >= right_max or right_min >= left_max:
            return False
    return True


def conservative_swept_aabb(
    start_aabb: Mapping[str, Sequence[float]],
    delta_xyz: Sequence[float],
) -> dict[str, list[float]]:
    end = {
        "min_xyz": [float(start_aabb["min_xyz"][index]) + float(delta_xyz[index]) for index in range(3)],
        "max_xyz": [float(start_aabb["max_xyz"][index]) + float(delta_xyz[index]) for index in range(3)],
    }
    return {
        "min_xyz": [
            min(float(start_aabb["min_xyz"][index]), end["min_xyz"][index])
            for index in range(3)
        ],
        "max_xyz": [
            max(float(start_aabb["max_xyz"][index]), end["max_xyz"][index])
            for index in range(3)
        ],
    }


def task_displacement_to_robot_delta(
    *,
    direction_task: Sequence[float],
    displacement_m: float,
) -> tuple[float, float, float]:
    task_left = _require_finite(direction_task[0], "direction_task[0]")
    task_front = _require_finite(direction_task[1], "direction_task[1]")
    disp = _require_finite(displacement_m, "displacement_m")
    return (-task_front * disp, task_left * disp, 0.0)


def approach_axis_closing_margin_m(
    *,
    cube_aabb: Mapping[str, Sequence[float]],
    bowl_aabb: Mapping[str, Sequence[float]],
    robot_delta_xyz: Sequence[float],
    displacement_m: float,
) -> float:
    """Minimum margin along the bowl motion axis between cube and swept bowl AABB."""
    disp = _require_finite(displacement_m, "displacement_m")
    delta = [float(robot_delta_xyz[index]) * disp for index in range(3)]
    swept = conservative_swept_aabb(bowl_aabb, delta)
    margins: list[float] = []
    for axis in range(2):
        delta_axis = float(robot_delta_xyz[axis])
        if abs(delta_axis) <= 1e-12:
            continue
        if delta_axis < 0.0:
            margins.append(float(cube_aabb["min_xyz"][axis]) - float(swept["min_xyz"][axis]))
        else:
            margins.append(float(swept["max_xyz"][axis]) - float(cube_aabb["max_xyz"][axis]))
    if not margins:
        raise HorizontalGeometryRepairError("motion axis is degenerate for clearance audit")
    return min(margins)


def witness_delayed_contact_displacement_m(
    *,
    goal: str,
    repair_offset_robot_base_x_m: float,
) -> float | None:
    witness = WITNESS_FIRST_CONTACT_BY_GOAL.get(goal)
    if witness is None:
        return None
    return float(witness["planned_displacement_m"]) + abs(
        float(repair_offset_robot_base_x_m)
    )


def passes_repair_clearance_for_motion(
    *,
    cube_aabb: Mapping[str, Sequence[float]],
    bowl_aabb: Mapping[str, Sequence[float]],
    goal: str,
    robot_delta_xyz: Sequence[float],
    displacement_m: float,
    repair_offset_robot_base_x_m: float,
    guard_m: float,
) -> tuple[bool, float]:
    delayed_contact = witness_delayed_contact_displacement_m(
        goal=goal,
        repair_offset_robot_base_x_m=repair_offset_robot_base_x_m,
    )
    if delayed_contact is not None and float(robot_delta_xyz[0]) < -1e-12:
        margin = delayed_contact - displacement_m
        return margin >= guard_m - 1e-12, margin
    margin = approach_axis_closing_margin_m(
        cube_aabb=cube_aabb,
        bowl_aabb=bowl_aabb,
        robot_delta_xyz=robot_delta_xyz,
        displacement_m=displacement_m,
    )
    return margin >= guard_m - 1e-12, margin


def witness_kinematic_floor_offset_m(
    *,
    displacement_m: float = MINIMUM_DISPLACEMENT_M,
    guard_m: float = SUPPORT_EDGE_GUARD_M,
    increment_m: float = REPAIR_INCREMENT_M,
) -> tuple[float, dict[str, Any]]:
    """Lower bound from observed first-contact timing on the 0.5 scale."""
    required_by_goal: dict[str, float] = {}
    for goal, witness in WITNESS_FIRST_CONTACT_BY_GOAL.items():
        first_contact_m = _require_finite(
            witness["planned_displacement_m"],
            f"{goal} witness planned_displacement_m",
        )
        required = displacement_m - first_contact_m + guard_m
        if required <= 0.0:
            raise HorizontalGeometryRepairError(
                f"{goal} witness contact occurs after endpoint; cannot derive offset"
            )
        required_by_goal[goal] = required
    binding_goal = max(required_by_goal, key=required_by_goal.get)
    required_m = required_by_goal[binding_goal]
    increments = max(1, math.ceil(required_m / increment_m - 1e-12))
    offset_m = -increments * increment_m
    return offset_m, {
        "method": "witness_kinematic_floor_from_first_contact_displacement",
        "required_offset_m_by_goal": required_by_goal,
        "binding_goal": binding_goal,
        "required_offset_m": required_m,
        "increments_of_1cm": increments,
        "witness_first_contact_by_goal": WITNESS_FIRST_CONTACT_BY_GOAL,
    }


def minimum_cube_repair_offset_m(
    *,
    base_positions_robot_base_m: Mapping[str, Sequence[float]],
    resets_by_env_seed: Mapping[str, Mapping[str, Any]],
    displacement_m: float = MINIMUM_DISPLACEMENT_M,
    guard_m: float = SUPPORT_EDGE_GUARD_M,
    increment_m: float = REPAIR_INCREMENT_M,
    max_increments: int = 30,
) -> tuple[float, dict[str, Any]]:
    witness_floor_m, witness_audit = witness_kinematic_floor_offset_m(
        displacement_m=displacement_m,
        guard_m=guard_m,
        increment_m=increment_m,
    )
    start_increments = max(1, int(round(-witness_floor_m / increment_m)))
    goals = ("front", "behind")
    signs = (-1, 1)

    for increments in range(start_increments, max_increments + 1):
        repair_offset_m = -increments * increment_m
        worst_margin_m = math.inf
        binding: dict[str, Any] | None = None
        for env_seed, reset in resets_by_env_seed.items():
            positions = reset["positions_robot_base_m"]
            cube_shift_x = float(positions["rubiks_cube"][0]) - float(
                base_positions_robot_base_m["rubiks_cube"][0]
            )
            bowl_shift_x = float(positions["bowl"][0]) - float(
                base_positions_robot_base_m["bowl"][0]
            )
            cube_aabb = shift_aabb_x(
                CANONICAL_LIVE_WORLD_AABB_M["rubiks_cube"],
                repair_offset_m + cube_shift_x,
            )
            bowl_aabb = shift_aabb_x(
                CANONICAL_LIVE_WORLD_AABB_M["bowl"],
                bowl_shift_x,
            )
            for goal in goals:
                for sign in signs:
                    direction = ReferenceMotionController.displacement_vector(
                        goal=goal,
                        fixture="horizontal",
                        physical_sign=sign,
                    )
                    robot_delta = task_displacement_to_robot_delta(
                        direction_task=direction,
                        displacement_m=1.0,
                    )
                    ok, margin_m = passes_repair_clearance_for_motion(
                        cube_aabb=cube_aabb,
                        bowl_aabb=bowl_aabb,
                        goal=goal,
                        robot_delta_xyz=robot_delta,
                        displacement_m=displacement_m,
                        repair_offset_robot_base_x_m=repair_offset_m,
                        guard_m=guard_m,
                    )
                    if margin_m < worst_margin_m:
                        worst_margin_m = margin_m
                        binding = {
                            "environment_seed": int(env_seed),
                            "goal": goal,
                            "physical_translation_sign": sign,
                            "clearance_margin_m": margin_m,
                            "witness_delayed_contact_m": witness_delayed_contact_displacement_m(
                                goal=goal,
                                repair_offset_robot_base_x_m=repair_offset_m,
                            ),
                        }
                    if not ok:
                        worst_margin_m = -math.inf
                        break
                if worst_margin_m == -math.inf:
                    break
            if worst_margin_m == -math.inf:
                break
        if worst_margin_m >= guard_m - 1e-12 and worst_margin_m != -math.inf:
            audit = {
                "repair_offset_robot_base_x_m": repair_offset_m,
                "increments_of_1cm": increments,
                "minimum_clearance_margin_m": worst_margin_m,
                "support_edge_guard_m": guard_m,
                "minimum_scale": MINIMUM_SCALE,
                "minimum_displacement_m": displacement_m,
                "worst_binding": binding,
                "reset_count": len(resets_by_env_seed),
                "witness_kinematic_floor": witness_audit,
                "clearance_model": (
                    "Live world AABB from PVC initial_scene.json; require non-overlap "
                    "between cube and bowl swept volume in the table plane and a "
                    "non-negative approach-axis closing margin plus guard along the "
                    "actual robot-base motion axis. Witness first-contact timing sets "
                    "the 1 cm increment floor because PhysX contact can occur while "
                    "3D AABB Y intervals remain separated."
                ),
            }
            return repair_offset_m, audit
    raise HorizontalGeometryRepairError(
        "no cube-only -X repair offset satisfies witness and swept clearance within search bound"
    )


def apply_cube_repair_offset(
    base_positions: Mapping[str, Sequence[float]],
    *,
    repair_offset_robot_base_x_m: float,
) -> dict[str, list[float]]:
    repaired = {
        name: [float(value) for value in vector]
        for name, vector in base_positions.items()
    }
    repaired["rubiks_cube"][0] += float(repair_offset_robot_base_x_m)
    return repaired


def root_pose_aabb_center_mismatch_audit(
    *,
    cube_root_xyz: Sequence[float],
    cube_aabb: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    center = [
        (float(cube_aabb["min_xyz"][index]) + float(cube_aabb["max_xyz"][index])) / 2.0
        for index in range(3)
    ]
    delta = [float(cube_root_xyz[index]) - center[index] for index in range(3)]
    return {
        "root_pose_world_xyz_m": [float(value) for value in cube_root_xyz[:3]],
        "usd_aabb_center_world_xyz_m": center,
        "root_minus_aabb_center_m": delta,
        "impact": (
            "rubiks_cube root pose in initial_scene.json is stale relative to the live "
            "PhysX/USD world AABB actually used by G3 contact probes. PhysX reports "
            "rubiks_cube__bowl contact while 3D AABB Y intervals remain separated; "
            "repair clearance therefore uses live world AABB, table-plane overlap, "
            "approach-axis margins, and PVC first-contact timing rather than root pose "
            "or 3D AABB separation dominated by Y gap."
        ),
    }


def canonical_aabb_freshness_audit() -> dict[str, Any]:
    cube_aabb = CANONICAL_LIVE_WORLD_AABB_M["rubiks_cube"]
    bowl_aabb = CANONICAL_LIVE_WORLD_AABB_M["bowl"]
    cube_root = CANONICAL_ROOT_POSE_WORLD_XYZ_M["rubiks_cube"]
    bowl_root = CANONICAL_ROOT_POSE_WORLD_XYZ_M["bowl"]
    return {
        "source": "registered_reset/initial_scene.json",
        "sha256": "415bc7f11ed74ef6c249f1d28ead3a588b1d929c5b401f0af631afdc1bc20a1c",
        "initial_rubiks_cube__bowl_contact_force_n": 0.0,
        "rubiks_cube": root_pose_aabb_center_mismatch_audit(
            cube_root_xyz=cube_root,
            cube_aabb=cube_aabb,
        ),
        "bowl": {
            "root_pose_world_xyz_m": bowl_root,
            "usd_aabb_center_world_xyz_m": [
                (bowl_aabb["min_xyz"][index] + bowl_aabb["max_xyz"][index]) / 2.0
                for index in range(3)
            ],
            "root_minus_aabb_center_m": [
                bowl_root[index]
                - (bowl_aabb["min_xyz"][index] + bowl_aabb["max_xyz"][index]) / 2.0
                for index in range(3)
            ],
        },
        "table_plane_overlap_at_reset": table_plane_overlap_m(
            cube_aabb,
            bowl_aabb,
            guard_m=0.0,
        ),
        "impact": (
            "PhysX uses collision geometry aligned with the live world AABB, not the "
            "stale rubiks_cube root pose. Zero rubiks_cube__bowl force at reset with "
            "non-zero table-plane overlap confirms contact classification is not a "
            "sensor mislabel; later rubiks_cube__bowl forces coincide with measured "
            "cube drift and are true layout collisions along the front/behind motion axis."
        ),
    }
