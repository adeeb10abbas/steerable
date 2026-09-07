"""Deterministic horizontal cube-only geometry repair for V4 G3 unblock."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from experiments.online_correction_v4.motion import ReferenceMotionController

FIXTURE_VERSION = "horizontal_geometry_repair_v1"
COHORT = "confirmatory_horizontal_geometry_repair_v1"
REPAIR_INCREMENT_M = 0.01
SUPPORT_EDGE_GUARD_M = 0.005
MINIMUM_SCALE = 0.5
NOMINAL_TRANSLATION_M = 0.12
MINIMUM_DISPLACEMENT_M = NOMINAL_TRANSLATION_M * MINIMUM_SCALE

# Independently verified live USD AABB minus root pose for seed 2100000000 /
# attempt g3p20260905h scale 0.5 (PVC initial_scene.json, 2026-09-07).
CANONICAL_ROOT_TO_AABB_CENTER_M = {
    "rubiks_cube": [0.1269652053614106, -0.18999092307103022, -0.002408766732739445],
    "bowl": [-0.013616085052490123, 0.002325847744941767, 2.3178756237043907e-05],
}
CANONICAL_AABB_HALF_EXTENTS_M = {
    "rubiks_cube": [0.029207949648032135, 0.028942831187189904, 0.02909460770741018],
    "bowl": [0.08075668872790975, 0.080576183893762825, 0.02754670205340101],
}


class HorizontalGeometryRepairError(ValueError):
    """Raised when repair selection or clearance checks fail."""


def _require_finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise HorizontalGeometryRepairError(f"{label} must be finite")
    return float(value)


def aabb_from_root(*, root_xyz: Sequence[float], object_name: str) -> dict[str, list[float]]:
    center_offset = CANONICAL_ROOT_TO_AABB_CENTER_M[object_name]
    half = CANONICAL_AABB_HALF_EXTENTS_M[object_name]
    center = [
        float(root_xyz[index]) + float(center_offset[index])
        for index in range(3)
    ]
    return {
        "min_xyz": [center[index] - half[index] for index in range(3)],
        "max_xyz": [center[index] + half[index] for index in range(3)],
    }


def aabb_separation_m(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
) -> float:
    dx = max(
        0.0,
        max(float(left["min_xyz"][0]) - float(right["max_xyz"][0]), float(right["min_xyz"][0]) - float(left["max_xyz"][0])),
    )
    dy = max(
        0.0,
        max(float(left["min_xyz"][1]) - float(right["max_xyz"][1]), float(right["min_xyz"][1]) - float(left["max_xyz"][1])),
    )
    dz = max(
        0.0,
        max(float(left["min_xyz"][2]) - float(right["max_xyz"][2]), float(right["min_xyz"][2]) - float(left["max_xyz"][2])),
    )
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        ox = min(float(left["max_xyz"][0]), float(right["max_xyz"][0])) - max(
            float(left["min_xyz"][0]), float(right["min_xyz"][0])
        )
        oy = min(float(left["max_xyz"][1]), float(right["max_xyz"][1])) - max(
            float(left["min_xyz"][1]), float(right["min_xyz"][1])
        )
        oy = max(0.0, oy)
        oz = min(float(left["max_xyz"][2]), float(right["max_xyz"][2])) - max(
            float(left["min_xyz"][2]), float(right["min_xyz"][2])
        )
        oz = max(0.0, oz)
        return -math.sqrt(max(0.0, ox) ** 2 + oy ** 2 + oz ** 2)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


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


def minimum_cube_repair_offset_m(
    *,
    base_positions_robot_base_m: Mapping[str, Sequence[float]],
    resets_by_env_seed: Mapping[str, Mapping[str, Any]],
    displacement_m: float = MINIMUM_DISPLACEMENT_M,
    guard_m: float = SUPPORT_EDGE_GUARD_M,
    increment_m: float = REPAIR_INCREMENT_M,
    max_increments: int = 20,
) -> tuple[float, dict[str, Any]]:
    goals = ("left", "right", "front", "behind")
    signs = (-1, 1)
    for increments in range(1, max_increments + 1):
        repair_offset_m = -increments * increment_m
        worst_separation_m = math.inf
        binding: dict[str, Any] | None = None
        for env_seed, reset in resets_by_env_seed.items():
            positions = reset["positions_robot_base_m"]
            cube_root = [
                float(positions["rubiks_cube"][0]) + repair_offset_m,
                float(positions["rubiks_cube"][1]),
                float(positions["rubiks_cube"][2]),
            ]
            bowl_root = [float(value) for value in positions["bowl"]]
            cube_aabb = aabb_from_root(root_xyz=cube_root, object_name="rubiks_cube")
            bowl_start = aabb_from_root(root_xyz=bowl_root, object_name="bowl")
            for goal in goals:
                for sign in signs:
                    direction = ReferenceMotionController.displacement_vector(
                        goal=goal,
                        fixture="horizontal",
                        physical_sign=sign,
                    )
                    delta = task_displacement_to_robot_delta(
                        direction_task=direction,
                        displacement_m=displacement_m,
                    )
                    swept = conservative_swept_aabb(bowl_start, delta)
                    separation_m = aabb_separation_m(cube_aabb, swept)
                    if separation_m < worst_separation_m:
                        worst_separation_m = separation_m
                        binding = {
                            "environment_seed": int(env_seed),
                            "goal": goal,
                            "physical_translation_sign": sign,
                            "separation_m": separation_m,
                        }
        if worst_separation_m >= guard_m - 1e-12:
            audit = {
                "repair_offset_robot_base_x_m": repair_offset_m,
                "increments_of_1cm": increments,
                "minimum_swept_separation_m": worst_separation_m,
                "support_edge_guard_m": guard_m,
                "minimum_scale": MINIMUM_SCALE,
                "minimum_displacement_m": displacement_m,
                "worst_binding": binding,
                "reset_count": len(resets_by_env_seed),
            }
            return repair_offset_m, audit
    raise HorizontalGeometryRepairError(
        "no cube-only -X repair offset satisfies swept-AABB clearance within search bound"
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
            "Stale root-pose metadata only. G3 scoring and repair clearance use live "
            "USD world AABB projected into the registered task frame; projected "
            "half-extents remain usable."
        ),
    }
