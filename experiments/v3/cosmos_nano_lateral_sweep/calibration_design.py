"""Pure geometry and selection rules for the V3-B004 lateral sweep.

This module deliberately imports neither Isaac/RoboLab nor any model package.
All rounding is integer millimetres so the preregistered 10 mm scan and 30 mm
half-range grids cannot drift through binary floating-point arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


GRID_MM = 10
RADIUS_GRID_MM = 30
MINIMUM_RADIUS_MM = 90
CONTROL_BOWL_Y_MM = 127  # display/index anchor only; exact centre stays float below
CONTROL_BOWL_Y_M = 0.12658219039440155


class CalibrationDesignError(ValueError):
    """Raised when a model-blind scan cannot release seven numeric levels."""


def _mm(value_m: float) -> int:
    if not math.isfinite(value_m):
        raise CalibrationDesignError("coordinate must be finite")
    return round(value_m * 1000.0)


def dense_candidates(*, lower_y_m: float, upper_y_m: float) -> tuple[float, ...]:
    """Return 10 mm displacements about the exact control bowl coordinate."""

    if lower_y_m >= upper_y_m:
        raise CalibrationDesignError("live lower bound must be below upper bound")
    minimum_offset_mm = math.ceil((lower_y_m - CONTROL_BOWL_Y_M) * 1000 / GRID_MM) * GRID_MM
    maximum_offset_mm = math.floor((upper_y_m - CONTROL_BOWL_Y_M) * 1000 / GRID_MM) * GRID_MM
    if minimum_offset_mm > maximum_offset_mm:
        raise CalibrationDesignError("live support contains no aligned scan candidate")
    return tuple(
        CONTROL_BOWL_Y_M + offset_mm / 1000.0
        for offset_mm in range(minimum_offset_mm, maximum_offset_mm + GRID_MM, GRID_MM)
    )


def candidate_key(y_m: float) -> int:
    """Map an aligned candidate to its integer displacement in millimetres."""

    offset_mm = _mm((y_m - CONTROL_BOWL_Y_M))
    if offset_mm % GRID_MM:
        raise CalibrationDesignError("candidate is not on the frozen 10 mm displacement grid")
    return offset_mm


def seven_levels(radius_mm: int) -> tuple[float, ...]:
    if radius_mm < MINIMUM_RADIUS_MM or radius_mm % RADIUS_GRID_MM:
        raise CalibrationDesignError("radius must be at least 90 mm on the 30 mm grid")
    offsets = (-radius_mm, -2 * radius_mm // 3, -radius_mm // 3, 0,
               radius_mm // 3, 2 * radius_mm // 3, radius_mm)
    if any(offset % GRID_MM for offset in offsets):
        raise CalibrationDesignError("seven levels do not lie on the 10 mm scan grid")
    return tuple(CONTROL_BOWL_Y_M + offset / 1000.0 for offset in offsets)


def select_largest_radius(passing_y_m: Iterable[float]) -> tuple[int, tuple[float, ...]]:
    keys = {candidate_key(value) for value in passing_y_m}
    if not keys:
        raise CalibrationDesignError("no dense-scan candidate passed")
    max_abs = max(abs(value) for value in keys)
    candidates = range(
        MINIMUM_RADIUS_MM,
        max_abs - (max_abs % RADIUS_GRID_MM) + RADIUS_GRID_MM,
        RADIUS_GRID_MM,
    )
    valid: list[int] = []
    for radius in candidates:
        required = {-radius, -2 * radius // 3, -radius // 3, 0,
                    radius // 3, 2 * radius // 3, radius}
        if required <= keys:
            valid.append(radius)
    if not valid:
        raise CalibrationDesignError("no symmetric seven-level radius >=90 mm passed")
    radius = max(valid)
    return radius, seven_levels(radius)


def neutral_under_frozen_cones(
    *, cube_x_m: float, cube_y_m: float, bowl_x_m: float, bowl_y_m: float
) -> bool:
    """Exact complement of RoboLab's LEFT/RIGHT 45-degree cone at reset."""

    return abs(cube_y_m - bowl_y_m) < abs(cube_x_m - bowl_x_m)


def xy_aabb_separation_m(
    first_min_xy: Sequence[float],
    first_max_xy: Sequence[float],
    second_min_xy: Sequence[float],
    second_max_xy: Sequence[float],
) -> float:
    """Positive shortest XY gap; zero means projections touch or overlap."""

    if not all(len(value) == 2 for value in (first_min_xy, first_max_xy, second_min_xy, second_max_xy)):
        raise CalibrationDesignError("XY AABB endpoints must each contain two values")
    dx = max(second_min_xy[0] - first_max_xy[0], first_min_xy[0] - second_max_xy[0], 0.0)
    dy = max(second_min_xy[1] - first_max_xy[1], first_min_xy[1] - second_max_xy[1], 0.0)
    return math.hypot(dx, dy)
