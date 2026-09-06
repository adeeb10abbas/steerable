#!/usr/bin/env python3
"""Build the model-blind C8 displacement-ladder geometry plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.second_stack import (  # noqa: E402
    REFERENCE_OBJECT,
    RELATION_AXES_SCENE_XY,
    SOURCE_OBJECT,
    SUPPORT_CENTER_SCENE_M,
    SUPPORT_HALF_EXTENTS_M,
    reference_destination_xy,
)


SCALES = (2.0, 1.5, 1.0, 0.75, 0.5)
NOMINAL_TRANSLATION_M = 0.08
CUBE_HALF_EXTENT_M = 0.015
SUPPORT_EDGE_MARGIN_M = 0.005
RELATION_CLEARANCE_M = 0.01
PAIR_CLEARANCE_M = 0.005
MIN_SHRINKING_AREA_REMOVED_FRACTION = 0.20


class SecondStackG3PlanError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        sum(
            first[0] * second[1] - first[1] * second[0]
            for first, second in zip(polygon, polygon[1:] + polygon[:1])
        )
    )


def _clip_halfspace(
    polygon: list[tuple[float, float]],
    *,
    axis: tuple[float, float],
    threshold: float,
) -> list[tuple[float, float]]:
    if not polygon:
        return []

    def signed(point: tuple[float, float]) -> float:
        return point[0] * axis[0] + point[1] * axis[1] - threshold

    result: list[tuple[float, float]] = []
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        first_value = signed(first)
        second_value = signed(second)
        first_inside = first_value >= -1e-12
        second_inside = second_value >= -1e-12
        if first_inside:
            result.append(first)
        if first_inside != second_inside:
            denominator = first_value - second_value
            fraction = first_value / denominator
            result.append(
                (
                    first[0] + fraction * (second[0] - first[0]),
                    first[1] + fraction * (second[1] - first[1]),
                )
            )
    return result


def _workspace_polygon() -> list[tuple[float, float]]:
    half_x = SUPPORT_HALF_EXTENTS_M[0] - CUBE_HALF_EXTENT_M - SUPPORT_EDGE_MARGIN_M
    half_y = SUPPORT_HALF_EXTENTS_M[1] - CUBE_HALF_EXTENT_M - SUPPORT_EDGE_MARGIN_M
    center_x, center_y = SUPPORT_CENTER_SCENE_M[:2]
    return [
        (center_x - half_x, center_y - half_y),
        (center_x + half_x, center_y - half_y),
        (center_x + half_x, center_y + half_y),
        (center_x - half_x, center_y + half_y),
    ]


def goal_area_m2(
    *,
    reference_xy: tuple[float, float],
    relation: str,
) -> float:
    axis = RELATION_AXES_SCENE_XY[relation]
    projected_half = CUBE_HALF_EXTENT_M * (abs(axis[0]) + abs(axis[1]))
    threshold = (
        reference_xy[0] * axis[0]
        + reference_xy[1] * axis[1]
        + RELATION_CLEARANCE_M
        + 2.0 * projected_half
    )
    return _polygon_area(
        _clip_halfspace(
            _workspace_polygon(),
            axis=axis,
            threshold=threshold,
        )
    )


def point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta = (end[0] - start[0], end[1] - start[1])
    denominator = delta[0] ** 2 + delta[1] ** 2
    if denominator <= 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = (
        (point[0] - start[0]) * delta[0]
        + (point[1] - start[1]) * delta[1]
    ) / denominator
    fraction = min(max(fraction, 0.0), 1.0)
    nearest = (
        start[0] + fraction * delta[0],
        start[1] + fraction * delta[1],
    )
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def minimum_axis_separation_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    samples: int = 2000,
) -> float:
    if samples <= 0:
        raise SecondStackG3PlanError("segment samples must be positive")
    return min(
        max(
            abs(start[0] + index / samples * (end[0] - start[0]) - point[0]),
            abs(start[1] + index / samples * (end[1] - start[1]) - point[1]),
        )
        for index in range(samples + 1)
    )


def _inside_reference_workspace(point: tuple[float, float]) -> bool:
    center_x, center_y = SUPPORT_CENTER_SCENE_M[:2]
    half_x = SUPPORT_HALF_EXTENTS_M[0] - CUBE_HALF_EXTENT_M - SUPPORT_EDGE_MARGIN_M
    half_y = SUPPORT_HALF_EXTENTS_M[1] - CUBE_HALF_EXTENT_M - SUPPORT_EDGE_MARGIN_M
    return (
        center_x - half_x <= point[0] <= center_x + half_x
        and center_y - half_y <= point[1] <= center_y + half_y
    )


def build_plan(*, registry_path: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        not isinstance(registry, dict)
        or registry.get("fixture_id") != "second_stack"
        or registry.get("registered_env_seed_count") != 64
    ):
        raise SecondStackG3PlanError("C8 reset registry differs")
    resets = registry.get("resets_by_env_seed")
    if not isinstance(resets, Mapping):
        raise SecondStackG3PlanError("C8 reset registry lacks rows")
    scale_rows: list[dict[str, Any]] = []
    for scale in SCALES:
        displacement = NOMINAL_TRANSLATION_M * scale
        checks: list[dict[str, Any]] = []
        for seed_text in sorted(resets, key=int):
            row = resets[seed_text]
            positions = row["positions_scene_xy_m"]
            source = tuple(float(value) for value in positions[SOURCE_OBJECT])
            reference = tuple(float(value) for value in positions[REFERENCE_OBJECT])
            sign = int(row["physical_translation_sign"])
            for relation in RELATION_AXES_SCENE_XY:
                endpoint = reference_destination_xy(
                    initial_xy=reference,
                    relation=relation,
                    displacement_m=displacement,
                    physical_translation_sign=sign,
                )
                initial_area = goal_area_m2(
                    reference_xy=reference,
                    relation=relation,
                )
                endpoint_area = goal_area_m2(
                    reference_xy=endpoint,
                    relation=relation,
                )
                removed_fraction = (
                    max(initial_area - endpoint_area, 0.0) / initial_area
                    if initial_area > 0.0
                    else 0.0
                )
                minimum_pair_distance = point_segment_distance(
                    source,
                    reference,
                    endpoint,
                )
                minimum_axis_separation = minimum_axis_separation_on_segment(
                    source,
                    reference,
                    endpoint,
                )
                shrinking = sign == 1
                passed = (
                    _inside_reference_workspace(reference)
                    and _inside_reference_workspace(endpoint)
                    and minimum_axis_separation
                    >= 2.0 * CUBE_HALF_EXTENT_M + PAIR_CLEARANCE_M
                    and initial_area > 0.0
                    and endpoint_area > 0.0
                )
                checks.append(
                    {
                        "environment_seed": int(seed_text),
                        "relation": relation,
                        "physical_translation_sign": sign,
                        "initial_reference_scene_xy_m": list(reference),
                        "endpoint_reference_scene_xy_m": list(endpoint),
                        "minimum_source_reference_center_distance_m": (
                            minimum_pair_distance
                        ),
                        "minimum_source_reference_axis_separation_m": (
                            minimum_axis_separation
                        ),
                        "initial_goal_area_m2": initial_area,
                        "endpoint_goal_area_m2": endpoint_area,
                        "goal_area_removed_fraction": removed_fraction,
                        "shrinking_direction_case": shrinking,
                        "passed": passed,
                    }
                )
        scale_rows.append(
            {
                "scale": scale,
                "displacement_m": displacement,
                "check_count": len(checks),
                "passed": all(row["passed"] for row in checks),
                "checks": checks,
            }
        )
    selected = next((row for row in scale_rows if row["passed"]), None)
    return {
        "schema_version": "v4-second-stack-g3-geometry-plan-v1",
        "campaign_id": "online_correction_v4",
        "family_ids": ["C8"],
        "fixture_id": "second_stack",
        "qualification_scope": "model_blind_no_policy",
        "status": "candidate_not_released_for_inference",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "reset_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "scale_ladder": list(SCALES),
        "nominal_translation_m": NOMINAL_TRANSLATION_M,
        "selected_analytical_scale": (
            float(selected["scale"]) if selected is not None else None
        ),
        "support_workspace_scene_xy": _workspace_polygon(),
        "relation_axes_scene_xy": {
            key: list(value) for key, value in RELATION_AXES_SCENE_XY.items()
        },
        "thresholds": {
            "cube_half_extent_m": CUBE_HALF_EXTENT_M,
            "support_edge_margin_m": SUPPORT_EDGE_MARGIN_M,
            "relation_clearance_m": RELATION_CLEARANCE_M,
            "pair_clearance_m": PAIR_CLEARANCE_M,
            "minimum_shrinking_area_removed_fraction": (
                MIN_SHRINKING_AREA_REMOVED_FRACTION
            ),
                "shrinking_area_fraction_gate_applicable": False,
        },
        "scales": scale_rows,
        "release_boundary": (
            "Analytical C8 G3 geometry candidate only. Live swept-path and "
            "privileged scripted-controller receipts remain required."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite C8 G3 plan: {args.output}")
    payload = build_plan(registry_path=args.registry.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": sha256_file(args.output),
                "selected_analytical_scale": payload["selected_analytical_scale"],
                "passing_scales": [
                    row["scale"] for row in payload["scales"] if row["passed"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["selected_analytical_scale"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
