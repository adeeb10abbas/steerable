"""Pure horizontal RoboLab geometry and contact checks for model-blind G3."""

from __future__ import annotations

from itertools import product
import math
from typing import Any, Mapping

from experiments.online_correction_v4.droid_task_files.constants import (
    FixtureObjectSpec,
    fixture_object_spec,
)
from experiments.online_correction_v4.geometry import (
    AxisAlignedBox,
    ObjectFootprint,
    PlanarRelationSpec,
    TaskFrame,
    build_planar_goal_set,
    planar_goal_area,
    shrinking_area_fraction,
)


class DroidG3Error(ValueError):
    """Raised when live G3 evidence cannot support a fail-closed decision."""


def physics_sampling_stride(
    physics_dt_s: float, *, maximum_interval_s: float
) -> tuple[int, float]:
    """Return the largest integral physics stride no coarser than the gate cap."""
    if (
        isinstance(physics_dt_s, bool)
        or not isinstance(physics_dt_s, (int, float))
        or not math.isfinite(float(physics_dt_s))
        or physics_dt_s <= 0
    ):
        raise DroidG3Error("physics_dt_s must be positive and finite")
    if (
        isinstance(maximum_interval_s, bool)
        or not isinstance(maximum_interval_s, (int, float))
        or not math.isfinite(float(maximum_interval_s))
        or maximum_interval_s <= 0
    ):
        raise DroidG3Error("maximum_interval_s must be positive and finite")
    stride = int(math.floor(float(maximum_interval_s) / float(physics_dt_s) + 1e-12))
    if stride < 1:
        raise DroidG3Error("physics dt is coarser than the G3 path-sampling cap")
    interval = stride * float(physics_dt_s)
    if interval > float(maximum_interval_s) + 1e-12:
        raise DroidG3Error("derived G3 sample interval exceeds the gate cap")
    return stride, interval


def scenario_duration_s(
    scenario: str, motion_config: Mapping[str, Any]
) -> float:
    if scenario in {"original_sham", "destination_static", "move_stop"}:
        key = "move_stop_duration_s"
        duration = motion_config.get(key)
    elif scenario == "slow_drift":
        duration = motion_config.get("slow_drift_duration_s")
    elif scenario == "fast_drift":
        duration = motion_config.get("fast_drift_duration_s")
    elif scenario == "reversal":
        waypoints = motion_config.get("reversal_waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise DroidG3Error("reversal scenario lacks waypoints")
        duration = waypoints[-1].get("time_s")
    else:
        raise DroidG3Error(f"unsupported horizontal G3 scenario: {scenario!r}")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or duration <= 0
    ):
        raise DroidG3Error(f"invalid duration for G3 scenario {scenario!r}")
    return float(duration)


def task_frame_from_evidence(evidence: Mapping[str, Any]) -> TaskFrame:
    def vector(key: str) -> tuple[float, float, float]:
        raw = evidence.get(key)
        if not isinstance(raw, list) or len(raw) != 3:
            raise DroidG3Error(f"task-frame evidence lacks {key}")
        values = tuple(float(value) for value in raw)
        if not all(math.isfinite(value) for value in values):
            raise DroidG3Error(f"task-frame evidence {key} is non-finite")
        return values

    return TaskFrame(
        u_left=vector("u_left_world"),
        u_front=vector("u_front_world"),
        u_up=vector("u_up_world"),
        origin=vector("origin_world"),
    )


def bounds_world_to_task(
    frame: TaskFrame, bounds: Mapping[str, Any]
) -> AxisAlignedBox:
    minimum = bounds.get("min_xyz")
    maximum = bounds.get("max_xyz")
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 3
        or len(maximum) != 3
    ):
        raise DroidG3Error("world bounds must contain min_xyz/max_xyz 3-vectors")
    lo = tuple(float(value) for value in minimum)
    hi = tuple(float(value) for value in maximum)
    if not all(math.isfinite(value) for value in (*lo, *hi)):
        raise DroidG3Error("world bounds are non-finite")
    if any(left >= right for left, right in zip(lo, hi)):
        raise DroidG3Error("world bounds are empty")
    corners = [
        frame.world_to_task((x, y, z))
        for x, y, z in product(
            (lo[0], hi[0]),
            (lo[1], hi[1]),
            (lo[2], hi[2]),
        )
    ]
    return AxisAlignedBox(
        min(point[0] for point in corners),
        max(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[1] for point in corners),
        min(point[2] for point in corners),
        max(point[2] for point in corners),
    )


def footprint_from_task_bounds(bounds: AxisAlignedBox) -> ObjectFootprint:
    return ObjectFootprint(
        half_left=(bounds.x_max - bounds.x_min) / 2.0,
        half_front=(bounds.y_max - bounds.y_min) / 2.0,
        half_up=(bounds.z_max - bounds.z_min) / 2.0,
    )


def supported_center_workspace(
    table_bounds: AxisAlignedBox,
    target_footprint: ObjectFootprint,
    *,
    edge_margin_m: float,
) -> AxisAlignedBox:
    if not math.isfinite(edge_margin_m) or edge_margin_m < 0:
        raise DroidG3Error("support edge margin must be finite and non-negative")
    workspace = AxisAlignedBox(
        table_bounds.x_min + target_footprint.half_left + edge_margin_m,
        table_bounds.x_max - target_footprint.half_left - edge_margin_m,
        table_bounds.y_min + target_footprint.half_front + edge_margin_m,
        table_bounds.y_max - target_footprint.half_front - edge_margin_m,
        table_bounds.z_max,
        table_bounds.z_max + 2.0 * target_footprint.half_up,
    )
    if workspace.is_empty():
        raise DroidG3Error("target-eroded supported workspace is empty")
    return workspace


def planar_geometry_from_scene(
    *,
    task_frame_evidence: Mapping[str, Any],
    scene_state: Mapping[str, Any],
    support_edge_margin_m: float,
    target_object: str,
    reference_object: str,
) -> dict[str, Any]:
    frame = task_frame_from_evidence(task_frame_evidence)
    objects = scene_state.get("objects")
    if not isinstance(objects, Mapping):
        raise DroidG3Error("scene state lacks objects")
    target = objects.get(target_object)
    reference = objects.get(reference_object)
    if not isinstance(target, Mapping) or not isinstance(reference, Mapping):
        raise DroidG3Error("scene state lacks target/reference objects")
    table_bounds = bounds_world_to_task(frame, scene_state.get("table_world_aabb_m", {}))
    target_bounds = bounds_world_to_task(frame, target.get("world_aabb_m", {}))
    reference_bounds = bounds_world_to_task(frame, reference.get("world_aabb_m", {}))
    target_footprint = footprint_from_task_bounds(target_bounds)
    reference_footprint = footprint_from_task_bounds(reference_bounds)
    workspace = supported_center_workspace(
        table_bounds,
        target_footprint,
        edge_margin_m=support_edge_margin_m,
    )
    return {
        "frame": frame,
        "table_bounds_task": table_bounds,
        "target_footprint": target_footprint,
        "reference_footprint": reference_footprint,
        "target_workspace": workspace,
    }


def horizontal_geometry_from_scene(
    *,
    task_frame_evidence: Mapping[str, Any],
    scene_state: Mapping[str, Any],
    support_edge_margin_m: float,
) -> dict[str, Any]:
    spec = fixture_object_spec("horizontal")
    return planar_geometry_from_scene(
        task_frame_evidence=task_frame_evidence,
        scene_state=scene_state,
        support_edge_margin_m=support_edge_margin_m,
        target_object=spec.target_object,
        reference_object=spec.reference_object,
    )


def geometry_from_scene_for_fixture(
    *,
    fixture_id: str,
    task_frame_evidence: Mapping[str, Any],
    scene_state: Mapping[str, Any],
    support_edge_margin_m: float,
) -> dict[str, Any]:
    spec = fixture_object_spec(fixture_id)
    return planar_geometry_from_scene(
        task_frame_evidence=task_frame_evidence,
        scene_state=scene_state,
        support_edge_margin_m=support_edge_margin_m,
        target_object=spec.target_object,
        reference_object=spec.reference_object,
    )


def reference_is_supported(
    *,
    frame: TaskFrame,
    reference_position_world: tuple[float, float, float],
    table_bounds_task: AxisAlignedBox,
    reference_footprint: ObjectFootprint,
    edge_margin_m: float,
) -> bool:
    x, y, _z = frame.world_to_task(reference_position_world)
    return (
        x - reference_footprint.half_left
        >= table_bounds_task.x_min + edge_margin_m
        and x + reference_footprint.half_left
        <= table_bounds_task.x_max - edge_margin_m
        and y - reference_footprint.half_front
        >= table_bounds_task.y_min + edge_margin_m
        and y + reference_footprint.half_front
        <= table_bounds_task.y_max - edge_margin_m
    )


def goal_set_for_reference(
    *,
    geometry: Mapping[str, Any],
    relation: str,
    reference_position_world: tuple[float, float, float],
    clearance_m: float,
):
    if relation not in {"left", "right", "front", "behind"}:
        raise DroidG3Error(f"unsupported horizontal relation: {relation!r}")
    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise DroidG3Error("relation clearance must be finite and non-negative")
    spec = PlanarRelationSpec(
        relation=relation,  # type: ignore[arg-type]
        clearance_m=clearance_m,
        workspace=geometry["target_workspace"],
        object_footprint=geometry["target_footprint"],
        reference_footprint=geometry["reference_footprint"],
    )
    return build_planar_goal_set(
        geometry["frame"],
        spec,
        reference_position_world,
    )


def goal_area_case(
    *,
    geometry: Mapping[str, Any],
    relation: str,
    original_reference_world: tuple[float, float, float],
    endpoint_reference_world: tuple[float, float, float],
    clearance_m: float,
    minimum_shrinking_area_fraction: float,
) -> dict[str, Any]:
    before = goal_set_for_reference(
        geometry=geometry,
        relation=relation,
        reference_position_world=original_reference_world,
        clearance_m=clearance_m,
    )
    after = goal_set_for_reference(
        geometry=geometry,
        relation=relation,
        reference_position_world=endpoint_reference_world,
        clearance_m=clearance_m,
    )
    before_area = 0.0 if before.region is None else planar_goal_area(before.region)
    after_area = 0.0 if after.region is None else planar_goal_area(after.region)
    shrinking = after_area < before_area - 1e-12
    removed_fraction = (
        math.nan
        if before.region is None
        else shrinking_area_fraction(
            before.region,
            after.region
            if after.region is not None
            else AxisAlignedBox(1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        )
    )
    passes = (
        not before.empty
        and not after.empty
        and (
            not shrinking
            or removed_fraction + 1e-12 >= minimum_shrinking_area_fraction
        )
    )
    return {
        "relation": relation,
        "original_area_m2": before_area,
        "destination_area_m2": after_area,
        "shrinking_direction": shrinking,
        "removed_area_fraction": removed_fraction,
        "minimum_shrinking_area_fraction": minimum_shrinking_area_fraction,
        "original_goal_empty": before.empty,
        "destination_goal_empty": after.empty,
        "passes_information_gate": passes,
    }


def _contact_pairs_for_objects(object_names: tuple[str, ...]) -> set[str]:
    pairs: set[str] = set()
    for name in object_names:
        pairs.add(f"{name}__table")
        pairs.add(f"table__{name}")
    return pairs


def classify_contacts(
    force_n_by_pair: Mapping[str, Any],
    *,
    active_force_threshold_n: float,
    fixture_spec: FixtureObjectSpec | None = None,
) -> dict[str, Any]:
    if fixture_spec is None:
        fixture_spec = fixture_object_spec("horizontal")
    if not math.isfinite(active_force_threshold_n) or active_force_threshold_n <= 0:
        raise DroidG3Error("active contact threshold must be positive and finite")
    active: dict[str, float] = {}
    for name, value in force_n_by_pair.items():
        force = float(value)
        if not math.isfinite(force) or force < 0:
            raise DroidG3Error(f"invalid contact force for {name!r}")
        if force >= active_force_threshold_n:
            active[str(name)] = force
    support_objects = tuple(fixture_spec.movable_objects)
    if fixture_spec.distractor_object is not None:
        support_objects = support_objects + (fixture_spec.distractor_object,)
    supported = {
        name: any(
            pair in active
            for pair in (f"{name}__table", f"table__{name}")
        )
        for name in support_objects
    }
    reference_object = fixture_spec.reference_object
    reference_robot_pairs = [
        pair
        for pair in active
        if reference_object in pair
        and any(token in pair for token in ("robot_all", "gripper"))
    ]
    allowed_pairs = _contact_pairs_for_objects(support_objects)
    disallowed_pairs = sorted(pair for pair in active if pair not in allowed_pairs)
    return {
        "active_force_threshold_n": active_force_threshold_n,
        "active_force_n_by_pair": active,
        "supported_by_object": supported,
        "support_valid": all(supported.values()),
        "reference_robot_contact": bool(reference_robot_pairs),
        "reference_robot_contact_pairs": sorted(reference_robot_pairs),
        "unmodeled_collision": bool(disallowed_pairs),
        "disallowed_contact_pairs": disallowed_pairs,
    }
