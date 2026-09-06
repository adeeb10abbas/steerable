"""Pure-Python geometry for V4 goal regions, distances, and prompt equivalence.

All coordinates are meters in a right-handed world frame unless noted. Task-frame
axes follow the frozen campaign convention: ``u_front`` toward the robot,
``u_left`` to the robot's left, ``u_up`` opposite gravity. Fixture-specific
dimensions are supplied by callers; this module does not embed simulator geometry.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

Vec3 = tuple[float, float, float]

HORIZONTAL_RELATIONS = frozenset({"left", "right", "front", "behind"})
VERTICAL_RELATIONS = frozenset({"above", "below"})
CONTAINMENT_RELATIONS = frozenset({"inside", "contains"})
PLANAR_RELATIONS = HORIZONTAL_RELATIONS
ALL_RELATIONS = HORIZONTAL_RELATIONS | VERTICAL_RELATIONS | CONTAINMENT_RELATIONS

RelationKind = Literal[
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "inside",
    "contains",
]
ProjectionKind = Literal["terminal", "response_planar", "response_vertical", "response_opening"]


class UnsupportedReferenceOrientationError(ValueError):
    """Raised when a non-identity reference orientation is not yet supported."""


@dataclass(frozen=True)
class ReferenceOrientation:
    """Frozen reference-body orientation relative to the task frame.

    Columns are the reference-local basis vectors expressed in world coordinates.
    Only the identity orientation is supported for containment scoring today;
    callers must fail closed rather than silently ignore rotation.
    """

    u_left: Vec3 = (1.0, 0.0, 0.0)
    u_front: Vec3 = (0.0, 1.0, 0.0)
    u_up: Vec3 = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        for name, axis in (("u_left", self.u_left), ("u_front", self.u_front), ("u_up", self.u_up)):
            if not math.isfinite(_norm(axis)):
                raise ValueError(f"{name} must be finite and nonzero")

    @classmethod
    def identity(cls) -> ReferenceOrientation:
        return cls()

    def is_identity(self, tol: float = 1e-9) -> bool:
        identity = self.identity()
        for a, b in zip((self.u_left, self.u_front, self.u_up), (identity.u_left, identity.u_front, identity.u_up)):
            if any(abs(x - y) > tol for x, y in zip(a, b)):
                return False
        return True

    def local_offset_to_world(self, p_local: Vec3) -> Vec3:
        return (
            self.u_left[0] * p_local[0] + self.u_front[0] * p_local[1] + self.u_up[0] * p_local[2],
            self.u_left[1] * p_local[0] + self.u_front[1] * p_local[1] + self.u_up[1] * p_local[2],
            self.u_left[2] * p_local[0] + self.u_front[2] * p_local[1] + self.u_up[2] * p_local[2],
        )


@dataclass(frozen=True)
class TrajectorySample:
    simulation_time_s: float
    p_obj_world: Vec3
    p_named_ref_world: Vec3


@dataclass(frozen=True)
class TrajectorySeries:
    """Generic executed-trajectory samples, independent of analysis estimands."""

    samples: tuple[TrajectorySample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("trajectory must contain at least one sample")
        for sample in self.samples:
            if not math.isfinite(sample.simulation_time_s):
                raise ValueError("simulation_time_s must be finite")
        if any(self.samples[index].simulation_time_s > self.samples[index + 1].simulation_time_s for index in range(len(self.samples) - 1)):
            raise ValueError("trajectory samples must be sorted by simulation_time_s")


@dataclass(frozen=True)
class ResolvedTrajectorySample:
    sample: TrajectorySample
    terminal_extension_applied: bool


@dataclass(frozen=True)
class HorizonDistanceSample:
    simulation_time_s: float
    p_obj_world: Vec3
    p_ref_world: Vec3
    distance: "DistanceResult"
    terminal_extension_applied: bool



def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Vec3) -> Vec3:
    n = _norm(a)
    if n == 0.0:
        raise ValueError("zero-length vector cannot be normalized")
    return _scale(a, 1.0 / n)


def _axis_distance(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        return math.inf
    if value < lo:
        return lo - value
    if value > hi:
        return value - hi
    return 0.0


@dataclass(frozen=True)
class TaskFrame:
    """Orthonormal robot-relative task basis recorded in world coordinates."""

    u_left: Vec3
    u_front: Vec3
    u_up: Vec3
    origin: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name, axis in (("u_left", self.u_left), ("u_front", self.u_front), ("u_up", self.u_up)):
            norm = _norm(axis)
            if not math.isfinite(norm) or abs(norm - 1.0) > 1e-9:
                raise ValueError(f"{name} must be a finite unit vector")
        if abs(_dot(self.u_left, self.u_front)) > 1e-9:
            raise ValueError("u_left and u_front must be orthogonal")
        if abs(_dot(self.u_left, self.u_up)) > 1e-9:
            raise ValueError("u_left and u_up must be orthogonal")
        if abs(_dot(self.u_front, self.u_up)) > 1e-9:
            raise ValueError("u_front and u_up must be orthogonal")
        cross = (
            self.u_left[1] * self.u_front[2] - self.u_left[2] * self.u_front[1],
            self.u_left[2] * self.u_front[0] - self.u_left[0] * self.u_front[2],
            self.u_left[0] * self.u_front[1] - self.u_left[1] * self.u_front[0],
        )
        if any(abs(actual - expected) > 1e-9 for actual, expected in zip(cross, self.u_up)):
            raise ValueError("task frame must be right-handed: u_left × u_front = u_up")

    @classmethod
    def identity(cls) -> TaskFrame:
        return cls(u_left=(1.0, 0.0, 0.0), u_front=(0.0, 1.0, 0.0), u_up=(0.0, 0.0, 1.0))

    def world_to_task(self, p_world: Vec3) -> Vec3:
        local = _sub(p_world, self.origin)
        return (_dot(local, self.u_left), _dot(local, self.u_front), _dot(local, self.u_up))

    def task_to_world(self, p_task: Vec3) -> Vec3:
        rotated = _add(
            _add(_scale(self.u_left, p_task[0]), _scale(self.u_front, p_task[1])),
            _scale(self.u_up, p_task[2]),
        )
        return _add(self.origin, rotated)

    def relative_offset_task(self, p_obj_world: Vec3, p_ref_world: Vec3) -> Vec3:
        return _sub(self.world_to_task(p_obj_world), self.world_to_task(p_ref_world))

    def signed_axes(self, p_obj_world: Vec3, p_ref_world: Vec3) -> dict[str, float]:
        rel = self.relative_offset_task(p_obj_world, p_ref_world)
        return {"signed_left_m": rel[0], "signed_front_m": rel[1], "signed_up_m": rel[2]}


@dataclass(frozen=True)
class ObjectFootprint:
    """Conservative axis-aligned half-extents in the task frame."""

    half_left: float
    half_front: float
    half_up: float

    def __post_init__(self) -> None:
        for name, value in (
            ("half_left", self.half_left),
            ("half_front", self.half_front),
            ("half_up", self.half_up),
        ):
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class AxisAlignedBox:
    """Closed axis-aligned region expressed in task coordinates."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def is_empty(self) -> bool:
        return (
            self.x_min > self.x_max
            or self.y_min > self.y_max
            or self.z_min > self.z_max
        )

    def intersect(self, other: AxisAlignedBox) -> AxisAlignedBox:
        return AxisAlignedBox(
            x_min=max(self.x_min, other.x_min),
            x_max=min(self.x_max, other.x_max),
            y_min=max(self.y_min, other.y_min),
            y_max=min(self.y_max, other.y_max),
            z_min=max(self.z_min, other.z_min),
            z_max=min(self.z_max, other.z_max),
        )

    def point_inside(self, p: Vec3, tol: float = 0.0) -> bool:
        if self.is_empty():
            return False
        return (
            self.x_min - tol <= p[0] <= self.x_max + tol
            and self.y_min - tol <= p[1] <= self.y_max + tol
            and self.z_min - tol <= p[2] <= self.z_max + tol
        )

    def distance_to_point(self, p: Vec3) -> float:
        if self.is_empty():
            return math.inf
        dx = _axis_distance(p[0], self.x_min, self.x_max)
        dy = _axis_distance(p[1], self.y_min, self.y_max)
        dz = _axis_distance(p[2], self.z_min, self.z_max)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def planar_xy_footprint(self) -> AxisAlignedBox:
        return AxisAlignedBox(self.x_min, self.x_max, self.y_min, self.y_max, 0.0, 0.0)


@dataclass(frozen=True)
class ConvexPolygonPrism:
    """Closed convex XY polygon with a task-frame vertical interval."""

    vertices_xy: tuple[tuple[float, float], ...]
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        if len(self.vertices_xy) < 3:
            raise ValueError("convex polygon requires at least three vertices")
        if any(
            not math.isfinite(value)
            for point in self.vertices_xy
            for value in point
        ):
            raise ValueError("convex polygon vertices must be finite")
        if self.z_min > self.z_max:
            raise ValueError("convex polygon prism vertical interval is empty")
        signs = []
        for first, second, third in zip(
            self.vertices_xy,
            self.vertices_xy[1:] + self.vertices_xy[:1],
            self.vertices_xy[2:] + self.vertices_xy[:2],
        ):
            cross = (
                (second[0] - first[0]) * (third[1] - second[1])
                - (second[1] - first[1]) * (third[0] - second[0])
            )
            if abs(cross) > 1e-12:
                signs.append(math.copysign(1.0, cross))
        if not signs or min(signs) != max(signs):
            raise ValueError("polygon vertices must define a nondegenerate convex region")

    def is_empty(self) -> bool:
        return False

    def point_inside(self, p: Vec3, tol: float = 0.0) -> bool:
        if not self.z_min - tol <= p[2] <= self.z_max + tol:
            return False
        orientation = _polygon_orientation(self.vertices_xy)
        for first, second in zip(
            self.vertices_xy,
            self.vertices_xy[1:] + self.vertices_xy[:1],
        ):
            edge_x = second[0] - first[0]
            edge_y = second[1] - first[1]
            cross = edge_x * (p[1] - first[1]) - edge_y * (p[0] - first[0])
            if orientation * cross < -tol * math.hypot(edge_x, edge_y):
                return False
        return True

    def distance_to_point(self, p: Vec3) -> float:
        nearest = self.nearest_point(p)
        return math.dist(p, nearest)

    def nearest_point(self, p: Vec3) -> Vec3:
        z = min(max(p[2], self.z_min), self.z_max)
        planar = (p[0], p[1], z)
        if self.point_inside(planar):
            return planar
        nearest_xy = min(
            (
                _nearest_point_on_segment_xy((p[0], p[1]), first, second)
                for first, second in zip(
                    self.vertices_xy,
                    self.vertices_xy[1:] + self.vertices_xy[:1],
                )
            ),
            key=lambda point: math.hypot(point[0] - p[0], point[1] - p[1]),
        )
        return (nearest_xy[0], nearest_xy[1], z)


def _polygon_orientation(vertices: tuple[tuple[float, float], ...]) -> float:
    twice_area = sum(
        first[0] * second[1] - first[1] * second[0]
        for first, second in zip(vertices, vertices[1:] + vertices[:1])
    )
    return 1.0 if twice_area > 0.0 else -1.0


def _nearest_point_on_segment_xy(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float]:
    delta_x = second[0] - first[0]
    delta_y = second[1] - first[1]
    denominator = delta_x * delta_x + delta_y * delta_y
    if denominator <= 0.0:
        return first
    fraction = (
        (point[0] - first[0]) * delta_x
        + (point[1] - first[1]) * delta_y
    ) / denominator
    fraction = min(max(fraction, 0.0), 1.0)
    return (
        first[0] + fraction * delta_x,
        first[1] + fraction * delta_y,
    )


def _clip_polygon_axis(
    region: ConvexPolygonPrism,
    *,
    axis: str,
    min_value: float | None,
    max_value: float | None,
) -> ConvexPolygonPrism | None:
    vertices = list(region.vertices_xy)
    axis_index = 0 if axis == "x" else 1
    for threshold, keep_greater in (
        (min_value, True),
        (max_value, False),
    ):
        if threshold is None:
            continue
        result: list[tuple[float, float]] = []
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            first_delta = first[axis_index] - threshold
            second_delta = second[axis_index] - threshold
            first_inside = first_delta >= -1e-12 if keep_greater else first_delta <= 1e-12
            second_inside = second_delta >= -1e-12 if keep_greater else second_delta <= 1e-12
            if first_inside:
                result.append(first)
            if first_inside != second_inside:
                fraction = first_delta / (first_delta - second_delta)
                result.append(
                    (
                        first[0] + fraction * (second[0] - first[0]),
                        first[1] + fraction * (second[1] - first[1]),
                    )
                )
        vertices = result
        if len(vertices) < 3:
            return None
    return ConvexPolygonPrism(tuple(vertices), region.z_min, region.z_max)


@dataclass(frozen=True)
class PlanarRelationSpec:
    relation: RelationKind
    clearance_m: float
    workspace: AxisAlignedBox
    object_footprint: ObjectFootprint
    reference_footprint: ObjectFootprint

    def __post_init__(self) -> None:
        if self.relation not in HORIZONTAL_RELATIONS:
            raise ValueError(f"unsupported planar relation {self.relation!r}")
        if self.clearance_m < 0.0 or not math.isfinite(self.clearance_m):
            raise ValueError("clearance_m must be finite and nonnegative")


@dataclass(frozen=True)
class PolygonPlanarRelationSpec:
    relation: RelationKind
    clearance_m: float
    workspace: ConvexPolygonPrism
    object_footprint: ObjectFootprint
    reference_footprint: ObjectFootprint

    def __post_init__(self) -> None:
        if self.relation not in HORIZONTAL_RELATIONS:
            raise ValueError(f"unsupported planar relation {self.relation!r}")
        if self.clearance_m < 0.0 or not math.isfinite(self.clearance_m):
            raise ValueError("clearance_m must be finite and nonnegative")


@dataclass(frozen=True)
class ShelfRelationSpec:
    relation: RelationKind
    top_shelf: AxisAlignedBox
    bottom_shelf: AxisAlignedBox
    reference_footprint: ObjectFootprint
    object_footprint: ObjectFootprint
    horizontal_overlap_min_m: float
    support_surface_tol_m: float = 0.002

    def __post_init__(self) -> None:
        if self.relation not in VERTICAL_RELATIONS:
            raise ValueError(f"unsupported shelf relation {self.relation!r}")
        if self.horizontal_overlap_min_m < 0.0:
            raise ValueError("horizontal_overlap_min_m must be nonnegative")


@dataclass(frozen=True)
class ContainmentSpec:
    interior_reference_local: AxisAlignedBox
    object_footprint: ObjectFootprint
    wall_clearance_m: float
    orientation: ReferenceOrientation = ReferenceOrientation.identity()
    support_surface_tol_m: float = 0.002

    def __post_init__(self) -> None:
        if self.wall_clearance_m < 0.0:
            raise ValueError("wall_clearance_m must be nonnegative")
        if not self.orientation.is_identity():
            raise UnsupportedReferenceOrientationError(
                "non-identity reference orientation is not supported for containment scoring"
            )

    @property
    def interior(self) -> AxisAlignedBox:
        """Backward-compatible alias for the reference-local interior declaration."""
        return self.interior_reference_local


@dataclass(frozen=True)
class GoalSetResult:
    region: AxisAlignedBox | ConvexPolygonPrism | None
    empty: bool
    empty_cause: str | None
    projection_kind: ProjectionKind
    component_id: str


@dataclass(frozen=True)
class DistanceResult:
    distance_m: float
    capped_distance_m: float
    d_cap_m: float
    goal_set_empty: bool
    cap_applied: bool
    nearest_point_task: Vec3 | None
    component_id: str | None
    projection_kind: ProjectionKind


def canonical_relation(relation: str) -> RelationKind:
    normalized = relation.strip().lower()
    if normalized == "contains":
        return "inside"
    if normalized not in ALL_RELATIONS:
        raise ValueError(f"unsupported relation {relation!r}")
    return normalized  # type: ignore[return-value]


def relation_from_wording(relation: str, wording: str) -> RelationKind:
    canonical = canonical_relation(relation)
    if wording not in {"direct", "inverse"}:
        raise ValueError("wording must be 'direct' or 'inverse'")
    return canonical


def _reference_center_task(frame: TaskFrame, p_ref_world: Vec3) -> Vec3:
    return frame.world_to_task(p_ref_world)


def _object_center_task(frame: TaskFrame, p_obj_world: Vec3) -> Vec3:
    return frame.world_to_task(p_obj_world)


def _planar_thresholds(
    relation: RelationKind,
    ref_center: Vec3,
    ref_fp: ObjectFootprint,
    obj_fp: ObjectFootprint,
    clearance_m: float,
) -> tuple[str, float | None, float | None]:
    """Return (primary_axis, min_value, max_value) in task coordinates for object center."""
    x_ref, y_ref, z_ref = ref_center
    if relation == "left":
        return ("x", x_ref + ref_fp.half_left + obj_fp.half_left + clearance_m, None)
    if relation == "right":
        return ("x", None, x_ref - ref_fp.half_left - obj_fp.half_left - clearance_m)
    if relation == "front":
        return ("y", y_ref + ref_fp.half_front + obj_fp.half_front + clearance_m, None)
    if relation == "behind":
        return ("y", None, y_ref - ref_fp.half_front - obj_fp.half_front - clearance_m)
    raise ValueError(relation)


def _apply_center_constraint(
    workspace: AxisAlignedBox, axis: str, min_value: float | None, max_value: float | None
) -> AxisAlignedBox:
    box = workspace
    if axis == "x":
        if min_value is not None:
            box = AxisAlignedBox(min_value, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max)
        if max_value is not None:
            box = AxisAlignedBox(box.x_min, max_value, box.y_min, box.y_max, box.z_min, box.z_max)
    elif axis == "y":
        if min_value is not None:
            box = AxisAlignedBox(box.x_min, box.x_max, min_value, box.y_max, box.z_min, box.z_max)
        if max_value is not None:
            box = AxisAlignedBox(box.x_min, box.x_max, box.y_min, max_value, box.z_min, box.z_max)
    else:
        raise ValueError(axis)
    return box


def build_planar_goal_set(
    frame: TaskFrame,
    spec: PlanarRelationSpec | PolygonPlanarRelationSpec,
    p_ref_world: Vec3,
    *,
    projection_kind: ProjectionKind = "terminal",
) -> GoalSetResult:
    ref_center = _reference_center_task(frame, p_ref_world)
    axis, min_value, max_value = _planar_thresholds(
        spec.relation,
        ref_center,
        spec.reference_footprint,
        spec.object_footprint,
        spec.clearance_m,
    )
    if isinstance(spec.workspace, ConvexPolygonPrism):
        region = _clip_polygon_axis(
            spec.workspace,
            axis=axis,
            min_value=min_value,
            max_value=max_value,
        )
    else:
        region = _apply_center_constraint(
            spec.workspace,
            axis,
            min_value,
            max_value,
        )
    if region is None or region.is_empty():
        return GoalSetResult(
            region=None,
            empty=True,
            empty_cause=f"planar_{spec.relation}_halfspace_misses_workspace",
            projection_kind=projection_kind,
            component_id=f"planar:{spec.relation}",
        )
    return GoalSetResult(
        region=region,
        empty=False,
        empty_cause=None,
        projection_kind=projection_kind,
        component_id=f"planar:{spec.relation}",
    )


def _horizontal_overlap_amount(a: AxisAlignedBox, b: AxisAlignedBox) -> float:
    overlap_x = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    overlap_y = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    return overlap_x * overlap_y


def _object_horizontal_footprint_at(center: Vec3, footprint: ObjectFootprint) -> AxisAlignedBox:
    return AxisAlignedBox(
        center[0] - footprint.half_left,
        center[0] + footprint.half_left,
        center[1] - footprint.half_front,
        center[1] + footprint.half_front,
        0.0,
        0.0,
    )


def build_shelf_goal_set(
    frame: TaskFrame,
    spec: ShelfRelationSpec,
    p_ref_world: Vec3,
    *,
    projection_kind: ProjectionKind = "terminal",
) -> GoalSetResult:
    ref_center = _reference_center_task(frame, p_ref_world)
    ref_xy = _object_horizontal_footprint_at(ref_center, spec.reference_footprint)
    shelf = spec.top_shelf if spec.relation == "above" else spec.bottom_shelf
    overlap_region = shelf.intersect(
        AxisAlignedBox(
            ref_xy.x_min,
            ref_xy.x_max,
            ref_xy.y_min,
            ref_xy.y_max,
            shelf.z_min,
            shelf.z_max,
        )
    )
    if overlap_region.is_empty():
        return GoalSetResult(
            region=None,
            empty=True,
            empty_cause=f"shelf_{spec.relation}_horizontal_overlap_empty",
            projection_kind=projection_kind,
            component_id=f"shelf:{spec.relation}",
        )
    obj_fp = spec.object_footprint
    region = AxisAlignedBox(
        overlap_region.x_min + obj_fp.half_left,
        overlap_region.x_max - obj_fp.half_left,
        overlap_region.y_min + obj_fp.half_front,
        overlap_region.y_max - obj_fp.half_front,
        overlap_region.z_min + obj_fp.half_up,
        overlap_region.z_max - obj_fp.half_up,
    )
    if region.is_empty():
        return GoalSetResult(
            region=None,
            empty=True,
            empty_cause=f"shelf_{spec.relation}_eroded_region_empty",
            projection_kind=projection_kind,
            component_id=f"shelf:{spec.relation}",
        )
    return GoalSetResult(
        region=region,
        empty=False,
        empty_cause=None,
        projection_kind=projection_kind,
        component_id=f"shelf:{spec.relation}",
    )


def _eroded_reference_local_interior(spec: ContainmentSpec) -> AxisAlignedBox:
    obj = spec.object_footprint
    margin = spec.wall_clearance_m
    interior = spec.interior_reference_local
    return AxisAlignedBox(
        interior.x_min + obj.half_left + margin,
        interior.x_max - obj.half_left - margin,
        interior.y_min + obj.half_front + margin,
        interior.y_max - obj.half_front - margin,
        interior.z_min + obj.half_up + margin,
        interior.z_max - obj.half_up - margin,
    )


def _reference_local_box_to_task(
    frame: TaskFrame,
    spec: ContainmentSpec,
    p_ref_world: Vec3,
    box_local: AxisAlignedBox,
) -> AxisAlignedBox:
    if not spec.orientation.is_identity():
        raise UnsupportedReferenceOrientationError(
            "non-identity reference orientation is not supported for containment scoring"
        )
    ref_center_task = _reference_center_task(frame, p_ref_world)
    return AxisAlignedBox(
        ref_center_task[0] + box_local.x_min,
        ref_center_task[0] + box_local.x_max,
        ref_center_task[1] + box_local.y_min,
        ref_center_task[1] + box_local.y_max,
        ref_center_task[2] + box_local.z_min,
        ref_center_task[2] + box_local.z_max,
    )


def build_containment_goal_set(
    frame: TaskFrame,
    spec: ContainmentSpec,
    p_ref_world: Vec3,
    *,
    projection_kind: ProjectionKind = "terminal",
) -> GoalSetResult:
    region = _reference_local_box_to_task(frame, spec, p_ref_world, _eroded_reference_local_interior(spec))
    if region.is_empty():
        return GoalSetResult(
            region=None,
            empty=True,
            empty_cause="containment_eroded_interior_empty",
            projection_kind=projection_kind,
            component_id="containment:inside",
        )
    return GoalSetResult(
        region=region,
        empty=False,
        empty_cause=None,
        projection_kind=projection_kind,
        component_id="containment:inside",
    )


def _projection_region(
    region: AxisAlignedBox | ConvexPolygonPrism,
    projection_kind: ProjectionKind,
) -> AxisAlignedBox | ConvexPolygonPrism:
    if isinstance(region, ConvexPolygonPrism):
        if projection_kind == "response_planar":
            return ConvexPolygonPrism(
                region.vertices_xy,
                -math.inf,
                math.inf,
            )
        if projection_kind == "response_opening":
            return ConvexPolygonPrism(region.vertices_xy, 0.0, 0.0)
        return region
    if projection_kind == "response_planar":
        return AxisAlignedBox(region.x_min, region.x_max, region.y_min, region.y_max, -math.inf, math.inf)
    if projection_kind == "response_opening":
        return region.planar_xy_footprint()
    return region


def goal_distance(
    frame: TaskFrame,
    p_obj_world: Vec3,
    goal: GoalSetResult,
    *,
    d_cap_m: float,
) -> DistanceResult:
    if d_cap_m <= 0.0 or not math.isfinite(d_cap_m):
        raise ValueError("d_cap_m must be finite and positive")
    if goal.empty or goal.region is None:
        return DistanceResult(
            distance_m=math.inf,
            capped_distance_m=d_cap_m,
            d_cap_m=d_cap_m,
            goal_set_empty=True,
            cap_applied=True,
            nearest_point_task=None,
            component_id=goal.component_id,
            projection_kind=goal.projection_kind,
        )
    center = _object_center_task(frame, p_obj_world)
    region = _projection_region(goal.region, goal.projection_kind)
    distance = region.distance_to_point(center)
    cap_applied = distance > d_cap_m or not math.isfinite(distance)
    capped = min(distance, d_cap_m) if math.isfinite(distance) else d_cap_m
    nearest = (
        region.nearest_point(center)
        if isinstance(region, ConvexPolygonPrism)
        else _nearest_point_on_box(center, region)
    )
    return DistanceResult(
        distance_m=distance,
        capped_distance_m=capped,
        d_cap_m=d_cap_m,
        goal_set_empty=False,
        cap_applied=cap_applied,
        nearest_point_task=nearest,
        component_id=goal.component_id,
        projection_kind=goal.projection_kind,
    )


def _nearest_point_on_box(p: Vec3, box: AxisAlignedBox) -> Vec3:
    if box.is_empty():
        return p
    return (
        min(max(p[0], box.x_min), box.x_max),
        min(max(p[1], box.y_min), box.y_max),
        min(max(p[2], box.z_min), box.z_max),
    )


def response_projection_distance(
    frame: TaskFrame,
    p_obj_world: Vec3,
    goal: GoalSetResult,
    *,
    d_cap_m: float,
) -> DistanceResult:
    if goal.projection_kind == "terminal":
        projected = GoalSetResult(
            region=goal.region,
            empty=goal.empty,
            empty_cause=goal.empty_cause,
            projection_kind="response_planar",
            component_id=goal.component_id,
        )
    else:
        projected = goal
    return goal_distance(frame, p_obj_world, projected, d_cap_m=d_cap_m)


def point_in_goal_set(frame: TaskFrame, p_obj_world: Vec3, goal: GoalSetResult, *, tol: float = 0.0) -> bool:
    if goal.empty or goal.region is None:
        return False
    center = _object_center_task(frame, p_obj_world)
    region = (
        goal.region
        if goal.projection_kind == "terminal"
        else _projection_region(goal.region, goal.projection_kind)
    )
    return region.point_inside(center, tol=tol)


def shelf_support_ok(
    frame: TaskFrame,
    spec: ShelfRelationSpec,
    p_obj_world: Vec3,
    *,
    tol: float | None = None,
) -> bool:
    tol = spec.support_surface_tol_m if tol is None else tol
    center = _object_center_task(frame, p_obj_world)
    shelf = spec.top_shelf if spec.relation == "above" else spec.bottom_shelf
    target_z = shelf.z_min if spec.relation == "above" else shelf.z_max
    supported = abs(center[2] - target_z) <= tol + spec.object_footprint.half_up
    if not supported:
        return False
    overlap = _horizontal_overlap_amount(
        _object_horizontal_footprint_at(center, spec.object_footprint),
        shelf.planar_xy_footprint(),
    )
    return overlap >= spec.horizontal_overlap_min_m


def containment_support_ok(
    spec: ContainmentSpec,
    frame: TaskFrame,
    p_obj_world: Vec3,
    p_ref_world: Vec3,
    *,
    tol: float | None = None,
) -> bool:
    tol = spec.support_surface_tol_m if tol is None else tol
    center = _object_center_task(frame, p_obj_world)
    bottom_local = (
        spec.interior_reference_local.z_min
        + spec.object_footprint.half_up
        + spec.wall_clearance_m
    )
    ref_center_task = _reference_center_task(frame, p_ref_world)
    bottom_task = ref_center_task[2] + bottom_local
    return center[2] <= bottom_task + tol


def inside_containment(goal: GoalSetResult, frame: TaskFrame, p_obj_world: Vec3, *, partial_tol: float = 0.0) -> bool:
    if goal.empty or goal.region is None:
        return False
    center = _object_center_task(frame, p_obj_world)
    return goal.region.point_inside(center, tol=partial_tol)


def inverse_relation(relation: RelationKind) -> RelationKind:
    mapping: dict[RelationKind, RelationKind] = {
        "left": "right",
        "right": "left",
        "front": "behind",
        "behind": "front",
        "above": "below",
        "below": "above",
        "inside": "inside",
        "contains": "inside",
    }
    return mapping[canonical_relation(relation)]


_PROMPT_CARRIER = "Place the {object} so that {clause}."
_HORIZONTAL_SUFFIX = " Use the robot's fixed viewpoint for left, right, front, and behind."

_DIRECT_CLAUSES: dict[tuple[str, str], str] = {
    ("left", "direct"): "the {object} is left of the {reference}",
    ("right", "direct"): "the {object} is right of the {reference}",
    ("front", "direct"): "the {object} is in front of the {reference}",
    ("behind", "direct"): "the {object} is behind the {reference}",
    ("above", "direct"): "the {object} is above the {reference}",
    ("below", "direct"): "the {object} is below the {reference}",
    ("inside", "direct"): "the {object} is inside the {reference}",
    ("inside", "inverse"): "the {reference} contains the {object}",
}

for rel in ("left", "right", "front", "behind", "above", "below"):
    if rel == "left":
        _DIRECT_CLAUSES[(rel, "inverse")] = "the {reference} is right of the {object}"
    elif rel == "right":
        _DIRECT_CLAUSES[(rel, "inverse")] = "the {reference} is left of the {object}"
    elif rel == "front":
        _DIRECT_CLAUSES[(rel, "inverse")] = "the {reference} is behind the {object}"
    elif rel == "behind":
        _DIRECT_CLAUSES[(rel, "inverse")] = "the {reference} is in front of the {object}"
    elif rel == "above":
        _DIRECT_CLAUSES[(rel, "inverse")] = "the {reference} is below the {object}"
    elif rel == "below":
        _DIRECT_CLAUSES[(rel, "inverse")] = "the {reference} is above the {object}"


def build_prompt(
    manipulated_object: str,
    reference_object: str,
    relation: RelationKind,
    wording: Literal["direct", "inverse"],
    *,
    horizontal: bool = True,
) -> str:
    rel = canonical_relation(relation)
    clause_template = _DIRECT_CLAUSES[(rel, wording)]
    clause = clause_template.format(object=manipulated_object, reference=reference_object)
    prompt = _PROMPT_CARRIER.format(object=manipulated_object, clause=clause)
    if horizontal and rel in HORIZONTAL_RELATIONS:
        prompt += _HORIZONTAL_SUFFIX
    return prompt


def parse_prompt(prompt: str) -> dict[str, str]:
    text = prompt.strip()
    carrier_match = re.match(
        r"Place the (?P<object>.+?) so that (?P<clause>.+?)\.(?: Use the robot's fixed viewpoint for left, right, front, and behind\.)?$",
        text,
    )
    if not carrier_match:
        raise ValueError("prompt does not match the frozen carrier")
    obj = carrier_match.group("object")
    clause = carrier_match.group("clause")
    patterns = [
        ("direct", rel, rf"the {re.escape(obj)} is {token} the (?P<reference>.+)")
        for rel, token in (
            ("left", "left of"),
            ("right", "right of"),
            ("front", "in front of"),
            ("behind", "behind"),
            ("above", "above"),
            ("below", "below"),
            ("inside", "inside"),
        )
    ] + [
        ("inverse", rel, rf"the (?P<reference>.+) is {token} the {re.escape(obj)}")
        for rel, token in (
            ("left", "right of"),
            ("right", "left of"),
            ("front", "behind"),
            ("behind", "in front of"),
            ("above", "below"),
            ("below", "above"),
        )
    ] + [("inverse", "inside", rf"the (?P<reference>.+) contains the {re.escape(obj)}")]
    for wording, relation, pattern in patterns:
        match = re.fullmatch(pattern, clause)
        if match:
            return {
                "manipulated_object": obj,
                "reference_object": match.group("reference"),
                "relation": relation,
                "wording": wording,
            }
    raise ValueError("clause does not match a registered relation form")


def semantic_goal_key(parsed: Mapping[str, str]) -> tuple[str, str, str]:
    relation = canonical_relation(parsed["relation"])
    return (parsed["manipulated_object"], parsed["reference_object"], relation)


def prompts_semantically_equivalent(a: str, b: str) -> bool:
    return semantic_goal_key(parse_prompt(a)) == semantic_goal_key(parse_prompt(b))


def direct_inverse_pair_equivalent(
    manipulated_object: str,
    reference_object: str,
    relation: RelationKind,
    *,
    horizontal: bool = True,
) -> bool:
    direct = build_prompt(manipulated_object, reference_object, relation, "direct", horizontal=horizontal)
    inverse = build_prompt(manipulated_object, reference_object, relation, "inverse", horizontal=horizontal)
    return prompts_semantically_equivalent(direct, inverse)


def reference_membership(
    *,
    named_goal: GoalSetResult,
    other_goal: GoalSetResult,
    frame: TaskFrame,
    p_obj_world: Vec3,
    tol: float = 0.0,
) -> Literal["named", "other", "both", "neither"]:
    named_ok = point_in_goal_set(frame, p_obj_world, named_goal, tol=tol)
    other_ok = point_in_goal_set(frame, p_obj_world, other_goal, tol=tol)
    if named_ok and other_ok:
        return "both"
    if named_ok:
        return "named"
    if other_ok:
        return "other"
    return "neither"


def planar_goal_area(box: AxisAlignedBox) -> float:
    if box.is_empty():
        return 0.0
    return max(0.0, box.x_max - box.x_min) * max(0.0, box.y_max - box.y_min)


def shrinking_area_fraction(before: AxisAlignedBox, after: AxisAlignedBox) -> float:
    before_area = planar_goal_area(before)
    if before_area <= 0.0:
        return math.nan
    if after.is_empty():
        return 1.0
    overlap = before.intersect(after)
    removed = before_area - planar_goal_area(overlap)
    return max(0.0, removed / before_area)


def resolve_trajectory_sample(
    trajectory: TrajectorySeries,
    query_time_s: float,
    *,
    terminal_extension: TrajectorySample | None = None,
) -> ResolvedTrajectorySample:
    """Return the latest sample at or before ``query_time_s``.

    When the query falls after the last logged sample and ``terminal_extension`` is
    supplied, hold the settled terminal pose instead of extrapolating motion.
    """
    if not math.isfinite(query_time_s):
        raise ValueError("query_time_s must be finite")
    chosen = trajectory.samples[0]
    for sample in trajectory.samples:
        if sample.simulation_time_s <= query_time_s + 1e-12:
            chosen = sample
        else:
            break
    if query_time_s > trajectory.samples[-1].simulation_time_s + 1e-12:
        if terminal_extension is None:
            return ResolvedTrajectorySample(sample=chosen, terminal_extension_applied=False)
        return ResolvedTrajectorySample(sample=terminal_extension, terminal_extension_applied=True)
    return ResolvedTrajectorySample(sample=chosen, terminal_extension_applied=False)


def build_response_planar_goal_set(
    frame: TaskFrame,
    spec: PlanarRelationSpec,
    p_ref_world: Vec3,
) -> GoalSetResult:
    return build_planar_goal_set(frame, spec, p_ref_world, projection_kind="response_planar")


def horizon_planar_capped_distance(
    frame: TaskFrame,
    spec: PlanarRelationSpec,
    *,
    p_obj_world: Vec3,
    p_ref_world: Vec3,
    d_cap_m: float,
) -> HorizonDistanceSample:
    goal = build_response_planar_goal_set(frame, spec, p_ref_world)
    distance = response_projection_distance(frame, p_obj_world, goal, d_cap_m=d_cap_m)
    return HorizonDistanceSample(
        simulation_time_s=math.nan,
        p_obj_world=p_obj_world,
        p_ref_world=p_ref_world,
        distance=distance,
        terminal_extension_applied=False,
    )


def horizon_planar_capped_distance_at_time(
    frame: TaskFrame,
    spec: PlanarRelationSpec,
    *,
    query_time_s: float,
    p_obj_world: Vec3,
    p_ref_world: Vec3,
    d_cap_m: float,
) -> HorizonDistanceSample:
    result = horizon_planar_capped_distance(
        frame,
        spec,
        p_obj_world=p_obj_world,
        p_ref_world=p_ref_world,
        d_cap_m=d_cap_m,
    )
    return HorizonDistanceSample(
        simulation_time_s=query_time_s,
        p_obj_world=result.p_obj_world,
        p_ref_world=result.p_ref_world,
        distance=result.distance,
        terminal_extension_applied=result.terminal_extension_applied,
    )
