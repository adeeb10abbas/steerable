"""Privileged zero-policy horizontal grasp/transport/place controller for G3."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Callable, Mapping, Sequence

from experiments.online_correction_v4.adapters import TerminalPhysicalPredicates
from experiments.online_correction_v4.droid_task_files.constants import (
    REFERENCE_OBJECT,
    TARGET_OBJECT,
)
from experiments.online_correction_v4.geometry import GoalSetResult, TaskFrame, Vec3, point_in_goal_set
from experiments.online_correction_v4.motion import minimum_jerk_scalar

def trajectory_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-scripted-trajectory-v1"


TRAJECTORY_SCHEMA = trajectory_schema("horizontal")
PHASES = (
    "approach",
    "descend",
    "close_dwell",
    "lift",
    "transport",
    "place_descend",
    "open_dwell",
    "retreat",
    "settle",
)
HORIZONTAL_RELATIONS = frozenset({"left", "right", "front", "behind"})
ReferenceMotionCallback = Callable[[int, float], None]


class DroidG3ScriptedError(ValueError):
    """Raised when scripted-controller configuration or geometry is invalid."""


@dataclass(frozen=True)
class ScriptedControllerConfig:
    """Frozen phase counts and geometry offsets for the horizontal scripted controller."""

    approach_ticks: int
    descend_ticks: int
    close_dwell_ticks: int
    lift_ticks: int
    transport_ticks: int
    place_descend_ticks: int
    open_dwell_ticks: int
    retreat_ticks: int
    settle_ticks: int
    approach_height_m: float
    descend_offset_m: float
    lift_height_m: float
    transport_height_m: float
    place_descend_offset_m: float
    retreat_height_m: float
    target_inset_m: float
    min_grasp_lift_m: float = 0.04
    gripper_open: float = 0.0
    gripper_close: float = 0.785398
    eef_tool_length_m: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ScriptedControllerConfig:
        ticks = raw.get("phase_ticks")
        if not isinstance(ticks, Mapping):
            raise DroidG3ScriptedError("config.phase_ticks must be a mapping")
        offsets = raw.get("geometry_offsets")
        if not isinstance(offsets, Mapping):
            raise DroidG3ScriptedError("config.geometry_offsets must be a mapping")
        kwargs: dict[str, Any] = {}
        for phase in PHASES:
            key = f"{phase}_ticks"
            kwargs[key] = _require_positive_int(ticks.get(phase), f"phase_ticks.{phase}")
        for name in (
            "approach_height_m",
            "descend_offset_m",
            "lift_height_m",
            "transport_height_m",
            "place_descend_offset_m",
            "retreat_height_m",
            "target_inset_m",
        ):
            kwargs[name] = _require_positive_finite(offsets.get(name), f"geometry_offsets.{name}")
        if "min_grasp_lift_m" in raw:
            kwargs["min_grasp_lift_m"] = _require_positive_finite(
                raw.get("min_grasp_lift_m"), "min_grasp_lift_m"
            )
        if "gripper_open" in raw:
            kwargs["gripper_open"] = _require_finite_number(raw.get("gripper_open"), "gripper_open")
        if "gripper_close" in raw:
            kwargs["gripper_close"] = _require_finite_number(raw.get("gripper_close"), "gripper_close")
        if "eef_tool_length_m" in raw:
            kwargs["eef_tool_length_m"] = _require_positive_finite(
                raw.get("eef_tool_length_m"),
                "eef_tool_length_m",
            )
        return cls(**kwargs)

    def phase_tick(self, phase: str) -> int:
        if phase not in PHASES:
            raise DroidG3ScriptedError(f"unsupported phase {phase!r}")
        return int(getattr(self, f"{phase}_ticks"))


def default_scripted_controller_config() -> ScriptedControllerConfig:
    return ScriptedControllerConfig.from_mapping(
        {
            "phase_ticks": {phase: 12 for phase in PHASES},
            "geometry_offsets": {
                "approach_height_m": 0.12,
                "descend_offset_m": 0.025,
                "lift_height_m": 0.12,
                "transport_height_m": 0.12,
                "place_descend_offset_m": 0.04,
                "retreat_height_m": 0.10,
                "target_inset_m": 0.015,
            },
            "min_grasp_lift_m": 0.04,
        }
    )


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DroidG3ScriptedError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise DroidG3ScriptedError(f"{label} must be finite")
    return number


def _require_positive_finite(value: Any, label: str) -> float:
    number = _require_finite_number(value, label)
    if number <= 0.0:
        raise DroidG3ScriptedError(f"{label} must be positive")
    return number


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DroidG3ScriptedError(f"{label} must be a positive integer")
    if value <= 0:
        raise DroidG3ScriptedError(f"{label} must be positive")
    return value


def _as_vector3(value: Any) -> Vec3:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    elif hasattr(value, "detach"):
        raw = list(_host_floats(value))
    elif hasattr(value, "tolist"):
        raw = list(value.tolist())
    else:
        raise DroidG3ScriptedError("pose vector must be a 3-vector")
    if len(raw) != 3:
        raise DroidG3ScriptedError("pose vector must have length 3")
    xyz = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in xyz):
        raise DroidG3ScriptedError("pose vector must be finite")
    return xyz


def _as_quaternion_wxyz(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    elif hasattr(value, "detach"):
        raw = list(_host_floats(value))
    elif hasattr(value, "tolist"):
        raw = list(value.tolist())
    else:
        raise DroidG3ScriptedError("quaternion must be a 4-vector")
    if len(raw) != 4:
        raise DroidG3ScriptedError("quaternion must have length 4")
    quat = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in quat):
        raise DroidG3ScriptedError("quaternion must be finite")
    return quat


def _host_floats(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return [float(item) for item in value.tolist()]
    return [float(item) for item in value]


def _lerp(a: Vec3, b: Vec3, fraction: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * fraction,
        a[1] + (b[1] - a[1]) * fraction,
        a[2] + (b[2] - a[2]) * fraction,
    )


def minimum_jerk_waypoints(start: Vec3, end: Vec3, tick_count: int) -> list[Vec3]:
    """Return Cartesian waypoints sampled with scalar minimum-jerk interpolation."""
    count = _require_positive_int(tick_count, "tick_count")
    if count == 1:
        return [end]
    points: list[Vec3] = []
    for index in range(count):
        parameter = index / (count - 1)
        fraction = minimum_jerk_scalar(parameter)
        points.append(_lerp(start, end, fraction))
    return points


def select_robust_target_task(
    *,
    frame: TaskFrame,
    goal: GoalSetResult,
    cube_position_world: Vec3,
    relation: str,
    inset_m: float,
    table_top_z_task: float,
    object_half_up: float,
) -> Vec3:
    """Pick a robust task-frame placement point inside a nonempty planar goal set."""
    normalized = relation.strip().lower()
    if normalized not in HORIZONTAL_RELATIONS:
        raise DroidG3ScriptedError(f"unsupported relation {relation!r}")
    inset = _require_positive_finite(inset_m, "target_inset_m")
    half_up = _require_positive_finite(object_half_up, "object_half_up")
    table_top = _require_finite_number(table_top_z_task, "table_top_z_task")
    if goal.empty or goal.region is None:
        raise DroidG3ScriptedError("goal set is empty")
    region = goal.region
    cube_task = frame.world_to_task(cube_position_world)

    def inset_coordinate(low: float, high: float, *, from_low: bool) -> float:
        if high < low:
            raise DroidG3ScriptedError("goal region has inverted bounds")
        edge_guard = min(0.005, 0.5 * (high - low))
        if from_low:
            return min(low + inset, high - edge_guard)
        return max(high - inset, low + edge_guard)

    if normalized == "left":
        x = inset_coordinate(region.x_min, region.x_max, from_low=True)
        y = min(max(cube_task[1], region.y_min), region.y_max)
    elif normalized == "right":
        x = inset_coordinate(region.x_min, region.x_max, from_low=False)
        y = min(max(cube_task[1], region.y_min), region.y_max)
    elif normalized == "front":
        y = inset_coordinate(region.y_min, region.y_max, from_low=True)
        x = min(max(cube_task[0], region.x_min), region.x_max)
    else:
        y = inset_coordinate(region.y_min, region.y_max, from_low=False)
        x = min(max(cube_task[0], region.x_min), region.x_max)
    z = table_top + half_up
    z = min(max(z, region.z_min), region.z_max)
    target = (x, y, z)
    if not region.point_inside(target):
        raise DroidG3ScriptedError("selected target lies outside goal region")
    return target


def select_robust_target_world(
    *,
    frame: TaskFrame,
    goal: GoalSetResult,
    cube_position_world: Vec3,
    relation: str,
    inset_m: float,
    table_top_z_task: float,
    object_half_up: float,
) -> Vec3:
    target_task = select_robust_target_task(
        frame=frame,
        goal=goal,
        cube_position_world=cube_position_world,
        relation=relation,
        inset_m=inset_m,
        table_top_z_task=table_top_z_task,
        object_half_up=object_half_up,
    )
    return frame.task_to_world(target_task)


def _read_body_row(table: Any, index: int) -> Any:
    try:
        return table[0, index]
    except TypeError:
        return table[0][index]


def _read_eef_pose(env: Any) -> tuple[Vec3, tuple[float, float, float, float]]:
    backend = env.backend
    frames = backend.env.scene["frames"]
    frame_names = list(frames.data.target_frame_names)
    if "eef_frame" not in frame_names:
        raise DroidG3ScriptedError("robot scene data lacks eef_frame")
    index = frame_names.index("eef_frame")
    position = _as_vector3(_read_body_row(frames.data.target_pos_w, index))
    quaternion = _as_quaternion_wxyz(_read_body_row(frames.data.target_quat_w, index))
    return position, quaternion


def _read_object_pose(env: Any, name: str) -> Vec3:
    get_world = env.backend.modules["get_world"]
    world = get_world(env.backend.env)
    position, _quat = world.get_pose(name, env_id=0)
    return _as_vector3(position)


def _object_grabbed(env: Any, target_object: str) -> bool:
    probe = env.backend.modules["object_grabbed"]
    return bool(probe(env.backend.env, object=target_object, env_id=0))


def _object_dropped(env: Any, target_object: str) -> bool:
    probe = env.backend.modules["object_dropped"]
    return bool(probe(env.backend.env, object=target_object, env_id=0))


def _quaternion_multiply(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quaternion_inverse(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm_squared = sum(value * value for value in quaternion)
    if norm_squared <= 0.0 or not math.isfinite(norm_squared):
        raise DroidG3ScriptedError("EEF offset quaternion must have finite nonzero norm")
    w, x, y, z = quaternion
    return (
        w / norm_squared,
        -x / norm_squared,
        -y / norm_squared,
        -z / norm_squared,
    )


def _compose_action(
    position: Vec3,
    measured_eef_quaternion: tuple[float, float, float, float],
    eef_offset_rotation: tuple[float, float, float, float],
    gripper: float,
) -> tuple[float, ...]:
    # RoboLab's absolute-IK command is the desired EEF quaternion with the
    # fixed tool-frame offset removed, matching the established V3 controller.
    quaternion = _quaternion_multiply(
        measured_eef_quaternion,
        _quaternion_inverse(eef_offset_rotation),
    )
    return (
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
        float(gripper),
    )


@dataclass(frozen=True)
class _PhaseSegment:
    phase: str
    waypoints: tuple[Vec3, ...]
    gripper: float


def _build_phase_segments(
    *,
    config: ScriptedControllerConfig,
    start_position: Vec3,
    cube_position: Vec3,
    target_position: Vec3,
) -> list[_PhaseSegment]:
    cube_approach = (
        cube_position[0],
        cube_position[1],
        cube_position[2] + config.approach_height_m + config.eef_tool_length_m,
    )
    cube_grasp = (
        cube_position[0],
        cube_position[1],
        cube_position[2] + config.descend_offset_m + config.eef_tool_length_m,
    )
    cube_lift = (
        cube_position[0],
        cube_position[1],
        cube_position[2] + config.lift_height_m + config.eef_tool_length_m,
    )
    target_transport = (
        target_position[0],
        target_position[1],
        target_position[2] + config.transport_height_m + config.eef_tool_length_m,
    )
    target_place = (
        target_position[0],
        target_position[1],
        target_position[2] + config.place_descend_offset_m + config.eef_tool_length_m,
    )
    target_retreat = (
        target_position[0],
        target_position[1],
        target_position[2] + config.retreat_height_m + config.eef_tool_length_m,
    )
    cursor = start_position
    segments: list[_PhaseSegment] = []
    plan: list[tuple[str, Vec3, Vec3, float]] = [
        ("approach", cursor, cube_approach, config.gripper_open),
        ("descend", cube_approach, cube_grasp, config.gripper_open),
        ("close_dwell", cube_grasp, cube_grasp, config.gripper_close),
        ("lift", cube_grasp, cube_lift, config.gripper_close),
        ("transport", cube_lift, target_transport, config.gripper_close),
        ("place_descend", target_transport, target_place, config.gripper_close),
        ("open_dwell", target_place, target_place, config.gripper_open),
        ("retreat", target_place, target_retreat, config.gripper_open),
        ("settle", target_retreat, target_retreat, config.gripper_open),
    ]
    for phase, start, end, gripper in plan:
        tick_count = config.phase_tick(phase)
        if phase.endswith("_dwell") or phase == "settle":
            waypoints = tuple(end for _ in range(tick_count))
        else:
            waypoints = tuple(minimum_jerk_waypoints(start, end, tick_count))
        segments.append(_PhaseSegment(phase=phase, waypoints=waypoints, gripper=gripper))
        cursor = end
    return segments


def _evaluate_stages(
    *,
    frame: TaskFrame,
    goal: GoalSetResult,
    config: ScriptedControllerConfig,
    grasp_contact_after_close: bool,
    object_z_after_lift: float | None,
    object_z_at_grasp: float | None,
    grasp_retained_through_transport: bool,
    released_after_open: bool,
    terminal_predicates: TerminalPhysicalPredicates,
    final_object_position: Vec3,
    geometric_tol_m: float = 0.0,
) -> tuple[dict[str, bool], list[str]]:
    reasons: list[str] = []
    lift_ok = (
        object_z_after_lift is not None
        and object_z_at_grasp is not None
        and object_z_after_lift - object_z_at_grasp + 1e-12 >= config.min_grasp_lift_m
    )
    grasped = grasp_contact_after_close and lift_ok
    if not grasp_contact_after_close:
        reasons.append("no_contact_after_close")
    if not lift_ok:
        reasons.append("insufficient_lift_after_close")
    transported = grasped and grasp_retained_through_transport
    if grasped and not grasp_retained_through_transport:
        reasons.append("grasp_lost_before_target_approach")
    released = released_after_open
    if not released_after_open:
        reasons.append("object_not_dropped_after_open")
    stably_placed = (
        terminal_predicates.available
        and terminal_predicates.allowed_support
        and terminal_predicates.stable_for_dwell
        and not terminal_predicates.boundary_violation
        and not terminal_predicates.collision_terminal_failure
    )
    if not terminal_predicates.available:
        reasons.append("terminal_predicates_unavailable")
    elif not terminal_predicates.allowed_support:
        reasons.append("support_not_allowed")
    elif not terminal_predicates.stable_for_dwell:
        reasons.append("object_not_stable_for_dwell")
    if terminal_predicates.boundary_violation:
        reasons.append("boundary_violation")
    if terminal_predicates.collision_terminal_failure:
        reasons.append("collision_terminal_failure")
    goal_satisfied = point_in_goal_set(
        frame,
        final_object_position,
        goal,
        tol=geometric_tol_m,
    )
    if not goal_satisfied:
        reasons.append("object_center_outside_goal_set")
    stages = {
        "grasped": grasped,
        "transported": transported,
        "released": released,
        "stably_placed": stably_placed,
        "goal_satisfied": goal_satisfied,
    }
    return stages, reasons


def run_scripted_check(
    env: Any,
    *,
    target_object: str,
    reference_object: str,
    relation: str,
    goal: GoalSetResult,
    frame: TaskFrame,
    config: Mapping[str, Any] | ScriptedControllerConfig,
    table_top_z_task: float,
    object_half_up: float,
    geometric_tol_m: float = 0.0,
    reference_motion_callback: ReferenceMotionCallback | None = None,
    fixture_id: str = "horizontal",
) -> dict[str, Any]:
    """Execute one privileged scripted check with no model/policy calls."""
    controller_config = (
        config
        if isinstance(config, ScriptedControllerConfig)
        else ScriptedControllerConfig.from_mapping(config)
    )
    start_position, measured_eef_quaternion = _read_eef_pose(env)
    eef_offset_rotation = _as_quaternion_wxyz(
        env.backend.modules["eef_offset_rotation"]
    )
    cube_position = _read_object_pose(env, target_object)
    placement_target = select_robust_target_world(
        frame=frame,
        goal=goal,
        cube_position_world=cube_position,
        relation=relation,
        inset_m=controller_config.target_inset_m,
        table_top_z_task=table_top_z_task,
        object_half_up=object_half_up,
    )
    segments = _build_phase_segments(
        config=controller_config,
        start_position=start_position,
        cube_position=cube_position,
        target_position=placement_target,
    )
    control_dt_s = float(getattr(env, "control_dt_s", 0.0) or 0.0)
    trajectory: list[dict[str, Any]] = []
    tick = 0
    terminated_early = False
    termination_reason: str | None = None
    grasp_contact_after_close = False
    object_z_at_grasp: float | None = None
    object_z_after_lift: float | None = None
    grasp_retained_through_transport = True
    released_after_open = False
    grasp_xy_correction: tuple[float, float] | None = None
    for segment in segments:
        for target_position_tick in segment.waypoints:
            next_tick = tick + 1
            next_sim_time_s = (
                next_tick * control_dt_s if control_dt_s > 0.0 else float(next_tick)
            )
            if reference_motion_callback is not None:
                reference_motion_callback(next_tick, next_sim_time_s)
            if segment.phase == "place_descend" and grasp_xy_correction is None:
                measured_position, _ = _read_eef_pose(env)
                measured_object = _read_object_pose(env, target_object)
                grasp_xy_correction = (
                    min(max(measured_position[0] - measured_object[0], -0.02), 0.02),
                    min(max(measured_position[1] - measured_object[1], -0.02), 0.02),
                )
            commanded_position = target_position_tick
            if (
                grasp_xy_correction is not None
                and segment.phase
                in {"place_descend", "open_dwell", "retreat", "settle"}
            ):
                commanded_position = (
                    target_position_tick[0] + grasp_xy_correction[0],
                    target_position_tick[1] + grasp_xy_correction[1],
                    target_position_tick[2],
                )
            action = _compose_action(
                commanded_position,
                measured_eef_quaternion,
                eef_offset_rotation,
                segment.gripper,
            )
            _obs, info = env.step(action)
            tick = next_tick
            sim_time_s = tick * control_dt_s if control_dt_s > 0.0 else float(tick)
            measured_position, _ = _read_eef_pose(env)
            reference_position = _read_object_pose(env, reference_object)
            object_position = _read_object_pose(env, target_object)
            grabbed = _object_grabbed(env, target_object)
            dropped = _object_dropped(env, target_object)
            trajectory.append(
                {
                    "tick": tick,
                    "sim_time_s": sim_time_s,
                    "phase": segment.phase,
                    "action": [float(value) for value in action],
                    "target_eef_xyz": [float(value) for value in commanded_position],
                    "measured_eef_xyz": [float(value) for value in measured_position],
                    "reference_xyz": [float(value) for value in reference_position],
                    "object_xyz": [float(value) for value in object_position],
                    "gripper_command": float(segment.gripper),
                    "object_grabbed": grabbed,
                    "object_dropped": dropped,
                }
            )
            if info.get("terminated") or info.get("truncated"):
                terminated_early = True
                termination_reason = "simulator_terminated" if info.get("terminated") else "simulator_truncated"
                break
            if segment.phase in {"transport", "place_descend"} and not grabbed:
                grasp_retained_through_transport = False
        if terminated_early:
            break
        if segment.phase == "close_dwell":
            grasp_contact_after_close = _object_grabbed(env, target_object)
            object_z_at_grasp = _read_object_pose(env, target_object)[2]
        elif segment.phase == "lift":
            object_z_after_lift = _read_object_pose(env, target_object)[2]
            if not _object_grabbed(env, target_object):
                grasp_retained_through_transport = False
        elif segment.phase == "open_dwell":
            released_after_open = _object_dropped(env, target_object)
    terminal_predicates = env.sample_terminal_predicates()
    final_object_position = _read_object_pose(env, target_object)
    stages, reasons = _evaluate_stages(
        frame=frame,
        goal=goal,
        config=controller_config,
        grasp_contact_after_close=grasp_contact_after_close,
        object_z_after_lift=object_z_after_lift,
        object_z_at_grasp=object_z_at_grasp,
        grasp_retained_through_transport=grasp_retained_through_transport,
        released_after_open=released_after_open,
        terminal_predicates=terminal_predicates,
        final_object_position=final_object_position,
        geometric_tol_m=geometric_tol_m,
    )
    if terminated_early and termination_reason is not None:
        reasons = [termination_reason, *reasons]
    passed = all(stages.values()) and not terminated_early
    return {
        "schema_version": trajectory_schema(fixture_id),
        "fixture_id": fixture_id,
        "model_request_count": 0,
        "relation": relation.strip().lower(),
        "target_world_xyz": [float(value) for value in placement_target],
        "grasp_xy_correction_m": (
            [float(value) for value in grasp_xy_correction]
            if grasp_xy_correction is not None
            else None
        ),
        "tick_count": tick,
        "terminated_early": terminated_early,
        "termination_reason": termination_reason,
        "trajectory": trajectory,
        "stages": stages,
        "reasons": reasons,
        "passed": passed,
    }


def run_scripted_horizontal_check(
    env: Any,
    *,
    relation: str,
    goal: GoalSetResult,
    frame: TaskFrame,
    config: Mapping[str, Any] | ScriptedControllerConfig,
    table_top_z_task: float,
    object_half_up: float,
    geometric_tol_m: float = 0.0,
    reference_motion_callback: ReferenceMotionCallback | None = None,
) -> dict[str, Any]:
    """Execute one privileged scripted horizontal check with no model/policy calls."""
    return run_scripted_check(
        env,
        target_object=TARGET_OBJECT,
        reference_object=REFERENCE_OBJECT,
        relation=relation,
        goal=goal,
        frame=frame,
        config=config,
        table_top_z_task=table_top_z_task,
        object_half_up=object_half_up,
        geometric_tol_m=geometric_tol_m,
        reference_motion_callback=reference_motion_callback,
        fixture_id="horizontal",
    )


def trajectory_json_compatible(result: Mapping[str, Any]) -> str:
    """Return a JSON string for a scripted-check result, validating serializability."""
    return json.dumps(dict(result), allow_nan=False, sort_keys=True)
