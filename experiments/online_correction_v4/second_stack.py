"""Pinned SimplerEnv/WidowX bindings for the model-blind C8 fixture."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


FIXTURE_ID = "second_stack"
ENV_NAME = "simpler_env_widowx/widowx_stack_cube"
SOURCE_OBJECT = "baked_green_cube_3cm"
REFERENCE_OBJECT = "baked_yellow_cube_3cm"
RELATIONS = ("left", "right", "front", "behind")
RELATION_AXES_SCENE_XY = {
    "left": (0.48393356800079346, -0.8751052618026733),
    "right": (-0.48393356800079346, 0.8751052618026733),
    "front": (0.8751052618026733, 0.48393356800079346),
    "behind": (-0.8751052618026733, -0.48393356800079346),
}


class SecondStackBindingError(ValueError):
    pass


def direct_prompt(relation: str) -> str:
    normalized = relation.strip().lower()
    if normalized not in RELATIONS:
        raise SecondStackBindingError(f"unsupported C8 relation: {relation!r}")
    return (
        "Place the green block so that it is "
        f"{normalized} of the yellow block. "
        "Use the robot's fixed viewpoint for left, right, front, and behind."
    )


def task_axes_from_camera_extrinsic(
    extrinsic: Sequence[Sequence[Any]],
) -> dict[str, tuple[float, float]]:
    if not isinstance(extrinsic, Sequence) or len(extrinsic) < 2:
        raise SecondStackBindingError("camera extrinsic must contain at least two rows")
    image_right = _finite_vector(
        extrinsic[0][:2],
        length=2,
        label="camera extrinsic image-right axis",
    )
    image_down = _finite_vector(
        extrinsic[1][:2],
        length=2,
        label="camera extrinsic image-down axis",
    )

    def normalized(vector: tuple[float, float], label: str) -> tuple[float, float]:
        norm = math.hypot(*vector)
        if norm <= 0.0:
            raise SecondStackBindingError(f"{label} has zero planar norm")
        return (vector[0] / norm, vector[1] / norm)

    right = normalized(image_right, "camera image-right axis")
    front = normalized(image_down, "camera image-down axis")
    if abs(right[0] * front[0] + right[1] * front[1]) > 1e-5:
        raise SecondStackBindingError("camera task-frame axes are not orthogonal")
    left = (-right[0], -right[1])
    return {
        "left": left,
        "right": right,
        "front": front,
        "behind": (-front[0], -front[1]),
    }


def unwrap_simpler_env(env: Any) -> Any:
    outer = getattr(env, "unwrapped", env)
    inner = getattr(outer, "env", None)
    if inner is None:
        raise SecondStackBindingError("C8 environment lacks the SimplerEnv wrapper")
    raw = getattr(inner, "unwrapped", inner)
    required = ("episode_source_obj", "episode_target_obj", "_scene")
    missing = [name for name in required if not hasattr(raw, name)]
    if missing:
        raise SecondStackBindingError(
            f"C8 raw environment lacks required bindings: {missing}"
        )
    return raw


def _finite_vector(
    value: Sequence[Any],
    *,
    length: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SecondStackBindingError(f"{label} must be a finite vector")
    result = tuple(float(item) for item in value)
    if len(result) != length or any(not math.isfinite(item) for item in result):
        raise SecondStackBindingError(f"{label} must be a finite {length}-vector")
    return result


def actor_pose(actor: Any) -> tuple[tuple[float, float, float], tuple[float, ...]]:
    pose = actor.pose
    position = _finite_vector(pose.p, length=3, label=f"{actor.name}.position")
    quaternion = _finite_vector(pose.q, length=4, label=f"{actor.name}.quaternion")
    return position, quaternion


def fixture_actors(env: Any) -> tuple[Any, Any]:
    raw = unwrap_simpler_env(env)
    source = raw.episode_source_obj
    reference = raw.episode_target_obj
    if source.name != SOURCE_OBJECT or reference.name != REFERENCE_OBJECT:
        raise SecondStackBindingError(
            "C8 live object names differ from the released fixture binding"
        )
    return source, reference


def apply_registered_reset(
    env: Any,
    reset_row: Mapping[str, Any],
    *,
    settle_steps: int = 250,
) -> dict[str, list[float]]:
    if isinstance(settle_steps, bool) or not isinstance(settle_steps, int):
        raise SecondStackBindingError("settle_steps must be an integer")
    if settle_steps <= 0:
        raise SecondStackBindingError("settle_steps must be positive")
    positions = reset_row.get("positions_scene_xy_m")
    if not isinstance(positions, Mapping):
        raise SecondStackBindingError("C8 reset row lacks positions_scene_xy_m")
    raw = unwrap_simpler_env(env)
    source, reference = fixture_actors(env)
    live: dict[str, list[float]] = {}
    for actor in (source, reference):
        xy = _finite_vector(
            positions.get(actor.name),
            length=2,
            label=f"positions_scene_xy_m.{actor.name}",
        )
        current_position, current_quaternion = actor_pose(actor)
        destination = [xy[0], xy[1], current_position[2]]
        pose_type = type(actor.pose)
        actor.set_pose(pose_type(p=destination, q=current_quaternion))
        if hasattr(actor, "set_velocity"):
            actor.set_velocity([0.0, 0.0, 0.0])
        if hasattr(actor, "set_angular_velocity"):
            actor.set_angular_velocity([0.0, 0.0, 0.0])
        live[actor.name] = destination
    for _ in range(settle_steps):
        raw._scene.step()
    if hasattr(raw._scene, "update_render"):
        raw._scene.update_render()
    return {
        actor.name: list(actor_pose(actor)[0])
        for actor in (source, reference)
    }


def reference_destination_xy(
    *,
    initial_xy: Sequence[Any],
    relation: str,
    displacement_m: float,
    physical_translation_sign: int,
) -> tuple[float, float]:
    origin = _finite_vector(initial_xy, length=2, label="initial_xy")
    normalized = relation.strip().lower()
    if normalized not in RELATION_AXES_SCENE_XY:
        raise SecondStackBindingError(f"unsupported C8 relation: {relation!r}")
    if physical_translation_sign not in {-1, 1}:
        raise SecondStackBindingError("physical_translation_sign must be -1 or 1")
    distance = float(displacement_m)
    if not math.isfinite(distance) or distance <= 0.0:
        raise SecondStackBindingError("displacement_m must be positive and finite")
    axis = RELATION_AXES_SCENE_XY[normalized]
    return (
        origin[0] + physical_translation_sign * distance * axis[0],
        origin[1] + physical_translation_sign * distance * axis[1],
    )


def set_reference_xy(env: Any, destination_xy: Sequence[Any]) -> list[float]:
    destination = _finite_vector(
        destination_xy,
        length=2,
        label="destination_xy",
    )
    _source, reference = fixture_actors(env)
    position, quaternion = actor_pose(reference)
    command = [destination[0], destination[1], position[2]]
    reference.set_pose(type(reference.pose)(p=command, q=quaternion))
    if hasattr(reference, "set_velocity"):
        reference.set_velocity([0.0, 0.0, 0.0])
    if hasattr(reference, "set_angular_velocity"):
        reference.set_angular_velocity([0.0, 0.0, 0.0])
    return command


def active_contact_pairs(env: Any, *, impulse_threshold: float = 1e-6) -> list[dict[str, Any]]:
    threshold = float(impulse_threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise SecondStackBindingError("impulse_threshold must be finite and nonnegative")
    raw = unwrap_simpler_env(env)
    rows: list[dict[str, Any]] = []
    for contact in raw._scene.get_contacts():
        squared_norm = 0.0
        for point in contact.points:
            impulse = _finite_vector(point.impulse, length=3, label="contact impulse")
            squared_norm += sum(value * value for value in impulse)
        norm = math.sqrt(squared_norm)
        if norm > threshold:
            rows.append(
                {
                    "actor0": str(contact.actor0.name),
                    "actor1": str(contact.actor1.name),
                    "impulse_norm": norm,
                }
            )
    return rows
