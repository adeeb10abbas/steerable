"""Hash-pinned, timeout-only RoboLab tasks for the forecast-layout study.

Two intentionally separate entry points exist:

* ``build_candidate_gate_task_class`` constructs an unreleased numeric
  candidate for model-blind calibration only.
* ``build_timeout_only_task_class`` accepts only a frozen manifest backed by an
  accepted live-gate record.

Both task types omit success from the termination configuration.  LEFT/RIGHT
success remains available through :func:`success_measurements` for the
recording adapter's measurement-only event stream.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from .fixture_layouts import (
        COMMANDS,
        LAYOUT_ARMS,
        MOVABLE_OBJECTS,
        candidate_index,
        load_frozen_pose_manifest,
        load_json_file,
        require,
        validate_candidate_pool,
    )
except ImportError:  # Direct task-file execution adds this directory to sys.path.
    from fixture_layouts import (  # type: ignore
        COMMANDS,
        LAYOUT_ARMS,
        MOVABLE_OBJECTS,
        candidate_index,
        load_frozen_pose_manifest,
        load_json_file,
        require,
        validate_candidate_pool,
    )


LEFT_PROMPT = "Put the Rubik's cube to the left of the bowl."
RIGHT_PROMPT = "Put the Rubik's cube to the right of the bowl."
PROMPTS = {"left": LEFT_PROMPT, "right": RIGHT_PROMPT}
SCENE_NAME = "rubiks_cube_banana_bowl.usda"
ACTION_CAP = 450
CONTROL_HZ = 15
EPISODE_LENGTH_SECONDS = ACTION_CAP / CONTROL_HZ


def _layout_from_manifest(
    manifest_path: Path,
    manifest_sha256: str,
    layout_pair_id: str,
    layout_arm: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(layout_arm in LAYOUT_ARMS, "layout arm must be original or reflected")
    manifest = load_frozen_pose_manifest(manifest_path, manifest_sha256)
    row = manifest["layout_pairs"].get(layout_pair_id)
    require(row is not None, f"layout is absent from frozen pose manifest: {layout_pair_id}")
    return row["layouts"][layout_arm], {
        "input_kind": "live_qualified_frozen_pose_manifest",
        "pose_manifest_path": str(Path(manifest_path).resolve()),
        "pose_manifest_sha256": manifest_sha256,
        "layout_pair_id": layout_pair_id,
        "candidate_id": row["candidate_id"],
        "candidate_payload_sha256": row["candidate_payload_sha256"],
        "accepted_gate_record_sha256": row["accepted_gate_record_sha256"],
        "released_for_model_inference": False,
    }


def _layout_from_unreleased_candidate(
    candidate_pool_path: Path,
    candidate_pool_sha256: str,
    candidate_id: str,
    layout_arm: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(layout_arm in LAYOUT_ARMS, "layout arm must be original or reflected")
    pool, _ = load_json_file(candidate_pool_path, candidate_pool_sha256)
    pool = validate_candidate_pool(pool)
    candidate = candidate_index(pool).get(candidate_id)
    require(candidate is not None, f"candidate is absent from pool: {candidate_id}")
    return candidate["layouts"][layout_arm], {
        "input_kind": "unreleased_candidate_for_model_blind_gate_only",
        "candidate_pool_path": str(Path(candidate_pool_path).resolve()),
        "candidate_pool_sha256": candidate_pool_sha256,
        "layout_pair_id": candidate["layout_pair_id"],
        "candidate_id": candidate_id,
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "released_for_model_inference": False,
    }


def _scene(layout: Mapping[str, Any]):
    """Construct the scene lazily so pure contract tests need no Isaac install."""

    from robolab.core.scenes.utils import import_scene

    positions = layout.get("positions_robot_base_m")
    quaternions = layout.get("quaternions_wxyz")
    require(isinstance(positions, dict) and set(positions) == set(MOVABLE_OBJECTS), "task position inventory changed")
    require(isinstance(quaternions, dict) and set(quaternions) == set(MOVABLE_OBJECTS), "task quaternion inventory changed")
    scene = import_scene(SCENE_NAME, [*MOVABLE_OBJECTS, "table"])
    for name in MOVABLE_OBJECTS:
        asset = copy.deepcopy(getattr(scene, name))
        asset.init_state.pos = tuple(float(item) for item in positions[name])
        asset.init_state.rot = tuple(float(item) for item in quaternions[name])
        if hasattr(asset.init_state, "lin_vel"):
            asset.init_state.lin_vel = (0.0, 0.0, 0.0)
        if hasattr(asset.init_state, "ang_vel"):
            asset.init_state.ang_vel = (0.0, 0.0, 0.0)
        setattr(scene, name, asset)
    return scene


def _timeout_only_termination_class():
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils import configclass

    @configclass
    class ForecastLayoutTimeoutOnlyTermination:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Fail loudly if a simulator decorator unexpectedly injected another term.
    annotations = getattr(ForecastLayoutTimeoutOnlyTermination, "__annotations__", {})
    require("success" not in annotations and not hasattr(ForecastLayoutTimeoutOnlyTermination, "success"), "success termination was injected")
    return ForecastLayoutTimeoutOnlyTermination


def success_measurements(env: Any) -> dict[str, bool]:
    """Measure both goal predicates and release without terminating the task."""

    from robolab.core.task.conditionals import object_left_of, object_right_of
    from robolab.core.world.world_state import get_world

    common = {
        "object": "rubiks_cube",
        "reference_object": "bowl",
        "frame_of_reference": "robot",
        "mirrored": False,
        "require_gripper_detached": True,
        "env_id": 0,
    }
    return {
        "left": bool(object_left_of(env, **common)),
        "right": bool(object_right_of(env, **common)),
        "released": not bool(get_world(env).in_contact("rubiks_cube", "gripper", env_id=0)),
    }


def _host_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    require(isinstance(value, (list, tuple)), "simulator value is not vector-like")
    output = [float(item) for item in value]
    require(all(math.isfinite(item) for item in output), "simulator vector is non-finite")
    return output


def _array_identity(value: Any) -> dict[str, Any]:
    """Return an exact dtype/shape/byte identity without retaining the array."""

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    require(not array.dtype.hasobject, "reset observation contains an object array")
    header = {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"}
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {**header, "value_sha256": digest.hexdigest()}


def observation_identities(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the exact post-settle image/proprio arrays, excluding prompt bytes."""

    output: dict[str, Any] = {}
    for group_name in ("image_obs", "proprio_obs"):
        group = observation.get(group_name)
        require(isinstance(group, Mapping) and group, f"post-settle observation lacks {group_name}")
        output[group_name] = {
            str(name): _array_identity(value) for name, value in sorted(group.items())
        }
    output["combined_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()
    return output


def _hold_action(observation: Mapping[str, Any], device: str):
    import torch

    proprio = observation.get("proprio_obs")
    require(isinstance(proprio, Mapping), "hold action needs proprio_obs")
    arm = proprio["arm_joint_pos"].detach().to(device)
    gripper = proprio["gripper_pos"].detach().to(device)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    action = torch.cat((arm, gripper), dim=1)
    require(tuple(action.shape) == (1, 8), f"hold action shape changed: {tuple(action.shape)}")
    return action


def settle_for_recording_reset(
    env: Any,
    observation: Mapping[str, Any],
    info: Any,
    *,
    pose_manifest_sha256: str,
    reset_identity: str,
    collision_sampler,
    visibility_sampler,
    settle_steps: int = 60,
    stability_window_steps: int = 15,
    linear_speed_tolerance_m_s: float = 0.02,
    angular_speed_tolerance_rad_s: float = 0.2,
) -> tuple[Any, Any, dict[str, Any]]:
    """Settle a new physical reset and return the exact observation to record.

    The helper executes only joint-position hold actions, samples no model, and
    resets RoboLab's per-environment timeout buffer after settling.  Collision
    and visibility samplers are mandatory simulator-specific callbacks and must
    return mappings with ``passed: true``; absent evidence fails closed.
    """

    require(isinstance(reset_identity, str) and reset_identity, "reset identity is required")
    require(
        isinstance(pose_manifest_sha256, str)
        and len(pose_manifest_sha256) == 64
        and all(character in "0123456789abcdef" for character in pose_manifest_sha256),
        "pose manifest SHA-256 is required",
    )
    require(settle_steps >= 1 and stability_window_steps >= 2, "settle window is invalid")
    require(callable(collision_sampler) and callable(visibility_sampler), "collision and visibility samplers are required")
    from robolab.core.world.world_state import get_world

    hold = _hold_action(observation, env.device)
    settled_observation = observation
    settled_info = info
    for _ in range(settle_steps):
        settled_observation, _, terminated, truncated, settled_info = env.step(hold)
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError("timeout-only task terminated during reset settling")
    maxima = {
        name: {"max_linear_speed_m_s": 0.0, "max_angular_speed_rad_s": 0.0}
        for name in MOVABLE_OBJECTS
    }
    for _ in range(stability_window_steps):
        settled_observation, _, terminated, truncated, settled_info = env.step(hold)
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError("timeout-only task terminated during reset stability window")
        world = get_world(env)
        for name in MOVABLE_OBJECTS:
            velocity = _host_list(world.get_velocity(name, env_id=0))
            require(len(velocity) == 6, f"{name} velocity shape changed")
            maxima[name]["max_linear_speed_m_s"] = max(
                maxima[name]["max_linear_speed_m_s"],
                math.sqrt(sum(item * item for item in velocity[:3])),
            )
            maxima[name]["max_angular_speed_rad_s"] = max(
                maxima[name]["max_angular_speed_rad_s"],
                math.sqrt(sum(item * item for item in velocity[3:])),
            )
    stability_passed = all(
        row["max_linear_speed_m_s"] <= linear_speed_tolerance_m_s
        and row["max_angular_speed_rad_s"] <= angular_speed_tolerance_rad_s
        for row in maxima.values()
    )
    require(stability_passed, "post-reset fixture did not satisfy frozen stability tolerances")
    collision_evidence = collision_sampler(env, settled_observation)
    visibility_evidence = visibility_sampler(env, settled_observation)
    require(isinstance(collision_evidence, Mapping) and collision_evidence.get("passed") is True, "collision evidence failed or is missing")
    require(isinstance(visibility_evidence, Mapping) and visibility_evidence.get("passed") is True, "visibility evidence failed or is missing")
    success = success_measurements(env)
    require(success["left"] is False and success["right"] is False, "settled reset begins in a success state")
    episode_length_buf = getattr(env, "episode_length_buf", None)
    require(episode_length_buf is not None, "RoboLab episode_length_buf is unavailable")
    episode_length_buf[:] = 0
    require(bool((episode_length_buf == 0).all()), "failed to zero behavioral timeout counter")
    identities = observation_identities(settled_observation)
    receipt = {
        "schema_version": "wmf-forecast-layout-settled-reset-receipt-v1",
        "passed": True,
        "settled": True,
        "left_success": False,
        "right_success": False,
        "released": success["released"],
        "reset_identity": reset_identity,
        "pose_manifest_sha256": pose_manifest_sha256,
        "initial_observation_hashes": identities,
        "settle_evidence": {
            "settle_steps": settle_steps,
            "stability_window_steps": stability_window_steps,
            "maxima_by_object": maxima,
            "linear_speed_tolerance_m_s": linear_speed_tolerance_m_s,
            "angular_speed_tolerance_rad_s": angular_speed_tolerance_rad_s,
        },
        "collision_evidence": dict(collision_evidence),
        "visibility_evidence": dict(visibility_evidence),
        "settled_observation_returned": True,
        "model_request_count_during_settle": 0,
        "episode_length_buf_reset_to_zero": True,
    }
    return settled_observation, settled_info, receipt


def _build_task_class(
    *,
    layout: Mapping[str, Any],
    binding: Mapping[str, Any],
    layout_arm: str,
    command: str,
    class_name: str,
):
    require(layout_arm in LAYOUT_ARMS, "unknown layout arm")
    require(command in COMMANDS, "unknown command")
    from robolab.core.task.task import Task

    termination_config = _timeout_only_termination_class()
    task_scene = _scene(layout)

    @dataclass
    class ForecastLayoutTask(Task):
        contact_object_list = [*MOVABLE_OBJECTS, "table"]
        scene = task_scene
        terminations = termination_config
        instruction = {"default": PROMPTS[command]}
        attributes = [
            "spatial",
            "wmf_ablation_001_20260912",
            "forecast_layout",
            layout_arm,
            "timeout_only_450_actions",
        ]
        episode_length_s: float = EPISODE_LENGTH_SECONDS
        subtasks: tuple = ()
        wmf_action_cap = ACTION_CAP
        wmf_stop_on_success = False
        wmf_success_is_measurement_only = True
        wmf_success_sampler = staticmethod(success_measurements)
        wmf_fixture_binding = dict(binding)
        wmf_layout_arm = layout_arm
        wmf_command = command

    ForecastLayoutTask.__name__ = class_name
    ForecastLayoutTask.__qualname__ = class_name
    return ForecastLayoutTask


def build_timeout_only_task_class(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    layout_pair_id: str,
    layout_arm: str,
    command: str,
    class_name: str = "WMFForecastLayoutTask",
):
    """Build a runtime task from one accepted, hash-pinned frozen pose row."""

    layout, binding = _layout_from_manifest(
        manifest_path, manifest_sha256, layout_pair_id, layout_arm
    )
    return _build_task_class(
        layout=layout,
        binding=binding,
        layout_arm=layout_arm,
        command=command,
        class_name=class_name,
    )


def build_candidate_gate_task_class(
    *,
    candidate_pool_path: Path,
    candidate_pool_sha256: str,
    candidate_id: str,
    layout_arm: str,
    command: str,
    class_name: str = "WMFModelBlindCandidateGateTask",
):
    """Build a calibration-only task; callers must never attach a policy client."""

    layout, binding = _layout_from_unreleased_candidate(
        candidate_pool_path, candidate_pool_sha256, candidate_id, layout_arm
    )
    return _build_task_class(
        layout=layout,
        binding=binding,
        layout_arm=layout_arm,
        command=command,
        class_name=class_name,
    )
