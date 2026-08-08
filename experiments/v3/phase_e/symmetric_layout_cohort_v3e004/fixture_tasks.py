"""Hash-bound RoboLab task primitives for the V3-E004 DROID layouts.

Importing this module requires the model-blind candidate path and SHA.  The
candidate is validated by the pure E004 contract before Isaac objects are
constructed.  A live reset/visibility gate is still mandatory before model
inference; this module alone never releases a cell.
"""

from __future__ import annotations

import copy
import math
import os
from functools import partial
from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_dropped,
    object_grabbed,
    object_left_of,
    object_right_of,
)
from robolab.core.task.subtask import Subtask

from .layout_contract import E004Candidate, load_candidate


LEFT_PROMPT = "Put the Rubik's cube to the left of the bowl."
RIGHT_PROMPT = "Put the Rubik's cube to the right of the bowl."


def _candidate() -> E004Candidate:
    raw = os.environ.get("VLA_WAM_V3E004_FIXTURE_CANDIDATE")
    expected = os.environ.get("VLA_WAM_V3E004_FIXTURE_SHA256")
    if not raw or not expected:
        raise RuntimeError("E004 fixture candidate path and SHA-256 are required")
    return load_candidate(Path(raw), expected)


_CANDIDATE = _candidate()


def _level_from_environment() -> float:
    raw = os.environ.get("VLA_WAM_V3E004_SYMMETRY_LEVEL_S")
    if raw is None:
        raise RuntimeError("E004 symmetry level is required")
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError("E004 symmetry level is not numeric") from exc


def _yaw_quaternion_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))


def _scene():
    level = _level_from_environment()
    control = math.isclose(level, 0.0, abs_tol=1e-12)
    scene_path = os.environ.get(
        "VLA_WAM_V3E004_CONTROL_SCENE_ASSET" if control else "VLA_WAM_V3E004_PAIRED_SCENE_ASSET"
    )
    if not scene_path:
        kind = "control" if control else "paired-clutter"
        raise RuntimeError(f"E004 {kind} USDA scene path is required")
    poses = _CANDIDATE.layout(level)
    # Candidate logical names map to the two physical banana prim names used
    # by the registered paired-payload scene.  The mapping itself is supplied
    # explicitly so a runner cannot silently alias one prim twice.
    mapping_raw = os.environ.get("VLA_WAM_V3E004_SCENE_OBJECT_MAPPING")
    if not mapping_raw:
        raise RuntimeError("E004 scene-object mapping JSON is required")
    import json

    mapping = json.loads(mapping_raw)
    if set(mapping) != set(_CANDIDATE.symmetric_poses) or len(set(mapping.values())) != len(mapping):
        raise RuntimeError("E004 scene-object mapping must bind the complete one-to-one s>0 inventory")
    scene_names = [str(mapping[name]) for name in poses] + ["table"]
    scene = import_scene(scene_path, scene_names)
    for logical_name, pose in poses.items():
        scene_name = str(mapping[logical_name])
        asset = copy.deepcopy(getattr(scene, scene_name))
        asset.init_state.pos = (pose.x_m, pose.y_m, pose.z_m)
        asset.init_state.rot = _yaw_quaternion_wxyz(pose.yaw_rad)
        if hasattr(asset.init_state, "lin_vel"):
            asset.init_state.lin_vel = (0.0, 0.0, 0.0)
        if hasattr(asset.init_state, "ang_vel"):
            asset.init_state.ang_vel = (0.0, 0.0, 0.0)
        setattr(scene, scene_name, asset)
    return scene


def _subtask(relation: str) -> list[Subtask]:
    fn = object_left_of if relation == "left" else object_right_of
    return [
        Subtask(
            name=f"rubiks_cube_{relation}_of_bowl",
            conditions=[
                partial(object_grabbed, object="rubiks_cube"),
                partial(
                    fn,
                    object="rubiks_cube",
                    reference_object="bowl",
                    frame_of_reference="robot",
                    mirrored=False,
                ),
                partial(object_dropped, object="rubiks_cube"),
            ],
            logical="all",
            score=1.0,
        )
    ]


@configclass
class _LeftTermination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_left_of,
        params={
            "object": "rubiks_cube",
            "reference_object": "bowl",
            "frame_of_reference": "robot",
            "mirrored": False,
            "require_gripper_detached": True,
        },
    )


@configclass
class _RightTermination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_right_of,
        params={
            "object": "rubiks_cube",
            "reference_object": "bowl",
            "frame_of_reference": "robot",
            "mirrored": False,
            "require_gripper_detached": True,
        },
    )
