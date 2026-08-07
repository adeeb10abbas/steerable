"""E003 symmetric-object task scene and frozen direct-command predicates."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_dropped, object_grabbed, object_left_of, object_right_of
from robolab.core.task.subtask import Subtask

LEFT_PROMPT = "Put the Rubik's cube to the left of the bowl."
RIGHT_PROMPT = "Put the Rubik's cube to the right of the bowl."
OBJECTS = ("rubiks_cube", "bowl", "banana_left", "banana_right")
BASE_POSITIONS = {
    "bowl": (0.44258353114128113, 0.0, 0.07732785493135452),
    "rubiks_cube": (0.303364634513855, 0.0, 0.08113233000040054),
    "banana_left": (0.538878858089447, -0.22, 0.0684281587600708),
    "banana_right": (0.538878858089447, 0.22, 0.0684281587600708),
}


def _candidate() -> dict:
    raw = os.environ.get("VLA_WAM_V3E003_FIXTURE_CANDIDATE")
    expected = os.environ.get("VLA_WAM_V3E003_FIXTURE_SHA256")
    if not raw or not expected:
        raise RuntimeError("E003 fixture candidate path and SHA-256 are required")
    path = Path(raw).resolve()
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != expected:
        raise RuntimeError("E003 fixture candidate digest mismatch")
    value = json.loads(path.read_text())
    if value.get("schema_version") != "vla-wam-shared-v3e003-symmetric-object-layout-candidate-v1" or value.get("status") != "model_blind_candidate_not_released_for_inference":
        raise RuntimeError("E003 candidate is not the registered model-blind candidate")
    return value


_CANDIDATE = _candidate()


def _scene():
    # Keep the exact B001 scene asset and all non-movable geometry.  Add a
    # second YCB banana payload in the task config so the only extra movable
    # items are the registered mirrored clutter pair.
    scene_path = os.environ.get("VLA_WAM_V3E003_SCENE_ASSET")
    if not scene_path:
        raise RuntimeError("E003 custom USDA scene path is required")
    scene = import_scene(scene_path, ["rubiks_cube", "banana", "banana_right", "bowl", "table"])
    # The registered E003 USDA contains two distinct banana prims.  Keeping
    # both in the asset (rather than aliasing one config twice) is essential:
    # each mirrored clutter member must have its own physical prim.
    left = scene.banana
    right = scene.banana_right
    left.init_state.pos = tuple(BASE_POSITIONS["banana_left"])
    right.init_state.pos = tuple(BASE_POSITIONS["banana_right"])
    scene.banana_right.init_state.pos = tuple(BASE_POSITIONS["banana_right"])
    scene.bowl.init_state.pos = tuple(BASE_POSITIONS["bowl"])
    scene.rubiks_cube.init_state.pos = tuple(BASE_POSITIONS["rubiks_cube"])
    # The source USD contains a captured nonzero velocity state.  E003's
    # registered scene is a settled layout, so clear only the movable
    # objects' initial velocities; robot/camera/non-movable geometry is untouched.
    for asset in (scene.banana, scene.banana_right, scene.bowl, scene.rubiks_cube):
        if hasattr(asset.init_state, "lin_vel"):
            asset.init_state.lin_vel = (0.0, 0.0, 0.0)
        if hasattr(asset.init_state, "ang_vel"):
            asset.init_state.ang_vel = (0.0, 0.0, 0.0)
    return scene


def _subtask(relation: str) -> list[Subtask]:
    fn = object_left_of if relation == "left" else object_right_of
    return [Subtask(name=f"rubiks_cube_{relation}_of_bowl", conditions=[partial(object_grabbed, object="rubiks_cube"), partial(fn, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot", mirrored=False), partial(object_dropped, object="rubiks_cube")], logical="all", score=1.0)]


@configclass
class _LeftTermination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_left_of, params={"object":"rubiks_cube", "reference_object":"bowl", "frame_of_reference":"robot", "mirrored":False, "require_gripper_detached":True})


@configclass
class _RightTermination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_right_of, params={"object":"rubiks_cube", "reference_object":"bowl", "frame_of_reference":"robot", "mirrored":False, "require_gripper_detached":True})
