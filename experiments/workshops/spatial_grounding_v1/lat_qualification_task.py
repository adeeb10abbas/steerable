"""Candidate-bound, goal-independent LAT task overlay for RoboLab Abs-IK."""

from dataclasses import dataclass
import copy
import hashlib
import json
import os

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task


def _candidate() -> dict:
    raw = os.environ.get("SGW_LAT_CANDIDATE_JSON")
    expected = os.environ.get("SGW_LAT_CANDIDATE_SHA256")
    if not raw or not expected:
        raise RuntimeError("SGW LAT candidate JSON and SHA-256 are required")
    actual = hashlib.sha256(raw.encode()).hexdigest()
    if actual != expected:
        raise RuntimeError("SGW LAT candidate JSON digest mismatch")
    value = json.loads(raw)
    if value.get("family") != "LAT" or value.get("action_cap") != 450 or value.get("goal_termination"):
        raise RuntimeError("invalid goal-independent SGW LAT task payload")
    poses = value.get("object_poses")
    if set(poses or ()) != {"rubiks_cube", "bowl"}:
        raise RuntimeError("SGW LAT task requires only cube and bowl candidate poses")
    return value


_CANDIDATE = _candidate()


def _scene():
    scene = import_scene("rubiks_cube_banana_bowl.usda", ["rubiks_cube", "banana", "bowl", "table"])
    for name, payload in _CANDIDATE["object_poses"].items():
        position = payload["position_m"]
        quaternion = payload["quaternion_wxyz"]
        if len(position) != 3 or len(quaternion) != 4:
            raise RuntimeError(f"invalid {name} pose")
        asset = copy.deepcopy(getattr(scene, name))
        asset.init_state.pos = tuple(float(item) for item in position)
        asset.init_state.rot = tuple(float(item) for item in quaternion)
        setattr(scene, name, asset)
    return scene


@configclass
class _Termination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class SGWLatQualificationTask(Task):
    """A fixed 450-action task with no relation success termination."""

    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = _scene()
    terminations = _Termination
    instruction = {"default": "Put the Rubik's cube to the left of the bowl."}
    attributes = ["sgw_01", "lat", "model_blind_qualification"]
    episode_length_s = 450
    subtasks = []
