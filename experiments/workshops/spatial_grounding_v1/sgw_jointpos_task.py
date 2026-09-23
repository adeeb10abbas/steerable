"""Candidate-bound joint-position SGW task overlay; imported only by RoboLab."""

import copy
from dataclasses import dataclass
import hashlib
import json
import os

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task


raw = os.environ.get("SGW01_JOINTPOS_CELL_JSON", "")
if not raw or hashlib.sha256(raw.encode()).hexdigest() != os.environ.get("SGW01_JOINTPOS_CELL_SHA256"):
    raise RuntimeError("hash-bound SGW joint-position cell binding is required")
CELL = json.loads(raw)
candidate = CELL["candidate"]
family = candidate.get("family")
required = {"rubiks_cube", "bowl"} | ({"plate"} if family == "DIST" else set())
if (family not in {"LAT", "HEIGHT", "DIST"} or set(candidate.get("object_poses", ())) != required
        or candidate.get("action_cap") != 450 or candidate.get("goal_termination") is not False):
    raise RuntimeError("joint-position task requires goal-independent candidate geometry")
scene_spec = candidate.get("native_scene", {
    "asset": "rubiks_cube_banana_bowl.usda",
    "object_names": ["rubiks_cube", "banana", "bowl", "table"],
})
if not required.issubset(scene_spec["object_names"]) or "table" not in scene_spec["object_names"]:
    raise RuntimeError("joint-position scene omits scored objects or table")


def _scene():
    scene = import_scene(scene_spec["asset"], scene_spec["object_names"])
    for name, pose in candidate["object_poses"].items():
        asset = copy.deepcopy(getattr(scene, name))
        asset.init_state.pos = tuple(float(value) for value in pose["position_m"])
        asset.init_state.rot = tuple(float(value) for value in pose["quaternion_wxyz"])
        setattr(scene, name, asset)
    return scene


@configclass
class _Termination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class SGWJointPositionTask(Task):
    contact_object_list = scene_spec["object_names"]
    scene = _scene()
    instruction = {"default": CELL["prompt"]}
    episode_length_s = 450
    terminations = _Termination
    subtasks = []
