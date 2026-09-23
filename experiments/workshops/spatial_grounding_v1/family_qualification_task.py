"""Candidate-bound HEIGHT/DIST native RoboLab overlay with no goal termination."""

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
    raw = os.environ.get("SGW_FAMILY_CANDIDATE_JSON")
    expected = os.environ.get("SGW_FAMILY_CANDIDATE_SHA256")
    if not raw or hashlib.sha256(raw.encode()).hexdigest() != expected:
        raise RuntimeError("SGW family candidate JSON and SHA-256 are required")
    value = json.loads(raw)
    if value.get("family") not in {"HEIGHT", "DIST"} or value.get("action_cap") != 450 or value.get("goal_termination"):
        raise RuntimeError("invalid goal-independent SGW family task payload")
    scene = value.get("native_scene")
    required = {"rubiks_cube", "bowl"} | ({"plate"} if value["family"] == "DIST" else set())
    if not isinstance(scene, dict) or not isinstance(scene.get("object_names"), list) or not required.issubset(scene["object_names"]):
        raise RuntimeError("family task lacks measured native scene inventory")
    if set(value.get("object_poses", ())) != required:
        raise RuntimeError("family task object poses do not match the scored inventory")
    return value


_CANDIDATE = _candidate()


def _scene():
    scene_spec = _CANDIDATE["native_scene"]
    scene = import_scene(scene_spec["asset"], scene_spec["object_names"])
    for name, payload in _CANDIDATE["object_poses"].items():
        asset = copy.deepcopy(getattr(scene, name))
        asset.init_state.pos = tuple(float(item) for item in payload["position_m"])
        asset.init_state.rot = tuple(float(item) for item in payload["quaternion_wxyz"])
        setattr(scene, name, asset)
    return scene


@configclass
class _Termination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class SGWFamilyQualificationTask(Task):
    contact_object_list = _CANDIDATE["native_scene"]["object_names"]
    scene = _scene()
    terminations = _Termination
    instruction = {"default": "Move the Rubik's cube to the requested supported location."}
    attributes = ["sgw_01", _CANDIDATE["family"].lower(), "model_blind_qualification"]
    episode_length_s = 450
    subtasks = []
