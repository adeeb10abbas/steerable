"""Zero-model task overlay for a hash-bound prospective HEIGHT/DIST scene."""

from dataclasses import dataclass
import hashlib
import json
import os

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task


def _manifest() -> dict:
    path = os.environ.get("SGW_PROSPECTIVE_OVERLAY_MANIFEST")
    expected = os.environ.get("SGW_PROSPECTIVE_OVERLAY_MANIFEST_SHA256")
    if not path or not expected:
        raise RuntimeError("prospective overlay manifest path and SHA-256 are required")
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("prospective overlay manifest file digest mismatch")
    value = json.loads(raw)
    if value.get("status") != "prospective_scene_design_not_measured_or_qualified":
        raise RuntimeError("overlay manifest is not prospective-only")
    return value


_MANIFEST = _manifest()
_NAMES = _MANIFEST["native_import_contract"]["objects_of_interest"]


@configclass
class _Termination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class SGWProspectiveFamilyCaptureTask(Task):
    contact_object_list = _NAMES
    scene = import_scene(_MANIFEST["overlay_usda"]["path"], _NAMES)
    terminations = _Termination
    instruction = {"default": "Observe the scene without acting."}
    attributes = ["sgw_01", _MANIFEST["family"].lower(), "prospective_zero_model_capture"]
    episode_length_s = 5
    subtasks = []
