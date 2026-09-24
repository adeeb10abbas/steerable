"""Native scene-development task; the pinned robot/camera registration is reused."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task

_path = Path(os.environ['SGW_SCENE_DESIGN_INPUT'])
_raw = _path.read_bytes()
if hashlib.sha256(_raw).hexdigest() != os.environ['SGW_SCENE_DESIGN_SHA256']:
    raise RuntimeError('Scene input changed after launch')
_input = json.loads(_raw)
_scene = _input['scene']
if hashlib.sha256(Path(_scene['path']).read_bytes()).hexdigest() != _scene['sha256']:
    raise RuntimeError('Authored scene bytes changed')


@configclass
class _Termination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class SGWSceneDesignTask(Task):
    contact_object_list = _scene['object_names']
    scene = import_scene(_scene['path'], contact_object_list)
    terminations = _Termination
    instruction = {'default': "Move the Rubik's cube to the requested supported location."}
    attributes = ['sgw_01', 'scripted_scene_design']
    episode_length_s = 450
    subtasks = []
