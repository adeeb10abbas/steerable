"""Real neutral RoboLab scene used only for SGW-01 renderer preflight."""

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task


@configclass
class _ProbeTermination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class SGWRendererProbeTask(Task):
    """No success termination, no subtask, and no learned-policy interface."""

    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = import_scene("rubiks_cube_banana_bowl.usda", contact_object_list)
    terminations = _ProbeTermination
    instruction = {"default": "Put the Rubik's cube to the left of the bowl."}
    attributes = ["sgw_01", "model_blind_renderer_preflight"]
    episode_length_s = 5
    subtasks = []
