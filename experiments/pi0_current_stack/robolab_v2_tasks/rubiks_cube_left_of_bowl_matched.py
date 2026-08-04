"""Neutral-reset LEFT task with the frozen V2-A008 wording families."""

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_dropped, object_grabbed, object_left_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class RubiksCubeLeftOfBowlMatchedTermination:
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


@dataclass
class RubiksCubeLeftOfBowlMatchedTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = import_scene("rubiks_cube_banana_bowl.usda", contact_object_list)
    scene.rubiks_cube.init_state.pos = (
        0.303364634513855,
        0.12396888434886932,
        0.08113233000040054,
    )
    terminations = RubiksCubeLeftOfBowlMatchedTermination
    instruction = {
        "short_command": "Put the cube left of the bowl.",
        "goal_as_outcome": "The Rubik's cube should end up to the left of the bowl.",
        "desired_plus_negated_opposite": (
            "Put the Rubik's cube to the left of the bowl, not to the right of the bowl."
        ),
    }
    attributes = ["spatial", "vla_wam_v2a008"]
    episode_length_s: int = 30
    subtasks = [
        Subtask(
            name="rubiks_cube_left_of_bowl",
            conditions=[
                partial(object_grabbed, object="rubiks_cube"),
                partial(
                    object_left_of,
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
