from dataclasses import dataclass

from experiments.groot_droid.robolab_v2_tasks import rubiks_cube_left_of_bowl_matched as _base


@dataclass
class V3C001GrootGoalAsOutcomeLeftTask(_base.RubiksCubeLeftOfBowlMatchedTask):
    instruction = {"default": "The Rubik's cube should end up to the left of the bowl."}
    attributes = ["spatial", "vla_wam_v3c001", "goal_as_outcome"]
