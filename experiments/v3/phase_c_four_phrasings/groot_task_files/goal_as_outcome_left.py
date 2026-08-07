from dataclasses import dataclass

from experiments.groot_droid.robolab_v2_tasks.rubiks_cube_left_of_bowl_matched import RubiksCubeLeftOfBowlMatchedTask


@dataclass
class V3C001GrootGoalAsOutcomeLeftTask(RubiksCubeLeftOfBowlMatchedTask):
    instruction = {"default": "The Rubik's cube should end up to the left of the bowl."}
    attributes = ["spatial", "vla_wam_v3c001", "goal_as_outcome"]
