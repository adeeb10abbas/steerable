from dataclasses import dataclass

from experiments.groot_droid.robolab_v2_tasks.rubiks_cube_right_of_bowl_matched import RubiksCubeRightOfBowlMatchedTask


@dataclass
class V3C001GrootGoalAsOutcomeRightTask(RubiksCubeRightOfBowlMatchedTask):
    instruction = {"default": "The Rubik's cube should end up to the right of the bowl."}
    attributes = ["spatial", "vla_wam_v3c001", "goal_as_outcome"]
