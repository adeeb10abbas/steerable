from dataclasses import dataclass

from experiments.groot_droid.robolab_v2_tasks.rubiks_cube_right_of_bowl_matched import RubiksCubeRightOfBowlMatchedTask


@dataclass
class V3C001GrootShortCommandRightTask(RubiksCubeRightOfBowlMatchedTask):
    instruction = {"default": "Put the cube right of the bowl."}
    attributes = ["spatial", "vla_wam_v3c001", "short_command"]
