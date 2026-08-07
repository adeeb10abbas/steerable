from dataclasses import dataclass

from experiments.groot_droid.robolab_v2_tasks import rubiks_cube_right_of_bowl_matched as _base


@dataclass
class V3C001GrootShortCommandRightTask(_base.RubiksCubeRightOfBowlMatchedTask):
    instruction = {"default": "Put the cube right of the bowl."}
    attributes = ["spatial", "vla_wam_v3c001", "short_command"]
