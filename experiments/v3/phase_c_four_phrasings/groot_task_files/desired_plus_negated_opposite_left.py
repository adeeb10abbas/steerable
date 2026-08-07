from dataclasses import dataclass

from experiments.groot_droid.robolab_v2_tasks import rubiks_cube_left_of_bowl_matched as _base


@dataclass
class V3C001GrootContrastiveLeftTask(_base.RubiksCubeLeftOfBowlMatchedTask):
    instruction = {"default": "Put the Rubik's cube to the left of the bowl, not to the right of the bowl."}
    attributes = ["spatial", "vla_wam_v3c001", "desired_plus_negated_opposite"]
