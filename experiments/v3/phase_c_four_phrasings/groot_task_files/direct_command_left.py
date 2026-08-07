from dataclasses import dataclass

from experiments.groot_droid.robolab_v2_tasks.rubiks_cube_left_of_bowl_matched import RubiksCubeLeftOfBowlMatchedTask


@dataclass
class V3C001GrootDirectCommandLeftTask(RubiksCubeLeftOfBowlMatchedTask):
    instruction = {"default": "Put the Rubik's cube to the left of the bowl."}
    attributes = ["spatial", "vla_wam_v3c001", "direct_command"]
