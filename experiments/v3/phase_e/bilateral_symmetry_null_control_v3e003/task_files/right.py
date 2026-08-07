from dataclasses import dataclass
from experiments.v3.phase_e.bilateral_symmetry_null_control_v3e003 import fixture_tasks as fixture
from robolab.core.task.task import Task

@dataclass
class V3E003Pi05SymmetricRightTask(Task):
    contact_object_list = ["rubiks_cube", "banana_left", "banana_right", "bowl", "table"]
    scene = fixture._scene()
    terminations = fixture._RightTermination
    instruction = {"default": fixture.RIGHT_PROMPT}
    attributes = ["spatial", "vla_wam_v3e003", "symmetric_object_layout"]
    episode_length_s: int = 30
    subtasks = fixture._subtask("right")
