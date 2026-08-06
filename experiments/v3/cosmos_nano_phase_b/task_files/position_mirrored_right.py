"""Model-blind position-mirrored-layout RIGHT calibration task."""

from dataclasses import dataclass

from experiments.v3.cosmos_nano_phase_b import fixture_tasks as fixture
from robolab.core.task.task import Task


@dataclass
class V3BNanoPositionMirroredRightCalibrationTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = fixture._scene("position_mirrored")
    terminations = fixture._RightTermination
    instruction = {"default": fixture.RIGHT_PROMPT}
    attributes = [
        "spatial", "vla_wam_v3b", "model_blind_calibration", "position_mirrored"
    ]
    episode_length_s: int = 30
    subtasks = fixture._subtask("right")
