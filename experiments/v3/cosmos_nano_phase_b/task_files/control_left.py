"""Model-blind control-layout LEFT calibration task."""

from dataclasses import dataclass

from experiments.v3.cosmos_nano_phase_b import fixture_tasks as fixture
from robolab.core.task.task import Task


@dataclass
class V3BNanoControlLeftCalibrationTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = fixture._scene("control")
    terminations = fixture._LeftTermination
    instruction = {"default": fixture.LEFT_PROMPT}
    attributes = ["spatial", "vla_wam_v3b", "model_blind_calibration", "control"]
    episode_length_s: int = 30
    subtasks = fixture._subtask("left")
