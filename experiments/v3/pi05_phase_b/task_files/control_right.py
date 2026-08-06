"""Hash-bound V3-B002 control-layout RIGHT task."""

from dataclasses import dataclass

from experiments.v3.pi05_phase_b import fixture_tasks as fixture
from robolab.core.task.task import Task


@dataclass
class V3B002Pi05ControlRightTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = fixture._scene("control")
    terminations = fixture._RightTermination
    instruction = {"default": fixture.RIGHT_PROMPT}
    attributes = ["spatial", "vla_wam_v3b002", "control"]
    episode_length_s: int = 30
    subtasks = fixture._subtask("right")
