"""V3-B005 LEFT task at the process-bound frozen bowl level."""

from dataclasses import dataclass

from experiments.v3.cosmos_nano_lateral_sweep import fixture_tasks as fixture
from robolab.core.task.task import Task


@dataclass
class V3B005NanoLeftTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = fixture._scene()
    terminations = fixture._LeftTermination
    instruction = {"default": fixture.LEFT_PROMPT}
    attributes = ["spatial", "vla_wam_v3b005", f"level_{fixture._LEVEL_INDEX}"]
    episode_length_s: int = 30
    subtasks = fixture._subtask("left")
