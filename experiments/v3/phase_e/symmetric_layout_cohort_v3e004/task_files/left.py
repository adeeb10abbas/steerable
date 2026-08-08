from dataclasses import dataclass
import json
import os

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004 import fixture_tasks as fixture
from robolab.core.task.task import Task


def _contact_objects() -> list[str]:
    level = fixture._level_from_environment()
    logical = fixture._CANDIDATE.layout(level)
    mapping = json.loads(os.environ["VLA_WAM_V3E004_SCENE_OBJECT_MAPPING"])
    return [str(mapping[name]) for name in logical] + ["table"]


@dataclass
class V3E004DroidLeftTask(Task):
    contact_object_list = _contact_objects()
    scene = fixture._scene()
    terminations = fixture._LeftTermination
    instruction = {"default": fixture.LEFT_PROMPT}
    attributes = ["spatial", "vla_wam_v3e004", "graded_symmetric_object_layout"]
    episode_length_s: int = 30
    subtasks = fixture._subtask("left")
