"""V3-B008 target_start_left left zero-model gate task."""

from dataclasses import dataclass

from experiments.v3.prospective_tier_b_gates import droid_fixture_tasks as fixture
from robolab.core.task.task import Task


@dataclass
class V3B008TargetStartLeftLeftGateTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = fixture._scene("V3-B008", "target_start_left")
    terminations = fixture._termination("V3-B008", "target_start_left", "left")
    instruction = {"default": fixture._prompt("V3-B008", "target_start_left", "left")}
    attributes = ["spatial", "vla_wam_v3b", "model_blind_gate", "v3-b008"]
    episode_length_s: int = 30
    subtasks = fixture._subtask("V3-B008", "target_start_left", "left")

