"""V3-B009 cube_target_bowl_reference right zero-model gate task."""

from dataclasses import dataclass

from experiments.v3.prospective_tier_b_gates import droid_fixture_tasks as fixture
from robolab.core.task.task import Task


@dataclass
class V3B009CubeTargetBowlReferenceRightGateTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = fixture._scene("V3-B009", "cube_target_bowl_reference")
    terminations = fixture._termination("V3-B009", "cube_target_bowl_reference", "right")
    instruction = {"default": fixture._prompt("V3-B009", "cube_target_bowl_reference", "right")}
    attributes = ["spatial", "vla_wam_v3b", "model_blind_gate", "v3-b009"]
    episode_length_s: int = 30
    subtasks = fixture._subtask("V3-B009", "cube_target_bowl_reference", "right")

