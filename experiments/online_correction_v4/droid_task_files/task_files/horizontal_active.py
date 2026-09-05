"""V4 active horizontal task resolved from per-episode env binding at registration time."""

from dataclasses import dataclass

from robolab.core.task.task import Task

from experiments.online_correction_v4.droid_task_files.constants import EPISODE_LENGTH_S
from experiments.online_correction_v4.droid_task_files import horizontal_shared as shared


@dataclass
class V4HorizontalActiveTask(Task):
    contact_object_list = list(shared.CONTACT_OBJECT_LIST)
    scene = shared.scene_for_active_episode()
    terminations = shared.timeout_only_termination()
    instruction = shared.instruction_for_active_episode()
    attributes = shared.task_attributes_for_active_episode()
    episode_length_s: int = EPISODE_LENGTH_S
    subtasks = []
