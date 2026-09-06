"""V4 active C6 containment task resolved from the episode binding."""

from dataclasses import dataclass

from robolab.core.task.task import Task

from experiments.online_correction_v4.droid_task_files.constants import EPISODE_LENGTH_S
from experiments.online_correction_v4.droid_task_files import containment_shared as shared


@dataclass
class V4ContainmentActiveTask(Task):
    contact_object_list = list(shared.CONTAINMENT_CONTACT_OBJECTS)
    scene = shared.scene_for_active_episode()
    terminations = shared.timeout_only_termination()
    instruction = shared.instruction_for_active_episode()
    attributes = shared.task_attributes_for_active_episode()
    episode_length_s: int = EPISODE_LENGTH_S
    subtasks = []
