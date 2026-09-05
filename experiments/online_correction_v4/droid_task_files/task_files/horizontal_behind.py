"""V4 horizontal behind relation: timeout-only rubiks_cube_banana_bowl task."""

from dataclasses import dataclass

from robolab.core.task.task import Task

from experiments.online_correction_v4.droid_task_files.constants import EPISODE_LENGTH_S
from experiments.online_correction_v4.droid_task_files import horizontal_shared as shared


@dataclass
class V4HorizontalBehindTask(Task):
    contact_object_list = list(shared.CONTACT_OBJECT_LIST)
    scene = shared.scene_for_env_seed(
        shared.bound_instruction_for_relation("behind").env_seed
    )
    terminations = shared.timeout_only_termination()
    instruction = shared.instruction_dict_for_relation("behind")
    attributes = shared.task_attributes("behind")
    episode_length_s: int = EPISODE_LENGTH_S
    subtasks = []
