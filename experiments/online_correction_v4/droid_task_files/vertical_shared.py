"""RoboLab bindings for the model-blind V4 C5 vertical fixture."""

from experiments.online_correction_v4.droid_task_files import candidate_fixture_shared
from experiments.online_correction_v4.droid_task_files.constants import (
    EPISODE_LENGTH_S,
    VERTICAL_CONTACT_OBJECTS,
)
from experiments.online_correction_v4.droid_task_files.vertical_core import task_attributes


FIXTURE_ID = "vertical"


def clear_episode_caches() -> None:
    candidate_fixture_shared.clear_episode_caches()


def scene_for_active_episode():
    return candidate_fixture_shared.scene_for_active_episode(FIXTURE_ID)


def instruction_for_active_episode() -> dict[str, str]:
    return candidate_fixture_shared.instruction_for_active_episode(FIXTURE_ID)


def task_attributes_for_active_episode() -> list[str]:
    return task_attributes(candidate_fixture_shared.active_goal())


def timeout_only_termination():
    return candidate_fixture_shared.timeout_only_termination()
