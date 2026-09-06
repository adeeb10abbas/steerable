"""Shared RoboLab helpers for the V4 C7 sponge/tray timeout-only task."""

from __future__ import annotations

import copy
import os
from functools import lru_cache

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene

from experiments.online_correction_v4.droid_task_files.binding import (
    BoundEpisodeInstruction,
    load_bound_instruction,
)
from experiments.online_correction_v4.droid_task_files.constants import (
    ENV_ACTIVE_GOAL,
    EPISODE_LENGTH_S,
    OBJECT_PAIR_CONTACT_OBJECTS,
    OBJECT_PAIR_SCENE_PATH,
)
from experiments.online_correction_v4.droid_task_files.object_pair_core import task_attributes
from experiments.online_correction_v4.droid_task_files.reset_registry import (
    ResetRegistry,
    load_reset_registry,
)


@configclass
class _TimeoutOnlyTermination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@lru_cache(maxsize=1)
def _reset_registry() -> ResetRegistry:
    return load_reset_registry(expected_fixture_id="object_pair")


def clear_episode_caches() -> None:
    _reset_registry.cache_clear()


def active_goal() -> str:
    goal = os.environ.get(ENV_ACTIVE_GOAL)
    if not isinstance(goal, str) or not goal.strip():
        raise RuntimeError(
            f"{ENV_ACTIVE_GOAL} must be set before registering the active object-pair task"
        )
    return goal.strip()


def bound_instruction_for_active_episode() -> BoundEpisodeInstruction:
    return load_bound_instruction(
        expected_fixture="object_pair",
        expected_goal=active_goal(),
    )


def scene_for_env_seed(env_seed: int):
    registry = _reset_registry()
    if env_seed not in registry.positions_by_env_seed:
        raise RuntimeError(
            f"env_seed {env_seed} is not registered in the object-pair reset registry"
        )
    scene = import_scene(
        OBJECT_PAIR_SCENE_PATH,
        [*OBJECT_PAIR_CONTACT_OBJECTS, "table_visual"],
    )
    for name, position in registry.positions_by_env_seed[env_seed].items():
        asset = copy.deepcopy(getattr(scene, name))
        asset.init_state.pos = position
        setattr(scene, name, asset)
    return scene


def scene_for_active_episode():
    instruction = bound_instruction_for_active_episode()
    return scene_for_env_seed(instruction.env_seed)


def instruction_for_active_episode() -> dict[str, str]:
    return bound_instruction_for_active_episode().instruction


def task_attributes_for_active_episode() -> list[str]:
    return task_attributes(active_goal())


def timeout_only_termination():
    return _TimeoutOnlyTermination


__all__ = [
    "EPISODE_LENGTH_S",
    "OBJECT_PAIR_CONTACT_OBJECTS",
    "active_goal",
    "bound_instruction_for_active_episode",
    "clear_episode_caches",
    "instruction_for_active_episode",
    "scene_for_active_episode",
    "scene_for_env_seed",
    "task_attributes_for_active_episode",
    "timeout_only_termination",
]
