"""Shared RoboLab helpers for the V4 horizontal timeout-only fixture tasks."""

from __future__ import annotations

import copy
import os
from functools import lru_cache

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene

from experiments.online_correction_v4.droid_task_files.binding import BoundEpisodeInstruction, load_bound_instruction
from experiments.online_correction_v4.droid_task_files.constants import (
    CONTACT_OBJECT_LIST,
    ENV_ACTIVE_GOAL,
    EPISODE_LENGTH_S,
    SCENE_ASSET,
)
from experiments.online_correction_v4.droid_task_files.horizontal_core import task_attributes
from experiments.online_correction_v4.droid_task_files.reset_registry import ResetRegistry, load_reset_registry


@configclass
class _TimeoutOnlyTermination:
    """V4 first-placement episodes terminate on timeout only."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@lru_cache(maxsize=1)
def _reset_registry() -> ResetRegistry:
    return load_reset_registry()


def clear_episode_caches() -> None:
    """Drop per-process reset registry caches between live episodes."""
    _reset_registry.cache_clear()


def active_goal() -> str:
    goal = os.environ.get(ENV_ACTIVE_GOAL)
    if not isinstance(goal, str) or not goal.strip():
        raise RuntimeError(f"{ENV_ACTIVE_GOAL} must be set before registering the active horizontal task")
    return goal.strip()


def bound_instruction_for_active_episode() -> BoundEpisodeInstruction:
    return load_bound_instruction(expected_fixture="horizontal", expected_goal=active_goal())


def bound_instruction_for_relation(relation: str) -> BoundEpisodeInstruction:
    """Load one explicitly registered relation without changing prompt bytes."""
    return load_bound_instruction(
        expected_fixture="horizontal",
        expected_goal=relation,
    )


def instruction_dict_for_relation(relation: str) -> dict[str, str]:
    return bound_instruction_for_relation(relation).instruction


def scene_for_env_seed(env_seed: int):
    registry = _reset_registry()
    if env_seed not in registry.positions_by_env_seed:
        raise RuntimeError(f"env_seed {env_seed} is not registered in the horizontal reset registry")
    scene = import_scene(
        SCENE_ASSET,
        list(CONTACT_OBJECT_LIST),
    )
    positions = registry.positions_by_env_seed[env_seed]
    for name, position in positions.items():
        asset = copy.deepcopy(getattr(scene, name))
        asset.init_state.pos = position
        setattr(scene, name, asset)
    return scene


def scene_for_active_episode():
    instruction = bound_instruction_for_active_episode()
    return scene_for_env_seed(instruction.env_seed)


def instruction_for_active_episode() -> dict[str, str]:
    if os.environ.get("ONLINE_CORRECTION_V4_QUEUE_ROW") or os.environ.get("ONLINE_CORRECTION_V4_QUEUE_ROW_SHA256"):
        return bound_instruction_for_active_episode().instruction
    raise RuntimeError(
        "V4 horizontal task instruction requires "
        "ONLINE_CORRECTION_V4_QUEUE_ROW and ONLINE_CORRECTION_V4_QUEUE_ROW_SHA256"
    )


def task_attributes_for_active_episode() -> list[str]:
    return task_attributes(active_goal())


def task_attributes_for_relation(relation: str) -> list[str]:
    return task_attributes(relation)


def timeout_only_termination():
    return _TimeoutOnlyTermination


__all__ = [
    "CONTACT_OBJECT_LIST",
    "EPISODE_LENGTH_S",
    "active_goal",
    "bound_instruction_for_active_episode",
    "bound_instruction_for_relation",
    "instruction_for_active_episode",
    "instruction_dict_for_relation",
    "scene_for_active_episode",
    "scene_for_env_seed",
    "task_attributes",
    "task_attributes_for_active_episode",
    "task_attributes_for_relation",
    "timeout_only_termination",
]
