"""Shared RoboLab helpers for model-blind V4 fixture candidates."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

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
    fixture_object_spec,
)
from experiments.online_correction_v4.droid_task_files.reset_registry import (
    ResetRegistry,
    load_reset_registry,
)


@configclass
class _TimeoutOnlyTermination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@lru_cache(maxsize=None)
def reset_registry(fixture_id: str) -> ResetRegistry:
    return load_reset_registry(expected_fixture_id=fixture_id)


def clear_episode_caches() -> None:
    reset_registry.cache_clear()


def active_goal() -> str:
    goal = os.environ.get(ENV_ACTIVE_GOAL)
    if not isinstance(goal, str) or not goal.strip():
        raise RuntimeError(
            f"{ENV_ACTIVE_GOAL} must be set before registering the active V4 task"
        )
    return goal.strip()


def bound_instruction(
    fixture_id: str,
    *,
    goal: str | None = None,
) -> BoundEpisodeInstruction:
    return load_bound_instruction(
        expected_fixture=fixture_id,
        expected_goal=goal or active_goal(),
    )


def scene_for_env_seed(fixture_id: str, env_seed: int):
    registry = reset_registry(fixture_id)
    if env_seed not in registry.positions_by_env_seed:
        raise RuntimeError(
            f"env_seed {env_seed} is not registered in the {fixture_id} reset registry"
        )
    spec = fixture_object_spec(fixture_id)
    scene = import_scene(
        str(Path(__file__).resolve().parents[3] / spec.scene_asset),
        list(spec.contact_objects),
    )
    for name, position in registry.positions_by_env_seed[env_seed].items():
        asset = copy.deepcopy(getattr(scene, name))
        asset.init_state.pos = position
        setattr(scene, name, asset)
    return scene


def scene_for_active_episode(fixture_id: str):
    instruction = bound_instruction(fixture_id)
    return scene_for_env_seed(fixture_id, instruction.env_seed)


def instruction_for_active_episode(fixture_id: str) -> dict[str, str]:
    return bound_instruction(fixture_id).instruction


def timeout_only_termination():
    return _TimeoutOnlyTermination
