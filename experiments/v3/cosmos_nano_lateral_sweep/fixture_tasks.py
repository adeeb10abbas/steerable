"""Hash-bound RoboLab tasks for the V3-B005 Nano lateral sweep."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_dropped,
    object_grabbed,
    object_left_of,
    object_right_of,
)
from robolab.core.task.subtask import Subtask


EXPECTED_SCHEMA = "vla-wam-shared-v3b-nano-lateral-safe-distractor-fixture-v1"
EXPECTED_STATUS = "frozen_before_v3b005_model_blind_gate_and_before_any_v3b005_model_request"
LEFT_PROMPT = "Put the Rubik's cube to the left of the bowl."
RIGHT_PROMPT = "Put the Rubik's cube to the right of the bowl."
OBJECTS = ("rubiks_cube", "bowl", "banana")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fixture() -> tuple[dict, int]:
    raw_path = os.environ.get("VLA_WAM_V3B005_SAFE_FIXTURE")
    expected_hash = os.environ.get("VLA_WAM_V3B005_SAFE_FIXTURE_SHA256")
    level_raw = os.environ.get("VLA_WAM_V3B005_LEVEL_INDEX")
    if not raw_path or not expected_hash or level_raw is None:
        raise RuntimeError("V3-B005 safe-fixture path, SHA-256, and level are required")
    path = Path(raw_path).resolve()
    if _sha256(path) != expected_hash:
        raise RuntimeError("V3-B005 safe-fixture digest mismatch")
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPECTED_SCHEMA
        or value.get("amendment_id") != "V3-B005"
        or value.get("status") != EXPECTED_STATUS
        or value.get("model_request_count_before_registration") != 0
        or value.get("behavioral_episode_count_before_registration") != 0
    ):
        raise RuntimeError("V3-B005 safe fixture is not the prospective frozen input")
    try:
        level = int(level_raw)
    except ValueError as exc:
        raise RuntimeError("V3-B005 level index must be an integer") from exc
    levels = value.get("ordered_bowl_y_levels_m")
    if not isinstance(levels, list) or len(levels) != 7 or level not in range(7):
        raise RuntimeError("V3-B005 level index must select one of seven frozen levels")
    return value, level


_FIXTURE, _LEVEL_INDEX = _load_fixture()


def _positions() -> dict[str, tuple[float, float, float]]:
    raw = _FIXTURE["positions_robot_base_m"]
    cube = list(raw["rubiks_cube"])
    bowl = list(raw["bowl_center_at_level_0"])
    banana = list(raw["banana"])
    bowl[1] = _FIXTURE["ordered_bowl_y_levels_m"][_LEVEL_INDEX]
    rows = {"rubiks_cube": cube, "bowl": bowl, "banana": banana}
    output: dict[str, tuple[float, float, float]] = {}
    for name in OBJECTS:
        position = rows[name]
        if not isinstance(position, list) or len(position) != 3:
            raise RuntimeError(f"invalid V3-B005 position for {name}")
        output[name] = tuple(float(item) for item in position)
    return output


def _scene():
    scene = import_scene(
        "rubiks_cube_banana_bowl.usda",
        ["rubiks_cube", "banana", "bowl", "table"],
    )
    for name, position in _positions().items():
        asset = copy.deepcopy(getattr(scene, name))
        asset.init_state.pos = position
        setattr(scene, name, asset)
    return scene


def _subtask(relation: str) -> list[Subtask]:
    relation_fn = object_left_of if relation == "left" else object_right_of
    return [
        Subtask(
            name=f"rubiks_cube_{relation}_of_bowl",
            conditions=[
                partial(object_grabbed, object="rubiks_cube"),
                partial(
                    relation_fn,
                    object="rubiks_cube",
                    reference_object="bowl",
                    frame_of_reference="robot",
                    mirrored=False,
                ),
                partial(object_dropped, object="rubiks_cube"),
            ],
            logical="all",
            score=1.0,
        )
    ]


@configclass
class _LeftTermination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_left_of,
        params={
            "object": "rubiks_cube",
            "reference_object": "bowl",
            "frame_of_reference": "robot",
            "mirrored": False,
            "require_gripper_detached": True,
        },
    )


@configclass
class _RightTermination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_right_of,
        params={
            "object": "rubiks_cube",
            "reference_object": "bowl",
            "frame_of_reference": "robot",
            "mirrored": False,
            "require_gripper_detached": True,
        },
    )
