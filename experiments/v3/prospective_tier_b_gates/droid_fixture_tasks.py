"""Hash-pinned RoboLab fixtures for V3-B008 and V3-B009 zero-model gates."""

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


EXPECTED_SCHEMA = "vla-wam-shared-v3b-droid-tier-b-gate-candidate-v1"
EXPECTED_STATUS = "model_blind_candidate_not_released_for_inference"
OBJECTS = ("rubiks_cube", "bowl", "banana")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_candidate() -> dict:
    raw_path = os.environ.get("VLA_WAM_V3_TIERB_GATE_CANDIDATE")
    expected_hash = os.environ.get("VLA_WAM_V3_TIERB_GATE_SHA256")
    if not raw_path or not expected_hash:
        raise RuntimeError("Tier-B gate candidate path and SHA-256 are required")
    path = Path(raw_path).resolve()
    if _sha256(path) != expected_hash:
        raise RuntimeError("Tier-B gate candidate digest mismatch")
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPECTED_SCHEMA
        or value.get("status") != EXPECTED_STATUS
        or value.get("model_request_count") != 0
        or value.get("behavioral_episode_count") != 0
    ):
        raise RuntimeError("invalid Tier-B model-blind candidate")
    return value


_CANDIDATE = _load_candidate()


def _positions(study: str, arm: str) -> dict[str, tuple[float, float, float]]:
    raw = _CANDIDATE["studies"][study]["arms"][arm]["positions_robot_base_m"]
    if set(raw) != set(OBJECTS):
        raise RuntimeError(f"unexpected movable-object inventory for {study}/{arm}")
    return {name: tuple(float(item) for item in raw[name]) for name in OBJECTS}


def _scene(study: str, arm: str):
    scene = import_scene(
        "rubiks_cube_banana_bowl.usda",
        ["rubiks_cube", "banana", "bowl", "table"],
    )
    for name, position in _positions(study, arm).items():
        asset = copy.deepcopy(getattr(scene, name))
        asset.init_state.pos = position
        setattr(scene, name, asset)
    return scene


def _role(study: str, arm: str) -> tuple[str, str, str]:
    row = _CANDIDATE["studies"][study]["arms"][arm]
    return row["target_object"], row["reference_object"], row["target_label"]


def _prompt(study: str, arm: str, relation: str) -> str:
    target, _, target_label = _role(study, arm)
    reference = "bowl" if target == "rubiks_cube" else "Rubik's cube"
    return f"Put the {target_label} to the {relation} of the {reference}."


def _subtask(study: str, arm: str, relation: str) -> list[Subtask]:
    target, reference, _ = _role(study, arm)
    relation_fn = object_left_of if relation == "left" else object_right_of
    return [
        Subtask(
            name=f"{target}_{relation}_of_{reference}",
            conditions=[
                partial(object_grabbed, object=target),
                partial(
                    relation_fn,
                    object=target,
                    reference_object=reference,
                    frame_of_reference="robot",
                    mirrored=False,
                ),
                partial(object_dropped, object=target),
            ],
            logical="all",
            score=1.0,
        )
    ]


def _termination(study: str, arm: str, relation: str):
    target, reference, _ = _role(study, arm)
    relation_fn = object_left_of if relation == "left" else object_right_of

    @configclass
    class _Termination:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        success = DoneTerm(
            func=relation_fn,
            params={
                "object": target,
                "reference_object": reference,
                "frame_of_reference": "robot",
                "mirrored": False,
                "require_gripper_detached": True,
            },
        )

    return _Termination
