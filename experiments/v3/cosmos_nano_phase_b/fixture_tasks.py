"""Generic Phase-B calibration tasks loaded from one hash-pinned candidate.

This module is imported by RoboLab only after the model-blind calibration
driver has verified the candidate path and digest.  It contains no released
numeric fixture coordinates and cannot contact a policy endpoint.
"""

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


EXPECTED_SCHEMA = "vla-wam-shared-v3b-nano-position-mirror-candidate-v1"
EXPECTED_STATUS = "model_blind_candidate_not_released_for_inference"
LEFT_PROMPT = "Put the Rubik's cube to the left of the bowl."
RIGHT_PROMPT = "Put the Rubik's cube to the right of the bowl."
OBJECTS = ("rubiks_cube", "bowl", "banana")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_candidate() -> dict:
    raw_path = os.environ.get("VLA_WAM_V3B_FIXTURE_CANDIDATE")
    expected_hash = os.environ.get("VLA_WAM_V3B_FIXTURE_SHA256")
    if not raw_path or not expected_hash:
        raise RuntimeError("Phase-B fixture candidate path and SHA-256 are required")
    path = Path(raw_path).resolve()
    if _sha256(path) != expected_hash:
        raise RuntimeError("Phase-B fixture candidate digest mismatch")
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPECTED_SCHEMA
        or value.get("status") != EXPECTED_STATUS
        or value.get("model_request_count") != 0
        or value.get("behavioral_episode_count") != 0
        or value.get("exact_prompts") != {"left": LEFT_PROMPT, "right": RIGHT_PROMPT}
    ):
        raise RuntimeError("Phase-B fixture candidate is not a model-blind unreleased candidate")
    return value


_CANDIDATE = _load_candidate()


def _positions(arm: str) -> dict[str, tuple[float, float, float]]:
    rows = _CANDIDATE["layouts"][arm]["positions_robot_base_m"]
    if set(rows) != set(OBJECTS):
        raise RuntimeError(f"unexpected movable-object inventory for {arm}")
    output = {}
    for name, raw in rows.items():
        if not isinstance(raw, list) or len(raw) != 3:
            raise RuntimeError(f"invalid {arm} position for {name}")
        output[name] = tuple(float(item) for item in raw)
    return output


def _scene(arm: str):
    scene = import_scene(
        "rubiks_cube_banana_bowl.usda",
        ["rubiks_cube", "banana", "bowl", "table"],
    )
    for name, position in _positions(arm).items():
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


# RoboLab loads task files with ``exec_module`` without first inserting the
# transient module into ``sys.modules``.  Keep the shared, future-annotations
# helpers here and expose each dataclass Task through its own small wrapper in
# ``task_files``.  This also guarantees one Task subclass per file, which is
# the invariant assumed by RoboLab's task resolver.
