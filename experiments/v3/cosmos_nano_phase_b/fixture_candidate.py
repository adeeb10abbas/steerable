"""Pure model-blind derivation of the Nano position-mirror fixture candidate."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


STUDY_ID = "vla_wam_language_steerability_v3"
MODEL_ID = "cosmos3_nano_policy_droid"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
SCENE_NAME = "rubiks_cube_banana_bowl.usda"
MOVABLE_OBJECTS = ("rubiks_cube", "bowl", "banana")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}


class FixtureCandidateError(ValueError):
    """Raised when a model-blind fixture source is missing or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _literal_position(node: ast.AST) -> list[float] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return [float(item) for item in value]


def extract_neutral_cube_position(task_path: Path) -> list[float]:
    """Extract the sole literal ``scene.rubiks_cube.init_state.pos`` assignment."""

    task_path = Path(task_path)
    tree = ast.parse(task_path.read_text(), filename=str(task_path))
    values: list[list[float]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if not isinstance(target, ast.Attribute) or target.attr != "pos":
                continue
            init_state = target.value
            if not isinstance(init_state, ast.Attribute) or init_state.attr != "init_state":
                continue
            cube = init_state.value
            if not isinstance(cube, ast.Attribute) or cube.attr != "rubiks_cube":
                continue
            scene = cube.value
            if not isinstance(scene, ast.Name) or scene.id != "scene":
                continue
            position = _literal_position(value_node)
            if position is not None:
                values.append(position)
    if len(values) != 1:
        raise FixtureCandidateError(
            f"expected one literal neutral cube position in {task_path}, found {len(values)}"
        )
    return values[0]


def _scene_entries(metadata_path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(Path(metadata_path).read_text())
    rows = value.get(SCENE_NAME) if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise FixtureCandidateError(f"scene metadata does not contain {SCENE_NAME}")
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("name") in MOVABLE_OBJECTS:
            selected[row["name"]] = row
    if set(selected) != set(MOVABLE_OBJECTS):
        raise FixtureCandidateError("scene metadata is missing one or more movable objects")
    for name, row in selected.items():
        position = row.get("position")
        rotation = row.get("rotation")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or not isinstance(rotation, list)
            or len(rotation) != 4
        ):
            raise FixtureCandidateError(f"invalid pose metadata for {name}")
    return selected


def _position(value: list[Any], label: str) -> list[float]:
    if len(value) != 3 or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise FixtureCandidateError(f"{label} must be three finite numeric coordinates")
    output = [float(item) for item in value]
    if any(not (-10.0 < item < 10.0) for item in output):
        raise FixtureCandidateError(f"{label} contains an implausible coordinate")
    return output


def reflect_position_y(position: list[Any]) -> list[float]:
    x, y, z = _position(position, "position")
    mirrored_y = 0.0 if y == 0.0 else -y
    return [x, mirrored_y, z]


def derive_candidate(
    *,
    scene_metadata_path: Path,
    neutral_left_task_path: Path,
    neutral_right_task_path: Path,
    robolab_commit: str,
) -> dict[str, Any]:
    """Derive control/mirror coordinates without importing a policy or simulator."""

    if robolab_commit != ROBOLAB_COMMIT:
        raise FixtureCandidateError("candidate requires the exact frozen RoboLab commit")
    left_cube = extract_neutral_cube_position(neutral_left_task_path)
    right_cube = extract_neutral_cube_position(neutral_right_task_path)
    if left_cube != right_cube:
        raise FixtureCandidateError("LEFT and RIGHT neutral task files do not share one cube reset")
    source_rows = _scene_entries(scene_metadata_path)
    control_positions = {
        "rubiks_cube": left_cube,
        "bowl": _position(source_rows["bowl"]["position"], "bowl position"),
        "banana": _position(source_rows["banana"]["position"], "banana position"),
    }
    mirrored_positions = {
        name: reflect_position_y(position) for name, position in control_positions.items()
    }
    rotations = {
        name: [float(item) for item in source_rows[name]["rotation"]]
        for name in MOVABLE_OBJECTS
    }
    relative_y = control_positions["rubiks_cube"][1] - control_positions["bowl"][1]
    relative_x = control_positions["rubiks_cube"][0] - control_positions["bowl"][0]
    if abs(relative_y) >= abs(relative_x):
        raise FixtureCandidateError("control reset is not outside both 45-degree side cones")
    mirrored_relative_y = (
        mirrored_positions["rubiks_cube"][1] - mirrored_positions["bowl"][1]
    )
    if abs(mirrored_relative_y + relative_y) > 1e-12:
        raise FixtureCandidateError("mirrored relative lateral offset is not the exact negation")
    return {
        "schema_version": "vla-wam-shared-v3b-nano-position-mirror-candidate-v1",
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "phase": "B_confound_ablation",
        "status": "model_blind_candidate_not_released_for_inference",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "exact_prompts": PROMPTS,
        "source_identity": {
            "robolab_commit": robolab_commit,
            "scene": SCENE_NAME,
            "scene_metadata": file_record(scene_metadata_path),
            "neutral_left_task": file_record(neutral_left_task_path),
            "neutral_right_task": file_record(neutral_right_task_path),
        },
        "factor": {
            "name": "movable_object_center_position_reflection_about_robot_sagittal_plane",
            "transform": "for rubiks_cube, bowl, and banana only: (x,y,z) -> (x,-y,z)",
            "robot_base_plane_y_m": 0.0,
            "changed": "initial center positions of the three movable objects",
            "held_fixed": [
                "object identities",
                "object quaternions",
                "nonmovable scene geometry",
                "robot base and controller",
                "camera poses",
                "prompt bytes and scorer axes",
            ],
            "claim_boundary": (
                "This is a position-mirrored movable-object layout, not a full geometric "
                "reflection; an improper reflection is not represented by a quaternion."
            ),
        },
        "layouts": {
            "control": {
                "positions_robot_base_m": control_positions,
                "quaternions_wxyz_unchanged": rotations,
            },
            "position_mirrored": {
                "positions_robot_base_m": mirrored_positions,
                "quaternions_wxyz_unchanged": rotations,
            },
        },
        "analytic_neutrality_precheck": {
            "control_cube_minus_bowl_xy_m": [relative_x, relative_y],
            "position_mirrored_cube_minus_bowl_xy_m": [relative_x, mirrored_relative_y],
            "outside_left_and_right_45deg_cones": True,
            "live_simulator_reset_check_still_required": True,
        },
        "release_boundary": (
            "Candidate coordinates are not an inference release. A live model-blind reset, "
            "RTX renderer, fixture-settle, and raw-writer calibration report must pass and be "
            "hash-bound by a new amendment before any Nano model request."
        ),
    }


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()

