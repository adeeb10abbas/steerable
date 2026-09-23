"""CPU-only geometric design arithmetic; never a registered fixture or release."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .historical_root_nonmatch import _validated_position
from .lat_candidate_generator import workspace_digest
from .prospective_family_scene import _root_for_center, _validate_base_receipt

TARGET_DISTANCE_M = 0.50


def _vector(value: Any, label: str) -> tuple[float, float, float]:
    result = _validated_position(value)
    if result is None:
        raise ValueError(f"{label} must be a finite three-vector")
    return result


def translation_preview(
    objects: Mapping[str, Mapping[str, Any]], robot_root_env_local_xyz_m: list[float],
) -> dict[str, Any]:
    """Choose the smallest horizontal translation at fixed height and orientation."""
    if not {"rubiks_cube", "bowl", "table"}.issubset(objects):
        raise ValueError("design arithmetic requires measured cube, bowl and table")
    if any("robot" in name.lower() or "camera" in name.lower() for name in objects):
        raise ValueError("robot/camera transforms cannot enter the translated object inventory")
    robot = _vector(robot_root_env_local_xyz_m, "measured robot root")
    centers = {}
    for name, row in objects.items():
        center = _vector(row.get("geometric_center_env_local_xyz_m"), f"{name} geometric center")
        root = _vector(row.get("root_position_env_local_xyz_m"), f"{name} actor root")
        _vector(row.get("geometric_center_offset_root_local_xyz_m"), f"{name} center offset")
        q = row.get("root_quaternion_world_wxyz")
        if (not isinstance(q, (list, tuple)) or len(q) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in q)
                or not math.isclose(math.hypot(*q), 1.0, rel_tol=0, abs_tol=1e-3)):
            raise ValueError(f"{name} measured quaternion must be finite and normalized")
        if math.dist(_root_for_center(row, list(center)), root) > 1e-6:
            raise ValueError(f"{name} measured root/center/offset transform is inconsistent")
        minimum = _vector(row.get("bbox_env_local_min_xyz_m"), f"{name} bounds minimum")
        maximum = _vector(row.get("bbox_env_local_max_xyz_m"), f"{name} bounds maximum")
        if any(lo > hi for lo, hi in zip(minimum, maximum, strict=True)):
            raise ValueError(f"{name} measured bounds are inverted")
        centers[name] = center
    initial = centers["rubiks_cube"]
    dx, dy, dz = (initial[i] - robot[i] for i in range(3))
    horizontal = math.hypot(dx, dy)
    if not all(math.isfinite(x) for x in (horizontal, dz)) or abs(dz) > TARGET_DISTANCE_M:
        raise ValueError("target distance is unavailable at the retained geometric-center height")
    target_horizontal = math.sqrt((TARGET_DISTANCE_M - abs(dz)) * (TARGET_DISTANCE_M + abs(dz)))
    if horizontal == 0 and target_horizontal != 0:
        raise ValueError("horizontal direction is undefined; do not invent a displacement direction")
    if horizontal == 0:
        target = list(initial)
    else:
        target = [
            robot[0] + target_horizontal * (dx / horizontal),
            robot[1] + target_horizontal * (dy / horizontal),
            initial[2],
        ]
    if not math.isclose(math.dist(target, robot), TARGET_DISTANCE_M, rel_tol=0, abs_tol=1e-12):
        raise ValueError("target cannot be represented in the supplied coordinate frame")
    translation = [target[i] - initial[i] for i in range(3)]
    table = objects["table"]
    poses, proposed_centers, clearances = {}, {}, {}
    for name, row in objects.items():
        if name == "table":
            continue
        center = [centers[name][i] + translation[i] for i in range(3)]
        poses[name] = {
            "position_m": _root_for_center(row, center),
            "quaternion_wxyz": list(row["root_quaternion_world_wxyz"]),
        }
        proposed_centers[name] = center
        clearances[name] = min(
            *(row["bbox_env_local_min_xyz_m"][i] + translation[i] - table["bbox_env_local_min_xyz_m"][i]
              for i in (0, 1)),
            *(table["bbox_env_local_max_xyz_m"][i] - row["bbox_env_local_max_xyz_m"][i] - translation[i]
              for i in (0, 1)),
        )
    return {
        "schema_version": "sgw-01-paper-informed-translation-calculation-v1",
        "status": "arithmetic_design_preview_not_a_registered_candidate",
        "target_cube_geometric_center_distance_m": TARGET_DISTANCE_M,
        "initial_cube_geometric_center_distance_m": math.dist(initial, robot),
        "proposed_cube_geometric_center_distance_m": math.dist(proposed_centers["rubiks_cube"], robot),
        "proposed_cube_actor_root_distance_m": math.dist(poses["rubiks_cube"]["position_m"], robot),
        "measured_reference_robot_root_env_local_xyz_m": list(robot),
        "translation_env_local_xyz_m": translation,
        "translation_choice": "Minimum-length horizontal radial translation; retain geometric-center height.",
        "translated_actor_poses": poses, "proposed_geometric_centers_env_local_xyz_m": proposed_centers,
        "table_edge_clearance_xy_m": clearances,
        "objects_extending_outside_measured_table": sorted(name for name, margin in clearances.items() if margin < 0),
        "unchanged": ["all object orientations", "table", "robot", "cameras"],
        "arrangement_scope": "Translate every captured non-table object/support together, including the distractor; preserve relative geometry.",
        "new_scene_materialized": False, "new_registered_candidates": 0,
        "new_model_requests": 0, "new_simulator_runs": 0, "release_permitted": False,
        "claim_boundary": (
            "Authoring arithmetic from retained measurements, not native realization, an optimum, "
            "a new scoring/clearance gate or permission to refill a frozen pool. The inherited measured "
            "robot root must be rechecked at realization. Table-footprint margins are descriptive; "
            "both goal placements, grasp/loaded paths, joint limits, collisions, contacts, support "
            "stability and full qualification remain unverified."
        ),
    }


def from_files(workspace_path: Path, capture_path: Path, overlay_path: Path | None = None) -> dict[str, Any]:
    workspace_raw, capture_raw = workspace_path.read_bytes(), capture_path.read_bytes()
    workspace, capture = json.loads(workspace_raw), json.loads(capture_raw)
    _validate_base_receipt(workspace, workspace_path)
    inputs = {
        "robot_workspace": {"path": str(workspace_path), "sha256": hashlib.sha256(workspace_raw).hexdigest()},
        "layout_capture": {"path": str(capture_path), "sha256": hashlib.sha256(capture_raw).hexdigest()},
    }
    if capture_raw == workspace_raw:
        if overlay_path is not None:
            raise ValueError("a base-workspace calculation must not claim a separate family overlay")
    else:
        if overlay_path is None:
            raise ValueError("family capture requires its exact overlay-to-workspace binding")
        overlay_raw = overlay_path.read_bytes()
        overlay = json.loads(overlay_raw)
        binding = capture["overlay_manifest"]
        if (len(overlay_raw) != binding["bytes"] or hashlib.sha256(overlay_raw).hexdigest() != binding["sha256"]
                or overlay["base_workspace_receipt"]["sha256"] != inputs["robot_workspace"]["sha256"]
                or overlay["base_workspace_receipt"]["receipt_sha256"] != workspace["receipt_sha256"]):
            raise ValueError("native capture does not bind this overlay and measured robot workspace")
        if (capture.get("schema_version") != "sgw-01-prospective-family-native-capture-v1"
                or capture.get("receipt_sha256") != workspace_digest(capture)
                or capture.get("model_request_count") != 0 or capture.get("behavioral_episode_count") != 0
                or capture.get("environment_origin_world_xyz_m") != workspace["environment_origin_world_xyz_m"]
                or capture.get("robolab_commit") != workspace["robolab_commit"]):
            raise ValueError("family capture identity, frame or zero-model boundary differs")
        inputs["overlay"] = {"path": str(overlay_path), "sha256": binding["sha256"]}
    if workspace["robolab_commit"] != "0aef241fb088ca21bb4ebd24448940ed56620d17":
        raise ValueError("paper-informed preview must retain the pinned RoboLab version")
    report = translation_preview(capture["objects"], workspace["robot"]["base_position_env_local_xyz_m"])
    report["input_bindings"] = inputs
    report["reference_robot_quaternion_world_wxyz"] = workspace["robot"]["base_quaternion_world_wxyz"]
    report["robot_reference_scope"] = "Measured inherited base workspace, not a new contemporaneous robot-pose observation."
    report["producer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workspace-receipt", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = from_files(args.workspace_receipt, args.capture, args.overlay_manifest)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
