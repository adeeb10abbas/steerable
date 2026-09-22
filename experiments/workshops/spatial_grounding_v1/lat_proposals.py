"""Deterministic, explicitly unqualified LAT fixture proposals."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import math
from typing import Any, Mapping


def propose_lat_layouts(workspace: Mapping[str, Any], *, seed: int, count: int = 100) -> list[dict[str, Any]]:
    """Propose neutral root-pose layouts from measured geometry, never qualify them."""

    if workspace.get("measurement_schema_version") != "sgw-01-lat-measured-workspace-v2":
        raise ValueError("LAT proposals require a measured-workspace v2 receipt")
    if not 1 <= count <= 100:
        raise ValueError("LAT proposal count must be within the frozen 1..100 cap")
    objects = workspace["objects"]
    cube, bowl, table = (objects[name] for name in ("rubiks_cube", "bowl", "table"))
    for name, row in (("rubiks_cube", cube), ("bowl", bowl)):
        if "com_position_env_local_xyz_m" not in row:
            raise ValueError(f"{name} lacks required measured COM geometry")
    table_min, table_max = table["bbox_env_local_min_xyz_m"], table["bbox_env_local_max_xyz_m"]
    cube_half = _half_extent(cube)
    bowl_half = _half_extent(bowl)
    cube_offset, bowl_offset = cube["geometric_center_offset_root_local_xyz_m"], bowl["geometric_center_offset_root_local_xyz_m"]
    cube_q, bowl_q = cube["root_quaternion_world_wxyz"], bowl["root_quaternion_world_wxyz"]
    cube_z, bowl_z = cube["root_position_env_local_xyz_m"][2], bowl["root_position_env_local_xyz_m"][2]
    y_delta = _rot_y(bowl_q, bowl_offset) - _rot_y(cube_q, cube_offset)
    margin = 0.01
    low_x = table_min[0] + cube_half[0] + bowl_half[0] + margin
    high_x = table_max[0] - cube_half[0] - bowl_half[0] - margin
    low_y = table_min[1] + max(cube_half[1], bowl_half[1]) + margin
    high_y = table_max[1] - max(cube_half[1], bowl_half[1]) - margin
    if low_x >= high_x or low_y >= high_y:
        raise ValueError("measured table bounds cannot contain distinct neutral LAT proposals")
    rows = []
    for index in range(count):
        digest = hashlib.sha256(f"{seed}|LAT|{index}".encode()).digest()
        unit_x, unit_y = int.from_bytes(digest[:8], "big") / 2**64, int.from_bytes(digest[8:16], "big") / 2**64
        bowl_x, bowl_y = low_x + unit_x * (high_x - low_x), low_y + unit_y * (high_y - low_y)
        cube_x = bowl_x + (cube_half[0] + bowl_half[0] + margin) * (1 if index % 2 else -1)
        if not table_min[0] + cube_half[0] <= cube_x <= table_max[0] - cube_half[0]:
            cube_x = bowl_x - (cube_half[0] + bowl_half[0] + margin) * (1 if index % 2 else -1)
        rows.append({
            "proposal_id": f"LAT-PROPOSAL-{index + 1:03d}",
            "status": "proposed_unqualified_requires_physical_validation",
            "object_root_poses": {
                "rubiks_cube": {"position_m": [cube_x, bowl_y + y_delta, cube_z], "quaternion_wxyz": cube_q},
                "bowl": {"position_m": [bowl_x, bowl_y, bowl_z], "quaternion_wxyz": bowl_q},
            },
            "center_source": "pinned_robolab_geometric_center",
            "scoring_center_offsets_root_local_m": {"rubiks_cube": cube_offset, "bowl": bowl_offset},
            "waypoint_recipe": {
                "source": "experiments/v3/phase_e/reference_controller_runner.py",
                "phases": ["pregrasp", "grasp_approach", "close", "lift", "preplace", "place", "release"],
                "eef_start_env_local_xyz_m": workspace["eef_position_env_local_xyz_m"],
                "requires_live_abs_ik_validation": True,
            },
        })
    return rows


def proposal_to_candidate(proposal: Mapping[str, Any], workspace: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    """Make an explicitly unqualified proposal startable by physical validation."""
    poses = proposal["object_root_poses"]
    cube_center = _center(poses["rubiks_cube"], proposal["scoring_center_offsets_root_local_m"]["rubiks_cube"])
    bowl_center = _center(poses["bowl"], proposal["scoring_center_offsets_root_local_m"]["bowl"])
    paths = {}
    for sign, name in ((1, "positive"), (-1, "negative")):
        target = [bowl_center[0], bowl_center[1] + sign * .04, cube_center[2]]
        paths[name] = [
            _waypoint([cube_center[0], cube_center[1], cube_center[2] + .12], 0.0),
            _waypoint([cube_center[0], cube_center[1], cube_center[2] + .025], 0.0),
            _waypoint([cube_center[0], cube_center[1], cube_center[2] + .025], .785398),
            _waypoint([cube_center[0], cube_center[1], cube_center[2] + .12], .785398),
            _waypoint([target[0], target[1], target[2] + .12], .785398),
            _waypoint([target[0], target[1], target[2] + .04], .785398),
            _waypoint([target[0], target[1], target[2] + .04], 0.0),
        ]
    return {
        "candidate_id": proposal["proposal_id"].replace("PROPOSAL", "CANDIDATE"),
        "family": "LAT", "seed": seed, "task_asset": workspace["task_asset"],
        "asset_manifest_sha256": workspace["asset_manifest_sha256"], "object_poses": poses,
        "metadata": {
            "status": "unqualified_proposal_starting_physical_validation",
            "workspace_receipt_sha256": workspace["receipt_sha256"],
            "center_source": proposal["center_source"],
            "scoring_center_offsets_root_local_m": proposal["scoring_center_offsets_root_local_m"],
            "abs_ik_waypoints": paths, "historical_layout_fingerprint": None,
        },
    }


def _waypoint(position: list[float], gripper: float) -> dict[str, Any]:
    return {"position_world_xyz_m": position, "gripper_position": gripper, "hold_steps": 20}


def _center(pose: Mapping[str, Any], offset: list[float]) -> list[float]:
    q, p = pose["quaternion_wxyz"], pose["position_m"]
    w, x, y, z = q; vx, vy, vz = offset
    return [p[0] + (1-2*(y*y+z*z))*vx + 2*(x*y-z*w)*vy + 2*(x*z+y*w)*vz,
            p[1] + 2*(x*y+z*w)*vx + (1-2*(x*x+z*z))*vy + 2*(y*z-x*w)*vz,
            p[2] + 2*(x*z-y*w)*vx + 2*(y*z+x*w)*vy + (1-2*(x*x+y*y))*vz]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    workspace = json.loads(args.workspace_receipt.read_text())
    proposals = propose_lat_layouts(workspace, seed=args.seed, count=args.count)
    value = {"schema_version": "sgw-01-lat-proposals-v1", "status": "proposed_unqualified",
             "workspace_receipt_sha256": workspace["receipt_sha256"], "proposals": proposals,
             "candidates": [proposal_to_candidate(row, workspace, seed=args.seed) for row in proposals]}
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()


def _half_extent(row: Mapping[str, Any]) -> list[float]:
    return [(high - low) / 2 for low, high in zip(row["bbox_env_local_min_xyz_m"], row["bbox_env_local_max_xyz_m"], strict=True)]


def _rot_y(q: list[float], v: list[float]) -> float:
    w, x, y, z = q
    vx, vy, vz = v
    return 2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz
