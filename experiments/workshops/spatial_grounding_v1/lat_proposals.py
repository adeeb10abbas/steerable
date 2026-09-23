"""Deterministic LAT proposals; acceptance requires six recorded physical trials."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fixtures import ACTION_CAP, FixtureCandidate, candidate_order, write_json
from .lat_candidate_generator import workspace_digest


def _bounds(row: Mapping[str, Any], center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = center - np.asarray(row["geometric_center_env_local_xyz_m"])
    return (
        np.asarray(row["bbox_env_local_min_xyz_m"]) + delta,
        np.asarray(row["bbox_env_local_max_xyz_m"]) + delta,
    )


def _overlap(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.all(first[0] - 0.01 < second[1]) and np.all(second[0] - 0.01 < first[1]))


def _root_pose(row: Mapping[str, Any], center: np.ndarray) -> dict[str, Any]:
    delta = center - np.asarray(row["geometric_center_env_local_xyz_m"])
    return {
        "position_m": (np.asarray(row["root_position_env_local_xyz_m"]) + delta).tolist(),
        "quaternion_wxyz": row["root_quaternion_world_wxyz"],
    }


def _waypoints(cube: np.ndarray, target: np.ndarray, origin: np.ndarray) -> list[dict[str, Any]]:
    phases = (
        (cube, 0.12, 0.0, 20),
        (cube, 0.025, 0.0, 20),
        (cube, 0.025, 0.785398, 20),
        (cube, 0.12, 0.785398, 20),
        (target, 0.12, 0.785398, 20),
        (target, 0.04, 0.785398, 20),
        (target, 0.04, 0.0, ACTION_CAP - 120),
    )
    return [{
        "position_world_xyz_m": (point + origin + np.asarray([0.0, 0.0, lift])).tolist(),
        "gripper_position": grip,
        "hold_steps": steps,
    } for point, lift, grip, steps in phases]


def propose_lat_layouts(workspace: Mapping[str, Any], *, seed: int, count: int = 100) -> list[dict[str, Any]]:
    if workspace.get("measurement_schema_version") != "sgw-01-lat-measured-workspace-v2":
        raise ValueError("LAT proposals require a measured-workspace v2 receipt")
    if workspace.get("receipt_sha256") != workspace_digest(workspace):
        raise ValueError("workspace receipt digest mismatch")
    if workspace.get("model_request_count") != 0 or workspace.get("behavioral_episode_count") != 0:
        raise ValueError("workspace must be model blind")
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("LAT proposal count must be within the frozen 1..100 cap")
    objects = workspace["objects"]
    cube, bowl, table, banana = (objects[name] for name in ("rubiks_cube", "bowl", "table", "banana"))
    for row in (cube, bowl):
        if len(row.get("com_position_env_local_xyz_m", [])) != 3:
            raise ValueError("cube and bowl require measured COM geometry")
    origin = np.asarray(workspace["environment_origin_world_xyz_m"], dtype=float)
    if not np.allclose(workspace["robot"]["base_quaternion_world_wxyz"], [1, 0, 0, 0], atol=1e-6):
        raise ValueError("LAT proposal axes require the measured robot/world-axis alignment")
    measured_cube = np.asarray(cube["geometric_center_env_local_xyz_m"], dtype=float)
    measured_bowl = np.asarray(bowl["geometric_center_env_local_xyz_m"], dtype=float)
    table_min, table_max = np.asarray(table["bbox_env_local_min_xyz_m"]), np.asarray(table["bbox_env_local_max_xyz_m"])
    cube_half = (np.asarray(cube["bbox_env_local_max_xyz_m"]) - cube["bbox_env_local_min_xyz_m"]) / 2
    bowl_half = (np.asarray(bowl["bbox_env_local_max_xyz_m"]) - bowl["bbox_env_local_min_xyz_m"]) / 2
    gap_x = float(cube_half[0] + bowl_half[0] + 0.02)
    goal_depth = max(0.05, float(cube_half[1] + bowl_half[1] + 0.02))
    low = table_min[:2] + np.asarray([bowl_half[0], goal_depth + cube_half[1]]) + 0.02
    high = table_max[:2] - np.asarray([bowl_half[0], goal_depth + cube_half[1]]) - 0.02
    if not np.all(high > low):
        raise ValueError("measured table has no conservative proposal region")
    # The cap includes geometrically rejected proposals; never refill it.
    locations = list(itertools.product(np.linspace(low[0], high[0], 20), np.linspace(low[1], high[1], 20), (-1, 1)))
    locations.sort(key=lambda point: hashlib.sha256(json.dumps([seed, *map(float, point)]).encode()).digest())
    distractor = _bounds(banana, np.asarray(banana["geometric_center_env_local_xyz_m"]))
    rows: list[dict[str, Any]] = []
    for index, (x, y, side) in enumerate(locations[:count], 1):
        bowl_center = np.asarray([x, y, measured_bowl[2]])
        cube_center = np.asarray([x + side * gap_x, y, measured_cube[2]])
        targets = [cube_center + [0, sign * goal_depth, 0] for sign in (1, -1)]
        bounds = [_bounds(bowl, bowl_center), _bounds(cube, cube_center), *[_bounds(cube, point) for point in targets]]
        rejections = []
        if any(np.any(minimum[:2] < table_min[:2] + 0.01) or np.any(maximum[:2] > table_max[:2] - 0.01)
               for minimum, maximum in bounds):
            rejections.append("initial_or_target_bounds_outside_table_clearance")
        if _overlap(bounds[0], bounds[1]):
            rejections.append("initial_cube_bowl_overlap")
        if any(_overlap(bound, distractor) for bound in bounds):
            rejections.append("initial_or_target_distractor_overlap")
        corridor = (np.minimum(bounds[2][0], bounds[3][0]), np.maximum(bounds[2][1], bounds[3][1]))
        if _overlap(corridor, distractor):
            rejections.append("transport_corridor_distractor_overlap")
        if _overlap(corridor, bounds[0]):
            rejections.append("transport_corridor_bowl_overlap")
        candidate = {
            "candidate_id": f"LAT-CANDIDATE-{index:03d}",
            "family": "LAT", "seed": seed, "task_asset": workspace["task_asset"],
            "asset_manifest_sha256": workspace["asset_manifest_sha256"],
            "object_poses": {"rubiks_cube": _root_pose(cube, cube_center), "bowl": _root_pose(bowl, bowl_center)},
            "metadata": {
                "status": "proposed_unqualified_requires_physical_validation",
                "workspace_receipt_sha256": workspace["receipt_sha256"],
                "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "center_source": "pinned_robolab_geometric_center",
                "scoring_center_offsets_root_local_m": {
                    name: objects[name]["geometric_center_offset_root_local_xyz_m"]
                    for name in ("rubiks_cube", "bowl")
                },
                "abs_ik_waypoints": {
                    name: _waypoints(cube_center, target, origin)
                    for name, target in zip(("positive", "negative"), targets, strict=True)
                },
                "controller_recipe_source": "experiments/v3/phase_e/reference_controller_runner.py",
                "eef_start_env_local_xyz_m": workspace["eef_position_env_local_xyz_m"],
                "geometric_screen": "table_and_distractor_clearance_only_not_reachability",
                "geometric_screen_status": "rejected" if rejections else "passed",
                "geometric_rejection_reasons": rejections,
                "robot_reachability_status": "unqualified_requires_native_ik_and_physical_trials",
                "historical_layout_fingerprint": None,
            },
        }
        FixtureCandidate.from_json(candidate)
        rows.append(candidate)
    candidate_order(seed, [FixtureCandidate.from_json(row) for row in rows])
    return rows


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
    candidates = propose_lat_layouts(workspace, seed=args.seed, count=args.count)
    write_json(args.output, {
        "schema_version": "sgw-01-lat-proposals-v1",
        "status": "proposed_unqualified",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "workspace_receipt_sha256": workspace["receipt_sha256"],
        "seed": args.seed,
        "candidate_cap_includes_geometric_rejections": True,
        "qualification_order": [
            item.candidate_id for item in candidate_order(args.seed, [FixtureCandidate.from_json(row) for row in candidates])
        ],
        "candidates": candidates,
    })


if __name__ == "__main__":
    main()
