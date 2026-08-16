#!/usr/bin/env python3
"""Run the frozen V3-E007 CPU-only IK volume computation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.v3.phase_e.zero_model_reachability_v3e007.kinematics import (
    KinematicChain,
    deterministic_starts,
    forward,
    load_chain,
    pose_components,
    pose_errors,
    solve_pose,
)


_CHAIN: KinematicChain | None = None
_STARTS: np.ndarray | None = None
_ORIENTATION: np.ndarray | None = None
_IK: dict[str, Any] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def centers(edges: list[float]) -> list[float]:
    return [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:], strict=True)]


def _init_worker(chain: KinematicChain, starts: np.ndarray, orientation: np.ndarray, ik: dict[str, Any]) -> None:
    global _CHAIN, _STARTS, _ORIENTATION, _IK
    _CHAIN = chain
    _STARTS = starts
    _ORIENTATION = orientation
    _IK = ik


def _solve_task(task: dict[str, Any]) -> dict[str, Any]:
    if _CHAIN is None or _STARTS is None or _ORIENTATION is None or _IK is None:
        raise RuntimeError("worker not initialized")
    solution = solve_pose(
        _CHAIN,
        target_position=task["target_position_world_m"],
        target_wxyz=_ORIENTATION,
        starts=_STARTS,
        position_tolerance_m=float(_IK["position_error_m_inclusive"]),
        orientation_tolerance_deg=float(_IK["orientation_error_deg_inclusive"]),
        max_function_evaluations=int(_IK["max_function_evaluations_per_start"]),
    )
    return {**task, **solution}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--robot-usd", type=Path, required=True)
    parser.add_argument("--r005-schedule", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {args.output_root}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty or -1 for V3-E007")
    if "torch" in sys.modules or "isaacsim" in sys.modules or "isaaclab" in sys.modules:
        raise RuntimeError("GPU/simulator module imported in CPU-only analysis")
    if sha256(args.registration) != args.registration_sha256:
        raise RuntimeError("registration SHA differs")
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    if registration["status"] != "frozen_before_reachability_computation":
        raise RuntimeError("registration is not frozen")
    if sha256(args.robot_usd) != registration["ik"]["robot_asset"]["sha256"]:
        raise RuntimeError("robot USD differs")
    if args.robot_usd.stat().st_size != registration["ik"]["robot_asset"]["bytes"]:
        raise RuntimeError("robot USD byte count differs")

    chain = load_chain(args.robot_usd)
    reset_q = np.asarray(registration["workspace"]["reset_arm_joint_position_rad"], dtype=np.float64)
    reset_transform = forward(chain, reset_q)
    _, orientation = pose_components(reset_transform)
    starts = deterministic_starts(chain, reset_q, int(registration["ik"]["deterministic_starts"]))

    r005_binding = next(
        row for row in registration["source_bindings"] if row["path"].endswith("v3e006_r005/gates/candidate_schedule.json")
    )
    if sha256(args.r005_schedule) != r005_binding["sha256"]:
        raise RuntimeError("R005 schedule differs")
    schedule = json.loads(args.r005_schedule.read_text(encoding="utf-8"))
    fk_validation = []
    for row in schedule["known_reachable_diagnostics"]:
        source = row["source"]
        transform = forward(chain, source["joint_position_rad"][:7])
        position_error, orientation_error = pose_errors(
            transform,
            source["base_link_position_world_m"],
            source["base_link_quaternion_world_wxyz"],
        )
        passed = position_error <= 1e-5 and orientation_error <= 1e-3
        fk_validation.append(
            {
                "diagnostic_index_one_based": row["diagnostic_index_one_based"],
                "position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "passed": passed,
            }
        )
    if len(fk_validation) != 4 or not all(row["passed"] for row in fk_validation):
        raise RuntimeError("exact USD FK failed known-pose validation")

    domain = registration["workspace"]["reference_relative_domain_m"]
    x_centers = centers(domain["forward_x_edges"])
    y_centers = centers(domain["absolute_lateral_y_edges"])
    z_centers = centers(domain["vertical_z_edges"])
    voxel_volume = (
        (domain["forward_x_edges"][1] - domain["forward_x_edges"][0])
        * (domain["absolute_lateral_y_edges"][1] - domain["absolute_lateral_y_edges"][0])
        * (domain["vertical_z_edges"][1] - domain["vertical_z_edges"][0])
    )
    tasks: list[dict[str, Any]] = []
    for layout_index, layout in enumerate(registration["layouts"]):
        reference = np.asarray(layout["reference_position_world_m"], dtype=np.float64)
        for side in ("left", "right"):
            sign = 1.0 if side == "left" else -1.0
            for x_index, x_value in enumerate(x_centers):
                for y_index, y_value in enumerate(y_centers):
                    if abs(x_value) > y_value:
                        continue
                    for z_index, z_value in enumerate(z_centers):
                        target = reference + np.asarray([x_value, sign * y_value, z_value])
                        tasks.append(
                            {
                                "layout_index": layout_index,
                                "layout_id": layout["layout_id"],
                                "family": layout["family"],
                                "level": layout["level"],
                                "side": side,
                                "voxel_index": [x_index, y_index, z_index],
                                "relative_center_m": [x_value, sign * y_value, z_value],
                                "target_position_world_m": target.tolist(),
                            }
                        )

    args.output_root.mkdir(parents=True)
    raw_path = args.output_root / "workspace_points.jsonl"
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(chain, starts, orientation, registration["ik"]),
    ) as executor:
        rows = list(executor.map(_solve_task, tasks, chunksize=4))
    rows.sort(key=lambda row: (row["layout_index"], row["side"], row["voxel_index"]))
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summaries = []
    for layout in registration["layouts"]:
        side_rows = {
            side: [row for row in rows if row["layout_id"] == layout["layout_id"] and row["side"] == side]
            for side in ("left", "right")
        }
        side_summary = {}
        for side, group in side_rows.items():
            feasible = [row for row in group if row["feasible"]]
            side_summary[side] = {
                "voxel_count": len(group),
                "feasible_voxel_count": len(feasible),
                "feasible_fraction": len(feasible) / len(group),
                "feasible_volume_m3": len(feasible) * voxel_volume,
                "median_minimum_normalized_joint_limit_margin": float(
                    np.median([row["minimum_normalized_joint_limit_margin"] for row in feasible])
                ) if feasible else None,
                "median_minimum_translational_jacobian_singular_value_m_per_rad": float(
                    np.median([row["minimum_translational_jacobian_singular_value_m_per_rad"] for row in feasible])
                ) if feasible else None,
            }
        summaries.append(
            {
                **layout,
                "sides": side_summary,
                "right_minus_left_feasible_volume_m3": side_summary["right"]["feasible_volume_m3"] - side_summary["left"]["feasible_volume_m3"],
                "right_minus_left_feasible_fraction": side_summary["right"]["feasible_fraction"] - side_summary["left"]["feasible_fraction"],
            }
        )
    summary = {
        "schema_version": "vla-wam-shared-v3e007-zero-model-reachability-raw-summary-v1",
        "amendment_id": "V3-E007",
        "registration_sha256": args.registration_sha256,
        "robot_usd_sha256": sha256(args.robot_usd),
        "runtime": {
            "python": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "workers": args.workers,
            "torch_imported": "torch" in sys.modules,
            "isaacsim_imported": "isaacsim" in sys.modules,
            "isaaclab_imported": "isaaclab" in sys.modules,
        },
        "learned_model_request_count": 0,
        "behavioral_episode_count": 0,
        "fk_validation": fk_validation,
        "reset_base_link_pose_world_wxyz": [*reset_transform[:3, 3].tolist(), *orientation.tolist()],
        "voxel_volume_m3": voxel_volume,
        "point_count": len(rows),
        "layouts": summaries,
        "raw_points": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
        },
        "claim_boundary": "Strict joint-limit pose-IK volume only; no collision, dynamics, policy, or task-success inference.",
    }
    summary_path = args.output_root / "workspace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "sha256": sha256(summary_path), "points": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

