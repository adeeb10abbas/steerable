"""Derive a virtual grasp frame from recorded pad geometry, not nominal offsets."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .fixtures import ACTION_CAP, FixtureCandidate
from .lat_candidate_generator import workspace_digest
from .lat_workspace_capture import _root_local_offset, _rotate_wxyz
from .recorder import atomic_json


SCHEMA = "sgw-01-lat-closed-pad-midpoint-v1"


def derive_calibration(workspace: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    if workspace.get("receipt_sha256") != workspace_digest(workspace):
        raise ValueError("gripper workspace digest mismatch")
    if workspace.get("model_request_count") != 0 or workspace.get("behavioral_episode_count") != 0:
        raise ValueError("gripper workspace must be model blind")
    probe = workspace["gripper_calibration"]
    if (probe["status"] != "measured_empty_gripper_motion_not_fixture_qualification"
            or probe["issued_controller_commands"] != 60 or probe["observed_controller_actions"] != 60
            or probe["model_requests"] != 0 or probe["behavioral_episodes"] != 0):
        raise ValueError("virtual grasp frame requires the complete empty-gripper probe")
    records = probe["records"]
    if len(records) != 61 or any(
        row["action_step"] != index or not np.isclose(row["sim_time_s"], index * probe["control_step_dt_s"], atol=1e-5, rtol=0)
        for index, row in enumerate(records)
    ):
        raise ValueError("gripper records lack the complete physical-time sequence")
    pads = {}
    for side in ("left", "right"):
        matches = [
            row for row in probe["finger_geometry"]
            if row["body"] == f"{side}_inner_finger"
            and row["prim"].endswith("/Defeatured_2F_85_PAD_OPEN_fingertipsstep")
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one measured {side} fingertip pad")
        pads[side] = matches[0]
    samples = {}
    for index, label, expected_joint in ((0, "open", 0), (30, "closed", 0.785398), (60, "reopened", 0)):
        record = records[index]
        joint_index = record["joint_names"].index("finger_joint")
        if not np.isclose(record["joint_position_rad"][joint_index], expected_joint, atol=1e-4, rtol=0):
            raise ValueError(f"gripper did not reach the measured {label} state")
        bodies = record["robot_body_frames"]["bodies"]
        flange = bodies["base_link"]
        positions = []
        for pad in pads.values():
            body = bodies[pad["body"]]
            world = np.asarray(
                _rotate_wxyz(body["quaternion_world_wxyz"], pad["body_local_bounds_center_m"]), dtype=np.float64,
            )
            world += body["position_world_xyz_m"]
            positions.append(_root_local_offset(
                flange["position_world_xyz_m"], flange["quaternion_world_wxyz"], world,
            ))
        if not np.isfinite(positions).all():
            raise ValueError("pad geometry is nonfinite")
        samples[label] = {
            "action_step": index, "pad_centers_flange_xyz_m": positions,
            "midpoint_flange_xyz_m": np.mean(positions, axis=0).tolist(),
            "pad_center_separation_m": float(np.linalg.norm(np.subtract(*positions))),
        }
    result = {
        "schema_version": SCHEMA,
        "status": "prospective_static_controller_not_fixture_qualified",
        "source_workspace_sha256": source_sha256,
        "source_workspace_content_digest": workspace["receipt_sha256"],
        "derivation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "robolab_commit": workspace["robolab_commit"],
        "robot_asset": workspace["robot"]["asset_usd"],
        "pad_geometry": pads,
        "measured_samples": samples,
        "virtual_tcp_flange_xyz_m": samples["closed"]["midpoint_flange_xyz_m"],
        "lift_height_m": 0.12,
        "phase_hold_steps": [20, 20, 20, 20, 20, 20, 20, ACTION_CAP - 140],
        "phase_names": ["approach", "descend", "close", "lift", "transport", "lower", "open", "retreat_and_settle"],
        "model_requests": 0, "behavioral_episodes": 0,
        "claim_boundary": "Virtual TCP at the measured closed visual-pad midpoint; not a measured contact surface, fixture acceptance, or learned-policy result.",
    }
    result["receipt_sha256"] = workspace_digest(result)
    return result


def calibrated_actions(
    candidate: FixtureCandidate, goal_sign: int, calibration: dict[str, Any], *,
    flange_quaternion_world_wxyz: np.ndarray, environment_origin_world_xyz_m: np.ndarray,
    robot_root_world_xyz_m: np.ndarray,
) -> list[np.ndarray]:
    if (calibration.get("schema_version") != SCHEMA
            or calibration.get("receipt_sha256") != workspace_digest(calibration)):
        raise ValueError("invalid grasp calibration identity")
    if candidate.family != "LAT" or goal_sign not in (-1, 1):
        raise ValueError("calibrated controller requires a LAT goal")
    key = "positive" if goal_sign == 1 else "negative"
    cube = np.asarray(candidate.scoring_poses()["rubiks_cube"].position_m) + environment_origin_world_xyz_m
    frozen_target = np.asarray(candidate.metadata["abs_ik_waypoints"][key][-1]["position_world_xyz_m"])
    target = np.asarray([frozen_target[0], frozen_target[1], cube[2]])
    offset = np.asarray(_rotate_wxyz(flange_quaternion_world_wxyz, calibration["virtual_tcp_flange_xyz_m"]))
    lift = np.asarray([0, 0, calibration["lift_height_m"]])
    points = (cube + lift, cube, cube, cube + lift, target + lift, target, target, target + lift)
    grips = (0, 0, 0.785398, 0.785398, 0.785398, 0.785398, 0, 0)
    actions = []
    for point, grip, count in zip(points, grips, calibration["phase_hold_steps"], strict=True):
        position = point - offset - robot_root_world_xyz_m
        command = np.concatenate((position, flange_quaternion_world_wxyz, [grip])).astype(np.float32).reshape(1, 8)
        if not np.isfinite(command).all() or type(count) is not int or count < 1:
            raise ValueError("invalid calibrated command")
        actions.extend(command.copy() for _ in range(count))
    if len(actions) != ACTION_CAP:
        raise ValueError("calibrated plan must contain exactly 450 actions")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite grasp calibration: {args.output}")
    raw = args.workspace_receipt.read_bytes()
    atomic_json(args.output, derive_calibration(json.loads(raw), hashlib.sha256(raw).hexdigest()))


if __name__ == "__main__":
    main()
