"""Build the finite, contact-consistent V3-E006-R002 candidate schedule.

R002 uses only committed historical E004 state evidence.  It never evaluates a
live simulator scene and never makes a policy/model request.  Unlike R001, it
does not average robot joints independently of the cube.  It preserves one
observed cube-in-EEF SE(3) contact transform, constructs a direction-balanced
target cube pose on the robot midline, and derives the corresponding EEF IK
target by exact SE(3) composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "vla-wam-shared-v3e006-r002-contact-consistent-candidate-schedule-v1"
REGISTRATION_SCHEMA = "vla-wam-shared-v3e006-r002-prospective-construction-repair-registration-v1"
REGISTRATION_STATUS = "prospectively_registered_before_any_r002_live_candidate_or_model_request"
REPAIR_ID = "V3-E006-R002"
PREDECESSOR_ID = "V3-E006-R001"
MAXIMUM_CANDIDATE_PAIRS = 8
MIRROR = np.diag([1.0, -1.0, 1.0])
ANCHORS = {
    "canonical_grasp": {
        "environment_seed": 9521,
        "left": {"state_capture_index": 30, "hdf5_index": 104},
        "right": {"state_capture_index": 31, "hdf5_index": 105},
    },
    "canonical_carry": {
        "environment_seed": 9442,
        "left": {"state_capture_index": 39, "hdf5_index": 113},
        "right": {"state_capture_index": 38, "hdf5_index": 112},
    },
}

# The first four candidates use direct contact-consistent initialization.  The
# second four are the prospectively registered normal-contact approach fallback.
# Within each method, source anchoring is balanced before same-side variants.
VARIANTS = (
    (1, "direct_contact_initialization", "left_observed", "reflected_right_observed"),
    (2, "direct_contact_initialization", "reflected_right_observed", "left_observed"),
    (3, "direct_contact_initialization", "left_observed", "left_observed"),
    (4, "direct_contact_initialization", "reflected_right_observed", "reflected_right_observed"),
    (5, "open_approach_close_lift", "left_observed", "reflected_right_observed"),
    (6, "open_approach_close_lift", "reflected_right_observed", "left_observed"),
    (7, "open_approach_close_lift", "left_observed", "left_observed"),
    (8, "open_approach_close_lift", "reflected_right_observed", "reflected_right_observed"),
)


class ScheduleError(RuntimeError):
    """The immutable R002 schedule could not be built exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScheduleError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_bound_json(path: Path, expected: Mapping[str, Any]) -> Any:
    path = Path(path).resolve()
    require(path.is_file(), f"bound JSON is missing: {path}")
    require(path.stat().st_size == expected["bytes"], f"bound JSON byte count changed: {path}")
    require(sha256_file(path) == expected["sha256"], f"bound JSON digest changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_quaternion_wxyz(value: Sequence[float]) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    require(q.shape == (4,) and bool(np.all(np.isfinite(q))), "quaternion is malformed")
    norm = float(np.linalg.norm(q))
    require(norm > 1e-12, "quaternion has zero norm")
    q = q / norm
    for component in q:
        if abs(float(component)) > 1e-15:
            if float(component) < 0.0:
                q = -q
            break
    return q


def quaternion_to_matrix_wxyz(value: Sequence[float]) -> np.ndarray:
    w, x, y, z = canonical_quaternion_wxyz(value)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(value: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.shape == (3, 3) and bool(np.all(np.isfinite(matrix))), "rotation matrix is malformed")
    require(float(np.linalg.det(matrix)) > 0.999999, "rotation matrix is not proper")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            q = np.asarray([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                            (matrix[0, 1] + matrix[1, 0]) / scale,
                            (matrix[0, 2] + matrix[2, 0]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            q = np.asarray([(matrix[0, 2] - matrix[2, 0]) / scale,
                            (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                            (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            q = np.asarray([(matrix[1, 0] - matrix[0, 1]) / scale,
                            (matrix[0, 2] + matrix[2, 0]) / scale,
                            (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    return canonical_quaternion_wxyz(q)


def pose_matrix(position: Sequence[float], quaternion_wxyz: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix_wxyz(quaternion_wxyz)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    require(bool(np.all(np.isfinite(transform))), "pose contains a nonfinite number")
    return transform


def pose_record(transform: np.ndarray) -> dict[str, list[float]]:
    return {
        "position_world_m": [float(value) for value in transform[:3, 3]],
        "quaternion_world_wxyz": [float(value) for value in matrix_to_quaternion_wxyz(transform[:3, :3])],
    }


def relative_pose_record(transform: np.ndarray) -> dict[str, list[float]]:
    return {
        "translation_m": [float(value) for value in transform[:3, 3]],
        "quaternion_wxyz": [float(value) for value in matrix_to_quaternion_wxyz(transform[:3, :3])],
    }


def reflect_world_pose(transform: np.ndarray) -> np.ndarray:
    reflected = np.eye(4, dtype=np.float64)
    reflected[:3, :3] = MIRROR @ transform[:3, :3] @ MIRROR
    reflected[:3, 3] = MIRROR @ transform[:3, 3]
    require(abs(float(np.linalg.det(reflected[:3, :3])) - 1.0) < 1e-10, "reflection produced an improper rotation")
    return reflected


def geodesic_midpoint_rotation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    q_left = matrix_to_quaternion_wxyz(left)
    q_right = matrix_to_quaternion_wxyz(right)
    dot = float(np.dot(q_left, q_right))
    if dot < 0.0:
        q_right = -q_right
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-12:
        midpoint = canonical_quaternion_wxyz(q_left + q_right)
    else:
        theta = math.acos(dot)
        scale = math.sin(0.5 * theta) / math.sin(theta)
        midpoint = canonical_quaternion_wxyz(scale * q_left + scale * q_right)
    return quaternion_to_matrix_wxyz(midpoint)


def source_transforms(source: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    eef = pose_matrix(source["eef_position_world_m"], source["eef_quaternion_world_wxyz"])
    cube_pose = source["cube_pose_world_wxyz"]
    cube = pose_matrix(cube_pose[:3], cube_pose[3:])
    return eef, cube


def _assert_anchor(stage: str, row: Mapping[str, Any]) -> None:
    anchor = ANCHORS[stage]
    require(row["environment_seed"] == anchor["environment_seed"], f"{stage} seed differs from frozen anchor")
    for side in ("left", "right"):
        source = row["source_states"][side]
        expected = anchor[side]
        require(
            source["state_capture_index"] == expected["state_capture_index"]
            and source["hdf5_index"] == expected["hdf5_index"],
            f"{stage} {side} historical row differs from frozen anchor",
        )
        require(len(source["joint_position_rad"]) == 13, f"{stage} {side} joint vector differs")
        require(all(math.isfinite(float(v)) for v in source["joint_position_rad"]), f"{stage} {side} joint vector is nonfinite")


def derive_stage(stage: str, row: Mapping[str, Any], selector: str, reset_cube: np.ndarray) -> dict[str, Any]:
    _assert_anchor(stage, row)
    left_eef, left_cube = source_transforms(row["source_states"]["left"])
    right_eef_raw, right_cube_raw = source_transforms(row["source_states"]["right"])
    right_eef = reflect_world_pose(right_eef_raw)
    right_cube = reflect_world_pose(right_cube_raw)
    contacts = {
        "left_observed": np.linalg.inv(left_eef) @ left_cube,
        "reflected_right_observed": np.linalg.inv(right_eef) @ right_cube,
    }
    require(selector in contacts, f"unsupported contact selector: {selector}")
    contact = contacts[selector]

    target_cube = np.eye(4, dtype=np.float64)
    target_cube[:3, 3] = 0.5 * (left_cube[:3, 3] + right_cube[:3, 3])
    target_cube[1, 3] = 0.0
    target_cube[:3, :3] = geodesic_midpoint_rotation(left_cube[:3, :3], right_cube[:3, :3])
    target_eef = target_cube @ np.linalg.inv(contact)
    reconstructed_cube = target_eef @ contact
    residual_position = float(np.linalg.norm(reconstructed_cube[:3, 3] - target_cube[:3, 3]))
    residual_rotation = float(np.linalg.norm(reconstructed_cube[:3, :3] - target_cube[:3, :3], ord="fro"))
    require(abs(float(target_cube[1, 3])) == 0.0, "target cube is not exactly on the centerline")
    require(residual_position <= 1e-12 and residual_rotation <= 1e-12, "SE(3) reconstruction is not exact")

    contact_eef_at_reset = reset_cube @ np.linalg.inv(contact)
    approach_eef_at_reset = contact_eef_at_reset.copy()
    approach_eef_at_reset[2, 3] += 0.060
    selected_side = "left" if selector == "left_observed" else "right"
    selected_source = row["source_states"][selected_side]
    return {
        "stage": stage,
        "source_environment_seed": row["environment_seed"],
        "both_direction_sources": {
            "left": row["source_states"]["left"],
            "right": row["source_states"]["right"],
        },
        "reflection_definition": {
            "matrix_M": MIRROR.tolist(),
            "position": "p_reflected = M @ p",
            "orientation": "R_reflected = M @ R @ M",
            "right_world_eef_reflected": pose_record(right_eef),
            "right_world_cube_reflected": pose_record(right_cube),
        },
        "contact_transform_selector": selector,
        "selected_observed_cube_in_eef_transform": relative_pose_record(contact),
        "unselected_observed_cube_in_eef_transform": relative_pose_record(
            contacts["reflected_right_observed" if selector == "left_observed" else "left_observed"]
        ),
        "target_cube_pose": pose_record(target_cube),
        "centerline_constrained_eef_ik_target": pose_record(target_eef),
        "se3_reconstruction": {
            "reconstructed_cube_pose": pose_record(reconstructed_cube),
            "position_residual_m": residual_position,
            "rotation_matrix_frobenius_residual": residual_rotation,
            "cube_midline_residual_m": abs(float(reconstructed_cube[1, 3])),
        },
        "open_approach_targets": {
            "contact_eef_at_exact_e004_reset_cube": pose_record(contact_eef_at_reset),
            "approach_eef_at_exact_e004_reset_cube": pose_record(approach_eef_at_reset),
            "world_vertical_clearance_m": 0.060,
        },
        "selected_historical_gripper_joint_position_rad": [
            float(value) for value in selected_source["joint_position_rad"][7:]
        ],
        "selected_historical_source_side": selected_side,
        "historical_source_runtime_assertions": {
            "hdf5_action_last_component_exact": 1.0,
            "state_capture_object_grabbed": True,
            "source_raw_hdf5_and_state_capture_rehashed_before_AppLauncher": True,
        },
    }


def build(
    *,
    registration_path: Path,
    r001_schedule_path: Path,
    r001_results_path: Path,
    r001_evidence_path: Path,
    predecessor_closure_path: Path,
    e004_reset_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    registration_path = Path(registration_path).resolve()
    output_path = Path(output_path).resolve()
    require(not output_path.exists(), f"refusing to overwrite schedule: {output_path}")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    require(registration.get("schema_version") == REGISTRATION_SCHEMA, "registration schema differs")
    require(registration.get("status") == REGISTRATION_STATUS, "registration status differs")
    require(registration.get("repair_amendment_id") == REPAIR_ID, "registration repair ID differs")
    require(registration.get("counts_at_registration") == {
        "r002_live_candidate_evaluations": 0,
        "model_requests": 0,
        "behavioral_episodes": 0,
    }, "registration counts are nonzero")
    require(registration["candidate_search"]["maximum_candidate_pairs"] == MAXIMUM_CANDIDATE_PAIRS, "candidate budget differs")
    require(registration["candidate_search"]["variant_order"] == [
        {"rank": rank, "method": method, "grasp_contact": grasp, "carry_contact": carry}
        for rank, method, grasp, carry in VARIANTS
    ], "variant order differs from code")

    frozen = registration["frozen_inputs"]
    predecessor_closure = load_bound_json(predecessor_closure_path, frozen["predecessor_closure_binding"])
    r001_schedule = load_bound_json(r001_schedule_path, frozen["r001_candidate_schedule"])
    r001_results = load_bound_json(r001_results_path, frozen["r001_results"])
    r001_evidence = load_bound_json(r001_evidence_path, frozen["r001_evidence_manifest"])
    e004_reset = load_bound_json(e004_reset_path, frozen["e004_full_reset_reference"])
    require(r001_schedule.get("repair_amendment_id") == PREDECESSOR_ID, "R001 schedule identity differs")
    require(
        predecessor_closure.get("status")
        == "original_and_r001_predecessors_byte_identical_before_r002_live_candidate_or_model_request",
        "predecessor closure status differs",
    )
    require(predecessor_closure.get("r001_exhaustion_closure_commit") == registration["source_ancestry"]["r001_exhaustion_closure_commit"], "predecessor commit differs")
    require(r001_results.get("status") == "r001_candidate_budget_exhausted_no_valid_state_pair", "R001 was not exhausted")
    require(r001_results.get("candidate_gate_passed") is False, "R001 unexpectedly passed")
    require(r001_results.get("model_request_count") == r001_results.get("behavioral_episode_count") == 0, "R001 counts differ")
    require(r001_evidence.get("status") == "hash_bound_r001_finite_exhaustion_zero_model_zero_behavior", "R001 evidence status differs")
    require(r001_evidence["raw_attempt"]["result"] == r001_results["raw_result"], "R001 raw result bindings disagree")

    reset_cube_row = e004_reset["rigid_objects"]["rubiks_cube"]
    reset_cube = pose_matrix(
        reset_cube_row["root_position"]["values"],
        reset_cube_row["root_quaternion_wxyz"]["values"],
    )
    anchor_rows = {
        stage: r001_schedule["candidate_pairs"][0][stage]
        for stage in ("canonical_grasp", "canonical_carry")
    }
    candidate_pairs = []
    for rank, method, grasp_selector, carry_selector in VARIANTS:
        candidate_pairs.append(
            {
                "candidate_rank": rank,
                "construction_method": method,
                "canonical_grasp": derive_stage(
                    "canonical_grasp", anchor_rows["canonical_grasp"], grasp_selector, reset_cube
                ),
                "canonical_carry": derive_stage(
                    "canonical_carry", anchor_rows["canonical_carry"], carry_selector, reset_cube
                ),
            }
        )

    schedule = {
        "schema_version": SCHEMA,
        "repair_amendment_id": REPAIR_ID,
        "status": "frozen_before_any_r002_live_candidate_or_model_request",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r002_live_candidate_evaluation_count": 0,
        "repair_registration": binding(registration_path),
        "r001_predecessor": {
            "closure_binding": binding(predecessor_closure_path),
            "candidate_schedule": binding(r001_schedule_path),
            "results": binding(r001_results_path),
            "evidence_manifest": binding(r001_evidence_path),
            "outcome": "finite_r001_budget_exhausted_without_valid_state_pair",
        },
        "e004_full_reset_reference": binding(e004_reset_path),
        "original_v3e006_closure_binding": frozen["original_v3e006_closure_binding"],
        "unchanged_gate_bindings": {
            name: frozen[name]
            for name in ("state_contract", "ood_reference", "ood_freeze")
        },
        "candidate_budget": MAXIMUM_CANDIDATE_PAIRS,
        "candidate_pairs": candidate_pairs,
        "selection_rule": registration["candidate_search"],
        "construction_contract": registration["construction_contract"],
        "historical_policy_provenance_disclosure": (
            "Both-direction source transforms are from committed successful historical E004 pi0.5 trajectories. "
            "R002 schedule generation is repair-behavior-blind and makes no new model request; it is not "
            "model-blind provenance unqualified."
        ),
        "schedule_canonical_sha256": None,
    }
    schedule["schedule_canonical_sha256"] = canonical_json_sha256(
        {**schedule, "schedule_canonical_sha256": None}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schedule, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--r001-schedule", type=Path, required=True)
    parser.add_argument("--r001-results", type=Path, required=True)
    parser.add_argument("--r001-evidence", type=Path, required=True)
    parser.add_argument("--predecessor-closure", type=Path, required=True)
    parser.add_argument("--e004-reset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        registration_path=args.registration,
        r001_schedule_path=args.r001_schedule,
        r001_results_path=args.r001_results,
        r001_evidence_path=args.r001_evidence,
        predecessor_closure_path=args.predecessor_closure,
        e004_reset_path=args.e004_reset,
        output_path=args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha256_file(args.output),
                      "candidate_pairs": len(result["candidate_pairs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
