#!/usr/bin/env python3
"""Freeze V3-E006-R009 collision-pinch construction before any live run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r009.pinch_geometry import (
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r009.predecessor_contract import (
    R008_CLOSURE_COMMIT,
    R008_RESULTS_SHA256,
    validate_r008_exhaustion_closure,
)


ROOT = Path(__file__).resolve().parents[1]
R008 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008"
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009"
COLLISION_ASSET_BINDINGS = {
    "robot_usd": {
        "path": "/data/users/ali/vla_wam/external/RoboLab-11142d4/assets/robots/franka_robotiq_2f_85_flattened.usd",
        "bytes": 14156362,
        "sha256": "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41",
    },
    "cube_usd": {
        "path": "/data/users/ali/vla_wam/external/RoboLab-11142d4/assets/objects/hot3d/rubiks_cube.usd",
        "bytes": 682045,
        "sha256": "d9497c0a01c51df76d8c69e595ab91637fa028140f2656628549283267e65024",
    },
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        retained_path = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        retained_path = str(path)
    return {"path": retained_path, "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-at-utc", required=True)
    args = parser.parse_args()
    if ARTIFACT.exists():
        parser.error(f"refusing to overwrite prospective R009 freeze: {ARTIFACT}")
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{R008_CLOSURE_COMMIT}^{{commit}}"],
        check=False,
    ).returncode:
        parser.error("R008 closure commit is absent")

    registration_path = R008 / "repair_registration.json"
    schedule_path = R008 / "gates/candidate_schedule.json"
    source_gate_path = R008 / "source_push_gate.json"
    results_path = R008 / "results/results.json"
    manifest_path = R008 / "results/evidence_manifest.json"
    memo_path = R008 / "results/DECISION_MEMO.md"
    receipt_path = R008 / "results/target_validation_receipt.json"
    amendment_path = R008 / "postexecution_validator_amendment_v1.json"
    r008_registration = load(registration_path)
    r008_schedule = load(schedule_path)
    r008_results = load(results_path)
    if sha256(results_path) != R008_RESULTS_SHA256:
        parser.error("R008 result digest differs")
    validate_r008_exhaustion_closure(r008_results)

    pinch = {
        "algorithm_version": "uniform-tensor-reconstructed-collision-pinch-acquisition-v1",
        "applies_identically_to": "all four ranks and canonical_grasp/canonical_carry",
        "robot_body_resolution": "under the unique env_0 robot prim, require exactly one tensor rigid body whose path ends left_inner_finger and exactly one ending right_inner_finger; each selected owner must have UsdPhysics.RigidBodyAPI",
        "collision_resolution": "once before candidate actions, resolve enabled UsdPhysics.CollisionAPI inventories and transform each finite collision-local corner into its exact owning rigid-body frame; reject nested or different rigid-body boundaries",
        "dynamic_geometry_source": "at every action step reconstruct pad/cube centers and AABBs only from IsaacLab tensor rigid-body/root poses minus the explicitly retained scene env origin plus the frozen body-local corners",
        "dynamic_usd_world_bounds_used": False,
        "controller_coordinate_semantics": "env-local world-axis positions; tensor world position minus scene env origin; quaternions remain world-axis WXYZ",
        "left_inner_finger_body_suffix": "left_inner_finger",
        "right_inner_finger_body_suffix": "right_inner_finger",
        "target_midpoint_rule": "live tensor-reconstructed cube collision center in env-local world-axis coordinates",
        "approach_clearance_rule": "two times live tensor-reconstructed cube AABB half-extent z",
        "open_approach_target": "live cube collision center plus env-local world-z approach clearance, with the frozen rank-stage acquisition quaternion",
        "open_descent_target": "live cube collision center, with the same frozen acquisition quaternion",
        "normal_close_target": "live cube collision center, with unchanged normal binary close action",
        "closed_vertical_lift_target": "reset cube collision center with z replaced by target cube collision-center z; acquisition quaternion remains fixed",
        "closed_stage_transport_target": "target cube collision center; quaternion shortest-arc interpolates from acquisition quaternion to the unchanged frozen stage base quaternion",
        "target_cube_collision_center": "transform the frozen cube-body-local collision-center offset by the unchanged frozen target cube env-local world-axis pose",
        "translation_gain": 0.2,
        "rotation_gain": 0.2,
        "translation_cap_m_per_step": 0.002,
        "rotation_cap_deg_per_step": 2.0,
        "open_approach_steps": 180,
        "open_descent_steps": 180,
        "normal_close_steps": 120,
        "closed_vertical_lift_steps": 240,
        "closed_stage_transport_steps": 300,
        "gripper_open_command": 0.0,
        "gripper_closed_command": 1.0,
        "contact_or_grab_conditioned_branch": False,
        "early_stop": False,
        "contact_and_grab_trace_semantics": "diagnostic-only; never changes, skips, rejects, stops, or branches construction",
        "parameter_basis": "one symmetric collision-geometry controller using the same generic gain/caps as R008 and fixed phase lengths before R009; no offset or count is selected by rank/stage outcome",
    }
    validate_contract(pinch)
    pinch_sha = hashlib.sha256(canonical_bytes(pinch)).hexdigest()
    handoff = deepcopy(r008_schedule["joint_handoff_contract"])
    handoff.update(
        {
            "algorithm_version": "single-captured-q-normal-actuator-handoff-r009-v1",
            "capture": (
                "capture all 13 observed robot joint positions exactly once after "
                "closed_stage_transport step 300 (candidate action step 1020; episode step 1095)"
            ),
            "settle": (
                "advance exactly 600 normal physics/termination/observation steps under "
                "the captured invariant joint target without Cartesian action processing "
                "or state writes"
            ),
        }
    )
    handoff_sha = hashlib.sha256(canonical_bytes(handoff)).hexdigest()
    lifecycle = {
        "fresh_reset_steps": 75,
        "open_approach_steps": 180,
        "open_descent_steps": 180,
        "normal_close_steps": 120,
        "closed_vertical_lift_steps": 240,
        "closed_stage_transport_steps": 300,
        "joint_equilibrium_settle_steps": 600,
        "worst_case_steps": 1695,
        "registered_max_episode_length_steps": 1800,
        "registered_margin_steps": 105,
        "required_step_dt_s": 1.0 / 15.0,
        "registered_episode_length_s": 120.0,
        "only_construction_environment_timeout_changes": True,
        "behavioral_horizon_unchanged": True,
    }
    lifecycle_sha = hashlib.sha256(canonical_bytes(lifecycle)).hexdigest()

    predecessor = {
        "closure_commit": R008_CLOSURE_COMMIT,
        "registration": bind(registration_path),
        "candidate_schedule": bind(schedule_path),
        "source_push_gate": bind(source_gate_path),
        "results": bind(results_path),
        "evidence_manifest": bind(manifest_path),
        "decision_memo": bind(memo_path),
        "target_validation_receipt": bind(receipt_path),
        "postexecution_validator_amendment": bind(amendment_path),
        "raw_result": r008_results["raw_result"],
        "outcome": "four diagnostics passed; all eight stages had zero normal/intended gripper-cube contact and zero physics passes; zero model/behavior",
    }
    frozen_inputs = deepcopy(r008_registration["frozen_inputs"])
    frozen_inputs.update(
        {
            "r008_results": bind(results_path),
            "r008_evidence_manifest": bind(manifest_path),
            "r008_target_validation_receipt": bind(receipt_path),
            "r008_validator_amendment": bind(amendment_path),
            "r009_robot_collision_usd": deepcopy(
                COLLISION_ASSET_BINDINGS["robot_usd"]
            ),
            "r009_cube_collision_usd": deepcopy(
                COLLISION_ASSET_BINDINGS["cube_usd"]
            ),
        }
    )
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r009-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "original_amendment_id": "V3-E006",
        "repair_amendment_id": "V3-E006-R009",
        "predecessor_repair_amendment_id": "V3-E006-R008",
        "registered_at_utc": args.registered_at_utc,
        "status": "prospectively_registered_before_any_r009_live_diagnostic_candidate_or_model_request",
        "counts_at_registration": {
            "r009_live_diagnostics": 0,
            "r009_live_candidate_evaluations": 0,
            "model_requests": 0,
            "behavioral_episodes": 0,
        },
        "user_authorized_override": "Continue solving canonical-state construction after valid R008 exhaustion; behavior remains blocked until a state pair passes unchanged gates.",
        "engineering_rationale_only": (
            "Across all eight R008 stages, normal gripper contact and intended cube-gripper force were zero and the cube remained table-supported. "
            "R008 therefore attempted transport without first establishing acquisition. R009 changes only upstream construction: it aligns the tensor-reconstructed unique inner-finger collision-pad midpoint with the tensor-reconstructed live cube collision center before closing, then moves that pinch midpoint through one fixed lift/transport schedule."
        ),
        "r008_predecessor": predecessor,
        "pinch_geometry_contract": pinch,
        "pinch_geometry_contract_sha256": pinch_sha,
        "joint_handoff_contract": handoff,
        "joint_handoff_contract_sha256": handoff_sha,
        "construction_lifecycle_contract": lifecycle,
        "construction_lifecycle_contract_sha256": lifecycle_sha,
        "unchanged_science": {
            "r008_candidate_sources_target_cube_poses_rank_order_and_first_pass_rule_identical": True,
            "diagnostics_identical": True,
            "physics_ood_camera_companion_contact_midline_gates_and_thresholds_identical": True,
            "model_checkpoint_runtime_prompts_sample_size_behavioral_horizon_and_scorer_unchanged": True,
        },
        "prohibitions": [
            "no post-reset joint/object pose/velocity state write",
            "no weld/attachment/collision suppression/force injection",
            "no per-rank/stage/side geometry offset, gain, rate cap, or phase count",
            "no contact/grab-conditioned branch, early stop, skip, rejection, or command change",
            "no adaptive retry, threshold or target change",
            "no model request or behavioral episode",
        ],
        "frozen_inputs": frozen_inputs,
        "release_boundary": "One zero-model diagnostic and conditional finite four-pair search only; no inference or behavior.",
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": r008_registration[
            "downstream_behavioral_design_if_and_only_if_state_pair_passes"
        ],
    }
    ARTIFACT.mkdir(parents=True)
    registration_path_out = ARTIFACT / "repair_registration.json"
    registration_path_out.write_bytes(canonical_bytes(registration))

    schedule = deepcopy(r008_schedule)
    archived = deepcopy(schedule.get("archived_predecessor_contracts", {}))
    archived.update(
        {
            "status": "archived_lineage_only_not_active_r009_runtime_evidence",
            "r008_object_space_servo_contract": schedule.pop(
                "object_space_servo_contract"
            ),
            "r008_object_space_servo_contract_sha256": schedule.pop(
                "object_space_servo_contract_sha256"
            ),
            "r008_construction_lifecycle_contract": schedule.pop(
                "construction_lifecycle_contract"
            ),
            "r008_construction_lifecycle_contract_sha256": schedule.pop(
                "construction_lifecycle_contract_sha256"
            ),
        }
    )
    schedule.update(
        {
            "schema_version": "vla-wam-shared-v3e006-r009-candidate-schedule-v1",
            "repair_amendment_id": "V3-E006-R009",
            "status": "frozen_before_any_r009_live_diagnostic_candidate_or_model_request",
            "r009_live_diagnostic_count": 0,
            "r009_live_candidate_evaluation_count": 0,
            "repair_registration": bind(registration_path_out),
            "r008_predecessor": predecessor,
            "r008_candidate_schedule": bind(schedule_path),
            "pinch_geometry_contract": pinch,
            "pinch_geometry_contract_sha256": pinch_sha,
            "joint_handoff_contract": handoff,
            "joint_handoff_contract_sha256": handoff_sha,
            "construction_lifecycle_contract": lifecycle,
            "construction_lifecycle_contract_sha256": lifecycle_sha,
            "construction_asset_bindings": deepcopy(COLLISION_ASSET_BINDINGS),
            "archived_predecessor_contracts": archived,
        }
    )
    schedule.pop("r008_live_diagnostic_count", None)
    schedule.pop("r008_live_candidate_evaluation_count", None)
    schedule.pop("schedule_canonical_sha256_without_this_field", None)
    schedule["selection_rule"] = deepcopy(r008_schedule["selection_rule"])
    schedule["selection_rule"]["algorithm_version"] = (
        "r009-uniform-tensor-collision-pinch-first-passing-pair-v1"
    )
    schedule["selection_rule"]["exhaustion"] = (
        "Four valid rejected pairs exhaust R009 at zero model/behavior. Infrastructure-invalid attempts retry identically. No geometry rule, controller parameter, phase count, horizon, target, order, or scientific gate changes after live evaluation begins."
    )
    schedule["selection_rule"]["r009_construction_only_change"] = (
        "Resolve exact body-local collision geometry once, reconstruct all live pad/cube geometry from tensor poses minus the retained env origin, then execute the unconditional fixed open approach/descent, normal close, closed vertical lift, closed stage transport, and 600-step captured-q settle; evaluate unchanged final-ten gates. Contact/grab traces are diagnostic-only and never alter construction."
    )
    for pair in schedule["candidate_pairs"]:
        pair["construction_method"] = (
            "exact_reset_uniform_collision_pinch_acquire_lift_transport_q_handoff"
        )
        for stage_name in ("canonical_grasp", "canonical_carry"):
            stage = pair[stage_name]
            stage["r009_target_cube_pose"] = deepcopy(stage["target_cube_pose"])
            stage["r009_acquisition_base_quaternion_world_wxyz"] = deepcopy(
                stage["r008_precontact_targets"][
                    "contact_base_link_pose_at_exact_reset_cube"
                ]["quaternion_world_wxyz"]
            )
            stage["r009_final_base_quaternion_world_wxyz"] = deepcopy(
                stage["centerline_constrained_base_link_ik_target"][
                    "quaternion_world_wxyz"
                ]
            )
    schedule["schedule_canonical_sha256_without_this_field"] = hashlib.sha256(
        canonical_bytes(schedule)
    ).hexdigest()
    schedule_path_out = ARTIFACT / "gates/candidate_schedule.json"
    schedule_path_out.parent.mkdir()
    schedule_path_out.write_bytes(canonical_bytes(schedule))
    (ARTIFACT / "infrastructure_attempts.jsonl").write_bytes(b"")
    print(
        json.dumps(
            {
                "registration": bind(registration_path_out),
                "candidate_schedule": bind(schedule_path_out),
                "pinch_geometry_contract_sha256": pinch_sha,
                "joint_handoff_contract_sha256": handoff_sha,
                "construction_lifecycle_contract_sha256": lifecycle_sha,
                "candidate_budget": 4,
                "diagnostic_budget": 4,
                "model_request_count": 0,
                "behavioral_episode_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
