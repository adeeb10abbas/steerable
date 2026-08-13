#!/usr/bin/env python3
"""Freeze V3-E006-R010 collision-pinch construction before any live run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r010.pinch_geometry import (
    validate_attachment_preflight_contract,
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r010.predecessor_contract import (
    R009_CLOSURE_COMMIT,
    R009_RESULTS_SHA256,
    validate_r009_attachment_invalid_closure,
)


ROOT = Path(__file__).resolve().parents[1]
R009 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009"
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r010"
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
GEOMETRY_ORACLE_SOURCE_BINDINGS = {
    "physx_python_interface": {
        "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaacsim/extscache/omni.physx-107.3.18+107.3.1.lx64.r.cp311.u353/omni/physx/bindings/_physx.pyi",
        "bytes": 181192,
        "sha256": "ff13abb83480dcc707ac2ad60062306aef7a33f885d32ed4c8ee6dfea2008e79",
    },
    "nvidia_physx_camera_sync_test": {
        "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaacsim/extscache/omni.physx.camera-107.3.18+107.3.1.lx64.r.cp311.u353/omni/physxcamera/scripts/tests.py",
        "bytes": 14746,
        "sha256": "573cc70843dae3d261701e26f059d000f669cb8310d33abd9768fd2359d83425",
    },
    "isaac_simulation_manager": {
        "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/simulation_manager.py",
        "bytes": 42551,
        "sha256": "8802f455e5c0ee71c7a230b68d4a32adc55029254a717744471a7d0852d8585f",
    },
    "isaaclab_simulation_context": {
        "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py",
        "bytes": 50505,
        "sha256": "6de2674da9df40f0c030c8e101504d46da481b4572fcb4ce9cd7e1ca8ebcdc59",
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
        parser.error(f"refusing to overwrite prospective R010 freeze: {ARTIFACT}")
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{R009_CLOSURE_COMMIT}^{{commit}}"],
        check=False,
    ).returncode:
        parser.error("R009 closure commit is absent")

    registration_path = R009 / "repair_registration.json"
    schedule_path = R009 / "gates/candidate_schedule.json"
    source_gate_path = R009 / "source_push_gate.json"
    results_path = R009 / "results/results.json"
    manifest_path = R009 / "results/evidence_manifest.json"
    memo_path = R009 / "results/DECISION_MEMO.md"
    receipt_path = R009 / "results/target_validation_receipt.json"
    r009_registration = load(registration_path)
    r009_schedule = load(schedule_path)
    r009_results = load(results_path)
    if sha256(results_path) != R009_RESULTS_SHA256:
        parser.error("R009 result digest differs")
    validate_r009_attachment_invalid_closure(r009_results)

    pinch = {
        "algorithm_version": "uniform-relative-bound-tensor-collision-pinch-acquisition-v1",
        "applies_identically_to": "all four ranks and canonical_grasp/canonical_carry",
        "robot_body_resolution": "under the unique env_0 robot prim, require exactly one tensor rigid body whose path ends left_inner_finger and exactly one ending right_inner_finger; each selected owner must have UsdPhysics.RigidBodyAPI",
        "collision_resolution": "once before candidate actions, resolve each enabled UsdPhysics.CollisionAPI prim with ComputeRelativeBound(collisionPrim, owningRigidBody).ComputeAlignedRange; retain those corners directly as body-local and reject nested or different rigid-body boundaries",
        "relative_bound_api": "UsdGeom.BBoxCache.ComputeRelativeBound(collision_prim, owning_rigid_body).ComputeAlignedRange",
        "additional_transform_after_relative_bound": False,
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
        "parameter_basis": "R009 numeric controller, phases, ranks, and targets are retained exactly; R010 changes only the collision-bound attachment extraction from double-transformed ComputeLocalBound to direct body-relative ComputeRelativeBound",
    }
    validate_contract(pinch)
    pinch_sha = hashlib.sha256(canonical_bytes(pinch)).hexdigest()
    preflight = {
        "algorithm_version": "r010-relative-bound-world-oracle-preflight-v1",
        "preflight_budget": 1,
        "execution_order": "exactly one dedicated fresh environment, completed and closed before the first known-reachable diagnostic or candidate environment",
        "fresh_reset_steps": 75,
        "physics_to_usd_setting_path": "/physics/updateToUsd",
        "physics_to_usd_synchronization_call": "omni.physx.get_physx_interface().update_transformations(False, True, False, False)",
        "physics_to_usd_setting_must_remain_unchanged": True,
        "synchronization_timing": "snapshot all live tensors after fresh reset; make the one-shot writeback call; take no physics/action step; create fresh USD caches and read the oracle",
        "usd_oracle": "fresh BBoxCache.ComputeWorldBound(collisionPrim).ComputeAlignedRange after 75 synchronized reset steps",
        "oracle_scope": "every enabled collision prim under the unique left pad owner, right pad owner, and cube rigid body",
        "oracle_tolerance_m_inclusive": 1e-5,
        "owner_pose_tolerance_m_inclusive": 1e-6,
        "owner_orientation_tolerance_deg_inclusive": 1e-4,
        "tensor_reconstruction": "body-relative aligned-range corners transformed once by the live tensor rigid-body/root pose; world comparison adds the exact retained scene env origin",
        "oracle_is_controller_input": False,
        "candidate_environment_identity_rule": "before its first construction action, every rank-stage environment must reproduce the exact preflight collision inventory and body-relative geometry canonical SHA",
        "failure_policy": "a failed dedicated oracle produces a zero-candidate terminal preflight failure; any later rank-stage geometry identity mismatch is infrastructure-invalid before its first construction action; neither is a scientific candidate rejection",
        "dynamic_usd_world_bounds_used_by_controller": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    validate_attachment_preflight_contract(preflight)
    preflight_sha = hashlib.sha256(canonical_bytes(preflight)).hexdigest()
    handoff = deepcopy(r009_schedule["joint_handoff_contract"])
    handoff_sha = hashlib.sha256(canonical_bytes(handoff)).hexdigest()
    lifecycle = deepcopy(r009_schedule["construction_lifecycle_contract"])
    lifecycle_sha = hashlib.sha256(canonical_bytes(lifecycle)).hexdigest()

    predecessor = {
        "closure_commit": R009_CLOSURE_COMMIT,
        "registration": bind(registration_path),
        "candidate_schedule": bind(schedule_path),
        "source_push_gate": bind(source_gate_path),
        "results": bind(results_path),
        "evidence_manifest": bind(manifest_path),
        "decision_memo": bind(memo_path),
        "target_validation_receipt": bind(receipt_path),
        "raw_result": r009_results["raw_result"],
        "outcome": (
            "four diagnostics and four pairs mechanically completed with zero "
            "model/behavior, but the intended pinch attachment was invalid because "
            "ComputeLocalBound geometry was transformed twice; intended R009 "
            "construction was not scientifically exhausted"
        ),
    }
    frozen_inputs = deepcopy(r009_registration["frozen_inputs"])
    frozen_inputs.update(
        {
            "r009_results": bind(results_path),
            "r009_evidence_manifest": bind(manifest_path),
            "r009_target_validation_receipt": bind(receipt_path),
            "r010_robot_collision_usd": deepcopy(
                COLLISION_ASSET_BINDINGS["robot_usd"]
            ),
            "r010_cube_collision_usd": deepcopy(
                COLLISION_ASSET_BINDINGS["cube_usd"]
            ),
            "r010_geometry_oracle_source_bindings": deepcopy(
                GEOMETRY_ORACLE_SOURCE_BINDINGS
            ),
        }
    )
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r010-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "original_amendment_id": "V3-E006",
        "repair_amendment_id": "V3-E006-R010",
        "predecessor_repair_amendment_id": "V3-E006-R009",
        "registered_at_utc": args.registered_at_utc,
        "status": "prospectively_registered_before_any_r010_live_diagnostic_candidate_or_model_request",
        "counts_at_registration": {
            "r010_geometry_attachment_preflights": 0,
            "r010_live_diagnostics": 0,
            "r010_live_candidate_evaluations": 0,
            "model_requests": 0,
            "behavioral_episodes": 0,
        },
        "user_authorized_override": "Continue solving canonical-state construction after attachment-invalid R009; behavior remains blocked until a state pair passes unchanged gates.",
        "engineering_rationale_only": (
            "R009 mechanically completed its frozen schedule, but ComputeLocalBound already included each collision prim's authored transform and R009 applied that transform again. The resulting 0.34249–0.34518 m reconstructed pad separations were inconsistent with 0.07925–0.08179 m live inner-body separations. R010 corrects only this attachment: each collision bound is computed directly relative to its owning rigid body and receives no later prim/world transform. A dedicated one-shot PhysX-to-USD oracle must pass before any diagnostic or candidate."
        ),
        "r009_predecessor": predecessor,
        "pinch_geometry_contract": pinch,
        "pinch_geometry_contract_sha256": pinch_sha,
        "geometry_attachment_preflight_contract": preflight,
        "geometry_attachment_preflight_contract_sha256": preflight_sha,
        "geometry_oracle_source_bindings": deepcopy(
            GEOMETRY_ORACLE_SOURCE_BINDINGS
        ),
        "joint_handoff_contract": handoff,
        "joint_handoff_contract_sha256": handoff_sha,
        "construction_lifecycle_contract": lifecycle,
        "construction_lifecycle_contract_sha256": lifecycle_sha,
        "unchanged_science": {
            "r009_candidate_sources_target_cube_poses_rank_order_and_first_pass_rule_identical": True,
            "r009_controller_gains_caps_phase_counts_horizon_and_joint_handoff_identical": True,
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
        "release_boundary": "One zero-model attachment preflight, diagnostic, and conditional finite four-pair search only; no inference or behavior.",
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": r009_registration[
            "downstream_behavioral_design_if_and_only_if_state_pair_passes"
        ],
    }
    ARTIFACT.mkdir(parents=True)
    registration_path_out = ARTIFACT / "repair_registration.json"
    registration_path_out.write_bytes(canonical_bytes(registration))

    schedule = deepcopy(r009_schedule)
    archived = deepcopy(schedule.get("archived_predecessor_contracts", {}))
    archived.update(
        {
            "status": "archived_lineage_only_not_active_r010_runtime_evidence",
            "r009_attachment_invalid_pinch_geometry_contract": schedule[
                "pinch_geometry_contract"
            ],
            "r009_attachment_invalid_pinch_geometry_contract_sha256": schedule[
                "pinch_geometry_contract_sha256"
            ],
        }
    )
    schedule.update(
        {
            "schema_version": "vla-wam-shared-v3e006-r010-candidate-schedule-v1",
            "repair_amendment_id": "V3-E006-R010",
            "status": "frozen_before_any_r010_live_diagnostic_candidate_or_model_request",
            "r010_geometry_attachment_preflight_count": 0,
            "r010_live_diagnostic_count": 0,
            "r010_live_candidate_evaluation_count": 0,
            "repair_registration": bind(registration_path_out),
            "r009_predecessor": predecessor,
            "r009_candidate_schedule": bind(schedule_path),
            "pinch_geometry_contract": pinch,
            "pinch_geometry_contract_sha256": pinch_sha,
            "geometry_attachment_preflight_contract": preflight,
            "geometry_attachment_preflight_contract_sha256": preflight_sha,
            "geometry_oracle_source_bindings": deepcopy(
                GEOMETRY_ORACLE_SOURCE_BINDINGS
            ),
            "joint_handoff_contract": handoff,
            "joint_handoff_contract_sha256": handoff_sha,
            "construction_lifecycle_contract": lifecycle,
            "construction_lifecycle_contract_sha256": lifecycle_sha,
            "construction_asset_bindings": deepcopy(COLLISION_ASSET_BINDINGS),
            "archived_predecessor_contracts": archived,
        }
    )
    schedule.pop("r009_live_diagnostic_count", None)
    schedule.pop("r009_live_candidate_evaluation_count", None)
    schedule.pop("schedule_canonical_sha256_without_this_field", None)
    schedule["selection_rule"] = deepcopy(r009_schedule["selection_rule"])
    schedule["selection_rule"]["algorithm_version"] = (
        "r010-relative-bound-validated-collision-pinch-first-passing-pair-v1"
    )
    schedule["selection_rule"]["exhaustion"] = (
        "Four valid rejected pairs exhaust R010 at zero model/behavior. Infrastructure-invalid attempts retry identically. No geometry rule, controller parameter, phase count, horizon, target, order, or scientific gate changes after live evaluation begins."
    )
    schedule["selection_rule"]["r010_construction_only_change"] = (
        "Replace only R009's double-transformed ComputeLocalBound attachment with exact ComputeRelativeBound(collision, owning rigid body).ComputeAlignedRange corners, validate every prim in one fresh-reset tensor-vs-synchronized-USD oracle environment, require identical geometry in every candidate environment, then execute the unchanged unconditional R009 controller and unchanged final-ten gates."
    )
    for pair in schedule["candidate_pairs"]:
        pair["construction_method"] = (
            "exact_reset_uniform_collision_pinch_acquire_lift_transport_q_handoff"
        )
        for stage_name in ("canonical_grasp", "canonical_carry"):
            stage = pair[stage_name]
            stage["r010_target_cube_pose"] = deepcopy(stage["r009_target_cube_pose"])
            stage["r010_acquisition_base_quaternion_world_wxyz"] = deepcopy(
                stage["r009_acquisition_base_quaternion_world_wxyz"]
            )
            stage["r010_final_base_quaternion_world_wxyz"] = deepcopy(
                stage["r009_final_base_quaternion_world_wxyz"]
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
                "geometry_attachment_preflight_contract_sha256": preflight_sha,
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
