#!/usr/bin/env python3
"""Freeze V3-E006-R011 scene-specific synchronization before any live run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r011.pinch_geometry import (
    validate_attachment_preflight_contract,
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r011.predecessor_contract import (
    R010_CLOSURE_COMMIT,
    R010_RESULTS_SHA256,
    validate_r010_oracle_failure_closure,
)


ROOT = Path(__file__).resolve().parents[1]
R010 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r010"
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r011"

SCENE_SYNC_SOURCE_BINDINGS = {
    "physx_python_interface": {
        "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaacsim/extscache/omni.physx-107.3.18+107.3.1.lx64.r.cp311.u353/omni/physx/bindings/_physx.pyi",
        "bytes": 181192,
        "sha256": "ff13abb83480dcc707ac2ad60062306aef7a33f885d32ed4c8ee6dfea2008e79",
    },
    "nvidia_physx_scene_specific_transform_test": {
        "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaacsim/extscache/omni.physx.tests-107.3.18+107.3.1.cp311.u353/omni/physxtests/tests/PhysicsRigidBodyAPI.py",
        "bytes": 89537,
        "sha256": "38209f2c379f143c87cbd921a02b756145b6919bf3d5883fc98c9419e2466dc0",
    },
    "isaac_sim_core_simulation_context": {
        "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py",
        "bytes": 50858,
        "sha256": "1160d8e283f88abe88babe1725b2c19ad5d946215bf47caf062ee88edbd75b66",
    },
    "isaaclab_simulation_context": {
        "path": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py",
        "bytes": 50505,
        "sha256": "6de2674da9df40f0c030c8e101504d46da481b4572fcb4ce9cd7e1ca8ebcdc59",
    },
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


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
    if (ARTIFACT / "repair_registration.json").exists() or (
        ARTIFACT / "gates/candidate_schedule.json"
    ).exists():
        parser.error(f"refusing to overwrite prospective R011 freeze: {ARTIFACT}")
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{R010_CLOSURE_COMMIT}^{{commit}}"],
        check=False,
    ).returncode:
        parser.error("R010 closure commit is absent")

    registration_path = R010 / "repair_registration.json"
    schedule_path = R010 / "gates/candidate_schedule.json"
    source_gate_path = R010 / "source_push_gate.json"
    results_path = R010 / "results/results.json"
    manifest_path = R010 / "results/evidence_manifest.json"
    memo_path = R010 / "results/DECISION_MEMO.md"
    receipt_path = R010 / "results/target_validation_receipt.json"
    r010_registration = load(registration_path)
    r010_schedule = load(schedule_path)
    r010_results = load(results_path)
    if sha256(results_path) != R010_RESULTS_SHA256:
        parser.error("R010 result digest differs")
    validate_r010_oracle_failure_closure(r010_results)

    # The controller and relative-bound geometry are immutable R010 bytes/data.
    pinch = deepcopy(r010_schedule["pinch_geometry_contract"])
    validate_contract(pinch)
    pinch_sha = hashlib.sha256(canonical_bytes(pinch)).hexdigest()
    handoff = deepcopy(r010_schedule["joint_handoff_contract"])
    handoff_sha = hashlib.sha256(canonical_bytes(handoff)).hexdigest()
    lifecycle = deepcopy(r010_schedule["construction_lifecycle_contract"])
    lifecycle_sha = hashlib.sha256(canonical_bytes(lifecycle)).hexdigest()

    preflight = {
        "algorithm_version": "r011-scene-specific-relative-bound-world-oracle-preflight-v1",
        "preflight_budget": 1,
        "execution_order": "exactly one dedicated fresh environment, completed and closed before the first known-reachable diagnostic or candidate environment",
        "fresh_reset_steps": 75,
        "simulation_forward_before_tensor_snapshot": False,
        "simulation_context_api": "isaaclab.sim.SimulationContext.instance()",
        "stage_identity_rule": "the fresh environment sim must be the SimulationContext singleton, and SimulationContext.get_initial_stage() must be the exact same Python USD stage object as omni.usd.get_context().get_stage before and after synchronization",
        "stage_cache_identity_api": "pxr.UsdUtils.StageCache.Get().GetId(stage).ToLongInt()",
        "stage_cache_identity_rule": "the initial/current stage cache ID must be valid, nonzero, equal before and after synchronization, and retained in raw evidence",
        "physics_scene_resolution": "traverse the verified current stage and require exactly one valid UsdPhysics.Scene prim whose path exactly equals SimulationContext.cfg.physics_prim_path",
        "expected_configured_physics_scene_path": "/physicsScene",
        "physics_scene_path_integer_api": "pxr.PhysicsSchemaTools.sdfPathToInt(physics_scene_prim_path)",
        "physics_scene_path_integer_rule": "runtime computes scene_path_int directly as int(PhysicsSchemaTools.sdfPathToInt(configured_scene_path)); the same nonzero value and exact scene path are retained independently in stage identity and structured call arguments before the one registered call",
        "physics_to_usd_setting_path": "/physics/updateToUsd",
        "physics_to_usd_synchronization_call": "omni.physx.get_physx_interface().update_transformations_scene(scene_path_int, True, False)",
        "physics_to_usd_call_count": 1,
        "physics_to_usd_setting_must_remain_unchanged": True,
        "synchronization_timing": "snapshot all live tensors after exact fresh reset; call the one resolved scene-specific writeback exactly once; take no physics/action step; create new USD Xform/BBox caches from the same verified stage and read the oracle",
        "usd_oracle": "new post-writeback BBoxCache.ComputeWorldBound(collisionPrim).ComputeAlignedRange plus new XformCache owner poses on the same verified stage",
        "oracle_scope": "every enabled collision prim under the unique left pad owner, right pad owner, and cube rigid body",
        "oracle_tolerance_m_inclusive": 1e-5,
        "owner_pose_tolerance_m_inclusive": 1e-6,
        "owner_orientation_tolerance_deg_inclusive": 1e-4,
        "tensor_reconstruction": "unchanged R010 body-relative aligned-range corners transformed once by the live tensor rigid-body/root pose; world comparison adds the exact retained scene env origin",
        "oracle_is_controller_input": False,
        "candidate_environment_identity_rule": "before its first construction action, every rank-stage environment must reproduce the exact preflight collision inventory and unchanged R010 body-relative geometry canonical SHA",
        "failure_policy": "any stage/scene/API/call/setting/owner/AABB mismatch produces one zero-candidate terminal preflight failure; no fallback or alternate synchronization is permitted",
        "dynamic_usd_world_bounds_used_by_controller": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    validate_attachment_preflight_contract(preflight)
    preflight_sha = hashlib.sha256(canonical_bytes(preflight)).hexdigest()

    predecessor = {
        "closure_commit": R010_CLOSURE_COMMIT,
        "registration": bind(registration_path),
        "candidate_schedule": bind(schedule_path),
        "source_push_gate": bind(source_gate_path),
        "results": bind(results_path),
        "evidence_manifest": bind(manifest_path),
        "decision_memo": bind(memo_path),
        "target_validation_receipt": bind(receipt_path),
        "raw_result": r010_results["raw_result"],
        "raw_preflight": r010_results["raw_preflight"],
        "outcome": "R010 failed closed before diagnostics/candidates because the registered global PhysX-to-USD writeback left USD owner/AABB oracle values inconsistent with simultaneous live tensors; relative-bound geometry and the controller were not evaluated",
    }
    frozen_inputs = deepcopy(r010_registration["frozen_inputs"])
    frozen_inputs.update(
        {
            "r010_results": bind(results_path),
            "r010_evidence_manifest": bind(manifest_path),
            "r010_target_validation_receipt": bind(receipt_path),
            "r011_scene_sync_source_bindings": deepcopy(SCENE_SYNC_SOURCE_BINDINGS),
        }
    )
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r011-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "original_amendment_id": "V3-E006",
        "repair_amendment_id": "V3-E006-R011",
        "predecessor_repair_amendment_id": "V3-E006-R010",
        "registered_at_utc": args.registered_at_utc,
        "status": "prospectively_registered_before_any_r011_live_preflight_diagnostic_candidate_or_model_request",
        "counts_at_registration": {
            "r011_geometry_attachment_preflights": 0,
            "r011_live_diagnostics": 0,
            "r011_live_candidate_evaluations": 0,
            "model_requests": 0,
            "behavioral_episodes": 0,
        },
        "user_authorized_override": "Continue solving canonical-state construction after R010's fail-closed oracle synchronization failure; behavior remains blocked until a state pair passes unchanged gates.",
        "engineering_rationale_only": "R010 proved that the global PhysX writeback did not synchronize the replicated/Fabric scene used by the fresh-reset environment. R011 changes only the validation oracle's stage/scene mapping: it binds the active SimulationContext stage and unique configured PhysicsScene, converts that exact scene path to its PhysX integer ID, and performs one scene-specific writeback. There is no fallback or controller change.",
        "r010_predecessor": predecessor,
        "pinch_geometry_contract": pinch,
        "pinch_geometry_contract_sha256": pinch_sha,
        "geometry_attachment_preflight_contract": preflight,
        "geometry_attachment_preflight_contract_sha256": preflight_sha,
        "scene_sync_source_bindings": deepcopy(SCENE_SYNC_SOURCE_BINDINGS),
        "joint_handoff_contract": handoff,
        "joint_handoff_contract_sha256": handoff_sha,
        "construction_lifecycle_contract": lifecycle,
        "construction_lifecycle_contract_sha256": lifecycle_sha,
        "unchanged_science": {
            "r010_relative_bound_geometry_and_candidate_aliases_identical": True,
            "r010_candidate_sources_targets_rank_order_and_first_pass_rule_identical": True,
            "r010_controller_gains_caps_phase_counts_horizon_and_joint_handoff_identical": True,
            "diagnostics_identical": True,
            "physics_ood_camera_companion_contact_midline_gates_and_thresholds_identical": True,
            "model_checkpoint_runtime_prompts_sample_size_behavioral_horizon_and_scorer_unchanged": True,
        },
        "prohibitions": [
            "no post-reset joint/object pose/velocity state write",
            "no weld/attachment/collision suppression/force injection",
            "no alternate synchronization API, fallback, scene search, or persistent setting change",
            "no physics/action step between tensor snapshot and oracle",
            "no controller, target, geometry, rank, horizon, scientific threshold, or gate change",
            "no model request or behavioral episode",
        ],
        "frozen_inputs": frozen_inputs,
        "release_boundary": "One zero-model scene-specific attachment preflight, diagnostic suite, and conditional finite four-pair search only; no inference or behavior.",
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": r010_registration[
            "downstream_behavioral_design_if_and_only_if_state_pair_passes"
        ],
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    registration_out = ARTIFACT / "repair_registration.json"
    registration_out.write_bytes(canonical_bytes(registration))

    schedule = deepcopy(r010_schedule)
    archived = deepcopy(schedule.get("archived_predecessor_contracts", {}))
    archived.update(
        {
            "status": "archived_lineage_only_not_active_r011_runtime_evidence",
            "r010_failed_global_sync_preflight_contract": deepcopy(
                r010_schedule["geometry_attachment_preflight_contract"]
            ),
            "r010_failed_global_sync_preflight_contract_sha256": r010_schedule[
                "geometry_attachment_preflight_contract_sha256"
            ],
        }
    )
    schedule.update(
        {
            "schema_version": "vla-wam-shared-v3e006-r011-candidate-schedule-v1",
            "repair_amendment_id": "V3-E006-R011",
            "status": "frozen_before_any_r011_live_preflight_diagnostic_candidate_or_model_request",
            "r011_geometry_attachment_preflight_count": 0,
            "r011_live_diagnostic_count": 0,
            "r011_live_candidate_evaluation_count": 0,
            "repair_registration": bind(registration_out),
            "r010_predecessor": predecessor,
            "r010_candidate_schedule": bind(schedule_path),
            "geometry_attachment_preflight_contract": preflight,
            "geometry_attachment_preflight_contract_sha256": preflight_sha,
            "scene_sync_source_bindings": deepcopy(SCENE_SYNC_SOURCE_BINDINGS),
            "archived_predecessor_contracts": archived,
        }
    )
    for key in (
        "r010_geometry_attachment_preflight_count",
        "r010_live_diagnostic_count",
        "r010_live_candidate_evaluation_count",
        "geometry_oracle_source_bindings",
        "schedule_canonical_sha256_without_this_field",
    ):
        schedule.pop(key, None)
    # Candidate pairs, diagnostics, pinch geometry, handoff and lifecycle are
    # retained byte/data-identically from R010.  Only selection metadata names
    # the prospective synchronization amendment.
    schedule["selection_rule"] = deepcopy(r010_schedule["selection_rule"])
    schedule["selection_rule"]["algorithm_version"] = (
        "r011-scene-sync-validated-relative-bound-collision-pinch-first-passing-pair-v1"
    )
    schedule["selection_rule"]["exhaustion"] = (
        "Four valid rejected pairs exhaust R011 at zero model/behavior. A failed dedicated scene-specific oracle blocks every candidate. No fallback, synchronization, geometry, controller, target, order, horizon, or scientific gate change is permitted after registration."
    )
    schedule["selection_rule"]["r011_validation_only_change"] = (
        "Replace only R010's global writeback oracle call with the one frozen stage-identity-checked scene-specific update_transformations_scene call; retain the exact R010 relative-bound geometry, candidate aliases, controller, lifecycle, selection order, and final gates."
    )
    schedule["schedule_canonical_sha256_without_this_field"] = hashlib.sha256(
        canonical_bytes(schedule)
    ).hexdigest()
    schedule_out = ARTIFACT / "gates/candidate_schedule.json"
    schedule_out.parent.mkdir(exist_ok=True)
    schedule_out.write_bytes(canonical_bytes(schedule))
    (ARTIFACT / "infrastructure_attempts.jsonl").write_bytes(b"")
    print(
        json.dumps(
            {
                "registration": bind(registration_out),
                "candidate_schedule": bind(schedule_out),
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
