#!/usr/bin/env python3
"""Freeze the prospective V3-E006-R012 live-tensor-only amendment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r012.pinch_geometry import (
    validate_attachment_preflight_contract,
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r012.predecessor_contract import (
    R011_CLOSURE_COMMIT,
    R011_RESULTS_SHA256,
    validate_r011_scene_sync_failure_closure,
)


ROOT = Path(__file__).resolve().parents[1]
R011 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r011"
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012"


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
        retained = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        retained = str(path)
    return {"path": retained, "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-at-utc", required=True)
    args = parser.parse_args()
    if (ARTIFACT / "repair_registration.json").exists() or (ARTIFACT / "gates/candidate_schedule.json").exists():
        parser.error(f"refusing to overwrite prospective R012 freeze: {ARTIFACT}")
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{R011_CLOSURE_COMMIT}^{{commit}}"],
        check=False,
    ).returncode:
        parser.error("R011 closure commit is absent")

    registration_path = R011 / "repair_registration.json"
    schedule_path = R011 / "gates/candidate_schedule.json"
    source_gate_path = R011 / "source_push_gate.json"
    results_path = R011 / "results/results.json"
    manifest_path = R011 / "results/evidence_manifest.json"
    memo_path = R011 / "results/DECISION_MEMO.md"
    receipt_path = R011 / "results/target_validation_receipt.json"
    r011_registration = load(registration_path)
    r011_schedule = load(schedule_path)
    r011_results = load(results_path)
    if sha256(results_path) != R011_RESULTS_SHA256:
        parser.error("R011 result digest differs")
    validate_r011_scene_sync_failure_closure(r011_results)

    pinch = deepcopy(r011_schedule["pinch_geometry_contract"])
    validate_contract(pinch)
    pinch_sha = hashlib.sha256(canonical_bytes(pinch)).hexdigest()
    handoff = deepcopy(r011_schedule["joint_handoff_contract"])
    handoff_sha = hashlib.sha256(canonical_bytes(handoff)).hexdigest()
    lifecycle = deepcopy(r011_schedule["construction_lifecycle_contract"])
    lifecycle_sha = hashlib.sha256(canonical_bytes(lifecycle)).hexdigest()
    preflight = {
        "algorithm_version": "r012-live-physics-tensor-geometry-sanity-preflight-v1",
        "preflight_budget": 1,
        "execution_order": "exactly one dedicated fresh environment, completed and closed before the first known-reachable diagnostic or candidate environment",
        "fresh_reset_steps": 75,
        "dynamic_state_source": "finite IsaacLab live robot body_pos_w/body_quat_w and cube root_pos_w/root_quat_w tensors minus the exact retained scene env origin",
        "static_attachment_source": "the frozen R010 ComputeRelativeBound collision inventory and owning-rigid-body-local aligned-range corners",
        "exact_tensor_index_name_and_body_ownership_required": True,
        "left_right_tensor_indices_must_be_distinct": True,
        "pad_collision_center_separation_m_inclusive": [0.05, 0.2],
        "cube_aabb_dimension_m_each_inclusive": [0.03, 0.1],
        "fresh_reset_tensor_identity_required": True,
        "dynamic_usd_world_state_used": False,
        "physics_to_usd_sync_call_count": 0,
        "dynamic_usd_world_bound_or_xform_query_count": 0,
        "candidate_environment_identity_rule": "before its first construction action, every rank-stage environment must reproduce the exact preflight collision inventory and unchanged R010 body-relative geometry canonical SHA",
        "failure_policy": "any nonfinite tensor, tensor index/name/ownership mismatch, static inventory/hash mismatch, degenerate pad separation/cube dimension, or fresh-reset tensor identity mismatch produces one zero-candidate terminal preflight failure",
        "dynamic_usd_world_bounds_used_by_controller": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    validate_attachment_preflight_contract(preflight)
    preflight_sha = hashlib.sha256(canonical_bytes(preflight)).hexdigest()

    predecessor = {
        "closure_commit": R011_CLOSURE_COMMIT,
        "registration": bind(registration_path),
        "candidate_schedule": bind(schedule_path),
        "source_push_gate": bind(source_gate_path),
        "results": bind(results_path),
        "evidence_manifest": bind(manifest_path),
        "decision_memo": bind(memo_path),
        "target_validation_receipt": bind(receipt_path),
        "raw_result": r011_results["raw_result"],
        "raw_preflight": r011_results["raw_preflight"],
        "outcome": "R011 failed closed before actions because the final dynamic USD oracle did not agree with live tensors; R012 follows the user-authorized live-physics-tensor-only boundary and removes that oracle entirely.",
    }
    frozen_inputs = deepcopy(r011_registration["frozen_inputs"])
    frozen_inputs.update(
        {
            "r011_results": bind(results_path),
            "r011_evidence_manifest": bind(manifest_path),
            "r011_target_validation_receipt": bind(receipt_path),
        }
    )
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r012-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "original_amendment_id": "V3-E006",
        "repair_amendment_id": "V3-E006-R012",
        "predecessor_repair_amendment_id": "V3-E006-R011",
        "registered_at_utc": args.registered_at_utc,
        "status": "prospectively_registered_before_any_r012_live_preflight_diagnostic_candidate_or_model_request",
        "counts_at_registration": {
            "r012_geometry_attachment_preflights": 0,
            "r012_live_diagnostics": 0,
            "r012_live_candidate_evaluations": 0,
            "model_requests": 0,
            "behavioral_episodes": 0,
        },
        "user_authorized_override": "Use live physics tensors plus the frozen body-relative static collision inventory as the sole dynamic geometry truth; remove every dynamic USD world-state oracle and synchronization step.",
        "engineering_rationale_only": "R011 proved its registered dynamic USD oracle remained stale after the final scene-specific synchronization. R012 changes only the fail-closed preflight to finite tensor/index/ownership/geometry sanity and leaves the construction controller and scientific gates unchanged.",
        "r011_predecessor": predecessor,
        "pinch_geometry_contract": pinch,
        "pinch_geometry_contract_sha256": pinch_sha,
        "geometry_attachment_preflight_contract": preflight,
        "geometry_attachment_preflight_contract_sha256": preflight_sha,
        "joint_handoff_contract": handoff,
        "joint_handoff_contract_sha256": handoff_sha,
        "construction_lifecycle_contract": lifecycle,
        "construction_lifecycle_contract_sha256": lifecycle_sha,
        "unchanged_science": {
            "r011_r010_relative_bound_geometry_and_candidate_aliases_identical": True,
            "r011_r010_candidate_sources_targets_rank_order_and_first_pass_rule_identical": True,
            "r011_r010_controller_gains_caps_phase_counts_horizon_and_joint_handoff_identical": True,
            "diagnostics_identical": True,
            "physics_ood_camera_companion_contact_midline_gates_and_thresholds_identical": True,
            "model_checkpoint_runtime_prompts_sample_size_behavioral_horizon_and_scorer_unchanged": True,
        },
        "prohibitions": [
            "no post-reset joint/object pose/velocity state write",
            "no weld/attachment/collision suppression/force injection",
            "no dynamic USD world bound, Xform oracle, PhysX-to-USD synchronization, or persistent setting change",
            "no controller, target, geometry, rank, horizon, scientific threshold, or gate change",
            "no model request or behavioral episode",
        ],
        "frozen_inputs": frozen_inputs,
        "release_boundary": "One zero-model live-tensor sanity preflight, diagnostic suite, and conditional finite four-pair search only; no inference or behavior until a state pair passes.",
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": r011_registration[
            "downstream_behavioral_design_if_and_only_if_state_pair_passes"
        ],
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    registration_out = ARTIFACT / "repair_registration.json"
    registration_out.write_bytes(canonical_bytes(registration))

    schedule = deepcopy(r011_schedule)
    archived = deepcopy(schedule.get("archived_predecessor_contracts", {}))
    archived.update(
        {
            "status": "archived_lineage_only_not_active_r012_runtime_evidence",
            "r011_failed_scene_sync_preflight_contract": deepcopy(r011_schedule["geometry_attachment_preflight_contract"]),
            "r011_failed_scene_sync_preflight_contract_sha256": r011_schedule["geometry_attachment_preflight_contract_sha256"],
        }
    )
    schedule.update(
        {
            "schema_version": "vla-wam-shared-v3e006-r012-candidate-schedule-v1",
            "repair_amendment_id": "V3-E006-R012",
            "status": "frozen_before_any_r012_live_preflight_diagnostic_candidate_or_model_request",
            "r012_geometry_attachment_preflight_count": 0,
            "r012_live_diagnostic_count": 0,
            "r012_live_candidate_evaluation_count": 0,
            "repair_registration": bind(registration_out),
            "r011_predecessor": predecessor,
            "r011_candidate_schedule": bind(schedule_path),
            "geometry_attachment_preflight_contract": preflight,
            "geometry_attachment_preflight_contract_sha256": preflight_sha,
            "archived_predecessor_contracts": archived,
        }
    )
    for key in (
        "r011_geometry_attachment_preflight_count",
        "r011_live_diagnostic_count",
        "r011_live_candidate_evaluation_count",
        "scene_sync_source_bindings",
        "schedule_canonical_sha256_without_this_field",
    ):
        schedule.pop(key, None)
    schedule["selection_rule"] = deepcopy(r011_schedule["selection_rule"])
    schedule["selection_rule"]["algorithm_version"] = (
        "r012-live-tensor-relative-bound-collision-pinch-first-passing-pair-v1"
    )
    schedule["selection_rule"]["exhaustion"] = (
        "Four valid rejected pairs exhaust R012 at zero model/behavior. A failed dedicated live-tensor sanity preflight blocks every candidate. No controller, target, geometry, rank, order, horizon, or scientific gate change is permitted after registration."
    )
    schedule["selection_rule"]["r012_validation_only_change"] = (
        "Remove R011 dynamic USD synchronization/world-state oracle and use only finite live tensor poses plus the unchanged frozen R010 body-relative static collision inventory; retain all construction commands and scientific gates."
    )
    schedule["schedule_canonical_sha256_without_this_field"] = hashlib.sha256(
        canonical_bytes(schedule)
    ).hexdigest()
    schedule_out = ARTIFACT / "gates/candidate_schedule.json"
    schedule_out.parent.mkdir(exist_ok=True)
    schedule_out.write_bytes(canonical_bytes(schedule))
    (ARTIFACT / "infrastructure_attempts.jsonl").write_bytes(b"")
    print(json.dumps({
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
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
