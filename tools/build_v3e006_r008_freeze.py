#!/usr/bin/env python3
"""Freeze V3-E006-R008 object-space servo construction before any live run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r008.predecessor_contract import (
    R007_CLOSURE_COMMIT,
    R007_RESULTS_SHA256,
    validate_r007_exhaustion_closure,
)


ROOT = Path(__file__).resolve().parents[1]
R007 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007"
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008"


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
    if ARTIFACT.exists():
        parser.error(f"refusing to overwrite prospective R008 freeze: {ARTIFACT}")
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{R007_CLOSURE_COMMIT}^{{commit}}"],
        check=False,
    ).returncode:
        parser.error("R007 closure commit is absent")

    registration_path = R007 / "repair_registration.json"
    schedule_path = R007 / "gates/candidate_schedule.json"
    source_gate_path = R007 / "source_push_gate_v2.json"
    results_path = R007 / "results/results.json"
    manifest_path = R007 / "results/evidence_manifest.json"
    memo_path = R007 / "results/DECISION_MEMO.md"
    receipt_path = R007 / "results/target_validation_receipt.json"
    amendment_path = R007 / "postexecution_validator_amendment_v1.json"
    r007_registration = load(registration_path)
    r007_schedule = load(schedule_path)
    r007_results = load(results_path)
    if sha256(results_path) != R007_RESULTS_SHA256:
        parser.error("R007 result digest differs")
    validate_r007_exhaustion_closure(r007_results)

    servo = {
        "algorithm_version": "uniform-live-object-space-se3-servo-v1",
        "applies_identically_to": "all four ranks and canonical_grasp/canonical_carry",
        "target_cube_pose": "unchanged frozen target_cube_pose from the corresponding R007 stage",
        "live_relative_transform": "T_base_cube = inverse(T_world_base_live) * T_world_cube_live, recomputed before every servo action",
        "ideal_base_pose": "T_world_base_ideal = T_world_cube_target * inverse(T_base_cube_live)",
        "translation_update": "p_cmd = p_base_live + clip_norm(translation_gain*(p_ideal-p_base_live), translation_cap_m_per_step)",
        "rotation_update": "q_cmd = exp(clip_norm(rotation_gain*log(q_ideal*inverse(q_base_live)), rotation_cap_rad_per_step))*q_base_live using canonical shortest arc",
        "translation_gain": 0.2,
        "rotation_gain": 0.2,
        "translation_cap_m_per_step": 0.002,
        "rotation_cap_deg_per_step": 2.0,
        "servo_steps": 360,
        "gripper_command": 1.0,
        "early_stop": False,
        "gate_read_during_servo": False,
        "parameter_basis": "generic damped proportional Cartesian controller: dimensionless gain 0.2 with 0.002 m and 2 degree per-control-step rate caps; fixed before R008 and not selected from any R007 rank metric",
    }
    servo_sha = hashlib.sha256(canonical_bytes(servo)).hexdigest()
    handoff = {
        "algorithm_version": "single-captured-q-normal-actuator-handoff-v1",
        "capture": "capture all 13 observed robot joint positions exactly once after servo step 360",
        "target_write": "set_joint_position_target once to the captured finite 13-joint vector before settle",
        "settle": "advance exact normal physics/termination/observation cadence without Cartesian action processing or state writes",
        "settle_steps": 600,
        "gate_window": "exact final 10 settle steps",
        "joint_target_write_count": 1,
        "cartesian_action_manager_apply_count_during_settle": 0,
        "joint_or_object_state_write_count": 0,
        "gripper_semantics": "the six captured gripper joints remain the normal actuator target; no binary action is dispatched after handoff",
    }
    handoff_sha = hashlib.sha256(canonical_bytes(handoff)).hexdigest()
    lifecycle = {
        "fresh_reset_steps": 75,
        "open_approach_steps": 120,
        "open_descent_steps": 120,
        "normal_close_steps": 90,
        "object_space_servo_steps": 360,
        "joint_equilibrium_settle_steps": 600,
        "worst_case_steps": 1365,
        "registered_max_episode_length_steps": 1500,
        "registered_margin_steps": 135,
        "required_step_dt_s": 1.0 / 15.0,
        "registered_episode_length_s": 100.0,
        "only_construction_environment_timeout_changes": True,
        "behavioral_horizon_unchanged": True,
    }
    lifecycle_sha = hashlib.sha256(canonical_bytes(lifecycle)).hexdigest()

    predecessor = {
        "closure_commit": R007_CLOSURE_COMMIT,
        "registration": bind(registration_path),
        "candidate_schedule": bind(schedule_path),
        "source_push_gate_v2": bind(source_gate_path),
        "results": bind(results_path),
        "evidence_manifest": bind(manifest_path),
        "decision_memo": bind(memo_path),
        "target_validation_receipt": bind(receipt_path),
        "postexecution_validator_amendment": bind(amendment_path),
        "raw_result": r007_results["raw_result"],
        "outcome": "four diagnostics passed; four complete candidate pairs rejected; every OOD/camera/companion/frame gate passed; zero model/behavior",
    }
    frozen_inputs = deepcopy(r007_registration["frozen_inputs"])
    frozen_inputs.update({
        "r007_results": bind(results_path),
        "r007_evidence_manifest": bind(manifest_path),
        "r007_target_validation_receipt": bind(receipt_path),
        "r007_validator_amendment": bind(amendment_path),
    })
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r008-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "original_amendment_id": "V3-E006",
        "repair_amendment_id": "V3-E006-R008",
        "predecessor_repair_amendment_id": "V3-E006-R007",
        "registered_at_utc": args.registered_at_utc,
        "status": "prospectively_registered_before_any_r008_live_diagnostic_candidate_or_model_request",
        "counts_at_registration": {
            "r008_live_diagnostics": 0,
            "r008_live_candidate_evaluations": 0,
            "model_requests": 0,
            "behavioral_episodes": 0,
        },
        "user_authorized_override": "Continue solving canonical-state construction after valid R007 exhaustion; behavior remains blocked until a state pair passes unchanged gates.",
        "engineering_rationale_only": (
            "Across all eight R007 stages, OOD, cameras, companions, and frame identity passed. "
            "The fixed Cartesian stage hold left nondecaying kinetics and contact-family-specific midline/contact failures. "
            "R008 changes only construction control: a uniform live object-pose servo moves the cube toward its already frozen target, "
            "then a single captured-q normal-actuator handoff removes persistent Cartesian correction during settle."
        ),
        "r007_predecessor": predecessor,
        "object_space_servo_contract": servo,
        "object_space_servo_contract_sha256": servo_sha,
        "joint_handoff_contract": handoff,
        "joint_handoff_contract_sha256": handoff_sha,
        "construction_lifecycle_contract": lifecycle,
        "construction_lifecycle_contract_sha256": lifecycle_sha,
        "unchanged_science": {
            "r007_candidate_sources_targets_contact_transforms_rank_order_and_first_pass_rule_identical": True,
            "open_approach_open_descent_and_normal_close_commands_identical": True,
            "diagnostics_identical": True,
            "physics_ood_camera_companion_contact_midline_gates_and_thresholds_identical": True,
            "model_checkpoint_runtime_prompts_sample_size_behavioral_horizon_and_scorer_unchanged": True,
        },
        "prohibitions": [
            "no post-reset joint/object pose/velocity state write",
            "no weld/attachment/collision suppression/force injection",
            "no per-rank/stage/side gain or rate cap",
            "no early gate read, reacquire, adaptive retry, threshold or target change",
            "no model request or behavioral episode",
        ],
        "frozen_inputs": frozen_inputs,
        "release_boundary": "One zero-model diagnostic and conditional finite four-pair search only; no inference or behavior.",
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": r007_registration[
            "downstream_behavioral_design_if_and_only_if_state_pair_passes"
        ],
    }
    ARTIFACT.mkdir(parents=True)
    registration_path_out = ARTIFACT / "repair_registration.json"
    registration_path_out.write_bytes(canonical_bytes(registration))

    schedule = deepcopy(r007_schedule)
    archived_predecessor_contracts = {
        "status": "archived_lineage_only_not_active_r008_runtime_evidence",
        "r005_construction_horizon_contract": schedule.pop("construction_horizon_contract"),
        "r005_construction_horizon_contract_sha256": schedule.pop(
            "construction_horizon_contract_sha256"
        ),
        "r007_open_contact_construction_contract": schedule.pop(
            "open_contact_construction_contract"
        ),
        "r007_open_contact_construction_contract_sha256": schedule.pop(
            "open_contact_construction_contract_sha256"
        ),
    }
    schedule.update({
        "schema_version": "vla-wam-shared-v3e006-r008-candidate-schedule-v1",
        "repair_amendment_id": "V3-E006-R008",
        "status": "frozen_before_any_r008_live_diagnostic_candidate_or_model_request",
        "r008_live_diagnostic_count": 0,
        "r008_live_candidate_evaluation_count": 0,
        "repair_registration": bind(registration_path_out),
        "r007_predecessor": predecessor,
        "r007_candidate_schedule": bind(schedule_path),
        "object_space_servo_contract": servo,
        "object_space_servo_contract_sha256": servo_sha,
        "joint_handoff_contract": handoff,
        "joint_handoff_contract_sha256": handoff_sha,
        "construction_lifecycle_contract": lifecycle,
        "construction_lifecycle_contract_sha256": lifecycle_sha,
        "archived_predecessor_contracts": archived_predecessor_contracts,
    })
    schedule.pop("r007_live_diagnostic_count", None)
    schedule.pop("r007_live_candidate_evaluation_count", None)
    schedule.pop("schedule_canonical_sha256_without_this_field", None)
    schedule["selection_rule"] = deepcopy(r007_schedule["selection_rule"])
    schedule["selection_rule"]["exhaustion"] = (
        "Four valid rejected pairs exhaust R008 at zero model/behavior. Infrastructure-invalid attempts retry identically. "
        "No controller parameter, phase count, horizon, target, order, or scientific gate changes after live evaluation begins."
    )
    schedule["selection_rule"]["r008_construction_only_change"] = (
        "After the unchanged open approach/descent/normal close, run exactly 360 uniform object-space servo steps, "
        "capture q once, set that normal actuator target once, run exactly 600 direct-physics settle steps, then evaluate unchanged final-ten gates."
    )
    for pair in schedule["candidate_pairs"]:
        pair["construction_method"] = "exact_reset_open_close_uniform_object_servo_q_handoff"
        for stage_name in ("canonical_grasp", "canonical_carry"):
            stage = pair[stage_name]
            stage["r008_target_cube_pose"] = deepcopy(stage["target_cube_pose"])
            stage["r008_precontact_targets"] = deepcopy(stage["r007_open_contact_targets"])
    schedule["schedule_canonical_sha256_without_this_field"] = hashlib.sha256(canonical_bytes(schedule)).hexdigest()
    schedule_path_out = ARTIFACT / "gates/candidate_schedule.json"
    schedule_path_out.parent.mkdir()
    schedule_path_out.write_bytes(canonical_bytes(schedule))
    (ARTIFACT / "infrastructure_attempts.jsonl").write_bytes(b"")
    print(json.dumps({
        "registration": bind(registration_path_out),
        "candidate_schedule": bind(schedule_path_out),
        "object_space_servo_contract_sha256": servo_sha,
        "joint_handoff_contract_sha256": handoff_sha,
        "construction_lifecycle_contract_sha256": lifecycle_sha,
        "candidate_budget": 4,
        "diagnostic_budget": 4,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
