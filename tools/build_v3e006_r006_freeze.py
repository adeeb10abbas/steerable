#!/usr/bin/env python3
"""Freeze prospective V3-E006-R006 joint-equilibrium construction repair."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r006.predecessor_contract import (
    R005_RAW_RESULT_SHA256,
    validate_r005_exhaustion_closure,
)


ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006"
R005 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005"
R005_CLOSURE_COMMIT = "040cf75c1d83a2e5f8383d87247fb096e8d2491a"
R005_RESULTS_SHA256 = "550665a234c378cbcb5c8022d16249a980d1a5b5368b08900568c959c51fb9f2"


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
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-at-utc", required=True)
    args = parser.parse_args()
    if ARTIFACT.exists():
        parser.error(f"refusing to overwrite prospective R006 freeze: {ARTIFACT}")
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{R005_CLOSURE_COMMIT}^{{commit}}"],
        check=False,
    ).returncode:
        parser.error("R005 closure commit is absent")

    registration_path = R005 / "repair_registration.json"
    schedule_path = R005 / "gates/candidate_schedule.json"
    source_gate_path = R005 / "source_push_gate_v2.json"
    results_path = R005 / "results/results.json"
    manifest_path = R005 / "results/evidence_manifest.json"
    memo_path = R005 / "results/DECISION_MEMO.md"
    receipt_path = R005 / "results/target_validation_receipt.json"
    r005_registration = load(registration_path)
    r005_schedule = load(schedule_path)
    r005_results = load(results_path)
    if sha256(results_path) != R005_RESULTS_SHA256:
        parser.error("R005 result digest differs")
    validate_r005_exhaustion_closure(r005_results)

    equilibrium = {
        "contract_version": "uniform-normal-joint-equilibrium-hold-v1",
        "scope": "zero-model R006 candidate materialization only; diagnostics and behavioral runtime are unchanged",
        "applies_identically_to": "all four frozen ranks and both canonical_grasp/canonical_carry stages",
        "initialization": "atomically write exact accepted IK arm q plus the exact frozen historical closed-gripper q and reconstructed cube/contact pose, with all robot/cube velocities zero; identical to R005",
        "normal_actuator_control": "set the robot joint-position target once to that exact 13-joint q and advance normal simulator physics directly; no Cartesian pose action during settle",
        "settle_steps": 780,
        "gate_window": "exact final 10 of 780 physics steps",
        "physics_step_semantics": "for each step: action_manager.apply_action is prohibited; scene.write_data_to_sim, sim.step(render=false), scheduled camera render, scene.update(physics_dt), episode counters increment, unchanged termination manager compute, and observation manager compute",
        "termination_semantics": "fail closed on any unchanged non-timeout termination or the inherited construction time_out; no automatic reset",
        "registered_construction_max_episode_length_steps": 900,
        "required_episode_length_buf_before_settle": 75,
        "worst_case_materialization_steps": 855,
        "fixed_margin_after_reset_and_settle_steps": 45,
        "command_invariance": "the exact same 13-joint target is retained for all 780 steps; no measured-pose, rank, stage, side, or outcome adaptation",
        "prohibitions": [
            "no weld or attachment", "no collision suppression", "no force injection",
            "no learned-policy/model request", "no Cartesian pose-controller action during settle",
            "no joint or cube state rewrite after settle begins", "no threshold or target change",
        ],
        "scientific_gates": "R005 settled_gate, stage_ood, camera, companion, contact and cube-midline functions/thresholds remain byte/AST identical and evaluate the final ten samples",
    }
    equilibrium_sha = hashlib.sha256(canonical_bytes(equilibrium)).hexdigest()

    frozen_inputs = deepcopy(r005_registration["frozen_inputs"])
    frozen_inputs.update({
        "r005_registration": bind(registration_path),
        "r005_candidate_schedule": bind(schedule_path),
        "r005_source_push_gate": bind(source_gate_path),
        "r005_results": bind(results_path),
        "r005_evidence_manifest": bind(manifest_path),
        "r005_decision_memo": bind(memo_path),
        "r005_target_validation_receipt": bind(receipt_path),
    })
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r006-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "original_amendment_id": "V3-E006",
        "repair_amendment_id": "V3-E006-R006",
        "predecessor_repair_amendment_id": "V3-E006-R005",
        "registered_at_utc": args.registered_at_utc,
        "status": "prospectively_registered_before_any_r006_live_diagnostic_candidate_or_model_request",
        "counts_at_registration": {
            "r006_live_diagnostics": 0, "r006_live_candidate_evaluations": 0,
            "model_requests": 0, "behavioral_episodes": 0,
        },
        "user_authorized_override": "Continue solving the canonical-state construction controller after R005 while preserving every scientific gate and running no model until a valid state pair exists.",
        "engineering_rationale_only": (
            "R005 completed all four diagnostics and four pairs. Every stage retained passing OOD, camera and companion gates. "
            "The left-contact grasp/carry variants also retained normal gripper contact and cube |y|<1mm, but the Cartesian "
            "pose action kept driving residual arm/cube motion during the final ten of 260 steps despite post-write FK errors "
            "of only micrometres and millidegrees. R006 changes only the construction settle control and duration; it does not "
            "change any candidate, target, contact transform, acceptance gate or threshold."
        ),
        "r005_predecessor": {
            "closure_commit": R005_CLOSURE_COMMIT,
            "registration": bind(registration_path), "candidate_schedule": bind(schedule_path),
            "source_push_gate": bind(source_gate_path), "results": bind(results_path),
            "evidence_manifest": bind(manifest_path), "decision_memo": bind(memo_path),
            "target_validation_receipt": bind(receipt_path),
            "raw_result": r005_results["raw_result"],
            "outcome": "four diagnostics passed; four complete candidate pairs rejected by unchanged physics gates; zero model/behavior",
        },
        "joint_equilibrium_hold_contract": equilibrium,
        "joint_equilibrium_hold_contract_sha256": equilibrium_sha,
        "unchanged_candidate_and_scientific_contract": {
            "r005_schedule": bind(schedule_path),
            "diagnostic_sources_targets_and_runtime_byte_identical": True,
            "candidate_sources_targets_contact_transforms_rank_order_and_ik_byte_identical": True,
            "construction_horizon_900_byte_identical": True,
            "final_ten_physics_ood_camera_companion_contact_midline_gates_byte_or_ast_identical": True,
            "candidate_budget": 4, "diagnostic_budget": 4,
            "first_complete_passing_pair_stops": True,
        },
        "release_boundary": "This registration authorizes one zero-model diagnostic and conditional finite state search only. It does not release a policy request or behavioral episode.",
        "frozen_inputs": frozen_inputs,
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": r005_registration[
            "downstream_behavioral_design_if_and_only_if_state_pair_passes"
        ],
    }
    ARTIFACT.mkdir(parents=True)
    out_registration = ARTIFACT / "repair_registration.json"
    out_registration.write_bytes(canonical_bytes(registration))

    schedule = deepcopy(r005_schedule)
    schedule.update({
        "schema_version": "vla-wam-shared-v3e006-r006-candidate-schedule-v1",
        "repair_amendment_id": "V3-E006-R006",
        "status": "frozen_before_any_r006_live_diagnostic_candidate_or_model_request",
        "r006_live_diagnostic_count": 0, "r006_live_candidate_evaluation_count": 0,
        "repair_registration": bind(out_registration),
        "r005_predecessor": registration["r005_predecessor"],
        "r005_target_solver_horizon_schedule": bind(schedule_path),
        "joint_equilibrium_hold_contract": equilibrium,
        "joint_equilibrium_hold_contract_sha256": equilibrium_sha,
    })
    schedule.pop("r005_live_diagnostic_count", None)
    schedule.pop("r005_live_candidate_evaluation_count", None)
    schedule.pop("schedule_canonical_sha256_without_this_field", None)
    schedule["selection_rule"] = deepcopy(r005_schedule["selection_rule"])
    schedule["selection_rule"]["exhaustion"] = (
        "Four valid rejected pairs close R006 at zero model requests and zero behavioral episodes. Infrastructure-invalid "
        "attempts are retried identically and are not rank rejection. No settle controller/count, horizon, source, target, "
        "waypoint, contact transform, solver/correction parameter, order, or scientific gate may change after live evaluation begins."
    )
    schedule["selection_rule"]["r006_construction_control_only_change"] = (
        "After the unchanged accepted IK/contact state write, hold the exact 13-joint q through normal joint actuators for exactly "
        "780 direct physics steps; no Cartesian action during settle; evaluate the unchanged final-ten gates."
    )
    schedule["schedule_canonical_sha256_without_this_field"] = hashlib.sha256(canonical_bytes(schedule)).hexdigest()
    out_schedule = ARTIFACT / "gates/candidate_schedule.json"
    out_schedule.parent.mkdir()
    out_schedule.write_bytes(canonical_bytes(schedule))
    print(json.dumps({
        "registration": bind(out_registration), "candidate_schedule": bind(out_schedule),
        "joint_equilibrium_hold_contract_sha256": equilibrium_sha,
        "settle_steps": 780, "diagnostic_budget": 4, "candidate_budget": 4,
        "model_request_count": 0, "behavioral_episode_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
