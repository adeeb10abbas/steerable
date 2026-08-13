#!/usr/bin/env python3
"""Freeze the prospective V3-E006-R004 symmetric residual-correction repair."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r004"
R003 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r003"
R003_CLOSURE_COMMIT = "fc2f23a915a98f368181dd897994b67d64a35eeb"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path, *, relative: bool = True) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path.relative_to(ROOT)) if relative else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-at-utc", required=True)
    args = parser.parse_args()
    if ARTIFACT.exists():
        parser.error(f"refusing to overwrite prospective R004 freeze: {ARTIFACT}")
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{R003_CLOSURE_COMMIT}^{{commit}}"],
        check=False,
    ).returncode:
        parser.error("R003 closure commit is absent")

    r003_registration_path = R003 / "repair_registration.json"
    r003_schedule_path = R003 / "gates/candidate_schedule.json"
    r003_results_path = R003 / "results/results.json"
    r003_manifest_path = R003 / "results/evidence_manifest.json"
    r003_memo_path = R003 / "DECISION_MEMO.md"
    r003_registration = load(r003_registration_path)
    r003_schedule = load(r003_schedule_path)
    r003_results = load(r003_results_path)
    r003_manifest = load(r003_manifest_path)
    if (
        sha256(r003_results_path) != "d30ed1dae4f69992cf257b9430fc9165129185d2be9a2af3bc1a2dd9ea5d1261"
        or sha256(r003_manifest_path) != "b6734b41cba5367f578982e94c1a438c561b128a162d4de721bb2247eaae9818"
        or r003_results.get("status")
        != "r003_known_reachable_diagnostic_failed_candidates_not_evaluated"
        or r003_results.get("model_request_count") != 0
        or r003_results.get("behavioral_episode_count") != 0
        or r003_results.get("diagnostic_evaluation_count") != 1
        or r003_results.get("candidate_pair_evaluation_count") != 0
        or r003_manifest.get("status") != "hash_closed_registered_diagnostic_failure"
    ):
        parser.error("R003 predecessor is not the immutable zero-request diagnostic closure")

    correction = {
        "algorithm_version": "symmetric-se3-measured-residual-correction-v1",
        "desired_target_immutability": "The desired base_link target is byte-identical to R003 and never changes. Only the command sent to the fixed controller is corrected.",
        "initial_command": "command_position_0 = desired_position; command_quaternion_0 = desired_quaternion",
        "maximum_correction_rounds": 3,
        "hold_steps_per_round": 30,
        "required_final_consecutive_steps": 10,
        "position_error_m_inclusive": 0.001,
        "orientation_geodesic_error_deg_inclusive": 1.0,
        "translation_gain": 1.0,
        "translation_update": "command_position_next = command_position_current + 1.0 * (desired_position - measured_final_position)",
        "rotation_gain": 1.0,
        "rotation_update": "q_error_world = desired_quaternion * inverse(measured_final_quaternion); choose the q_error_world antipode with nonnegative scalar (w==0 uses lexicographically nonnegative xyz); q_delta = exp(1.0 * log(q_error_world)); command_quaternion_next = normalize(q_delta * command_quaternion_current)",
        "measurement": "env-local robot/base_link pose after the last completed step of the current 30-step hold",
        "early_stop": "stop at the first round whose final ten samples all satisfy the unchanged inclusive 1mm/1deg criterion",
        "termination": "any termination, nonfinite state, frame-identity failure, or arm soft-limit violation fails the pose hold without correction",
        "symmetry": "identical gains, equations, round budget, step budget, thresholds, and ordering for every direction, stage, diagnostic, candidate rank, and waypoint",
        "adaptation_prohibited": "No gain, round, target, waypoint, source, order, or threshold may change after the first R004 live diagnostic begins.",
    }
    correction_sha = hashlib.sha256(canonical_bytes(correction)).hexdigest()

    frozen_inputs = deepcopy(r003_registration["frozen_inputs"])
    frozen_inputs.update(
        {
            "r003_registration": bind(r003_registration_path),
            "r003_candidate_schedule": bind(r003_schedule_path),
            "r003_results": bind(r003_results_path),
            "r003_evidence_manifest": bind(r003_manifest_path),
            "r003_decision_memo": bind(r003_memo_path),
        }
    )
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r004-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "original_amendment_id": "V3-E006",
        "repair_amendment_id": "V3-E006-R004",
        "predecessor_repair_amendment_id": "V3-E006-R003",
        "registered_at_utc": args.registered_at_utc,
        "status": "prospectively_registered_before_any_r004_live_diagnostic_candidate_or_model_request",
        "counts_at_registration": {
            "r004_live_diagnostics": 0,
            "r004_live_candidate_evaluations": 0,
            "model_requests": 0,
            "behavioral_episodes": 0,
        },
        "user_authorized_override": "Continue solving the canonical-state construction gate after R003 while preserving every scientific gate and running no model until a valid state pair exists.",
        "engineering_rationale_only": "R003 corrected the historical frame mismatch and reproduced its anchor exactly before physics. Its first fixed-controller hold converged to a stable 1.2278mm position residual, 0.2278mm outside the unchanged 1mm diagnostic tolerance, while orientation, reset, cameras, frame identity, finiteness, and limits passed. This motivates a controller-residual correction, not a gate or target change.",
        "r003_predecessor": {
            "closure_commit": R003_CLOSURE_COMMIT,
            "results": bind(r003_results_path),
            "evidence_manifest": bind(r003_manifest_path),
            "decision_memo": bind(r003_memo_path),
            "raw_result": r003_results["raw_evidence"]["result"],
            "target_validation_receipt": r003_results["raw_evidence"]["target_validation_receipt"],
            "outcome": "one registered diagnostic failed; zero candidate/model/behavior evaluations",
        },
        "unchanged_target_and_schedule_contract": {
            "r003_schedule": bind(r003_schedule_path),
            "diagnostic_sources_targets_order_byte_identical": True,
            "candidate_sources_targets_contact_transforms_rank_order_byte_identical": True,
            "waypoint_desired_poses_byte_identical": True,
            "candidate_budget": 4,
            "diagnostic_budget": 4,
            "first_complete_passing_pair_stops": True,
        },
        "residual_correction_contract": correction,
        "residual_correction_contract_sha256": correction_sha,
        "unchanged_scientific_gate": {
            "diagnostic_pose_tolerance": "final 10 consecutive samples at <=1mm position and <=1deg geodesic orientation",
            "candidate_physics_ood_camera_companion_contact_reset_gates": "byte/AST-identical R003 functions and frozen E006 state_contract/OOD threshold inputs",
            "no_weld_attachment_collision_suppression_force_injection_model_or_prompt_conditioning": True,
        },
        "known_reachable_diagnostic": "All four frozen R003 historical base_link anchors must pass the same R004 correction algorithm. Any failure blocks every candidate.",
        "candidate_search": "Only after all four diagnostics pass, evaluate the same four complete grasp/carry pairs in the same rank order. Apply the same correction algorithm independently at every frozen waypoint. First complete pair passing all unchanged gates stops; four valid rejected pairs exhaust R004.",
        "release_boundary": "This registration authorizes one zero-model diagnostic and conditional finite state search only. It does not release any policy request or behavioral episode.",
        "frozen_inputs": frozen_inputs,
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": r003_registration[
            "downstream_behavioral_design_if_and_only_if_state_pair_passes"
        ],
    }
    ARTIFACT.mkdir(parents=True)
    registration_path = ARTIFACT / "repair_registration.json"
    registration_path.write_bytes(canonical_bytes(registration))

    schedule = deepcopy(r003_schedule)
    schedule.update(
        {
            "schema_version": "vla-wam-shared-v3e006-r004-candidate-schedule-v1",
            "repair_amendment_id": "V3-E006-R004",
            "status": "frozen_before_any_r004_live_diagnostic_candidate_or_model_request",
            "r004_live_diagnostic_count": 0,
            "r004_live_candidate_evaluation_count": 0,
            "repair_registration": bind(registration_path),
            "r003_predecessor": registration["r003_predecessor"],
            "r003_target_schedule": bind(r003_schedule_path),
            "residual_correction_contract": correction,
            "residual_correction_contract_sha256": correction_sha,
        }
    )
    schedule.pop("r003_live_diagnostic_count", None)
    schedule.pop("r003_live_candidate_evaluation_count", None)
    schedule.pop("schedule_canonical_sha256_without_this_field", None)
    schedule["selection_rule"] = deepcopy(r003_schedule["selection_rule"])
    schedule["selection_rule"]["exhaustion"] = (
        "Four valid rejected pairs close R004 at zero model requests and zero behavioral episodes. "
        "Infrastructure-invalid attempts are retried identically and are not rank rejection. No seed, "
        "waypoint, target, contact transform, solver parameter, correction parameter, or gate may be "
        "changed after the diagnostic or candidate run begins."
    )
    schedule["selection_rule"]["r004_solver_only_change"] = (
        "Apply the one frozen symmetric measured-residual correction contract to every diagnostic and every candidate waypoint; desired poses, pair order, and scientific selection are unchanged."
    )
    for diagnostic in schedule["known_reachable_diagnostics"]:
        diagnostic["r004_residual_correction_contract_sha256"] = correction_sha
        diagnostic["maximum_correction_rounds"] = 3
    for pair in schedule["candidate_pairs"]:
        for stage in ("canonical_grasp", "canonical_carry"):
            initialization = pair[stage].pop("r003_solver_initialization")
            initialization["residual_correction_contract_sha256"] = correction_sha
            for waypoint in initialization["waypoints"]:
                waypoint["maximum_correction_rounds"] = 3
                waypoint["r004_residual_correction_contract_sha256"] = correction_sha
            pair[stage]["r004_solver_initialization"] = initialization
    schedule["schedule_canonical_sha256_without_this_field"] = hashlib.sha256(
        canonical_bytes(schedule)
    ).hexdigest()
    schedule_path = ARTIFACT / "gates/candidate_schedule.json"
    schedule_path.parent.mkdir()
    schedule_path.write_bytes(canonical_bytes(schedule))
    print(json.dumps({
        "registration": bind(registration_path),
        "candidate_schedule": bind(schedule_path),
        "residual_correction_contract_sha256": correction_sha,
        "diagnostic_budget": 4,
        "candidate_budget": 4,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
