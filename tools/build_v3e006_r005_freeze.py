#!/usr/bin/env python3
"""Freeze the prospective V3-E006-R005 construction-horizon repair."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005"
R004 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r004"
R004_CLOSURE_COMMIT = "4775965d721f2c1e8c875bcec566d7436162cb91"
R004_RESULTS_SHA256 = "aa5eebfd76064b4aa2664b5d296461fd2a8cdbbf489175b3148475f9a2546006"
R004_RAW_RESULT_SHA256 = "54c2335c5c4339037bd5f7e7e76ab15c5485191d82de253cc61d27dd66ddb81d"


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
        parser.error(f"refusing to overwrite prospective R005 freeze: {ARTIFACT}")
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{R004_CLOSURE_COMMIT}^{{commit}}"],
        check=False,
    ).returncode:
        parser.error("R004 closure commit is absent")

    r004_registration_path = R004 / "repair_registration.json"
    r004_schedule_path = R004 / "gates/candidate_schedule.json"
    r004_source_gate_path = R004 / "source_push_gate.json"
    r004_results_path = R004 / "results/results.json"
    r004_manifest_path = R004 / "results/evidence_manifest.json"
    r004_memo_path = R004 / "DECISION_MEMO.md"
    r004_registration = load(r004_registration_path)
    r004_schedule = load(r004_schedule_path)
    r004_results = load(r004_results_path)
    if (
        sha256(r004_results_path) != R004_RESULTS_SHA256
        or r004_results.get("status")
        != "r004_candidate_budget_exhausted_construction_time_limit_before_materialization"
        or r004_results.get("model_request_count") != 0
        or r004_results.get("behavioral_episode_count") != 0
        or r004_results.get("diagnostic_evaluation_count") != 4
        or r004_results.get("candidate_pair_evaluation_count") != 4
        or r004_results.get("stage_solve_count") != 8
        or r004_results.get("accepted_candidate_rank") is not None
        or r004_results.get("raw_evidence", {}).get("result", {}).get("sha256")
        != R004_RAW_RESULT_SHA256
    ):
        parser.error("R004 predecessor is not the immutable zero-request timeout closure")
    signature = r004_results.get("termination_signature", {})
    if signature != {
        "all_eight_stages_identical": True,
        "common_step_counter": 450,
        "max_episode_length": 450,
        "phase": "registered_waypoint_07_of_08_correction_round_01",
        "phase_step_one_based": 15,
        "success": False,
        "terminated": False,
        "time_out": True,
        "truncated": True,
    }:
        parser.error("R004 timeout diagnosis differs")

    # One construction lifecycle parameter changes. The bound is derived before
    # any R005 live environment exists: 75 reset steps plus 8 waypoints times
    # 3 correction rounds times 30 steps = 795. Nine hundred leaves 105 steps.
    horizon = {
        "contract_version": "construction-timeout-only-v1",
        "scope": "zero-model R005 construction environments only; behavioral E004 horizon remains frozen and untouched",
        "activation_point": "immediately after create_env returns and before env.reset or env.step",
        "applies_to_roles": ["known_reachable_diagnostic", "ik_solve", "candidate_materialization"],
        "original_max_episode_length_steps": 450,
        "registered_max_episode_length_steps": 900,
        "original_episode_length_s": 30.0,
        "registered_episode_length_s": 60.0,
        "required_step_dt_s": 0.06666666666666667,
        "worst_case_derivation": {
            "fresh_reset_steps": 75,
            "waypoint_count": 8,
            "maximum_correction_rounds_per_waypoint": 3,
            "hold_steps_per_round": 30,
            "worst_case_steps": 795,
            "registered_margin_steps": 105,
        },
        "only_allowed_runtime_mutation": "env.cfg.episode_length_s = 900 * env.step_dt; returned_cfg must be the identical config object and env.max_episode_length must then derive to 900",
        "termination_contract": "time_out remains active and unchanged except for its max-episode-length input; success and every other termination term/config remain byte-identical",
        "automatic_episode_length_buffer_reset_prohibited": True,
        "waypoint_boundary_reset_prohibited": True,
        "behavioral_episode_horizon_unchanged": True,
        "adaptation_prohibited": "No horizon, target, correction, order, source, gate, or threshold may change after the first R005 live diagnostic begins.",
    }
    horizon_sha = hashlib.sha256(canonical_bytes(horizon)).hexdigest()

    frozen_inputs = deepcopy(r004_registration["frozen_inputs"])
    frozen_inputs.update(
        {
            "r004_registration": bind(r004_registration_path),
            "r004_candidate_schedule": bind(r004_schedule_path),
            "r004_source_push_gate": bind(r004_source_gate_path),
            "r004_results": bind(r004_results_path),
            "r004_evidence_manifest": bind(r004_manifest_path),
            "r004_decision_memo": bind(r004_memo_path),
        }
    )
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r005-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "original_amendment_id": "V3-E006",
        "repair_amendment_id": "V3-E006-R005",
        "predecessor_repair_amendment_id": "V3-E006-R004",
        "registered_at_utc": args.registered_at_utc,
        "status": "prospectively_registered_before_any_r005_live_diagnostic_candidate_or_model_request",
        "counts_at_registration": {
            "r005_live_diagnostics": 0,
            "r005_live_candidate_evaluations": 0,
            "model_requests": 0,
            "behavioral_episodes": 0,
        },
        "user_authorized_override": "Continue solving the canonical-state construction lifecycle after R004 while preserving every scientific gate and running no model until a valid state pair exists.",
        "engineering_rationale_only": (
            "R004 passed all four registered reachability diagnostics after its frozen residual correction. "
            "Every one of eight candidate stage solves then stopped identically at construction step 450: "
            "time_out=true, truncated=true, terminated=false, success=false, before materialization. "
            "No safety, contact, physics, OOD, camera, or companion candidate gate was reached."
        ),
        "r004_predecessor": {
            "closure_commit": R004_CLOSURE_COMMIT,
            "registration": bind(r004_registration_path),
            "candidate_schedule": bind(r004_schedule_path),
            "source_push_gate": bind(r004_source_gate_path),
            "results": bind(r004_results_path),
            "evidence_manifest": bind(r004_manifest_path),
            "decision_memo": bind(r004_memo_path),
            "raw_result": r004_results["raw_evidence"]["result"],
            "target_validation_receipt": r004_results["raw_evidence"]["target_validation_receipt"],
            "outcome": "four diagnostics passed; all eight stage solves hit only time_out at step 450; zero state/model/behavior",
        },
        "construction_horizon_contract": horizon,
        "construction_horizon_contract_sha256": horizon_sha,
        "unchanged_target_correction_schedule_and_gate_contract": {
            "r004_schedule": bind(r004_schedule_path),
            "diagnostic_sources_targets_order_byte_identical": True,
            "candidate_sources_targets_contact_transforms_rank_order_byte_identical": True,
            "waypoint_desired_poses_byte_identical": True,
            "residual_correction_contract_byte_identical": True,
            "diagnostic_tolerance_byte_identical": True,
            "candidate_physics_ood_camera_companion_contact_reset_gates_byte_or_ast_identical": True,
            "candidate_budget": 4,
            "diagnostic_budget": 4,
            "first_complete_passing_pair_stops": True,
        },
        "release_boundary": "This registration authorizes one zero-model diagnostic and conditional finite state search only. It does not release any policy request or behavioral episode.",
        "frozen_inputs": frozen_inputs,
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": r004_registration[
            "downstream_behavioral_design_if_and_only_if_state_pair_passes"
        ],
    }
    ARTIFACT.mkdir(parents=True)
    registration_path = ARTIFACT / "repair_registration.json"
    registration_path.write_bytes(canonical_bytes(registration))

    schedule = deepcopy(r004_schedule)
    schedule.update(
        {
            "schema_version": "vla-wam-shared-v3e006-r005-candidate-schedule-v1",
            "repair_amendment_id": "V3-E006-R005",
            "status": "frozen_before_any_r005_live_diagnostic_candidate_or_model_request",
            "r005_live_diagnostic_count": 0,
            "r005_live_candidate_evaluation_count": 0,
            "repair_registration": bind(registration_path),
            "r004_predecessor": registration["r004_predecessor"],
            "r004_target_correction_schedule": bind(r004_schedule_path),
            "construction_horizon_contract": horizon,
            "construction_horizon_contract_sha256": horizon_sha,
        }
    )
    schedule.pop("r004_live_diagnostic_count", None)
    schedule.pop("r004_live_candidate_evaluation_count", None)
    schedule.pop("schedule_canonical_sha256_without_this_field", None)
    schedule["selection_rule"] = deepcopy(r004_schedule["selection_rule"])
    schedule["selection_rule"]["exhaustion"] = (
        "Four valid rejected pairs close R005 at zero model requests and zero behavioral episodes. "
        "Infrastructure-invalid attempts are retried identically and are not rank rejection. No horizon, "
        "source, target, waypoint, contact transform, controller/correction parameter, order, or scientific "
        "gate may change after the diagnostic or candidate run begins."
    )
    schedule["selection_rule"]["r005_construction_lifecycle_only_change"] = (
        "Set the zero-model construction max episode length from 450 to 900 immediately after each fresh environment is created and before any reset/step; retain time_out and every other termination term, behavioral horizon, target, correction, order, and gate unchanged."
    )
    schedule["schedule_canonical_sha256_without_this_field"] = hashlib.sha256(
        canonical_bytes(schedule)
    ).hexdigest()
    schedule_path = ARTIFACT / "gates/candidate_schedule.json"
    schedule_path.parent.mkdir()
    schedule_path.write_bytes(canonical_bytes(schedule))
    print(json.dumps({
        "registration": bind(registration_path),
        "candidate_schedule": bind(schedule_path),
        "construction_horizon_contract_sha256": horizon_sha,
        "registered_max_episode_length_steps": 900,
        "worst_case_steps": 795,
        "diagnostic_budget": 4,
        "candidate_budget": 4,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
