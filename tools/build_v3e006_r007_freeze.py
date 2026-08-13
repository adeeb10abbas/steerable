#!/usr/bin/env python3
"""Freeze prospective V3-E006-R007 open-contact construction repair."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r007.predecessor_contract import (
    R006_RAW_RESULT_SHA256,
    validate_r006_exhaustion_closure,
)


ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007"
R006 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006"
R006_CLOSURE_COMMIT = "125e8f0d231ebd2e3c7d0d9b54dce83e1080cea1"
R006_RESULTS_SHA256 = "3c58721d11f669243690aaf3619121d1c348bf788ca56aacd2a009f727065e63"


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


def _quat_matrix(value: Any) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _matrix_quat(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=np.float64)
    w = np.sqrt(max(0.0, 1.0 + float(np.trace(m)))) / 2.0
    if w > 1e-9:
        q = np.asarray([w, (m[2, 1] - m[1, 2]) / (4 * w),
                        (m[0, 2] - m[2, 0]) / (4 * w),
                        (m[1, 0] - m[0, 1]) / (4 * w)])
    else:
        index = int(np.argmax(np.diag(m)))
        scale = np.sqrt(max(0.0, 1.0 + m[index, index] - m[(index + 1) % 3, (index + 1) % 3] - m[(index + 2) % 3, (index + 2) % 3])) * 2
        if index == 0:
            q = np.asarray([(m[2, 1] - m[1, 2]) / scale, scale / 4,
                            (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale])
        elif index == 1:
            q = np.asarray([(m[0, 2] - m[2, 0]) / scale,
                            (m[0, 1] + m[1, 0]) / scale, scale / 4,
                            (m[1, 2] + m[2, 1]) / scale])
        else:
            q = np.asarray([(m[1, 0] - m[0, 1]) / scale,
                            (m[0, 2] + m[2, 0]) / scale,
                            (m[1, 2] + m[2, 1]) / scale, scale / 4])
    q /= np.linalg.norm(q)
    if q[0] < 0 or (q[0] == 0 and tuple(q[1:]) < (0.0, 0.0, 0.0)):
        q = -q
    return [float(v) for v in q]


def _pose(position: Any, quaternion: Any) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = _quat_matrix(quaternion)
    value[:3, 3] = np.asarray(position, dtype=np.float64)
    return value


def _pose_record(value: np.ndarray) -> dict[str, Any]:
    return {
        "position_world_m": [float(v) for v in value[:3, 3]],
        "quaternion_world_wxyz": _matrix_quat(value[:3, :3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-at-utc", required=True)
    args = parser.parse_args()
    if ARTIFACT.exists():
        parser.error(f"refusing to overwrite prospective R007 freeze: {ARTIFACT}")
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{R006_CLOSURE_COMMIT}^{{commit}}"],
        check=False,
    ).returncode:
        parser.error("R006 closure commit is absent")

    registration_path = R006 / "repair_registration.json"
    schedule_path = R006 / "gates/candidate_schedule.json"
    source_gate_path = R006 / "source_push_gate.json"
    results_path = R006 / "results/results.json"
    manifest_path = R006 / "results/evidence_manifest.json"
    memo_path = R006 / "results/DECISION_MEMO.md"
    receipt_path = R006 / "results/target_validation_receipt.json"
    r006_registration = load(registration_path)
    r006_schedule = load(schedule_path)
    r006_results = load(results_path)
    if sha256(results_path) != R006_RESULTS_SHA256:
        parser.error("R006 result digest differs")
    validate_r006_exhaustion_closure(r006_results)

    construction = {
        "contract_version": "direction-neutral-open-approach-normal-close-lift-v1",
        "scope": "zero-model R007 candidate materialization only; diagnostics and behavioral runtime are unchanged",
        "applies_identically_to": "all four frozen ranks and both canonical_grasp/canonical_carry stages",
        "initial_state": "exact E004 full reset after its registered 75-step settle; cube remains on the table and the gripper is open; no post-reset joint, object-pose, or velocity state write",
        "controller": "exact E004 DroidIKActionCfg absolute base_link pose command, quaternion WXYZ, with normal binary gripper command",
        "waypoint_interpolation": "piecewise position lerp plus sign-invariant shortest-arc quaternion SLERP; endpoints and counts below are fixed",
        "phase_steps": {
            "open_approach": 120,
            "open_descent": 120,
            "normal_close": 90,
            "closed_lift_to_registered_stage_target": 180,
            "closed_settle_at_registered_stage_target": 300,
        },
        "phase_targets": {
            "reset_start": "the observed exact-E004-reset live robot/base_link pose",
            "contact": "T_world_reset_cube * inverse(frozen selected observed cube-in-base_link contact transform)",
            "approach": "contact base_link pose plus exactly +0.060 m in world z, same orientation",
            "stage_target": "the unchanged R006 centerline_constrained_base_link_ik_target",
        },
        "gripper_commands": {
            "open_approach": 0.0, "open_descent": 0.0, "normal_close": 1.0,
            "closed_lift_to_registered_stage_target": 1.0,
            "closed_settle_at_registered_stage_target": 1.0,
        },
        "pose_command_rule": "commands follow the fixed geometric waypoint schedule exactly; no measured-state residual correction, rank/stage/side-specific gain, or adaptive waypoint is used",
        "gate_window": "exact final 10 of the 300 closed-settle steps",
        "registered_construction_max_episode_length_steps": 900,
        "required_episode_length_buf_before_candidate_actions": 75,
        "candidate_action_steps": 810,
        "worst_case_materialization_steps": 885,
        "fixed_margin_steps": 15,
        "termination_semantics": "fail closed on any unchanged non-timeout termination or the inherited construction time_out; no automatic reset",
        "direction_neutrality": "cube target has world y exactly zero; no prompt, requested direction, or lateral tuning enters construction; ranks differ only by the already frozen paired historical contact-transform selector order",
        "prohibitions": [
            "no weld or attachment", "no collision suppression", "no force injection",
            "no learned-policy/model request", "no post-reset joint or object state write",
            "no prompt or requested-side construction input", "no threshold or target change",
        ],
        "scientific_gates": "R006 settled_gate, stage_ood, camera, companion, contact and cube-midline functions/thresholds remain byte/AST identical and evaluate the final ten samples",
    }
    construction_sha = hashlib.sha256(canonical_bytes(construction)).hexdigest()

    frozen_inputs = deepcopy(r006_registration["frozen_inputs"])
    frozen_inputs.update({
        "r006_registration": bind(registration_path),
        "r006_candidate_schedule": bind(schedule_path),
        "r006_source_push_gate": bind(source_gate_path),
        "r006_results": bind(results_path),
        "r006_evidence_manifest": bind(manifest_path),
        "r006_decision_memo": bind(memo_path),
        "r006_target_validation_receipt": bind(receipt_path),
    })
    registration = {
        "schema_version": "vla-wam-shared-v3e006-r007-repair-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "original_amendment_id": "V3-E006",
        "repair_amendment_id": "V3-E006-R007",
        "predecessor_repair_amendment_id": "V3-E006-R006",
        "registered_at_utc": args.registered_at_utc,
        "status": "prospectively_registered_before_any_r007_live_diagnostic_candidate_or_model_request",
        "counts_at_registration": {
            "r007_live_diagnostics": 0, "r007_live_candidate_evaluations": 0,
            "model_requests": 0, "behavioral_episodes": 0,
        },
        "user_authorized_override": "Continue solving the canonical-state construction controller after R006 while preserving every scientific gate and running no model until a valid state pair exists.",
        "engineering_rationale_only": (
            "R006 completed all four diagnostics and four pairs. Every stage retained passing OOD, camera and companion gates, "
            "but direct injection of independently reconstructed cube/contact state relaxed under ordinary physics into lateral "
            "drift, excess motion, or lost contact. R007 removes that injection: it starts at exact E004 reset and uses only the "
            "frozen absolute-IK action interface plus a normal gripper close and lift. No target, contact-transform source, rank, "
            "acceptance gate, or threshold changes."
        ),
        "r006_predecessor": {
            "closure_commit": R006_CLOSURE_COMMIT,
            "registration": bind(registration_path), "candidate_schedule": bind(schedule_path),
            "source_push_gate": bind(source_gate_path), "results": bind(results_path),
            "evidence_manifest": bind(manifest_path), "decision_memo": bind(memo_path),
            "target_validation_receipt": bind(receipt_path),
            "raw_result": r006_results["raw_result"],
            "outcome": "four diagnostics passed; four complete candidate pairs rejected by unchanged physics gates; zero model/behavior",
        },
        "open_contact_construction_contract": construction,
        "open_contact_construction_contract_sha256": construction_sha,
        "unchanged_candidate_and_scientific_contract": {
            "r006_schedule": bind(schedule_path),
            "diagnostic_sources_targets_and_runtime_byte_identical": True,
            "candidate_sources_targets_contact_transforms_and_rank_order_byte_identical": True,
            "candidate_state_write_and_equilibrium_hold_replaced_by_exact_reset_plus_normal_actions": True,
            "construction_horizon_900_byte_identical": True,
            "final_ten_physics_ood_camera_companion_contact_midline_gates_byte_or_ast_identical": True,
            "candidate_budget": 4, "diagnostic_budget": 4,
            "first_complete_passing_pair_stops": True,
        },
        "release_boundary": "This registration authorizes one zero-model diagnostic and conditional finite state search only. It does not release a policy request or behavioral episode.",
        "frozen_inputs": frozen_inputs,
        "downstream_behavioral_design_if_and_only_if_state_pair_passes": r006_registration[
            "downstream_behavioral_design_if_and_only_if_state_pair_passes"
        ],
    }
    ARTIFACT.mkdir(parents=True)
    out_registration = ARTIFACT / "repair_registration.json"
    out_registration.write_bytes(canonical_bytes(registration))

    schedule = deepcopy(r006_schedule)
    schedule.update({
        "schema_version": "vla-wam-shared-v3e006-r007-candidate-schedule-v1",
        "repair_amendment_id": "V3-E006-R007",
        "status": "frozen_before_any_r007_live_diagnostic_candidate_or_model_request",
        "r007_live_diagnostic_count": 0, "r007_live_candidate_evaluation_count": 0,
        "repair_registration": bind(out_registration),
        "r006_predecessor": registration["r006_predecessor"],
        "r006_target_solver_horizon_schedule": bind(schedule_path),
        "open_contact_construction_contract": construction,
        "open_contact_construction_contract_sha256": construction_sha,
    })
    schedule.pop("r006_live_diagnostic_count", None)
    schedule.pop("r006_live_candidate_evaluation_count", None)
    schedule.pop("schedule_canonical_sha256_without_this_field", None)
    schedule["selection_rule"] = deepcopy(r006_schedule["selection_rule"])
    schedule["selection_rule"]["exhaustion"] = (
        "Four valid rejected pairs close R007 at zero model requests and zero behavioral episodes. Infrastructure-invalid "
        "attempts are retried identically and are not rank rejection. No settle controller/count, horizon, source, target, "
        "waypoint, contact transform, solver/correction parameter, order, or scientific gate may change after live evaluation begins."
    )
    schedule["selection_rule"]["r007_construction_only_change"] = (
        "Start every candidate stage from exact E004 reset with an open gripper and table cube; execute the one frozen 810-step "
        "direction-neutral approach/descent/normal-close/lift/settle action schedule; perform no post-reset state write; evaluate "
        "the unchanged final-ten gates."
    )
    reset_reference_path = (
        ROOT
        / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/gates/e004_full_reset_reference.json"
    )
    reset_cube_row = load(reset_reference_path)["rigid_objects"]["rubiks_cube"]
    reset_cube = _pose(
        reset_cube_row["root_position"]["values"],
        reset_cube_row["root_quaternion_wxyz"]["values"],
    )
    for pair in schedule["candidate_pairs"]:
        pair["construction_method"] = "exact_reset_open_approach_normal_close_lift"
        for stage_name in ("canonical_grasp", "canonical_carry"):
            stage = pair[stage_name]
            relative = stage["selected_observed_cube_in_base_link_transform"]
            base_to_cube = _pose(relative["translation_m"], relative["quaternion_wxyz"])
            reset_contact = reset_cube @ np.linalg.inv(base_to_cube)
            approach = reset_contact.copy()
            approach[2, 3] += 0.060
            stage["r007_open_contact_targets"] = {
                "reset_cube_pose": _pose_record(reset_cube),
                "contact_base_link_pose_at_exact_reset_cube": _pose_record(reset_contact),
                "approach_base_link_pose": _pose_record(approach),
                "stage_target_base_link_pose": deepcopy(
                    stage["centerline_constrained_base_link_ik_target"]
                ),
                "world_vertical_clearance_m": 0.060,
                "target_reconstruction_rule": "T_world_base = T_world_reset_cube * inverse(T_base_cube)",
            }
    schedule["schedule_canonical_sha256_without_this_field"] = hashlib.sha256(canonical_bytes(schedule)).hexdigest()
    out_schedule = ARTIFACT / "gates/candidate_schedule.json"
    out_schedule.parent.mkdir()
    out_schedule.write_bytes(canonical_bytes(schedule))
    print(json.dumps({
        "registration": bind(out_registration), "candidate_schedule": bind(out_schedule),
        "open_contact_construction_contract_sha256": construction_sha,
        "candidate_action_steps": 810, "diagnostic_budget": 4, "candidate_budget": 4,
        "model_request_count": 0, "behavioral_episode_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
