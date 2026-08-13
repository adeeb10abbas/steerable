#!/usr/bin/env python3
"""Hash-close R009 as a mechanically valid but attachment-invalid exhaustion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009"
DEFAULT_OUTPUT = ARTIFACT / "results"
TERMINAL = "r009_candidate_budget_exhausted_no_valid_state_pair"
OFFICIAL_BBOX_DOC = "https://openusd.org/release/api/class_usd_geom_b_box_cache.html"
STATE_GATE = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/state_repair_gate.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


def binding(path: Path, *, repo_relative: bool = False) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"missing bound file: {path}")
    return {
        "path": str(path.relative_to(ROOT)) if repo_relative else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def gate_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    physics = state.get("physics_gate")
    require(isinstance(physics, Mapping), "stage lacks physics gate")
    return {
        "passed": state.get("passed"),
        "normalized_state_sha256": state.get("normalized_state_sha256"),
        "physics_gate": physics,
        "ood_gate": state.get("ood_gate"),
        "camera_gate_passed": state.get("camera_evidence", {}).get("passed"),
        "companion_gate": state.get("companion_pose_gate"),
        "frame_identity_passed": state.get("base_link_to_eef_frame_identity", {}).get("passed"),
    }


def stage_pass_counts(attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
    states = [
        attempt["stages"][stage]["candidate_state"]
        for attempt in attempts
        for stage in ("canonical_grasp", "canonical_carry")
    ]
    checks = sorted(states[0]["physics_gate"]["checks"])
    require(all(sorted(row["physics_gate"]["checks"]) == checks for row in states), "gate taxonomy differs")
    return {
        "evaluated_stage_count": len(states),
        "full_state_pass_count": sum(row.get("passed") is True for row in states),
        "physics_gate_pass_count": sum(row["physics_gate"].get("passed") is True for row in states),
        "ood_gate_pass_count": sum(row.get("ood_gate", {}).get("passed") is True for row in states),
        "camera_gate_pass_count": sum(row.get("camera_evidence", {}).get("passed") is True for row in states),
        "companion_gate_pass_count": sum(row.get("companion_pose_gate", {}).get("passed") is True for row in states),
        "frame_identity_pass_count": sum(
            row.get("base_link_to_eef_frame_identity", {}).get("passed") is True for row in states
        ),
        "physics_check_pass_counts": {
            key: sum(row["physics_gate"]["checks"].get(key) is True for row in states)
            for key in checks
        },
    }


def attachment_rows(attempts: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        rank = int(attempt["candidate_rank"])
        for stage in ("canonical_grasp", "canonical_carry"):
            state = attempt["stages"][stage]["candidate_state"]
            construction = state["construction"]
            trace = construction["acquisition_lift_transport_trace"]
            require(isinstance(trace, list) and len(trace) == 1020, "R009 action trace length differs")
            geometry = trace[-1]["pre_action_pinch_geometry"]
            live = geometry["live_tensor_collision_geometry"]
            command = geometry["pinch_alignment_command"]
            left_body = live["left"]["live_tensor_pose"]["position_env_local_m"]
            right_body = live["right"]["live_tensor_pose"]["position_env_local_m"]
            left_pad = live["left"]["reconstructed_bounds_env_local"]["collision_center_env_local_m"]
            right_pad = live["right"]["reconstructed_bounds_env_local"]["collision_center_env_local_m"]
            body_separation = math.dist(left_body, right_body)
            pad_separation = math.dist(left_pad, right_pad)
            recorded_separation = float(command["inner_finger_collision_center_separation_m"])
            require(abs(pad_separation - recorded_separation) <= 1e-12, "retained pad separation differs")
            observed = state["physics_gate"]["observed"]
            rows.append(
                {
                    "candidate_rank": rank,
                    "stage": stage,
                    "state_passed": state.get("passed"),
                    "final_inner_finger_body_origin_separation_m": body_separation,
                    "final_reconstructed_pad_center_separation_m": pad_separation,
                    "pad_to_body_origin_separation_ratio": pad_separation / body_separation,
                    "max_cube_table_contact_force_n": observed[
                        "max_unintended_contact_force_n_by_pair"
                    ]["rubiks_cube__table"],
                    "object_grabbed_all_final_steps": observed["object_grabbed_all_steps"],
                    "minimum_intended_cube_gripper_contact_force_n": observed[
                        "minimum_intended_cube_gripper_contact_force_n"
                    ],
                }
            )
    return rows


def attachment_finding(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 8, "attachment row count differs")
    body = [float(row["final_inner_finger_body_origin_separation_m"]) for row in rows]
    pads = [float(row["final_reconstructed_pad_center_separation_m"]) for row in rows]
    table = [float(row["max_cube_table_contact_force_n"]) for row in rows]
    grabbed = [bool(row["object_grabbed_all_final_steps"]) for row in rows]
    require(all(row.get("state_passed") is False for row in rows), "attachment table contains a passing state")
    require(min(pads) > 0.3 and max(body) < 0.1 and min(table) > 1.0, "attachment signature differs")
    return {
        "classification": "attachment_invalid_intended_collision_pinch_semantics",
        "mechanically_valid_frozen_execution": True,
        "intended_collision_pinch_semantics_attachment_valid": False,
        "intended_collision_pinch_algorithm_scientifically_exhausted": False,
        "behavioral_release_permitted": False,
        "root_cause": (
            "ComputeLocalBound already includes the queried collision prim transform; R009 then "
            "applied that prim transform again while mapping corners into the owning rigid body"
        ),
        "official_openusd_reference": {
            "url": OFFICIAL_BBOX_DOC,
            "compute_local_bound_semantics": (
                "The computed bound includes the transform authored on the prim itself but no ancestor transforms."
            ),
            "compute_relative_bound_repair_semantics": (
                "ComputeRelativeBound computes the bound directly in the space of the specified ancestor prim."
            ),
        },
        "buggy_source": {
            **binding(STATE_GATE, repo_relative=True),
            "function": "_collision_geometry_body_local",
            "compute_local_bound_line": 1327,
            "extra_prim_transform_line": 1342,
        },
        "quantitative_signature": {
            "evaluated_stage_count": 8,
            "finger_body_origin_separation_m_min": min(body),
            "finger_body_origin_separation_m_max": max(body),
            "reconstructed_pad_center_separation_m_min": min(pads),
            "reconstructed_pad_center_separation_m_max": max(pads),
            "cube_table_contact_force_n_min": min(table),
            "cube_table_contact_force_n_max": max(table),
            "grabbed_all_final_steps_stage_count": sum(grabbed),
            "all_stages_table_loaded": all(value > 1.0 for value in table),
        },
        "per_stage": rows,
        "deterministic_duplicate_metric_groups": [
            [[1, "canonical_grasp"], [3, "canonical_grasp"]],
            [[2, "canonical_carry"], [3, "canonical_carry"]],
            [[1, "canonical_carry"], [4, "canonical_carry"]],
            [[2, "canonical_grasp"], [4, "canonical_grasp"]],
        ],
    }


def verify_duplicate_metrics(attempts: list[Mapping[str, Any]]) -> None:
    by_key = {
        (int(attempt["candidate_rank"]), stage): attempt["stages"][stage]["candidate_state"]["physics_gate"]
        for attempt in attempts
        for stage in ("canonical_grasp", "canonical_carry")
    }
    groups = [
        ((1, "canonical_grasp"), (3, "canonical_grasp")),
        ((2, "canonical_carry"), (3, "canonical_carry")),
        ((1, "canonical_carry"), (4, "canonical_carry")),
        ((2, "canonical_grasp"), (4, "canonical_grasp")),
    ]
    require(all(by_key[left] == by_key[right] for left, right in groups), "duplicate metric evidence differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--authoritative-validation-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw_root = args.candidate_root.resolve()
    output = args.output_root.resolve()
    require(not output.exists(), f"refusing to overwrite R009 closure: {output}")

    launch_path = raw_root / "launch.json"
    harness_path = raw_root / "harness_result.json"
    runtime_path = raw_root / "runtime.log"
    harness = load(harness_path)
    child = harness.get("child_report")
    require(isinstance(child, Mapping), "harness lacks child report")
    child_path = Path(str(child["path"])).resolve()
    require(binding(child_path)["sha256"] == child.get("sha256"), "child binding changed")
    raw = load(child_path)
    require(raw.get("status") == TERMINAL and raw.get("passed") is False, "R009 terminal differs")
    require(raw.get("accepted_candidate_rank") is None and raw.get("accepted_states") is None, "R009 accepted")
    require(raw.get("model_request_count") == raw.get("behavioral_episode_count") == 0, "R009 is not zero-model")
    diagnostics = raw.get("known_reachable_diagnostics")
    attempts = raw.get("attempts")
    require(isinstance(diagnostics, list) and len(diagnostics) == 4, "R009 diagnostics differ")
    require(all(row.get("passed") is True for row in diagnostics), "R009 diagnostics failed")
    require(isinstance(attempts, list) and len(attempts) == 4, "R009 pair count differs")
    require([row.get("candidate_rank") for row in attempts] == [1, 2, 3, 4], "R009 order differs")
    require(all(row.get("passed") is False for row in attempts), "R009 contains a passing pair")
    verify_duplicate_metrics(attempts)

    receipt = load(args.authoritative_validation_receipt)
    evidence = receipt.get("candidate_evidence")
    require(receipt.get("passed") is True and isinstance(evidence, Mapping), "target validation failed")
    require(evidence.get("child_report", {}).get("sha256") == sha256_file(child_path), "receipt binds another result")
    require(receipt.get("candidate_pair_count") == 4 and receipt.get("diagnostic_count") == 4, "receipt counts differ")

    summaries = []
    for attempt in attempts:
        stages = attempt["stages"]
        summaries.append(
            {
                "candidate_rank": attempt["candidate_rank"],
                "passed": False,
                "stages": {
                    stage: gate_summary(stages[stage]["candidate_state"])
                    for stage in ("canonical_grasp", "canonical_carry")
                },
            }
        )
    finding = attachment_finding(attachment_rows(attempts))
    counts = stage_pass_counts(attempts)
    output.mkdir(parents=True, exist_ok=False)
    compact = {
        "schema_version": "vla-wam-shared-v3e006-r009-state-repair-closure-v1",
        "amendment_id": "V3-E006-R009",
        "status": TERMINAL,
        "passed": False,
        "accepted_candidate_rank": None,
        "accepted_state_hashes": None,
        "registered_diagnostic_budget": 4,
        "diagnostic_evaluation_count": 4,
        "diagnostics_all_passed": True,
        "registered_candidate_budget": 4,
        "candidate_pair_evaluation_count": 4,
        "candidate_attempts": summaries,
        "stage_gate_pass_counts": counts,
        "first_passing_rule_obeyed": raw.get("first_passing_rule_obeyed"),
        "selection_rule": raw.get("selection_rule"),
        "mechanically_valid_frozen_execution": True,
        "intended_collision_pinch_semantics_attachment_valid": False,
        "intended_collision_pinch_algorithm_scientifically_exhausted": False,
        "geometry_attachment_finding": finding,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_activation_released": False,
        "raw_result": binding(child_path),
        "raw_harness": binding(harness_path),
        "raw_launch": binding(launch_path),
        "raw_runtime_log": binding(runtime_path),
        "authoritative_target_validation_receipt": binding(args.authoritative_validation_receipt),
        "registration": raw.get("repair_registration"),
        "candidate_schedule": raw.get("candidate_schedule"),
        "source_push_gate_at_execution": raw.get("source_push_gate"),
        "source_commit_at_execution": load(launch_path).get("study_commit"),
    }
    result_path = output / "results.json"
    result_path.write_bytes(canonical_bytes(compact))
    receipt_path = output / "target_validation_receipt.json"
    receipt_path.write_bytes(args.authoritative_validation_receipt.read_bytes())
    memo_path = output / "DECISION_MEMO.md"
    sig = finding["quantitative_signature"]
    memo_path.write_text(
        "# V3-E006-R009 state-construction decision\n\n"
        "R009 mechanically completed its frozen zero-model execution: all four diagnostics passed, "
        "all four candidate pairs were evaluated in order, all eight stages failed at least one "
        "unchanged scientific gate, and no state was accepted. It made zero model requests and zero "
        "behavioral episodes. Behavioral activation remains blocked.\n\n"
        "This is not evidence that the intended collision-pinch construction algorithm was scientifically "
        "exhausted. The execution is attachment-invalid relative to that intended algorithm. OpenUSD "
        "documents that ComputeLocalBound includes the queried prim's authored transform. R009 then "
        "applied that prim transform again while converting the returned corners into the owning rigid "
        "body frame. The reconstructed pad-center separation was "
        f"{sig['reconstructed_pad_center_separation_m_min']:.15f}–"
        f"{sig['reconstructed_pad_center_separation_m_max']:.15f} m, despite live inner-finger body-origin "
        f"separation of {sig['finger_body_origin_separation_m_min']:.15f}–"
        f"{sig['finger_body_origin_separation_m_max']:.15f} m. All eight stages remained cube-table loaded "
        f"at {sig['cube_table_contact_force_n_min']:.15f}–{sig['cube_table_contact_force_n_max']:.15f} N; "
        f"{sig['grabbed_all_final_steps_stage_count']}/8 nevertheless reported grabbed across the final ten "
        "steps. Four rank/stage metric pairs repeat exactly, supporting a common deterministic attachment "
        "error rather than outcome-driven candidate tuning.\n\n"
        f"Official semantics: {OFFICIAL_BBOX_DOC}\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "vla-wam-shared-v3e006-r009-closure-manifest-v1",
        "repair_amendment_id": "V3-E006-R009",
        "status": TERMINAL,
        "passed": False,
        "mechanically_valid_frozen_execution": True,
        "intended_collision_pinch_semantics_attachment_valid": False,
        "accepted_candidate_rank": None,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "repo_result": binding(result_path, repo_relative=True),
        "repo_target_validation_receipt": binding(receipt_path, repo_relative=True),
        "decision_memo": binding(memo_path, repo_relative=True),
        "raw_evidence": {
            "launch": binding(launch_path),
            "harness": binding(harness_path),
            "child_result": binding(child_path),
            "runtime_log": binding(runtime_path),
            "authoritative_validation": binding(args.authoritative_validation_receipt),
        },
        "buggy_source": binding(STATE_GATE, repo_relative=True),
        "official_openusd_reference": OFFICIAL_BBOX_DOC,
        "closure_tool": binding(Path(__file__), repo_relative=True),
        "closure_validator": binding(ROOT / "tools/validate_v3e006_r009_closure.py", repo_relative=True),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "invocation": [sys.executable, *sys.argv],
    }
    manifest_path = output / "evidence_manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    print(json.dumps({
        "result": binding(result_path), "receipt": binding(receipt_path),
        "memo": binding(memo_path), "manifest": binding(manifest_path), "status": TERMINAL,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
