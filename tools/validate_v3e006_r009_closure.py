#!/usr/bin/env python3
"""Validate R009's mechanically valid, attachment-invalid hash closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009/results"
STATE_GATE = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/state_repair_gate.py"
TERMINAL = "r009_candidate_budget_exhausted_no_valid_state_pair"
OFFICIAL_BBOX_DOC = "https://openusd.org/release/api/class_usd_geom_b_box_cache.html"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = root / path
    require(
        path.is_file() and path.stat().st_size == row.get("bytes") and sha256(path) == row.get("sha256"),
        f"{label} binding differs",
    )
    return path


def gate_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    physics = state.get("physics_gate")
    require(isinstance(physics, Mapping), "raw stage lacks physics gate")
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
    require(all(sorted(row["physics_gate"]["checks"]) == checks for row in states), "raw gate taxonomy differs")
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
            trace = state["construction"]["acquisition_lift_transport_trace"]
            require(isinstance(trace, list) and len(trace) == 1020, "raw action trace length differs")
            geometry = trace[-1]["pre_action_pinch_geometry"]
            live = geometry["live_tensor_collision_geometry"]
            command = geometry["pinch_alignment_command"]
            left_body = live["left"]["live_tensor_pose"]["position_env_local_m"]
            right_body = live["right"]["live_tensor_pose"]["position_env_local_m"]
            left_pad = live["left"]["reconstructed_bounds_env_local"]["collision_center_env_local_m"]
            right_pad = live["right"]["reconstructed_bounds_env_local"]["collision_center_env_local_m"]
            body_separation = math.dist(left_body, right_body)
            pad_separation = math.dist(left_pad, right_pad)
            recorded = float(command["inner_finger_collision_center_separation_m"])
            require(abs(pad_separation - recorded) <= 1e-12, "raw pad separation differs")
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


def attachment_finding(rows: list[Mapping[str, Any]], source_binding: Mapping[str, Any]) -> dict[str, Any]:
    require(len(rows) == 8, "attachment row count differs")
    body = [float(row["final_inner_finger_body_origin_separation_m"]) for row in rows]
    pads = [float(row["final_reconstructed_pad_center_separation_m"]) for row in rows]
    table = [float(row["max_cube_table_contact_force_n"]) for row in rows]
    grabbed = [bool(row["object_grabbed_all_final_steps"]) for row in rows]
    require(all(row.get("state_passed") is False for row in rows), "attachment table contains pass")
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
            **source_binding,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    root = args.study_root.resolve()
    closure = root / CLOSURE.relative_to(ROOT)
    results = json.loads((closure / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((closure / "evidence_manifest.json").read_text(encoding="utf-8"))
    require(results.get("status") == manifest.get("status") == TERMINAL, "terminal status differs")
    require(
        results.get("passed") is False
        and results.get("accepted_candidate_rank") is None
        and results.get("accepted_state_hashes") is None,
        "closure acceptance differs",
    )
    require(
        results.get("mechanically_valid_frozen_execution") is True
        and results.get("intended_collision_pinch_semantics_attachment_valid") is False
        and results.get("intended_collision_pinch_algorithm_scientifically_exhausted") is False
        and results.get("behavioral_activation_released") is False,
        "attachment/release classification differs",
    )
    require(results.get("model_request_count") == results.get("behavioral_episode_count") == 0, "counts differ")
    require(results.get("diagnostic_evaluation_count") == results.get("registered_diagnostic_budget") == 4, "diagnostics differ")
    require(results.get("candidate_pair_evaluation_count") == results.get("registered_candidate_budget") == 4, "pairs differ")
    require([row.get("candidate_rank") for row in results["candidate_attempts"]] == [1, 2, 3, 4], "order differs")
    require(all(row.get("passed") is False for row in results["candidate_attempts"]), "closure contains pass")
    counts = results["stage_gate_pass_counts"]
    require(
        counts.get("evaluated_stage_count") == 8
        and counts.get("full_state_pass_count") == 0
        and counts.get("physics_gate_pass_count") == 0
        and counts.get("ood_gate_pass_count") == 8
        and counts.get("camera_gate_pass_count") == 8
        and counts.get("companion_gate_pass_count") == 8
        and counts.get("frame_identity_pass_count") == 8,
        "stage pass counts differ",
    )
    verify(root, manifest["repo_result"], "repo result")
    verify(root, manifest["repo_target_validation_receipt"], "repo receipt")
    verify(root, manifest["decision_memo"], "decision memo")
    verify(root, manifest["closure_tool"], "closure tool")
    verify(root, manifest["closure_validator"], "closure validator")
    source = verify(root, manifest["buggy_source"], "buggy source")
    source_text = source.read_text(encoding="utf-8")
    require("bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()" in source_text, "ComputeLocalBound evidence absent")
    require("world_body.Transform(prim_world.Transform(corner))" in source_text, "double-transform evidence absent")
    require(manifest.get("official_openusd_reference") == OFFICIAL_BBOX_DOC, "official reference differs")
    if args.verify_raw:
        raw_paths = {name: verify(root, row, f"raw {name}") for name, row in manifest["raw_evidence"].items()}
        correspondence = {
            "raw_result": "child_result",
            "raw_harness": "harness",
            "raw_launch": "launch",
            "raw_runtime_log": "runtime_log",
            "authoritative_target_validation_receipt": "authoritative_validation",
        }
        for result_key, manifest_key in correspondence.items():
            require(results.get(result_key) == manifest["raw_evidence"][manifest_key], f"cross-binding differs: {result_key}")
        raw = json.loads(raw_paths["child_result"].read_text(encoding="utf-8"))
        harness = json.loads(raw_paths["harness"].read_text(encoding="utf-8"))
        receipt = json.loads(raw_paths["authoritative_validation"].read_text(encoding="utf-8"))
        require(raw.get("status") == harness.get("child_status") == TERMINAL, "raw terminal differs")
        require(harness.get("process_completed") is True and harness.get("scientific_gate_passed") is False, "harness differs")
        require(raw.get("passed") is False and raw.get("accepted_candidate_rank") is None and raw.get("accepted_states") is None, "raw accepted")
        require(raw.get("model_request_count") == raw.get("behavioral_episode_count") == 0, "raw counts differ")
        diagnostics = raw.get("known_reachable_diagnostics")
        attempts = raw.get("attempts")
        require(isinstance(diagnostics, list) and len(diagnostics) == 4 and all(row.get("passed") is True for row in diagnostics), "raw diagnostics differ")
        require(isinstance(attempts, list) and [row.get("candidate_rank") for row in attempts] == [1, 2, 3, 4], "raw order differs")
        require(all(row.get("passed") is False for row in attempts), "raw contains pass")
        regenerated = [
            {
                "candidate_rank": attempt["candidate_rank"],
                "passed": False,
                "stages": {
                    stage: gate_summary(attempt["stages"][stage]["candidate_state"])
                    for stage in ("canonical_grasp", "canonical_carry")
                },
            }
            for attempt in attempts
        ]
        require(regenerated == results.get("candidate_attempts"), "compact gate summaries differ")
        require(stage_pass_counts(attempts) == results.get("stage_gate_pass_counts"), "compact counts differ")
        verify_duplicate_metrics(attempts)
        source_binding = dict(manifest["buggy_source"])
        expected_finding = attachment_finding(attachment_rows(attempts), source_binding)
        require(expected_finding == results.get("geometry_attachment_finding"), "attachment finding does not regenerate")
        evidence = receipt.get("candidate_evidence", {})
        require(receipt.get("passed") is True and evidence.get("passed") is True, "target receipt failed")
        require(evidence.get("child_report") == manifest["raw_evidence"]["child_result"], "receipt binds another result")
        require(receipt.get("diagnostic_count") == 4 and receipt.get("candidate_pair_count") == 4, "receipt counts differ")
        repo_receipt = verify(root, manifest["repo_target_validation_receipt"], "repo receipt")
        require(repo_receipt.read_bytes() == raw_paths["authoritative_validation"].read_bytes(), "repo receipt differs")
    print(json.dumps({
        "passed": True,
        "results": manifest["repo_result"],
        "manifest_sha256": sha256(closure / "evidence_manifest.json"),
        "status": TERMINAL,
        "attachment_valid": False,
        "behavioral_release": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
