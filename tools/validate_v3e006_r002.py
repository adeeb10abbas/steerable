#!/usr/bin/env python3
"""Fail-closed validator for prospective and raw V3-E006-R002 evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


R001_COMMIT = "bbabac55dfd54f7a0b7d8a2693673a4b06409f21"
BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
REGISTRATION_STATUS = "prospectively_registered_before_any_r002_live_candidate_or_model_request"
SOURCE_GATE_STATUS = "passed_before_first_r002_live_candidate_or_model_request"
TERMINAL_STATUSES = {
    "passed_r002_state_repair_not_released_for_behavior": True,
    "r002_candidate_budget_exhausted_no_valid_state_pair": False,
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_binding(path: Path, expected: Mapping[str, Any], label: str) -> None:
    require(path.is_file(), f"{label} is missing: {path}")
    require(path.stat().st_size == expected.get("bytes"), f"{label} bytes differ")
    require(sha256(path) == expected.get("sha256"), f"{label} digest differs")


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON: {path}: {exc}") from exc


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name]
    require(len(nodes) == 1, f"{path} must contain exactly one {name}")
    return ast.dump(nodes[0], include_attributes=False)


def validate_static(root: Path, *, require_source_gate: bool = True) -> dict[str, Any]:
    root = root.resolve()
    relative = Path("artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r002")
    artifact = root / relative
    registration_path = artifact / "repair_registration.json"
    schedule_path = artifact / "gates/candidate_schedule.json"
    predecessor_path = artifact / "gates/predecessor_closure_binding.json"
    source_gate_path = artifact / "source_push_gate.json"
    registration = load(registration_path)
    schedule = load(schedule_path)
    predecessor = load(predecessor_path)

    require(registration.get("repair_amendment_id") == "V3-E006-R002", "registration ID differs")
    require(registration.get("status") == REGISTRATION_STATUS, "registration status differs")
    require(registration.get("counts_at_registration") == {
        "r002_live_candidate_evaluations": 0, "model_requests": 0, "behavioral_episodes": 0
    }, "registration counts differ")
    ancestry = registration.get("source_ancestry", {})
    require(ancestry.get("required_repository_base") == BASE, "required base differs")
    require(ancestry.get("r001_exhaustion_closure_commit") == R001_COMMIT, "R001 predecessor commit differs")
    for commit in (BASE, ancestry.get("original_v3e006_closure_commit"), R001_COMMIT):
        require(
            subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                           check=False).returncode == 0,
            f"source lineage commit does not exist: {commit}",
        )
    frozen = registration["frozen_inputs"]
    for name, row in frozen.items():
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = root / path
        verify_binding(path, row, f"registration frozen input {name}")

    require(predecessor.get("status") == "original_and_r001_predecessors_byte_identical_before_r002_live_candidate_or_model_request", "predecessor status differs")
    require(predecessor.get("r001_tree_file_count") == len(predecessor.get("r001_tree_files", [])) == 20, "R001 predecessor inventory differs")
    require(predecessor.get("model_request_count") == predecessor.get("behavioral_episode_count") == predecessor.get("r002_live_candidate_evaluation_count") == 0, "predecessor counts differ")
    for row in predecessor["r001_tree_files"]:
        relative_path = Path(str(row["path"]))
        require(not relative_path.is_absolute() and ".." not in relative_path.parts, "unsafe predecessor path")
        current = root / relative_path
        verify_binding(current, row, f"R001 predecessor {relative_path}")
        committed = subprocess.check_output(["git", "-C", str(root), "show", f"{R001_COMMIT}:{relative_path}"])
        require(len(committed) == row["bytes"] and hashlib.sha256(committed).hexdigest() == row["sha256"], f"R001 predecessor changed from commit: {relative_path}")

    require(schedule.get("status") == "frozen_before_any_r002_live_candidate_or_model_request", "schedule status differs")
    require(schedule.get("candidate_budget") == 8, "schedule budget differs")
    require([row.get("candidate_rank") for row in schedule.get("candidate_pairs", [])] == list(range(1, 9)), "schedule ranks differ")
    require(schedule.get("model_request_count") == schedule.get("behavioral_episode_count") == schedule.get("r002_live_candidate_evaluation_count") == 0, "schedule counts differ")
    verify_binding(registration_path, schedule["repair_registration"], "schedule registration")
    verify_binding(predecessor_path, schedule["r001_predecessor"]["closure_binding"], "schedule predecessor")
    expected_methods = ["direct_contact_initialization"] * 4 + ["open_approach_close_lift"] * 4
    require([row["construction_method"] for row in schedule["candidate_pairs"]] == expected_methods, "construction method dispatch differs")
    for pair in schedule["candidate_pairs"]:
        for stage in ("canonical_grasp", "canonical_carry"):
            row = pair[stage]
            require(row["target_cube_pose"]["position_world_m"][1] == 0.0, "target cube not centered")
            residual = row["se3_reconstruction"]
            require(residual["cube_midline_residual_m"] <= 1e-12, "reconstructed cube not centered")
            require(residual["position_residual_m"] <= 1e-12, "SE3 position residual differs")
            require(residual["rotation_matrix_frobenius_residual"] <= 1e-12, "SE3 rotation residual differs")
            require(set(row["both_direction_sources"]) == {"left", "right"}, "direction source pair incomplete")

    state_contract = root / frozen["state_contract"]["path"]
    ood_reference = root / frozen["ood_reference"]["path"]
    require(sha256(state_contract) == "2476b28d2867c1b87f477fd5f89e545616be00d860d4144f8cbdb70af10f3c18", "state contract not unchanged")
    require(sha256(ood_reference) == "4df1ebf0061096a74b5eccd10b2a144e840f52fd50469b8bdae9369d1696fd04", "OOD source not unchanged")
    r001_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r001/state_repair_gate.py"
    r002_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r002/state_repair_gate.py"
    for name in ("_contact_forces", "_contact_coverage", "_sample", "_capture_state", "_reference_bounds", "_save_camera_evidence", "_companion_gate", "_fresh_reset_and_gate"):
        require(function_ast(r001_source, name) == function_ast(r002_source, name), f"unchanged helper differs: {name}")

    source_gate_summary = None
    if require_source_gate:
        require(source_gate_path.is_file(), "R002 source-push gate is missing")
        source_gate = load(source_gate_path)
        require(source_gate.get("status") == SOURCE_GATE_STATUS, "source-push status differs")
        require(source_gate.get("model_request_count") == source_gate.get("behavioral_episode_count") == source_gate.get("r002_live_candidate_evaluation_count") == 0, "source-push counts differ")
        implementation_commit = str(source_gate.get("implementation_commit", ""))
        require(
            subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{implementation_commit}^{{commit}}"], check=False).returncode == 0,
            "source-push implementation commit does not exist",
        )
        for row in source_gate.get("implementation_files", []):
            relative_path = Path(str(row["path"]))
            require(not relative_path.is_absolute() and ".." not in relative_path.parts, "unsafe source-push path")
            verify_binding(root / relative_path, row, f"source-push inventory {relative_path}")
        require(len(source_gate.get("implementation_files", [])) >= 9, "source-push inventory is incomplete")
        verify_binding(registration_path, source_gate["repair_registration"], "source-push registration")
        verify_binding(schedule_path, source_gate["candidate_schedule"], "source-push schedule")
        verify_binding(predecessor_path, source_gate["predecessor_closure_binding"], "source-push predecessor")
        source_gate_summary = binding(source_gate_path)

    return {
        "passed": True,
        "registration": binding(registration_path),
        "candidate_schedule": binding(schedule_path),
        "predecessor_closure": binding(predecessor_path),
        "source_push_gate": source_gate_summary,
        "candidate_pair_count": 8,
        "unchanged_helper_count": 8,
    }


def validate_candidate_root(root: Path, candidate_root: Path) -> dict[str, Any]:
    candidate_root = candidate_root.resolve()
    launch_path = candidate_root / "launch.json"
    harness_path = candidate_root / "harness_result.json"
    runtime_path = candidate_root / "runtime.log"
    launch = load(launch_path)
    harness = load(harness_path)
    require(harness.get("model_request_count") == harness.get("behavioral_episode_count") == 0, "raw harness counts differ")
    verify_binding(launch_path, harness["launch"], "raw harness launch")
    verify_binding(runtime_path, harness["runtime_log"], "raw harness runtime log")
    report_binding = harness.get("child_report")
    require(isinstance(report_binding, Mapping), "raw harness lacks child report")
    report_path = Path(str(report_binding["path"]))
    verify_binding(report_path, report_binding, "raw child report")
    report = load(report_path)
    if harness.get("process_completed") is True:
        require(report_path.name == "state_repair_result.json", "completed child report filename differs")
        require(report.get("status") in TERMINAL_STATUSES, "child terminal status differs")
        require(report.get("passed") is TERMINAL_STATUSES[report["status"]], "child pass boolean differs")
        require(report.get("model_request_count") == report.get("behavioral_episode_count") == 0, "child counts differ")
        attempts = report.get("attempts")
        require(isinstance(attempts, list) and 1 <= len(attempts) <= 8, "child attempt count differs")
        require([row["candidate_rank"] for row in attempts] == list(range(1, len(attempts) + 1)), "child rank order differs")
        require(report.get("repair_candidate_evaluation_count") == len(attempts), "child evaluated count differs")
        for attempt in attempts:
            require(set(attempt["stages"]) == {"canonical_grasp", "canonical_carry"}, "child stages differ")
            for stage in attempt["stages"].values():
                require(stage["ik_solve_environment"]["fresh_reset"]["passed"] is True, "IK reset failed")
                require(stage["ik_solve_environment"]["environment_lifecycle"]["closed_before_next_environment"] is True, "IK env not closed")
                if stage["ik_solution"]["passed"]:
                    require(stage["materialization_environment"]["fresh_reset"]["passed"] is True, "material reset failed")
                    require(stage["materialization_environment"]["environment_lifecycle"]["closed_before_next_environment"] is True, "material env not closed")
    else:
        require(report_path.name == "state_construction_failure.json", "invalid child report filename differs")
        require(report.get("status") == "infrastructure_invalid_r002_state_repair", "invalid child status differs")
        require(report.get("model_request_count") == report.get("behavioral_episode_count") == report.get("state_candidate_count") == 0, "invalid child counts differ")
    for name, expected in launch.get("input_bindings", {}).items():
        verify_binding(Path(str(expected["path"])), expected, f"raw launch input {name}")
    return {
        "passed": True,
        "candidate_root": str(candidate_root),
        "launch": binding(launch_path),
        "harness": binding(harness_path),
        "child_report": binding(report_path),
        "child_status": report.get("status"),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pre-source-gate", action="store_true")
    parser.add_argument("--candidate-root", type=Path)
    args = parser.parse_args()
    result = validate_static(args.study_root, require_source_gate=not args.pre_source_gate)
    if args.candidate_root is not None:
        result["candidate_evidence"] = validate_candidate_root(args.study_root, args.candidate_root)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
