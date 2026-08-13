#!/usr/bin/env python3
"""Fail-closed validator for prospective and raw V3-E006-R004 evidence."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r004.predecessor_contract import (
    validate_r003_diagnostic_closure,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r004.residual_correction import (
    corrected_command,
    normalize_quaternion,
    validate_contract,
)


BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
R003_CLOSURE_COMMIT = "fc2f23a915a98f368181dd897994b67d64a35eeb"
R003_RESULTS_SHA256 = "d30ed1dae4f69992cf257b9430fc9165129185d2be9a2af3bc1a2dd9ea5d1261"
R003_RAW_RESULT_SHA256 = "0bac94d3d1e5b93f3eb00f94f1c4a6cc989cbe54d01f2b9247ed6f5ecf5a9392"
REGISTRATION_STATUS = (
    "prospectively_registered_before_any_r004_live_diagnostic_candidate_or_model_request"
)
SOURCE_GATE_STATUS = "passed_before_first_r004_live_diagnostic_candidate_or_model_request"
TERMINAL_STATUSES = {
    "passed_r004_state_repair_not_released_for_behavior": True,
    "r004_candidate_budget_exhausted_no_valid_state_pair": False,
    "r004_known_reachable_diagnostic_failed_candidates_not_evaluated": False,
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


def resolve_bound(root: Path, expected: Mapping[str, Any], label: str) -> Path:
    path = Path(str(expected.get("path", "")))
    if not path.is_absolute():
        path = root / path
    verify_binding(path, expected, label)
    return path


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON: {path}: {exc}") from exc


def canonical_sha(value: Any) -> str:
    payload = (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name]
    require(len(nodes) == 1, f"{path} must contain exactly one {name}")
    return ast.dump(nodes[0], include_attributes=False)


def qclose(left: Sequence[float], right: Sequence[float], atol: float = 1e-12) -> bool:
    a, b = normalize_quaternion(left), normalize_quaternion(right)
    return min(float(np.linalg.norm(a - b)), float(np.linalg.norm(a + b))) <= atol


def pclose(left: Sequence[float], right: Sequence[float], atol: float = 1e-12) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=atol))


def validate_pose_hold(hold: Mapping[str, Any], label: str) -> bool:
    contract = hold.get("residual_correction_contract")
    require(isinstance(contract, Mapping), f"{label} correction contract absent")
    try:
        validate_contract(contract)
    except ValueError as exc:
        raise ValidationError(f"{label}: {exc}") from exc
    require(
        canonical_sha(contract) == hold.get("residual_correction_contract_sha256"),
        f"{label} correction digest differs",
    )
    require(hold.get("maximum_correction_rounds") == 3, f"{label} round budget differs")
    require(hold.get("hold_steps") == hold.get("hold_steps_per_round") == 30, f"{label} hold differs")
    require(hold.get("required_final_consecutive_steps") == 10, f"{label} window differs")
    require(hold.get("position_error_m_inclusive") == 0.001, f"{label} position threshold differs")
    require(
        hold.get("orientation_geodesic_error_deg_inclusive") == 1.0,
        f"{label} orientation threshold differs",
    )
    rounds = hold.get("correction_rounds")
    require(isinstance(rounds, list) and 1 <= len(rounds) <= 3, f"{label} rounds differ")
    require(hold.get("completed_correction_rounds") == len(rounds), f"{label} round count differs")
    desired = hold.get("target_base_link_pose_world_wxyz")
    require(isinstance(desired, list) and len(desired) == 7, f"{label} desired target differs")
    desired_p, desired_q = desired[:3], desired[3:]
    all_errors: list[Mapping[str, Any]] = []
    all_trace: list[Mapping[str, Any]] = []
    aggregate_finite = True
    aggregate_inside = True
    aggregate_frame = True
    observed_termination = None
    for index, row in enumerate(rounds, start=1):
        require(row.get("round_one_based") == index, f"{label} round order differs")
        require(pclose(row.get("desired_target_position_world_m"), desired_p), f"{label} desired position changed")
        require(qclose(row.get("desired_target_quaternion_world_wxyz"), desired_q), f"{label} desired quaternion changed")
        errors = row.get("errors")
        require(isinstance(errors, list) and 0 <= len(errors) <= 30, f"{label} round errors differ")
        require(row.get("completed_steps") == len(errors), f"{label} completed steps differ")
        round_trace = row.get("construction_action_trace")
        require(isinstance(round_trace, list) and len(round_trace) == len(errors), f"{label} trace length differs")
        command_p = row.get("command_position_world_m")
        command_q = row.get("command_quaternion_world_wxyz")
        for error in errors:
            require(pclose(error.get("desired_target_position_world_m"), desired_p), f"{label} error desired position changed")
            require(qclose(error.get("desired_target_quaternion_world_wxyz"), desired_q), f"{label} error desired quaternion changed")
            require(pclose(error.get("command_position_world_m"), command_p), f"{label} command position drifted within round")
            require(qclose(error.get("command_quaternion_world_wxyz"), command_q), f"{label} command quaternion drifted within round")
            desired_position = np.asarray(error["desired_target_position_world_m"], dtype=np.float64)
            measured_position = np.asarray(error["measured_position_world_m"], dtype=np.float64)
            recomputed_position_error = float(np.linalg.norm(measured_position - desired_position))
            require(
                math.isclose(float(error.get("position_error_m")), recomputed_position_error,
                             rel_tol=0.0, abs_tol=1e-12),
                f"{label} logged position error differs",
            )
            desired_quaternion = normalize_quaternion(error["desired_target_quaternion_world_wxyz"])
            measured_quaternion = normalize_quaternion(error["measured_quaternion_world_wxyz"])
            dot = min(1.0, max(-1.0, abs(float(np.dot(desired_quaternion, measured_quaternion)))))
            recomputed_orientation_error = math.degrees(2.0 * math.acos(dot))
            require(
                math.isclose(float(error.get("orientation_geodesic_error_deg")),
                             recomputed_orientation_error, rel_tol=0.0, abs_tol=1e-9),
                f"{label} logged orientation error differs",
            )
            row_finite = error.get("finite") is True
            row_inside = error.get("arm_inside_soft_joint_limits") is True
            row_frame = error.get("base_link_to_eef_frame_identity", {}).get("passed") is True
            aggregate_finite = aggregate_finite and row_finite
            aggregate_inside = aggregate_inside and row_inside
            aggregate_frame = aggregate_frame and row_frame
        final = errors[-10:]
        recomputed_window = len(final) == 10 and all(
            value.get("position_error_m") <= 0.001
            and value.get("orientation_geodesic_error_deg") <= 1.0
            for value in final
        )
        require(row.get("final_window_passed") is recomputed_window, f"{label} round window flag differs")
        round_unsafe = bool(
            row.get("termination") is not None
            or not aggregate_finite
            or not aggregate_inside
            or not aggregate_frame
            or not errors
        )
        if row.get("termination") is not None:
            require(observed_termination is None, f"{label} multiple terminations")
            observed_termination = row["termination"]
        correction = row.get("measured_residual_correction")
        if not recomputed_window and not round_unsafe and index < 3:
            require(isinstance(correction, Mapping), f"{label} missing correction")
            last = errors[-1]
            expected = corrected_command(
                desired_position=desired_p,
                desired_quaternion=desired_q,
                measured_position=last["measured_position_world_m"],
                measured_quaternion=last["measured_quaternion_world_wxyz"],
                current_command_position=command_p,
                current_command_quaternion=command_q,
                translation_gain=1.0,
                rotation_gain=1.0,
            )
            require(pclose(correction.get("next_command_position_world_m"), expected["next_command_position_world_m"]), f"{label} translation correction differs")
            require(qclose(correction.get("next_command_quaternion_world_wxyz"), expected["next_command_quaternion_world_wxyz"]), f"{label} rotation correction differs")
            require(index < len(rounds), f"{label} correction lacks next round")
            require(pclose(rounds[index]["command_position_world_m"], expected["next_command_position_world_m"]), f"{label} next position command differs")
            require(qclose(rounds[index]["command_quaternion_world_wxyz"], expected["next_command_quaternion_world_wxyz"]), f"{label} next quaternion command differs")
        else:
            require(correction is None, f"{label} correction after pass/final round")
        all_errors.extend(errors)
        all_trace.extend(round_trace)
        if recomputed_window or round_unsafe:
            require(index == len(rounds), f"{label} did not stop at pass/unsafe round")
    require(hold.get("completed_steps") == len(all_errors), f"{label} aggregate step count differs")
    require(hold.get("errors") == all_errors, f"{label} aggregate errors differ")
    require(hold.get("construction_action_trace") == all_trace, f"{label} aggregate trace differs")
    require(hold.get("termination") == observed_termination, f"{label} termination differs")
    require(hold.get("all_states_finite") is aggregate_finite, f"{label} finite aggregate differs")
    require(hold.get("all_arm_states_inside_live_soft_joint_limits") is aggregate_inside,
            f"{label} soft-limit aggregate differs")
    require(hold.get("all_base_link_to_eef_frame_identity_checks_passed") is aggregate_frame,
            f"{label} frame aggregate differs")
    expected_pass = bool(
        rounds[-1].get("final_window_passed") is True
        and observed_termination is None
        and aggregate_finite
        and aggregate_inside
        and aggregate_frame
    )
    require(hold.get("final_window_passed") is rounds[-1].get("final_window_passed"), f"{label} final window differs")
    require(hold.get("desired_target_invariant_across_rounds") is True, f"{label} target-invariance flag differs")
    require(hold.get("passed") is expected_pass, f"{label} pass does not recompute")
    return expected_pass


def validate_scientific_selection(report: Mapping[str, Any], harness: Mapping[str, Any]) -> None:
    require(report.get("status") in TERMINAL_STATUSES, "child terminal status differs")
    terminal_pass = TERMINAL_STATUSES[str(report["status"])]
    require(report.get("passed") is terminal_pass, "child pass boolean differs")
    require(report.get("model_request_count") == report.get("behavioral_episode_count") == 0, "child counts differ")
    diagnostics = report.get("known_reachable_diagnostics")
    require(isinstance(diagnostics, list) and 1 <= len(diagnostics) <= 4, "diagnostics differ")
    require(
        [row.get("diagnostic_index_one_based") for row in diagnostics]
        == list(range(1, len(diagnostics) + 1)),
        "diagnostic order differs",
    )
    diagnostic_passes = []
    for row in diagnostics:
        hold_pass = validate_pose_hold(row.get("pose_hold", {}), "diagnostic pose hold")
        require(row.get("passed") is hold_pass, "diagnostic pass differs from hold")
        diagnostic_passes.append(hold_pass)
    require(report.get("r004_live_diagnostic_count") == len(diagnostics), "child diagnostic count differs")
    require(harness.get("r004_live_diagnostic_count") == len(diagnostics), "harness diagnostic count differs")
    attempts = report.get("attempts")
    if report.get("status") == "r004_known_reachable_diagnostic_failed_candidates_not_evaluated":
        require(not all(diagnostic_passes), "diagnostic failure status has no failure")
        require(attempts == [], "failed diagnostic evaluated candidates")
        require(report.get("repair_candidate_evaluation_count") == 0, "failed diagnostic candidate count differs")
        require(report.get("accepted_candidate_rank") is None, "failed diagnostic accepted rank")
    else:
        require(len(diagnostics) == 4 and all(diagnostic_passes), "candidate ran before four passed diagnostics")
        require(isinstance(attempts, list) and 1 <= len(attempts) <= 4, "attempt count differs")
        require([row.get("candidate_rank") for row in attempts] == list(range(1, len(attempts) + 1)), "rank order differs")
        rank_passes: list[bool] = []
        for attempt in attempts:
            stages = attempt.get("stages", {})
            require(set(stages) == {"canonical_grasp", "canonical_carry"}, "stage set differs")
            stage_passes = []
            for stage_name, stage in stages.items():
                ik = stage.get("ik_solution", {})
                for waypoint_index, waypoint in enumerate(ik.get("waypoint_results", []), start=1):
                    validate_pose_hold(waypoint, f"{stage_name} waypoint {waypoint_index}")
                state = stage.get("candidate_state", {})
                evaluated = all(isinstance(state.get(key), Mapping) for key in (
                    "physics_gate", "ood_gate", "camera_evidence", "companion_pose_gate"
                ))
                expected = evaluated and all(state[key].get("passed") is True for key in (
                    "physics_gate", "ood_gate", "camera_evidence", "companion_pose_gate"
                ))
                require(state.get("passed") is expected, f"{stage_name} pass differs")
                stage_passes.append(expected)
            rank_pass = all(stage_passes)
            require(attempt.get("passed") is rank_pass, "rank pass differs")
            rank_passes.append(rank_pass)
        require(report.get("repair_candidate_evaluation_count") == len(attempts), "evaluated count differs")
        if terminal_pass:
            require(rank_passes[-1] and not any(rank_passes[:-1]), "first-pass selection differs")
            require(report.get("accepted_candidate_rank") == len(attempts), "accepted rank differs")
            require(report.get("accepted_states") == attempts[-1]["stages"], "accepted states differ")
        else:
            require(len(attempts) == 4 and not any(rank_passes), "exhaustion differs")
            require(report.get("accepted_candidate_rank") is None and report.get("accepted_states") is None, "exhaustion accepted state")
        require(report.get("first_passing_rule_obeyed") is True, "first-pass flag differs")
    require(harness.get("process_completed") is True, "harness process not complete")
    require(harness.get("status") == "completed_r004_candidate_search", "harness status differs")
    require(harness.get("scientific_gate_passed") is terminal_pass, "harness scientific flag differs")
    require(harness.get("child_status") == report.get("status"), "harness child status differs")


def validate_static(root: Path, *, require_source_gate: bool = True) -> dict[str, Any]:
    root = root.resolve()
    artifact = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r004"
    registration_path = artifact / "repair_registration.json"
    schedule_path = artifact / "gates/candidate_schedule.json"
    source_gate_path = artifact / "source_push_gate.json"
    registration, schedule = load(registration_path), load(schedule_path)
    require(registration.get("repair_amendment_id") == "V3-E006-R004", "registration ID differs")
    require(registration.get("status") == REGISTRATION_STATUS, "registration status differs")
    require(registration.get("predecessor_repair_amendment_id") == "V3-E006-R003", "predecessor ID differs")
    require(registration.get("counts_at_registration") == {
        "r004_live_diagnostics": 0, "r004_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }, "registration counts differ")
    for commit in (BASE, R003_CLOSURE_COMMIT):
        require(subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0, f"lineage commit absent: {commit}")
    for name, row in registration["frozen_inputs"].items():
        resolve_bound(root, row, f"registration frozen input {name}")
    predecessor = registration["r003_predecessor"]
    require(predecessor.get("closure_commit") == R003_CLOSURE_COMMIT, "R003 closure commit differs")
    require(predecessor["results"].get("sha256") == R003_RESULTS_SHA256, "R003 results digest differs")
    require(predecessor["raw_result"].get("sha256") == R003_RAW_RESULT_SHA256, "R003 raw digest differs")
    predecessor_results_path = resolve_bound(root, predecessor["results"], "R003 predecessor results")
    try:
        validate_r003_diagnostic_closure(load(predecessor_results_path))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    require(schedule.get("repair_amendment_id") == "V3-E006-R004", "schedule ID differs")
    require(schedule.get("status") == "frozen_before_any_r004_live_diagnostic_candidate_or_model_request", "schedule status differs")
    require(schedule.get("candidate_budget") == schedule.get("diagnostic_budget") == 4, "budgets differ")
    require([row.get("candidate_rank") for row in schedule.get("candidate_pairs", [])] == [1, 2, 3, 4], "rank order differs")
    require([row.get("diagnostic_index_one_based") for row in schedule.get("known_reachable_diagnostics", [])] == [1, 2, 3, 4], "diagnostic order differs")
    require(all(schedule.get(key) == 0 for key in (
        "model_request_count", "behavioral_episode_count", "r004_live_diagnostic_count",
        "r004_live_candidate_evaluation_count",
    )), "schedule counts differ")
    verify_binding(registration_path, schedule["repair_registration"], "schedule registration")
    require(schedule.get("r003_predecessor") == predecessor, "schedule predecessor differs")
    correction = schedule.get("residual_correction_contract", {})
    try:
        validate_contract(correction)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    correction_sha = canonical_sha(correction)
    require(correction_sha == schedule.get("residual_correction_contract_sha256"), "correction digest differs")
    require(registration.get("residual_correction_contract") == correction, "registration correction differs")
    require(registration.get("residual_correction_contract_sha256") == correction_sha, "registration correction digest differs")

    r003_schedule_path = resolve_bound(root, schedule["r003_target_schedule"], "R003 target schedule")
    r003_schedule = load(r003_schedule_path)
    for new, old in zip(schedule["known_reachable_diagnostics"], r003_schedule["known_reachable_diagnostics"], strict=True):
        stripped = deepcopy(new)
        require(stripped.pop("r004_residual_correction_contract_sha256") == correction_sha, "diagnostic correction binding differs")
        require(stripped.pop("maximum_correction_rounds") == 3, "diagnostic round budget differs")
        require(stripped == old, "R003 diagnostic target/source/order changed")
    for new_pair, old_pair in zip(schedule["candidate_pairs"], r003_schedule["candidate_pairs"], strict=True):
        require(new_pair["candidate_rank"] == old_pair["candidate_rank"], "candidate rank changed")
        require(new_pair["construction_method"] == old_pair["construction_method"], "construction method changed")
        for stage_name in ("canonical_grasp", "canonical_carry"):
            new_stage, old_stage = deepcopy(new_pair[stage_name]), deepcopy(old_pair[stage_name])
            new_init = new_stage.pop("r004_solver_initialization")
            old_init = old_stage.pop("r003_solver_initialization")
            require(new_stage == old_stage, f"{stage_name} target/contact/source changed")
            require(new_init.pop("residual_correction_contract_sha256") == correction_sha, f"{stage_name} correction binding differs")
            for waypoint, old_waypoint in zip(new_init["waypoints"], old_init["waypoints"], strict=True):
                stripped_waypoint = deepcopy(waypoint)
                require(stripped_waypoint.pop("maximum_correction_rounds") == 3, "waypoint round budget differs")
                require(stripped_waypoint.pop("r004_residual_correction_contract_sha256") == correction_sha, "waypoint correction binding differs")
                require(stripped_waypoint == old_waypoint, "R003 waypoint desired pose changed")
            new_init_without_waypoints = deepcopy(new_init); old_init_without_waypoints = deepcopy(old_init)
            new_init_without_waypoints.pop("waypoints"); old_init_without_waypoints.pop("waypoints")
            require(new_init_without_waypoints == old_init_without_waypoints, "R003 solver initialization changed")

    r003_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r003/state_repair_gate.py"
    r004_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r004/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence",
        "_companion_gate", "_fresh_reset_and_gate", "_direct_materialize_and_gate",
    ):
        require(function_ast(r003_source, name) == function_ast(r004_source, name), f"unchanged helper differs: {name}")
    source_text = r004_source.read_text(encoding="utf-8")
    require("requests.post" not in source_text and "httpx" not in source_text and "policy_server" not in source_text, "model endpoint exists")
    tree = ast.parse(source_text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    pose_dump = ast.dump(functions["_run_registered_pose_hold"], include_attributes=False)
    require("corrected_command" in pose_dump and "maximum_rounds" in pose_dump, "correction loop absent")
    require("translation_gain" in pose_dump and "rotation_gain" in pose_dump, "registered gains absent")
    require("candidate_schedule" in pose_dump, "correction does not load frozen schedule")
    for caller in ("_run_known_reachable_diagnostic", "_solve_registered_ik"):
        require("_run_registered_pose_hold" in ast.dump(functions[caller], include_attributes=False), f"{caller} bypasses correction")
    command_dump = ast.dump(functions["_command_base_link"], include_attributes=False)
    require("EEF_OFFSET" not in command_dump and "_quat_inverse" not in command_dump, "base-link command applies offset")

    source_gate_summary = None
    if require_source_gate:
        require(source_gate_path.is_file(), "R004 source-push gate missing")
        source_gate = load(source_gate_path)
        require(source_gate.get("schema_version") == "vla-wam-shared-v3e006-r004-source-push-gate-v1", "source-gate schema differs")
        require(source_gate.get("status") == SOURCE_GATE_STATUS, "source-gate status differs")
        require(all(source_gate.get(key) == 0 for key in (
            "model_request_count", "behavioral_episode_count", "r004_live_diagnostic_count",
            "r004_live_candidate_evaluation_count", "completed_candidate_pair_count",
            "accepted_state_candidate_count", "infrastructure_invalid_search_attempt_count",
        )), "source-gate counts differ")
        implementation = str(source_gate.get("implementation_commit", ""))
        require(subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{implementation}^{{commit}}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0, "implementation commit absent")
        inventory = source_gate.get("implementation_files")
        require(isinstance(inventory, list) and len(inventory) == source_gate.get("implementation_file_count") == 11, "source inventory differs")
        for row in inventory:
            relative = Path(str(row.get("path", "")))
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
            verify_binding(root / relative, row, f"source inventory {relative}")
        verify_binding(registration_path, source_gate["repair_registration"], "source-gate registration")
        verify_binding(schedule_path, source_gate["candidate_schedule"], "source-gate schedule")
        source_gate_summary = binding(source_gate_path)
    return {
        "passed": True,
        "registration": binding(registration_path),
        "candidate_schedule": binding(schedule_path),
        "r003_predecessor_results": predecessor["results"],
        "residual_correction_contract_sha256": correction_sha,
        "candidate_pair_count": 4,
        "diagnostic_count": 4,
        "source_push_gate": source_gate_summary,
    }


def validate_candidate_root(root: Path, candidate_root: Path) -> dict[str, Any]:
    candidate_root = candidate_root.resolve()
    launch_path, harness_path, runtime_path = (
        candidate_root / "launch.json", candidate_root / "harness_result.json",
        candidate_root / "runtime.log",
    )
    launch, harness = load(launch_path), load(harness_path)
    require(harness.get("model_request_count") == harness.get("behavioral_episode_count") == 0, "harness counts differ")
    verify_binding(launch_path, harness["launch"], "harness launch")
    verify_binding(runtime_path, harness["runtime_log"], "harness runtime")
    for name, row in launch.get("input_bindings", {}).items():
        verify_binding(Path(str(row["path"])), row, f"launch input {name}")
    for name, row in launch.get("formal_health_preflight", {}).items():
        verify_binding(Path(str(row["path"])), row, f"health input {name}")
    verify_binding(Path(str(launch["harness_source"]["path"])), launch["harness_source"], "harness source")
    child_binding = harness.get("child_report")
    require(isinstance(child_binding, Mapping), "child report absent")
    report_path = Path(str(child_binding["path"])); verify_binding(report_path, child_binding, "child report")
    report = load(report_path)
    if harness.get("process_completed") is True:
        require(report_path.name == "state_repair_result.json", "terminal report filename differs")
        validate_scientific_selection(report, harness)
        for diagnostic in report["known_reachable_diagnostics"]:
            require(diagnostic["environment_lifecycle"]["closed_before_next_environment"] is True, "diagnostic env not closed")
            require(diagnostic["fresh_reset"]["passed"] is True, "diagnostic reset failed")
            for camera in diagnostic["fresh_reset"]["camera_evidence"]["bindings"].values():
                verify_binding(Path(str(camera["rgb"]["path"])), camera["rgb"], "diagnostic camera")
        for attempt in report.get("attempts", []):
            for stage in attempt["stages"].values():
                require(stage["ik_solve_environment"]["environment_lifecycle"]["closed_before_next_environment"] is True, "IK env not closed")
                require(stage["ik_solve_environment"]["fresh_reset"]["passed"] is True, "IK reset failed")
                if stage["ik_solution"]["passed"]:
                    require(stage["materialization_environment"]["environment_lifecycle"]["closed_before_next_environment"] is True, "material env not closed")
                    require(stage["materialization_environment"]["fresh_reset"]["passed"] is True, "material reset failed")
        for key in (
            "repair_registration", "candidate_schedule", "source_push_gate",
            "original_v3e006_closure_binding", "r003_predecessor_results", "ood_freeze",
            "e004_full_reset_reference", "e004_candidate", "construction_source", "video",
        ):
            verify_binding(Path(str(report[key]["path"])), report[key], f"report {key}")
        verify_binding(Path(str(report["frozen_e004_runtime_bindings"]["path"])), report["frozen_e004_runtime_bindings"], "report runtime")
        for row in report["scene_assets"].values():
            verify_binding(Path(str(row["path"])), row, "report scene")
    else:
        require(report_path.name == "state_construction_failure.json", "invalid report filename differs")
        require(report.get("status") == "infrastructure_invalid_r004_state_repair", "invalid status differs")
        require(report.get("model_request_count") == report.get("behavioral_episode_count") == report.get("state_candidate_count") == 0, "invalid counts differ")
        for row in report.get("available_raw_artifacts", {}).values():
            verify_binding(Path(str(row["path"])), row, "invalid raw artifact")
    evidence = report.get("execution_evidence", report)
    require(evidence.get("passed_health_preflight") == launch.get("formal_health_preflight"), "child health differs")
    for group in ("input_bindings", "passed_health_preflight"):
        for name, row in evidence.get(group, {}).items():
            verify_binding(Path(str(row["path"])), row, f"child {group}/{name}")
    for source in evidence.get("historical_source_verification_before_AppLauncher", {}).values():
        for name, row in source.get("bindings", {}).items():
            verify_binding(Path(str(row["path"])), row, f"historical source {name}")
    for name, row in evidence.get("controller_source_verification_before_AppLauncher", {}).items():
        verify_binding(Path(str(row["path"])), row, f"controller source {name}")
    return {
        "passed": True, "candidate_root": str(candidate_root), "launch": binding(launch_path),
        "harness": binding(harness_path), "child_report": binding(report_path),
        "child_status": report.get("status"), "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--pre-source-gate", action="store_true")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    result = validate_static(args.study_root, require_source_gate=not args.pre_source_gate)
    if args.candidate_root is not None:
        require(args.verify_raw, "--candidate-root requires --verify-raw")
        result["candidate_evidence"] = validate_candidate_root(args.study_root, args.candidate_root)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
