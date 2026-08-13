#!/usr/bin/env python3
"""Fail-closed validator for prospective and raw V3-E006-R006 evidence."""

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

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r006.predecessor_contract import (
    validate_r005_exhaustion_closure,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r006.residual_correction import (
    corrected_command,
    normalize_quaternion,
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r006.source_gate_contract import (
    SCHEMA as SOURCE_GATE_SCHEMA,
    STATUS as SOURCE_GATE_STATUS,
    validate_source_gate,
)


BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
R005_CLOSURE_COMMIT = "040cf75c1d83a2e5f8383d87247fb096e8d2491a"
R005_RESULTS_SHA256 = "550665a234c378cbcb5c8022d16249a980d1a5b5368b08900568c959c51fb9f2"
R005_RAW_RESULT_SHA256 = "4cb6c3e7e19e510e2422368131c06074014ba5173caf2134b7d6b456043afc01"
REGISTRATION_STATUS = (
    "prospectively_registered_before_any_r006_live_diagnostic_candidate_or_model_request"
)
TERMINAL_STATUSES = {
    "passed_r006_state_repair_not_released_for_behavior": True,
    "r006_candidate_budget_exhausted_no_valid_state_pair": False,
    "r006_known_reachable_diagnostic_failed_candidates_not_evaluated": False,
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


def validate_construction_horizon_activation(lifecycle: Mapping[str, Any], label: str) -> None:
    activation = lifecycle.get("construction_horizon_activation")
    require(isinstance(activation, Mapping), f"{label} horizon activation absent")
    require(
        activation.get("status")
        == "registered_construction_timeout_extended_before_first_reset_or_step",
        f"{label} horizon activation status differs",
    )
    require(activation.get("only_mutated_field") == "env.cfg.episode_length_s", f"{label} mutation differs")
    require(activation.get("original_max_episode_length_steps") == 450, f"{label} original horizon differs")
    require(activation.get("registered_max_episode_length_steps") == 900, f"{label} registered horizon differs")
    require(activation.get("original_episode_length_s") == 30.0, f"{label} original seconds differ")
    require(activation.get("registered_episode_length_s") == 60.0, f"{label} registered seconds differ")
    require(
        math.isclose(float(activation.get("step_dt_s")), 1.0 / 15.0, rel_tol=0.0, abs_tol=1e-12),
        f"{label} step dt differs",
    )
    require(activation.get("common_step_counter_before_and_after") == 0, f"{label} stepped before activation")
    require(activation.get("episode_length_buf_before_and_after") == [0.0], f"{label} episode counter differs")
    require(activation.get("termination_config_byte_equal") is True, f"{label} termination config changed")
    require(
        activation.get("termination_contract_before") == activation.get("termination_contract_after"),
        f"{label} termination snapshots differ",
    )
    require(
        activation.get("termination_contract_before", {}).get("active_terms") == ["success", "time_out"],
        f"{label} termination terms differ",
    )
    require(activation.get("registered_worst_case_steps") == 795, f"{label} worst case differs")
    require(activation.get("registered_margin_steps") == 105, f"{label} margin differs")
    require(activation.get("behavioral_horizon_mutated") is False, f"{label} behavioral horizon changed")


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
    require(report.get("r006_live_diagnostic_count") == len(diagnostics), "child diagnostic count differs")
    require(harness.get("r006_live_diagnostic_count") == len(diagnostics), "harness diagnostic count differs")
    attempts = report.get("attempts")
    if report.get("status") == "r006_known_reachable_diagnostic_failed_candidates_not_evaluated":
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
                if evaluated:
                    validate_joint_equilibrium_state(state, stage_name)
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
    require(harness.get("status") == "completed_r006_candidate_search", "harness status differs")
    require(harness.get("scientific_gate_passed") is terminal_pass, "harness scientific flag differs")
    require(harness.get("child_status") == report.get("status"), "harness child status differs")


def validate_terminal_selection_rule(
    report: Mapping[str, Any], schedule: Mapping[str, Any]
) -> None:
    require(
        report.get("selection_rule") == schedule.get("selection_rule"),
        "terminal selection rule is not the exact frozen candidate schedule",
    )


def validate_joint_equilibrium_state(state: Mapping[str, Any], label: str) -> None:
    construction = state.get("construction")
    require(isinstance(construction, Mapping), f"{label} construction absent")
    require(
        construction.get("method")
        == "direct_contact_initialization_then_uniform_normal_joint_equilibrium_hold",
        f"{label} equilibrium method differs",
    )
    require(construction.get("settle_steps") == 780, f"{label} settle count differs")
    require(construction.get("gate_window_final_steps") == 10, f"{label} gate window differs")
    require(construction.get("episode_length_buf_before_equilibrium") == [75], f"{label} pre-settle counter differs")
    require(construction.get("episode_length_buf_after_equilibrium") == [855], f"{label} post-settle counter differs")
    require(construction.get("joint_target_write_count_before_settle") == 1, f"{label} initial target write differs")
    require(construction.get("joint_target_write_count_during_settle") == 0, f"{label} target changed during settle")
    require(construction.get("cartesian_action_manager_apply_count_during_settle") == 0, f"{label} Cartesian action applied")
    require(construction.get("joint_or_cube_state_write_count_during_settle") == 0, f"{label} state rewritten")
    target = construction.get("normal_joint_position_target_rad")
    require(isinstance(target, list) and len(target) == 13, f"{label} joint target differs")
    require(construction.get("authoritative_closed_gripper_joint_target_rad") == target[7:], f"{label} gripper target differs")
    trace = construction.get("construction_action_trace")
    require(isinstance(trace, list) and len(trace) == 780, f"{label} trace count differs")
    for index, row in enumerate(trace, start=1):
        require(row.get("phase_step_one_based") == index, f"{label} trace order differs")
        require(row.get("phase") == "normal_joint_equilibrium_hold_780", f"{label} phase differs")
        require(row.get("normal_joint_position_target_rad") == target, f"{label} target adapted")
        require(row.get("cartesian_action_manager_applied") is False, f"{label} Cartesian action flag differs")
        require("command_action_8d" not in row, f"{label} retained a Cartesian command")


def validate_static(
    root: Path, *, require_source_gate: bool = True, verify_retry_history: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    artifact = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006"
    registration_path = artifact / "repair_registration.json"
    schedule_path = artifact / "gates/candidate_schedule.json"
    source_gate_path = artifact / "source_push_gate.json"
    registration, schedule = load(registration_path), load(schedule_path)
    require(registration.get("repair_amendment_id") == "V3-E006-R006", "registration ID differs")
    require(registration.get("status") == REGISTRATION_STATUS, "registration status differs")
    require(registration.get("predecessor_repair_amendment_id") == "V3-E006-R005", "predecessor ID differs")
    require(registration.get("counts_at_registration") == {
        "r006_live_diagnostics": 0, "r006_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }, "registration counts differ")
    for commit in (BASE, R005_CLOSURE_COMMIT):
        require(subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0, f"lineage commit absent: {commit}")
    for name, row in registration["frozen_inputs"].items():
        resolve_bound(root, row, f"registration frozen input {name}")
    predecessor = registration["r005_predecessor"]
    require(predecessor.get("closure_commit") == R005_CLOSURE_COMMIT, "R005 closure commit differs")
    require(predecessor["results"].get("sha256") == R005_RESULTS_SHA256, "R005 results digest differs")
    require(predecessor["raw_result"].get("sha256") == R005_RAW_RESULT_SHA256, "R005 raw digest differs")
    predecessor_results_path = resolve_bound(root, predecessor["results"], "R005 predecessor results")
    try:
        validate_r005_exhaustion_closure(load(predecessor_results_path))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    require(schedule.get("repair_amendment_id") == "V3-E006-R006", "schedule ID differs")
    require(schedule.get("status") == "frozen_before_any_r006_live_diagnostic_candidate_or_model_request", "schedule status differs")
    require(schedule.get("candidate_budget") == schedule.get("diagnostic_budget") == 4, "budgets differ")
    require([row.get("candidate_rank") for row in schedule.get("candidate_pairs", [])] == [1, 2, 3, 4], "rank order differs")
    require([row.get("diagnostic_index_one_based") for row in schedule.get("known_reachable_diagnostics", [])] == [1, 2, 3, 4], "diagnostic order differs")
    require(all(schedule.get(key) == 0 for key in (
        "model_request_count", "behavioral_episode_count", "r006_live_diagnostic_count",
        "r006_live_candidate_evaluation_count",
    )), "schedule counts differ")
    verify_binding(registration_path, schedule["repair_registration"], "schedule registration")
    require(schedule.get("r005_predecessor") == predecessor, "schedule predecessor differs")
    correction = schedule.get("residual_correction_contract", {})
    try:
        validate_contract(correction)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    correction_sha = canonical_sha(correction)
    require(correction_sha == schedule.get("residual_correction_contract_sha256"), "correction digest differs")
    horizon = schedule.get("construction_horizon_contract", {})
    horizon_sha = canonical_sha(horizon)
    require(horizon_sha == schedule.get("construction_horizon_contract_sha256"), "horizon digest differs")
    require(horizon.get("original_max_episode_length_steps") == 450, "original horizon differs")
    require(horizon.get("registered_max_episode_length_steps") == 900, "registered horizon differs")
    require(horizon.get("original_episode_length_s") == 30.0, "original seconds differ")
    require(horizon.get("registered_episode_length_s") == 60.0, "registered seconds differ")
    require(horizon.get("required_step_dt_s") == 1.0 / 15.0, "step dt differs")
    require(horizon.get("applies_to_roles") == [
        "known_reachable_diagnostic", "ik_solve", "candidate_materialization"
    ], "horizon roles differ")
    worst = horizon.get("worst_case_derivation", {})
    require(worst == {
        "fresh_reset_steps": 75,
        "waypoint_count": 8,
        "maximum_correction_rounds_per_waypoint": 3,
        "hold_steps_per_round": 30,
        "worst_case_steps": 795,
        "registered_margin_steps": 105,
    }, "worst-case derivation differs")
    require(75 + 8 * 3 * 30 == 795 < 900, "registered horizon does not cover worst case")
    require(horizon.get("behavioral_episode_horizon_unchanged") is True, "behavioral horizon changed")

    equilibrium = schedule.get("joint_equilibrium_hold_contract", {})
    equilibrium_sha = canonical_sha(equilibrium)
    require(equilibrium_sha == schedule.get("joint_equilibrium_hold_contract_sha256"), "equilibrium digest differs")
    require(registration.get("joint_equilibrium_hold_contract") == equilibrium, "registration equilibrium differs")
    require(registration.get("joint_equilibrium_hold_contract_sha256") == equilibrium_sha, "registration equilibrium digest differs")
    require(equilibrium.get("settle_steps") == 780, "equilibrium settle differs")
    require(equilibrium.get("required_episode_length_buf_before_settle") == 75, "pre-settle count differs")
    require(equilibrium.get("worst_case_materialization_steps") == 855, "materialization total differs")
    require(equilibrium.get("fixed_margin_after_reset_and_settle_steps") == 45, "horizon margin differs")
    require(75 + 780 == 855 and 900 - 855 == 45, "equilibrium horizon arithmetic differs")

    r005_schedule_path = resolve_bound(root, schedule["r005_target_solver_horizon_schedule"], "R005 target/solver/horizon schedule")
    r005_schedule = load(r005_schedule_path)
    require(schedule["known_reachable_diagnostics"] == r005_schedule["known_reachable_diagnostics"], "R005 diagnostics changed")
    require(schedule["candidate_pairs"] == r005_schedule["candidate_pairs"], "R005 candidate targets/order changed")
    require(correction == r005_schedule["residual_correction_contract"], "R005 residual correction changed")
    require(horizon == r005_schedule["construction_horizon_contract"], "R005 construction horizon changed")
    require(
        schedule.get("residual_correction_contract_sha256")
        == r005_schedule.get("residual_correction_contract_sha256"),
        "R005 residual-correction digest changed",
    )

    r005_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r005/state_repair_gate.py"
    r006_source = root / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence",
        "_companion_gate", "_fresh_reset_and_gate", "_finalize_unchanged_gates",
    ):
        require(function_ast(r005_source, name) == function_ast(r006_source, name), f"unchanged helper differs: {name}")
    source_text = r006_source.read_text(encoding="utf-8")
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
    equilibrium_step_dump = ast.dump(functions["_normal_joint_equilibrium_step"], include_attributes=False)
    require("set_joint_position_target" not in equilibrium_step_dump, "joint target is rewritten during settle")
    require("process_action" not in equilibrium_step_dump and "apply_action" not in equilibrium_step_dump, "Cartesian action entered equilibrium step")
    materialize_dump = ast.dump(functions["_direct_materialize_and_gate"], include_attributes=False)
    require(materialize_dump.count("set_joint_position_target") == 1, "joint target write count differs")
    require("_normal_joint_equilibrium_step" in materialize_dump and "780" in materialize_dump, "equilibrium loop differs")

    source_gate_summary = None
    if require_source_gate:
        require(source_gate_path.is_file(), "R006 source-push gate missing")
        source_gate = load(source_gate_path)
        require(source_gate.get("schema_version") == SOURCE_GATE_SCHEMA, "source-gate schema differs")
        require(source_gate.get("status") == SOURCE_GATE_STATUS, "source-gate status differs")
        try:
            validate_source_gate(
                source_gate, study_root=root, verify_raw_history=verify_retry_history
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        implementation = str(source_gate.get("implementation_commit", ""))
        require(subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{implementation}^{{commit}}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0, "implementation commit absent")
        inventory = source_gate.get("implementation_files")
        require(
            isinstance(inventory, list)
            and len(inventory) == source_gate.get("implementation_file_count") == 13,
            "source inventory differs",
        )
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
        "r005_predecessor_results": predecessor["results"],
        "residual_correction_contract_sha256": correction_sha,
        "construction_horizon_contract_sha256": horizon_sha,
        "joint_equilibrium_hold_contract_sha256": equilibrium_sha,
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
        schedule = load(
            root
            / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006/gates/candidate_schedule.json"
        )
        validate_terminal_selection_rule(report, schedule)
        validate_scientific_selection(report, harness)
        for diagnostic in report["known_reachable_diagnostics"]:
            validate_construction_horizon_activation(
                diagnostic["environment_lifecycle"], "diagnostic environment"
            )
            require(diagnostic["environment_lifecycle"]["closed_before_next_environment"] is True, "diagnostic env not closed")
            require(diagnostic["fresh_reset"]["passed"] is True, "diagnostic reset failed")
            for camera in diagnostic["fresh_reset"]["camera_evidence"]["bindings"].values():
                verify_binding(Path(str(camera["rgb"]["path"])), camera["rgb"], "diagnostic camera")
        for attempt in report.get("attempts", []):
            for stage in attempt["stages"].values():
                validate_construction_horizon_activation(
                    stage["ik_solve_environment"]["environment_lifecycle"], "IK environment"
                )
                require(stage["ik_solve_environment"]["environment_lifecycle"]["closed_before_next_environment"] is True, "IK env not closed")
                require(stage["ik_solve_environment"]["fresh_reset"]["passed"] is True, "IK reset failed")
                if stage["ik_solution"]["passed"]:
                    validate_construction_horizon_activation(
                        stage["materialization_environment"]["environment_lifecycle"],
                        "materialization environment",
                    )
                    require(stage["materialization_environment"]["environment_lifecycle"]["closed_before_next_environment"] is True, "material env not closed")
                    require(stage["materialization_environment"]["fresh_reset"]["passed"] is True, "material reset failed")
        for key in (
            "repair_registration", "candidate_schedule", "source_push_gate",
            "original_v3e006_closure_binding", "r005_predecessor_results", "ood_freeze",
            "e004_full_reset_reference", "e004_candidate", "construction_source", "video",
        ):
            verify_binding(Path(str(report[key]["path"])), report[key], f"report {key}")
        verify_binding(Path(str(report["frozen_e004_runtime_bindings"]["path"])), report["frozen_e004_runtime_bindings"], "report runtime")
        for row in report["scene_assets"].values():
            verify_binding(Path(str(row["path"])), row, "report scene")
    else:
        require(report_path.name == "state_construction_failure.json", "invalid report filename differs")
        require(report.get("status") == "infrastructure_invalid_r006_state_repair", "invalid status differs")
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
    result = validate_static(
        args.study_root,
        require_source_gate=not args.pre_source_gate,
        verify_retry_history=args.verify_raw,
    )
    if args.candidate_root is not None:
        require(args.verify_raw, "--candidate-root requires --verify-raw")
        result["candidate_evidence"] = validate_candidate_root(args.study_root, args.candidate_root)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
