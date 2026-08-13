from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r004.predecessor_contract import (
    validate_r003_diagnostic_closure,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r004.residual_correction import (
    corrected_command,
    validate_contract,
)
from tools.validate_v3e006_r004 import (
    ValidationError,
    canonical_sha,
    validate_pose_hold,
    validate_scientific_selection,
    validate_static,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r004"
REGISTRATION = ARTIFACT / "repair_registration.json"
SCHEDULE = ARTIFACT / "gates/candidate_schedule.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name)
    return ast.dump(node, include_attributes=False)


def _error(round_index: int, step: int, desired_p, desired_q, command_p, command_q,
           measured_p, measured_q, *, passed: bool) -> dict:
    return {
        "correction_round_one_based": round_index,
        "round_step_one_based": step,
        "step_one_based": (round_index - 1) * 30 + step,
        "desired_target_position_world_m": list(desired_p),
        "desired_target_quaternion_world_wxyz": list(desired_q),
        "command_position_world_m": list(command_p),
        "command_quaternion_world_wxyz": list(command_q),
        "measured_position_world_m": list(measured_p),
        "measured_quaternion_world_wxyz": list(measured_q),
        "position_error_m": 0.0005 if passed else 0.002,
        "orientation_geodesic_error_deg": 0.0,
        "finite": True,
        "arm_inside_soft_joint_limits": True,
        "base_link_to_eef_frame_identity": {"passed": True},
    }


def _pose_hold(*, passes_round: int = 2) -> dict:
    schedule = load(SCHEDULE)
    contract = schedule["residual_correction_contract"]
    desired_p, desired_q = [0.3, 0.0, 0.25], [1.0, 0.0, 0.0, 0.0]
    command_p, command_q = list(desired_p), list(desired_q)
    rounds, errors, trace = [], [], []
    for round_index in range(1, passes_round + 1):
        round_pass = round_index == passes_round
        measured_p = [0.2995, 0.0, 0.25] if round_pass else [0.298, 0.0, 0.25]
        measured_q = list(desired_q)
        rows = [
            _error(round_index, step, desired_p, desired_q, command_p, command_q,
                   measured_p, measured_q, passed=round_pass)
            for step in range(1, 31)
        ]
        traces = [{"base_link_to_eef_frame_identity": {"passed": True}} for _ in rows]
        correction = None
        if not round_pass and round_index < 3:
            correction = corrected_command(
                desired_position=desired_p, desired_quaternion=desired_q,
                measured_position=measured_p, measured_quaternion=measured_q,
                current_command_position=command_p, current_command_quaternion=command_q,
                translation_gain=1.0, rotation_gain=1.0,
            )
        rounds.append({
            "round_one_based": round_index,
            "desired_target_position_world_m": list(desired_p),
            "desired_target_quaternion_world_wxyz": list(desired_q),
            "command_position_world_m": list(command_p),
            "command_quaternion_world_wxyz": list(command_q),
            "completed_steps": 30,
            "final_window_passed": round_pass,
            "errors": rows,
            "construction_action_trace": traces,
            "termination": None,
            "measured_residual_correction": correction,
        })
        errors.extend(rows); trace.extend(traces)
        if correction:
            command_p = correction["next_command_position_world_m"]
            command_q = correction["next_command_quaternion_world_wxyz"]
    return {
        "passed": True,
        "failure_reason": None,
        "target_base_link_pose_world_wxyz": desired_p + desired_q,
        "desired_target_invariant_across_rounds": True,
        "residual_correction_contract": contract,
        "residual_correction_contract_sha256": canonical_sha(contract),
        "maximum_correction_rounds": 3,
        "completed_correction_rounds": len(rounds),
        "hold_steps": 30,
        "hold_steps_per_round": 30,
        "completed_steps": len(errors),
        "required_final_consecutive_steps": 10,
        "position_error_m_inclusive": 0.001,
        "orientation_geodesic_error_deg_inclusive": 1.0,
        "final_window_passed": True,
        "all_states_finite": True,
        "all_arm_states_inside_live_soft_joint_limits": True,
        "all_base_link_to_eef_frame_identity_checks_passed": True,
        "errors": errors,
        "correction_rounds": rounds,
        "termination": None,
        "construction_action_trace": trace,
    }


def test_prospective_static_freeze_and_zero_counts() -> None:
    result = validate_static(ROOT, require_source_gate=False)
    assert result["passed"] is True
    registration, schedule = load(REGISTRATION), load(SCHEDULE)
    assert registration["predecessor_repair_amendment_id"] == "V3-E006-R003"
    assert registration["counts_at_registration"] == {
        "r004_live_diagnostics": 0, "r004_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }
    assert schedule["candidate_budget"] == schedule["diagnostic_budget"] == 4
    assert "close R004" in schedule["selection_rule"]["exhaustion"]
    assert "close R003" not in schedule["selection_rule"]["exhaustion"]


def test_real_r003_closure_and_mutations() -> None:
    registration = load(REGISTRATION)
    row = registration["r003_predecessor"]["results"]
    path = ROOT / row["path"]
    assert digest(path) == row["sha256"]
    payload = load(path)
    validate_r003_diagnostic_closure(payload)
    for key, value in (
        ("diagnostic_evaluation_count", 2), ("candidate_pair_evaluation_count", 1),
        ("model_request_count", 1), ("behavioral_activation_released", True),
    ):
        bad = deepcopy(payload); bad[key] = value
        with pytest.raises(ValueError):
            validate_r003_diagnostic_closure(bad)
    bad = deepcopy(payload); bad["raw_evidence"]["result"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_r003_diagnostic_closure(bad)


def test_r003_targets_sources_order_and_waypoints_are_exact() -> None:
    schedule = load(SCHEDULE)
    predecessor = load(ROOT / schedule["r003_target_schedule"]["path"])
    assert [row["candidate_rank"] for row in schedule["candidate_pairs"]] == [1, 2, 3, 4]
    for new, old in zip(schedule["known_reachable_diagnostics"], predecessor["known_reachable_diagnostics"], strict=True):
        stripped = deepcopy(new)
        stripped.pop("maximum_correction_rounds")
        stripped.pop("r004_residual_correction_contract_sha256")
        assert stripped == old
    for new_pair, old_pair in zip(schedule["candidate_pairs"], predecessor["candidate_pairs"], strict=True):
        for stage in ("canonical_grasp", "canonical_carry"):
            new, old = deepcopy(new_pair[stage]), deepcopy(old_pair[stage])
            new_init, old_init = new.pop("r004_solver_initialization"), old.pop("r003_solver_initialization")
            assert new == old
            new_init.pop("residual_correction_contract_sha256")
            for waypoint, old_waypoint in zip(new_init["waypoints"], old_init["waypoints"], strict=True):
                waypoint.pop("maximum_correction_rounds")
                waypoint.pop("r004_residual_correction_contract_sha256")
                assert waypoint == old_waypoint


def test_correction_contract_rejects_any_registered_parameter_mutation() -> None:
    contract = load(SCHEDULE)["residual_correction_contract"]
    validate_contract(contract)
    for key, value in (
        ("translation_gain", 0.9), ("rotation_gain", 0.9),
        ("maximum_correction_rounds", 4), ("hold_steps_per_round", 29),
        ("required_final_consecutive_steps", 9), ("position_error_m_inclusive", 0.0011),
        ("orientation_geodesic_error_deg_inclusive", 1.1),
    ):
        bad = deepcopy(contract); bad[key] = value
        with pytest.raises(ValueError):
            validate_contract(bad)


def test_symmetric_translation_and_rotation_correction_math() -> None:
    result = corrected_command(
        desired_position=[1.0, 2.0, 3.0], desired_quaternion=[1.0, 0.0, 0.0, 0.0],
        measured_position=[0.9, 2.2, 2.7], measured_quaternion=[0.0, 0.0, 0.0, 1.0],
        current_command_position=[0.8, 2.1, 2.8], current_command_quaternion=[0.0, 0.0, 0.0, 1.0],
        translation_gain=1.0, rotation_gain=1.0,
    )
    assert np.allclose(result["next_command_position_world_m"], [0.9, 1.9, 3.1])
    observed = np.asarray(result["next_command_quaternion_world_wxyz"])
    assert abs(float(np.dot(observed, [1.0, 0.0, 0.0, 0.0]))) > 1.0 - 1e-12
    antipode = corrected_command(
        desired_position=[0, 0, 0], desired_quaternion=[-1, 0, 0, 0],
        measured_position=[0, 0, 0], measured_quaternion=[1, 0, 0, 0],
        current_command_position=[0, 0, 0], current_command_quaternion=[1, 0, 0, 0],
        translation_gain=1.0, rotation_gain=1.0,
    )
    assert abs(float(np.dot(antipode["next_command_quaternion_world_wxyz"], [1, 0, 0, 0]))) > 1 - 1e-12


def test_pose_hold_recomputes_rounds_and_rejects_mutations() -> None:
    hold = _pose_hold(passes_round=2)
    assert validate_pose_hold(hold, "fixture") is True
    for mutate in (
        lambda x: x["correction_rounds"][0]["measured_residual_correction"]["next_command_position_world_m"].__setitem__(0, 9.0),
        lambda x: x["correction_rounds"][1]["desired_target_position_world_m"].__setitem__(1, 0.1),
        lambda x: x.__setitem__("position_error_m_inclusive", 0.0011),
        lambda x: x.__setitem__("maximum_correction_rounds", 4),
    ):
        bad = deepcopy(hold); mutate(bad)
        with pytest.raises(ValidationError):
            validate_pose_hold(bad, "mutated")


def test_pose_hold_stops_unsafe_without_correction_and_never_runs_fourth_round() -> None:
    unsafe = _pose_hold(passes_round=2)
    first = unsafe["correction_rounds"][0]
    first["errors"][-1]["finite"] = False
    first["measured_residual_correction"] = None
    unsafe["correction_rounds"] = [first]
    unsafe["errors"] = first["errors"]
    unsafe["construction_action_trace"] = first["construction_action_trace"]
    unsafe["completed_correction_rounds"] = 1
    unsafe["completed_steps"] = 30
    unsafe["all_states_finite"] = False
    unsafe["final_window_passed"] = False
    unsafe["passed"] = False
    assert validate_pose_hold(unsafe, "unsafe") is False

    malformed = _pose_hold(passes_round=3)
    malformed["correction_rounds"].append(deepcopy(malformed["correction_rounds"][-1]))
    malformed["completed_correction_rounds"] = 4
    with pytest.raises(ValidationError):
        validate_pose_hold(malformed, "fourth-round")


def test_runtime_topology_uses_one_correction_function_for_diagnostic_and_waypoints() -> None:
    source = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r004/state_repair_gate.py"
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    pose = ast.dump(functions["_run_registered_pose_hold"], include_attributes=False)
    assert "corrected_command" in pose
    assert "translation_gain" in pose and "rotation_gain" in pose
    for caller in ("_run_known_reachable_diagnostic", "_solve_registered_ik"):
        assert "_run_registered_pose_hold" in ast.dump(functions[caller], include_attributes=False)
    assert "requests.post" not in text and "httpx" not in text and "policy_server" not in text


def test_unchanged_scientific_helpers_match_r003_ast() -> None:
    r003 = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r003/state_repair_gate.py"
    r004 = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r004/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence",
        "_companion_gate", "_fresh_reset_and_gate", "_direct_materialize_and_gate",
    ):
        assert function_ast(r003, name) == function_ast(r004, name)


def test_selection_blocks_candidates_until_all_four_diagnostics_pass() -> None:
    diagnostic = lambda i: {
        "diagnostic_index_one_based": i, "passed": True, "pose_hold": _pose_hold(passes_round=3)
    }
    report = {
        "status": "r004_known_reachable_diagnostic_failed_candidates_not_evaluated",
        "passed": False, "model_request_count": 0, "behavioral_episode_count": 0,
        "known_reachable_diagnostics": [diagnostic(1)], "r004_live_diagnostic_count": 1,
        "repair_candidate_evaluation_count": 0, "accepted_candidate_rank": None, "attempts": [],
    }
    report["known_reachable_diagnostics"][0]["passed"] = False
    failed_hold = report["known_reachable_diagnostics"][0]["pose_hold"]
    failed_hold["passed"] = False; failed_hold["final_window_passed"] = False
    failed_hold["correction_rounds"][-1]["final_window_passed"] = False
    for row in failed_hold["correction_rounds"][-1]["errors"][-10:]:
        row["measured_position_world_m"] = [0.298, 0.0, 0.25]
        row["position_error_m"] = 0.002
    harness = {
        "status": "completed_r004_candidate_search", "process_completed": True,
        "scientific_gate_passed": False, "child_status": report["status"],
        "r004_live_diagnostic_count": 1, "repair_candidate_evaluation_count": 0,
    }
    validate_scientific_selection(report, harness)
    bad = deepcopy(report); bad["attempts"] = [{"candidate_rank": 1}]
    with pytest.raises(ValidationError):
        validate_scientific_selection(bad, harness)


def test_stale_r002_primary_predecessor_key_is_rejected() -> None:
    source = (ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r004/state_repair_gate.py").read_text()
    assert '"r003_predecessor_results"' in source
    assert '"r002_predecessor_results"' not in source
    assert load(REGISTRATION)["predecessor_repair_amendment_id"] == "V3-E006-R003"
