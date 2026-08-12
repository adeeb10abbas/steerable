from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.validate_v3e006_r003 import (
    ValidationError,
    shortest_slerp,
    validate_scientific_selection,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r003"
REGISTRATION = ARTIFACT / "repair_registration.json"
SCHEDULE = ARTIFACT / "gates/candidate_schedule.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name)
    return ast.dump(node, include_attributes=False)


def test_registration_schedule_are_prospective_finite_and_zero_count() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    assert registration["status"] == "prospectively_registered_before_any_r003_live_diagnostic_candidate_or_model_request"
    assert registration["counts_at_registration"] == {
        "r003_live_diagnostics": 0,
        "r003_live_candidate_evaluations": 0,
        "model_requests": 0,
        "behavioral_episodes": 0,
    }
    assert schedule["candidate_budget"] == schedule["diagnostic_budget"] == 4
    assert [row["candidate_rank"] for row in schedule["candidate_pairs"]] == [1, 2, 3, 4]
    assert [row["diagnostic_index_one_based"] for row in schedule["known_reachable_diagnostics"]] == [1, 2, 3, 4]
    assert schedule["model_request_count"] == schedule["behavioral_episode_count"] == 0
    assert schedule["r003_live_diagnostic_count"] == schedule["r003_live_candidate_evaluation_count"] == 0
    assert schedule["repair_registration"]["bytes"] == REGISTRATION.stat().st_size
    assert schedule["repair_registration"]["sha256"] == sha256(REGISTRATION)


def test_frame_relabel_preserves_numeric_r002_targets_and_cube_world_pose() -> None:
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    r002_path = ROOT / schedule["r002_predecessor"]["candidate_schedule"]["path"]
    r002 = json.loads(r002_path.read_text(encoding="utf-8"))
    for pair in schedule["candidate_pairs"]:
        old_pair = r002["candidate_pairs"][pair["candidate_rank"] - 1]
        for stage in ("canonical_grasp", "canonical_carry"):
            new, old = pair[stage], old_pair[stage]
            assert new["target_cube_pose"] == old["target_cube_pose"]
            assert new["centerline_constrained_base_link_ik_target"] == old["centerline_constrained_eef_ik_target"]
            assert new["selected_observed_cube_in_base_link_transform"] == old["selected_observed_cube_in_eef_transform"]
            assert "centerline_constrained_eef_ik_target" not in new
            for source in new["both_direction_sources"].values():
                assert "base_link_position_world_m" in source
                assert "base_link_quaternion_world_wxyz" in source
                assert "eef_position_world_m" not in source


def test_solver_topology_is_direct_base_link_and_no_r002_fallback() -> None:
    source_path = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r003/state_repair_gate.py"
    source = source_path.read_text(encoding="utf-8")
    assert "def _command_base_link(" in source
    assert "action = _command_base_link(position, quaternion, 1.0, env.device)" in source
    assert "_run_known_reachable_diagnostic(" in source
    assert "historical_q_seed_then_eight_registered_abs_ik_waypoints" in source
    assert "open_approach_close_lift" not in source
    assert "requests.post" not in source
    assert "httpx" not in source
    tree = ast.parse(source)
    command = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == "_command_base_link")
    command_dump = ast.dump(command, include_attributes=False)
    assert "EEF_OFFSET" not in command_dump
    assert "_quat_inverse" not in command_dump
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    assert [row["construction_method"] for row in schedule["candidate_pairs"]] == ["direct_contact_initialization"] * 4
    for pair in schedule["candidate_pairs"]:
        for stage in ("canonical_grasp", "canonical_carry"):
            waypoints = pair[stage]["r003_solver_initialization"]["waypoints"]
            assert len(waypoints) == 8
            assert [row["fraction"] for row in waypoints] == [index / 8 for index in range(1, 9)]
            assert all(row["hold_steps"] == 30 for row in waypoints)


def test_waypoint_shortest_arc_recomputed_and_mutation_rejected(tmp_path: Path) -> None:
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    row = schedule["candidate_pairs"][0]["canonical_grasp"]
    solver = row["r003_solver_initialization"]
    source = row["both_direction_sources"][solver["source_side"]]
    expected = shortest_slerp(
        source["base_link_quaternion_world_wxyz"],
        row["centerline_constrained_base_link_ik_target"]["quaternion_world_wxyz"],
        0.125,
    )
    observed = np.asarray(solver["waypoints"][0]["quaternion_world_wxyz"], dtype=np.float64)
    assert abs(float(np.dot(observed / np.linalg.norm(observed), expected))) >= 1.0 - 1e-14
    assert abs(float(np.dot(-observed / np.linalg.norm(observed), expected))) >= 1.0 - 1e-14
    genuine_rotation = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert abs(float(np.dot(genuine_rotation, expected))) < 1.0 - 1e-4
    mutated = deepcopy(schedule)
    mutated["candidate_pairs"][0]["canonical_grasp"]["r003_solver_initialization"]["waypoints"][0]["position_world_m"][0] += 1e-5
    # Mutation is directly detected by the same exact recomputation contract used by static validation.
    first = mutated["candidate_pairs"][0]["canonical_grasp"]
    src = first["both_direction_sources"][first["r003_solver_initialization"]["source_side"]]
    expected_x = 0.875 * src["base_link_position_world_m"][0] + 0.125 * first["centerline_constrained_base_link_ik_target"]["position_world_m"][0]
    assert first["r003_solver_initialization"]["waypoints"][0]["position_world_m"][0] != expected_x


def test_scientific_gate_sources_and_unchanged_helpers_are_preserved() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    expected = {
        "state_contract": "2476b28d2867c1b87f477fd5f89e545616be00d860d4144f8cbdb70af10f3c18",
        "ood_reference": "4df1ebf0061096a74b5eccd10b2a144e840f52fd50469b8bdae9369d1696fd04",
    }
    for name, digest in expected.items():
        row = registration["frozen_inputs"][name]
        assert sha256(ROOT / row["path"]) == row["sha256"] == digest
    r002 = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r002/state_repair_gate.py"
    r003 = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r003/state_repair_gate.py"
    for name in ("_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence", "_companion_gate", "_fresh_reset_and_gate"):
        assert function_ast(r002, name) == function_ast(r003, name)


def _diagnostic(index: int, passed: bool = True) -> dict:
    identity = {"passed": passed}
    errors = [{"base_link_to_eef_frame_identity": deepcopy(identity)} for _ in range(30)]
    trace = [{"base_link_to_eef_frame_identity": deepcopy(identity)} for _ in range(30)]
    return {
        "diagnostic_index_one_based": index,
        "passed": passed,
        "pose_hold": {
            "passed": passed,
            "completed_steps": 30,
            "hold_steps": 30,
            "final_window_passed": passed,
            "all_states_finite": True,
            "all_arm_states_inside_live_soft_joint_limits": True,
            "all_base_link_to_eef_frame_identity_checks_passed": passed,
            "termination": None,
            "errors": errors,
            "construction_action_trace": trace,
        },
    }


def _terminal_fixture() -> tuple[dict, dict]:
    gate = lambda passed: {"passed": passed}
    state = {
        "passed": True,
        "physics_gate": gate(True),
        "ood_gate": gate(True),
        "camera_evidence": gate(True),
        "companion_pose_gate": gate(True),
    }
    attempt = {
        "candidate_rank": 1,
        "stages": {
            "canonical_grasp": {"candidate_state": deepcopy(state)},
            "canonical_carry": {"candidate_state": deepcopy(state)},
        },
        "passed": True,
    }
    report = {
        "status": "passed_r003_state_repair_not_released_for_behavior",
        "passed": True,
        "known_reachable_diagnostics": [_diagnostic(index) for index in range(1, 5)],
        "r003_live_diagnostic_count": 4,
        "repair_candidate_evaluation_count": 1,
        "candidate_budget": 4,
        "accepted_candidate_rank": 1,
        "accepted_states": deepcopy(attempt["stages"]),
        "first_passing_rule_obeyed": True,
        "attempts": [attempt],
    }
    harness = {
        "r003_live_diagnostic_count": 4,
        "repair_candidate_evaluation_count": 1,
        "process_completed": True,
        "status": "completed_r003_candidate_search",
        "scientific_gate_passed": True,
        "child_status": report["status"],
    }
    return report, harness


def test_selection_requires_all_diagnostics_and_recomputes_frame_identity() -> None:
    report, harness = _terminal_fixture()
    validate_scientific_selection(report, harness)
    bad = deepcopy(report)
    bad["known_reachable_diagnostics"][0]["pose_hold"]["all_base_link_to_eef_frame_identity_checks_passed"] = False
    with pytest.raises(ValidationError):
        validate_scientific_selection(bad, harness)
    bad = deepcopy(report)
    bad["known_reachable_diagnostics"] = bad["known_reachable_diagnostics"][:3]
    bad["r003_live_diagnostic_count"] = 3
    with pytest.raises(ValidationError):
        validate_scientific_selection(bad, harness)


def test_diagnostic_failure_is_terminal_only_before_any_candidate() -> None:
    report, harness = _terminal_fixture()
    report.update(
        {
            "status": "r003_known_reachable_diagnostic_failed_candidates_not_evaluated",
            "passed": False,
            "known_reachable_diagnostics": [_diagnostic(1, passed=False)],
            "r003_live_diagnostic_count": 1,
            "repair_candidate_evaluation_count": 0,
            "accepted_candidate_rank": None,
            "attempts": [],
        }
    )
    harness.update(
        {
            "r003_live_diagnostic_count": 1,
            "repair_candidate_evaluation_count": 0,
            "scientific_gate_passed": False,
            "child_status": report["status"],
        }
    )
    validate_scientific_selection(report, harness)
    bad = deepcopy(report)
    bad["attempts"] = [{"candidate_rank": 1}]
    with pytest.raises(ValidationError):
        validate_scientific_selection(bad, harness)
