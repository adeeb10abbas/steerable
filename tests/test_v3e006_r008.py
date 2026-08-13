from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r008.object_servo import (
    object_space_servo_command,
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r008.predecessor_contract import (
    validate_r007_exhaustion_closure,
)
from tools.validate_v3e006_r008 import (
    ValidationError,
    expected_precontact,
    validate_candidate_state,
    validate_construction_lifecycle,
    validate_static,
)


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008"
R007 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == name)
    return ast.dump(node, include_attributes=False)


def test_real_r007_closure_and_mutations() -> None:
    value = load(R007 / "results/results.json")
    validate_r007_exhaustion_closure(value)
    for key, replacement in (
        ("candidate_pair_evaluation_count", 3),
        ("diagnostics_all_passed", False),
        ("accepted_candidate_rank", 1),
    ):
        bad = deepcopy(value); bad[key] = replacement
        with pytest.raises(ValueError):
            validate_r007_exhaustion_closure(bad)
    bad = deepcopy(value); bad["raw_result"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_r007_exhaustion_closure(bad)


def test_registration_and_finite_schedule_are_prospective() -> None:
    registration = load(ART / "repair_registration.json")
    schedule = load(ART / "gates/candidate_schedule.json")
    assert registration["predecessor_repair_amendment_id"] == "V3-E006-R007"
    assert registration["counts_at_registration"] == {
        "r008_live_diagnostics": 0, "r008_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }
    assert registration["r007_predecessor"]["closure_commit"] == "7cc3acc120027bdd181340b443633d8a03d6858d"
    assert [row["candidate_rank"] for row in schedule["candidate_pairs"]] == [1,2,3,4]
    assert {row["construction_method"] for row in schedule["candidate_pairs"]} == {
        "exact_reset_open_close_uniform_object_servo_q_handoff"
    }
    assert schedule["construction_lifecycle_contract"]["worst_case_steps"] == 1365
    assert schedule["construction_lifecycle_contract"]["registered_max_episode_length_steps"] == 1500
    for pair in schedule["candidate_pairs"]:
        for stage in ("canonical_grasp", "canonical_carry"):
            assert pair[stage]["r008_target_cube_pose"] == pair[stage]["target_cube_pose"]
            assert pair[stage]["r008_target_cube_pose"]["position_world_m"][1] == 0.0


def test_object_servo_is_uniform_capped_and_quaternion_sign_invariant() -> None:
    contract = load(ART / "gates/candidate_schedule.json")["object_space_servo_contract"]
    validate_contract(contract)
    kwargs = dict(
        live_base_position=[0.3, 0.0, 0.2], live_base_quaternion=[1,0,0,0],
        live_cube_position=[0.32, 0.02, 0.10], live_cube_quaternion=[1,0,0,0],
        target_cube_position=[0.31, 0.0, 0.13], target_cube_quaternion=[1,0,0,0],
        translation_gain=0.2, rotation_gain=0.2,
        translation_cap_m_per_step=0.002, rotation_cap_deg_per_step=2.0,
    )
    result = object_space_servo_command(**kwargs)
    assert np.linalg.norm(result["applied_translation_correction_world_m"]) <= 0.002 + 1e-15
    antipode = dict(kwargs); antipode["target_cube_quaternion"] = [-1,0,0,0]
    other = object_space_servo_command(**antipode)
    assert np.allclose(result["command_base_position_world_m"], other["command_base_position_world_m"])
    assert abs(np.dot(result["command_base_quaternion_world_wxyz"], other["command_base_quaternion_world_wxyz"])) == pytest.approx(1.0)
    for key, value in (("translation_gain", 0.3), ("servo_steps", 359), ("early_stop", True)):
        bad = deepcopy(contract); bad[key] = value
        with pytest.raises(ValueError):
            validate_contract(bad)


def _trace_row(phase: str, step: int) -> dict:
    return {
        "phase": phase, "phase_step_one_based": step,
        "base_link_to_eef_frame_identity": {
            "passed": True,
            "position_composition_residual_m": 0.0,
            "orientation_composition_residual_deg": 0.0,
        },
        "base_link_position_world_m": [0.3, 0.0, 0.2],
        "base_link_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
        "eef_position_world_m": [0.3, 0.0, 0.2],
        "eef_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
        "joint_position_rad": [0.0] * 13, "joint_velocity_rad_s": [0.0] * 13,
        "cube_position_world_m": [0.31, 0.0, 0.13],
        "cube_quaternion_world_wxyz": [1.0,0.0,0.0,0.0],
        "cube_linear_velocity_m_s": [0.0] * 3, "cube_angular_velocity_rad_s": [0.0] * 3,
    }


def _synthetic_state() -> tuple[dict, dict]:
    schedule = load(ART / "gates/candidate_schedule.json")
    stage = schedule["candidate_pairs"][0]["canonical_grasp"]
    retained = deepcopy(stage); retained["candidate_rank"] = 1
    construction = {
        "method": "exact_reset_open_close_uniform_object_servo_q_handoff",
        "registered_stage_schedule": retained,
        "registered_target_cube_pose": stage["r008_target_cube_pose"],
        "registered_precontact_targets": stage["r008_precontact_targets"],
        "start_base_link_pose_world_wxyz": [0.3,0.0,0.3,1.0,0.0,0.0,0.0],
        "phase_steps": {"open_approach":120,"open_descent":120,"normal_close":90,"closed_object_space_servo":360,"captured_q_normal_joint_settle":600},
        "episode_length_buf_before_candidate_actions": [75],
        "episode_length_buf_before_handoff": [765],
        "episode_length_buf_after_candidate_actions": [1365],
        "post_reset_joint_state_write_count": 0, "post_reset_object_state_write_count": 0,
        "captured_joint_target_write_count": 1,
        "cartesian_action_manager_apply_count_during_joint_settle": 0,
        "joint_or_object_state_write_count": 0,
        "captured_joint_position_target_rad": [0.0] * 13,
        "object_space_servo_contract": schedule["object_space_servo_contract"],
        "joint_handoff_contract": schedule["joint_handoff_contract"],
        "construction_lifecycle_contract": schedule["construction_lifecycle_contract"],
    }
    trace = []
    for phase, step, action in expected_precontact(construction):
        row = _trace_row(phase, step); row["command_action_8d"] = action; trace.append(row)
    contract = construction["object_space_servo_contract"]
    servo = object_space_servo_command(
        live_base_position=[0.3,0,0.2], live_base_quaternion=[1,0,0,0],
        live_cube_position=[0.31,0,0.13], live_cube_quaternion=[1,0,0,0],
        target_cube_position=stage["r008_target_cube_pose"]["position_world_m"],
        target_cube_quaternion=stage["r008_target_cube_pose"]["quaternion_world_wxyz"],
        translation_gain=contract["translation_gain"], rotation_gain=contract["rotation_gain"],
        translation_cap_m_per_step=contract["translation_cap_m_per_step"],
        rotation_cap_deg_per_step=contract["rotation_cap_deg_per_step"],
    )
    servo_rows = []
    for step in range(1,361):
        row = _trace_row("closed_object_space_servo", step)
        row["pre_action_object_space_servo"] = deepcopy(servo)
        row["command_action_8d"] = [*servo["command_base_position_world_m"], *servo["command_base_quaternion_world_wxyz"], 1.0]
        trace.append(row); servo_rows.append(row)
    for step in range(1,601):
        row = _trace_row("captured_q_normal_joint_settle", step)
        row["normal_joint_position_target_rad"] = [0.0] * 13
        row["cartesian_action_manager_applied"] = False
        trace.append(row)
    construction["construction_action_trace"] = trace
    construction["object_space_servo_trace"] = servo_rows
    construction["settled_gate_samples"] = [
        {
            "cube_position_world_m": list(row["cube_position_world_m"]),
            "cube_linear_velocity_m_s": list(row["cube_linear_velocity_m_s"]),
            "cube_angular_velocity_rad_s": list(row["cube_angular_velocity_rad_s"]),
            "eef_position_world_m": list(row["base_link_position_world_m"]),
            "base_link_quaternion_world_wxyz": list(row["base_link_quaternion_world_wxyz"]),
            "live_eef_frame_position_world_m": list(row["eef_position_world_m"]),
            "live_eef_frame_quaternion_world_wxyz": list(row["eef_quaternion_world_wxyz"]),
            "base_link_to_eef_frame_identity": deepcopy(row["base_link_to_eef_frame_identity"]),
            "arm_joint_velocity_rad_s": list(row["joint_velocity_rad_s"][:7]),
            "object_grabbed": True,
            "contact_force_n": {"gripper__rubiks_cube": 2.0},
        }
        for row in trace[-10:]
    ]
    state = {
        "construction": construction,
        "robot": {
            "joint_position_rad": list(trace[-1]["joint_position_rad"]),
            "joint_velocity_rad_s": list(trace[-1]["joint_velocity_rad_s"]),
        },
        "objects": {
            "rubiks_cube": {
                "position_world_m": list(trace[-1]["cube_position_world_m"]),
                "quaternion_world_wxyz": list(trace[-1]["cube_quaternion_world_wxyz"]),
                "linear_velocity_m_s": list(trace[-1]["cube_linear_velocity_m_s"]),
                "angular_velocity_rad_s": list(trace[-1]["cube_angular_velocity_rad_s"]),
            },
        },
        "contact_evidence": {
            "settled_force_snapshots_n": [{"gripper__rubiks_cube": 2.0} for _ in range(10)],
            "object_grabbed_by_step": [True] * 10,
        },
        "eef": {
            "position_world_m": list(trace[-1]["base_link_position_world_m"]),
            "quaternion_world_wxyz": list(trace[-1]["base_link_quaternion_world_wxyz"]),
        },
        "physics_gate": {"passed": True, "settled_window_steps": 10},
        "ood_gate": {"passed": True}, "camera_evidence": {"passed": True},
        "companion_pose_gate": {"passed": True}, "passed": True,
    }
    return state, stage


def test_full_trace_recomputation_rejects_command_q_and_target_mutations() -> None:
    state, stage = _synthetic_state()
    schedule = load(ART / "gates/candidate_schedule.json")
    validate_candidate_state(state, stage, 1, schedule)
    mutations = []
    bad = deepcopy(state); bad["construction"]["construction_action_trace"][12]["command_action_8d"][0] += 0.01; mutations.append(bad)
    bad = deepcopy(state); bad["construction"]["object_space_servo_trace"][5]["pre_action_object_space_servo"]["translation_gain"] = 0.4; mutations.append(bad)
    bad = deepcopy(state); bad["construction"]["construction_action_trace"][329]["cube_position_world_m"][0] += 0.01; mutations.append(bad)
    bad = deepcopy(state); bad["construction"]["construction_action_trace"][340]["base_link_to_eef_frame_identity"]["orientation_composition_residual_deg"] = 0.1; mutations.append(bad)
    bad = deepcopy(state); bad["construction"]["construction_action_trace"][700]["normal_joint_position_target_rad"][0] = 1.0; mutations.append(bad)
    bad = deepcopy(state); bad["construction"]["captured_joint_position_target_rad"][0] = 1.0; mutations.append(bad)
    bad = deepcopy(state); bad["construction"]["joint_handoff_contract"]["settle_steps"] = 599; mutations.append(bad)
    bad = deepcopy(state); bad["construction"]["settled_gate_samples"][-1]["cube_position_world_m"][0] += 0.01; mutations.append(bad)
    bad = deepcopy(state); bad["contact_evidence"]["object_grabbed_by_step"][-1] = False; mutations.append(bad)
    bad = deepcopy(state); bad["eef"]["position_world_m"][0] += 0.01; mutations.append(bad)
    bad = deepcopy(state); bad["construction"]["registered_target_cube_pose"]["position_world_m"][1] = 0.01; mutations.append(bad)
    for value in mutations:
        with pytest.raises(ValidationError):
            validate_candidate_state(value, stage, 1, schedule)


def test_repo_bindings_survive_relocated_checkout(tmp_path: Path) -> None:
    relocated = tmp_path / "relocated-checkout"
    bindings: list[dict] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            if set(("path", "bytes", "sha256")) <= set(value):
                bindings.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(load(ART / "repair_registration.json"))
    collect(load(ART / "gates/candidate_schedule.json"))
    for row in bindings:
        path = Path(str(row["path"]))
        if path.is_absolute():
            assert not str(path).startswith(str(ROOT.resolve()))
            continue
        source = ROOT / path
        assert source.is_file()
        destination = relocated / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        assert destination.stat().st_size == row["bytes"]
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == row["sha256"]


def test_authoritative_1500_step_lifecycle_and_mutations() -> None:
    termination = {"active_terms": ["success", "time_out"], "term_cfg": "unchanged"}
    activation = {
        "status": "registered_construction_timeout_extended_before_first_reset_or_step",
        "only_mutated_field": "env.cfg.episode_length_s",
        "step_dt_s": 1.0 / 15.0,
        "original_episode_length_s": 30.0,
        "registered_episode_length_s": 100.0,
        "original_max_episode_length_steps": 450,
        "registered_max_episode_length_steps": 1500,
        "common_step_counter_before_and_after": 0,
        "episode_length_buf_before_and_after": [0],
        "termination_contract_before": termination,
        "termination_contract_after": deepcopy(termination),
        "termination_config_byte_equal": True,
        "registered_worst_case_steps": 1365,
        "registered_margin_steps": 135,
        "behavioral_horizon_mutated": False,
    }
    lifecycle = {"construction_horizon_activation": activation}
    validate_construction_lifecycle(lifecycle, "fixture")
    for key, replacement in (
        ("registered_max_episode_length_steps", 900),
        ("registered_worst_case_steps", 885),
        ("registered_margin_steps", 15),
        ("behavioral_horizon_mutated", True),
    ):
        bad = deepcopy(lifecycle)
        bad["construction_horizon_activation"][key] = replacement
        with pytest.raises(ValidationError):
            validate_construction_lifecycle(bad, "fixture")


def test_runtime_topology_and_scientific_helpers() -> None:
    old = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r007/state_repair_gate.py"
    new = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_reference_bounds", "_save_camera_evidence",
        "_companion_gate", "_fresh_reset_and_gate", "_finalize_unchanged_gates",
    ):
        assert function_ast(old, name) == function_ast(new, name)
    body = function_ast(new, "_open_contact_materialize_and_gate")
    assert "object_space_servo_command" in body and "_normal_joint_equilibrium_step" in body
    for prohibited in ("write_joint_state_to_sim", "write_root_pose_to_sim", "write_root_velocity_to_sim"):
        assert prohibited not in body
    assert 'REMOTE_BRANCH = "experiment/v3e006-r008-object-servo-repair"' in (ROOT / "tools/run_v3e006_r008_state_repair.py").read_text()


def test_pre_source_package_validator() -> None:
    value = validate_static(ROOT, source_gate_required=False)
    assert value["passed"] is True
    assert value["candidate_pair_count"] == value["diagnostic_count"] == 4
