from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r007.predecessor_contract import (
    validate_r006_exhaustion_closure,
)
from tools.validate_v3e006_r007 import (
    ValidationError,
    expected_open_contact_commands,
    validate_construction_horizon_activation,
    validate_open_contact_state,
)


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007"
R006 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: object) -> str:
    raw = (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.dump(node, include_attributes=False)


def test_real_r006_closure_and_mutations() -> None:
    payload = load(R006 / "results/results.json")
    validate_r006_exhaustion_closure(payload)
    for key, value in (("candidate_pair_evaluation_count", 3), ("diagnostics_all_passed", False)):
        bad = deepcopy(payload)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_r006_exhaustion_closure(bad)
    bad = deepcopy(payload)
    bad["raw_result"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_r006_exhaustion_closure(bad)


def test_registration_is_prospective_and_binds_r006() -> None:
    registration = load(ART / "repair_registration.json")
    assert registration["repair_amendment_id"] == "V3-E006-R007"
    assert registration["predecessor_repair_amendment_id"] == "V3-E006-R006"
    assert registration["counts_at_registration"] == {
        "r007_live_diagnostics": 0, "r007_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }
    assert registration["r006_predecessor"]["closure_commit"] == "125e8f0d231ebd2e3c7d0d9b54dce83e1080cea1"
    assert registration["r006_predecessor"]["raw_result"]["sha256"] == "7eae75c38a7b65ba4b8fbc44f3ca4c565c3af5675134c93570b1dc0e85176011"


def test_finite_direction_neutral_open_contact_schedule() -> None:
    schedule = load(ART / "gates/candidate_schedule.json")
    contract = schedule["open_contact_construction_contract"]
    assert canonical_sha(contract) == schedule["open_contact_construction_contract_sha256"]
    assert contract["phase_steps"] == {
        "open_approach": 120, "open_descent": 120, "normal_close": 90,
        "closed_lift_to_registered_stage_target": 180,
        "closed_settle_at_registered_stage_target": 300,
    }
    assert sum(contract["phase_steps"].values()) == 810
    assert contract["worst_case_materialization_steps"] == 885
    assert [row["candidate_rank"] for row in schedule["candidate_pairs"]] == [1, 2, 3, 4]
    assert {row["construction_method"] for row in schedule["candidate_pairs"]} == {
        "exact_reset_open_approach_normal_close_lift"
    }
    selectors = [
        (row["canonical_grasp"]["contact_transform_selector"], row["canonical_carry"]["contact_transform_selector"])
        for row in schedule["candidate_pairs"]
    ]
    assert selectors == [
        ("left_observed", "reflected_right_observed"),
        ("reflected_right_observed", "left_observed"),
        ("left_observed", "left_observed"),
        ("reflected_right_observed", "reflected_right_observed"),
    ]
    for pair in schedule["candidate_pairs"]:
        for stage_name in ("canonical_grasp", "canonical_carry"):
            stage = pair[stage_name]
            assert stage["target_cube_pose"]["position_world_m"][1] == 0.0
            assert stage["r007_open_contact_targets"]["world_vertical_clearance_m"] == 0.060


def test_contact_pose_reconstructs_exact_reset_cube() -> None:
    schedule = load(ART / "gates/candidate_schedule.json")
    for pair in schedule["candidate_pairs"]:
        for name in ("canonical_grasp", "canonical_carry"):
            stage = pair[name]
            target = stage["r007_open_contact_targets"]
            # Translation reconstruction is tested independently of quaternion sign.
            base_p = np.asarray(target["contact_base_link_pose_at_exact_reset_cube"]["position_world_m"])
            base_q = np.asarray(target["contact_base_link_pose_at_exact_reset_cube"]["quaternion_world_wxyz"])
            w, x, y, z = base_q / np.linalg.norm(base_q)
            rotation = np.asarray([
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
            ])
            relative = stage["selected_observed_cube_in_base_link_transform"]
            reconstructed = base_p + rotation @ np.asarray(relative["translation_m"])
            assert np.allclose(reconstructed, target["reset_cube_pose"]["position_world_m"], atol=1e-12)


def test_runtime_dispatches_only_open_contact_candidate_path() -> None:
    path = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r007/state_repair_gate.py"
    source = path.read_text(encoding="utf-8")
    body = function_ast(path, "_open_contact_materialize_and_gate")
    assert "write_joint_state_to_sim" not in body
    assert "write_root_pose_to_sim" not in body
    assert "write_root_velocity_to_sim" not in body
    assert "810" in body and "885" in body
    assert source.count("state = _open_contact_materialize_and_gate(") == 1
    assert "state = _direct_materialize_and_gate(" not in source


def test_outer_launcher_uses_the_registered_remote_branch() -> None:
    path = ROOT / "tools/run_v3e006_r007_state_repair.py"
    source = path.read_text(encoding="utf-8")
    assert 'REMOTE_BRANCH = "experiment/v3e006-r007-open-contact-repair"' in source
    assert "experiment/v3e006-r007-state-repair" not in source


def _synthetic_valid_open_contact_state() -> tuple[dict, dict]:
    stage = load(ART / "gates/candidate_schedule.json")["candidate_pairs"][0]["canonical_grasp"]
    retained_stage = deepcopy(stage)
    retained_stage["candidate_rank"] = 1
    construction = {
        "method": "exact_reset_open_approach_normal_close_lift",
        "stage": "canonical_grasp",
        "candidate_rank": 1,
        "registered_stage_schedule": retained_stage,
        "registered_targets": stage["r007_open_contact_targets"],
        "phase_steps": {
            "open_approach": 120, "open_descent": 120, "normal_close": 90,
            "closed_lift_to_registered_stage_target": 180,
            "closed_settle_at_registered_stage_target": 300,
        },
        "start_base_link_pose_world_wxyz": [0.30, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0],
        "episode_length_buf_before_candidate_actions": [75],
        "episode_length_buf_after_candidate_actions": [885],
        "post_reset_joint_state_write_count": 0,
        "post_reset_object_state_write_count": 0,
        "gate_window_final_steps": 10,
    }
    frame = {
        "passed": True,
        "position_composition_residual_m": 0.0,
        "orientation_composition_residual_deg": 0.0,
    }
    trace = []
    for phase, step, _, action in expected_open_contact_commands(construction, stage):
        trace.append({
            "phase": phase,
            "phase_step_one_based": step,
            "command_action_8d": action,
            "eef_position_world_m": [0.31, 0.0, 0.20],
            "eef_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
            "base_link_position_world_m": [0.31, 0.0, 0.20],
            "base_link_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
            "base_link_to_eef_frame_identity": frame,
            "joint_position_rad": [0.0] * 13,
            "joint_velocity_rad_s": [0.0] * 13,
            "cube_position_world_m": [0.31, 0.0, 0.10],
            "cube_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
            "cube_linear_velocity_m_s": [0.0] * 3,
            "cube_angular_velocity_rad_s": [0.0] * 3,
        })
    samples = [{
        "cube_position_world_m": row["cube_position_world_m"],
        "cube_linear_velocity_m_s": row["cube_linear_velocity_m_s"],
        "cube_angular_velocity_rad_s": row["cube_angular_velocity_rad_s"],
        "eef_position_world_m": row["base_link_position_world_m"],
        "base_link_quaternion_world_wxyz": row["base_link_quaternion_world_wxyz"],
        "live_eef_frame_position_world_m": row["eef_position_world_m"],
        "live_eef_frame_quaternion_world_wxyz": row["eef_quaternion_world_wxyz"],
        "base_link_to_eef_frame_identity": row["base_link_to_eef_frame_identity"],
        "arm_joint_velocity_rad_s": row["joint_velocity_rad_s"][:7],
        "object_grabbed": True,
        "contact_force_n": {"gripper__rubiks_cube": 2.0},
    } for row in trace[-10:]]
    construction["construction_action_trace"] = trace
    construction["settled_gate_samples"] = samples
    last = trace[-1]
    state = {
        "construction": construction,
        "robot": {
            "joint_position_rad": last["joint_position_rad"],
            "joint_velocity_rad_s": last["joint_velocity_rad_s"],
        },
        "objects": {"rubiks_cube": {
            "position_world_m": last["cube_position_world_m"],
            "quaternion_world_wxyz": last["cube_quaternion_world_wxyz"],
            "linear_velocity_m_s": last["cube_linear_velocity_m_s"],
            "angular_velocity_rad_s": last["cube_angular_velocity_rad_s"],
        }},
        "eef": {
            "position_world_m": last["base_link_position_world_m"],
            "quaternion_world_wxyz": last["base_link_quaternion_world_wxyz"],
        },
        "contact_evidence": {
            "settled_force_snapshots_n": [row["contact_force_n"] for row in samples],
            "object_grabbed_by_step": [row["object_grabbed"] for row in samples],
        },
    }
    return state, stage


def test_raw_open_contact_trace_recomputed_and_mutations_rejected() -> None:
    state, stage = _synthetic_valid_open_contact_state()
    validate_open_contact_state(state, "canonical_grasp", stage, candidate_rank=1)
    mutations = []
    bad_action = deepcopy(state); bad_action["construction"]["construction_action_trace"][17]["command_action_8d"][2] += 0.01
    mutations.append(bad_action)
    bad_schedule = deepcopy(state); bad_schedule["construction"]["registered_stage_schedule"]["candidate_rank"] = 4
    mutations.append(bad_schedule)
    missing_retained_rank = deepcopy(state); missing_retained_rank["construction"]["registered_stage_schedule"].pop("candidate_rank")
    mutations.append(missing_retained_rank)
    bad_rank = deepcopy(state); bad_rank["construction"]["candidate_rank"] = 2
    mutations.append(bad_rank)
    bad_trace_state = deepcopy(state); bad_trace_state["construction"]["construction_action_trace"][42]["joint_velocity_rad_s"][0] = float("nan")
    mutations.append(bad_trace_state)
    bad_final = deepcopy(state)
    changed_final = list(bad_final["construction"]["settled_gate_samples"][-1]["cube_position_world_m"])
    changed_final[0] += 0.01
    bad_final["construction"]["settled_gate_samples"][-1]["cube_position_world_m"] = changed_final
    mutations.append(bad_final)
    for bad in mutations:
        with pytest.raises(ValidationError):
            validate_open_contact_state(bad, "canonical_grasp", stage, candidate_rank=1)


def test_r007_horizon_evidence_is_885_steps_with_15_step_margin() -> None:
    termination = {"active_terms": ["success", "time_out"]}
    lifecycle = {"construction_horizon_activation": {
        "status": "registered_construction_timeout_extended_before_first_reset_or_step",
        "only_mutated_field": "env.cfg.episode_length_s",
        "original_max_episode_length_steps": 450,
        "registered_max_episode_length_steps": 900,
        "original_episode_length_s": 30.0,
        "registered_episode_length_s": 60.0,
        "step_dt_s": 1.0 / 15.0,
        "common_step_counter_before_and_after": 0,
        "episode_length_buf_before_and_after": [0.0],
        "termination_config_byte_equal": True,
        "termination_contract_before": termination,
        "termination_contract_after": termination,
        "registered_worst_case_steps": 885,
        "registered_margin_steps": 15,
        "behavioral_horizon_mutated": False,
    }}
    validate_construction_horizon_activation(lifecycle, "synthetic")
    bad = deepcopy(lifecycle)
    bad["construction_horizon_activation"]["registered_worst_case_steps"] = 795
    with pytest.raises(ValidationError):
        validate_construction_horizon_activation(bad, "synthetic")


def test_scientific_gate_functions_are_ast_identical_to_r006() -> None:
    old = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/state_repair_gate.py"
    new = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r007/state_repair_gate.py"
    for name in (
        "_sample", "_capture_state", "_save_camera_evidence", "_companion_gate",
        "_fresh_reset_and_gate", "_finalize_unchanged_gates", "_contact_coverage",
    ):
        assert function_ast(old, name) == function_ast(new, name)


def test_package_validator_pre_source_gate() -> None:
    from tools.validate_v3e006_r007 import validate_static

    result = validate_static(ROOT, require_source_gate=False)
    assert result["passed"] is True
    assert result["candidate_pair_count"] == 4
    assert result["candidate_action_steps"] == 810
