from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.v3.phase_e.canonical_stage_localization_v3e006_r002 import candidate_schedule as schedule_code


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r002"
REGISTRATION = ARTIFACT / "repair_registration.json"
SCHEDULE = ARTIFACT / "gates/candidate_schedule.json"
PREDECESSOR = ARTIFACT / "gates/predecessor_closure_binding.json"
R001_COMMIT = "bbabac55dfd54f7a0b7d8a2693673a4b06409f21"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(row for row in tree.body if isinstance(row, (ast.FunctionDef, ast.AsyncFunctionDef)) and row.name == name)
    return ast.dump(node, include_attributes=False)


def test_predecessor_tree_is_byte_identical_and_zero_count() -> None:
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    assert predecessor["r001_exhaustion_closure_commit"] == R001_COMMIT
    assert predecessor["r001_tree_file_count"] == 20
    assert predecessor["model_request_count"] == 0
    assert predecessor["behavioral_episode_count"] == 0
    assert predecessor["r002_live_candidate_evaluation_count"] == 0
    for row in predecessor["r001_tree_files"]:
        current = ROOT / row["path"]
        assert current.stat().st_size == row["bytes"]
        assert sha256(current) == row["sha256"]
        committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{R001_COMMIT}:{row['path']}"])
        assert hashlib.sha256(committed).hexdigest() == row["sha256"]
        assert len(committed) == row["bytes"]


def test_registration_and_schedule_are_finite_prospective_and_exact() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    assert registration["status"] == schedule_code.REGISTRATION_STATUS
    assert registration["counts_at_registration"] == {
        "r002_live_candidate_evaluations": 0, "model_requests": 0, "behavioral_episodes": 0
    }
    assert registration["candidate_search"]["maximum_candidate_pairs"] == 8
    assert schedule["candidate_budget"] == 8
    assert [row["candidate_rank"] for row in schedule["candidate_pairs"]] == list(range(1, 9))
    observed = [
        (row["candidate_rank"], row["construction_method"],
         row["canonical_grasp"]["contact_transform_selector"],
         row["canonical_carry"]["contact_transform_selector"])
        for row in schedule["candidate_pairs"]
    ]
    assert observed == list(schedule_code.VARIANTS)
    assert schedule["model_request_count"] == schedule["behavioral_episode_count"] == 0
    assert schedule["r002_live_candidate_evaluation_count"] == 0
    assert schedule["repair_registration"]["bytes"] == REGISTRATION.stat().st_size
    assert schedule["repair_registration"]["sha256"] == sha256(REGISTRATION)
    assert schedule["r001_predecessor"]["closure_binding"]["sha256"] == sha256(PREDECESSOR)


def test_historical_anchors_and_contact_consistent_se3_are_exact() -> None:
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    rank_one = schedule["candidate_pairs"][0]
    expected = {
        "canonical_grasp": (9521, 30, 104, 31, 105),
        "canonical_carry": (9442, 39, 113, 38, 112),
    }
    for stage, values in expected.items():
        row = rank_one[stage]
        left = row["both_direction_sources"]["left"]
        right = row["both_direction_sources"]["right"]
        assert (row["source_environment_seed"], left["state_capture_index"], left["hdf5_index"],
                right["state_capture_index"], right["hdf5_index"]) == values
    for pair in schedule["candidate_pairs"]:
        for stage in ("canonical_grasp", "canonical_carry"):
            row = pair[stage]
            assert row["target_cube_pose"]["position_world_m"][1] == 0.0
            assert row["se3_reconstruction"]["cube_midline_residual_m"] <= 1e-12
            assert row["se3_reconstruction"]["position_residual_m"] <= 1e-12
            assert row["se3_reconstruction"]["rotation_matrix_frobenius_residual"] <= 1e-12
            assert row["contact_transform_selector"] in {"left_observed", "reflected_right_observed"}
            assert row["historical_source_runtime_assertions"]["hdf5_action_last_component_exact"] == 1.0


def test_reflection_is_full_world_se3_and_proper() -> None:
    position = np.asarray([0.3, -0.04, 0.2])
    quaternion = schedule_code.canonical_quaternion_wxyz([0.7, 0.2, -0.3, 0.6])
    original = schedule_code.pose_matrix(position, quaternion)
    reflected = schedule_code.reflect_world_pose(original)
    assert np.allclose(reflected[:3, 3], schedule_code.MIRROR @ position, atol=0, rtol=0)
    assert np.allclose(
        reflected[:3, :3], schedule_code.MIRROR @ original[:3, :3] @ schedule_code.MIRROR,
        atol=1e-15, rtol=0,
    )
    assert abs(np.linalg.det(reflected[:3, :3]) - 1.0) < 1e-12
    assert np.allclose(schedule_code.reflect_world_pose(reflected), original, atol=1e-15, rtol=0)


def test_unchanged_gate_helpers_and_threshold_sources_are_preserved() -> None:
    r001 = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r001/state_repair_gate.py"
    r002 = ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r002/state_repair_gate.py"
    for name in (
        "_contact_forces", "_contact_coverage", "_sample", "_capture_state", "_reference_bounds",
        "_save_camera_evidence", "_companion_gate", "_fresh_reset_and_gate",
    ):
        assert function_ast(r002, name) == function_ast(r001, name)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name in ("state_contract", "ood_reference", "ood_freeze"):
        row = registration["frozen_inputs"][name]
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]


def test_runtime_implements_both_registered_methods_and_is_zero_model() -> None:
    source = (ROOT / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r002/state_repair_gate.py").read_text(encoding="utf-8")
    assert "_solve_registered_ik(" in source
    assert "direct_contact_initialization" in source
    assert "open_approach_close_lift" in source
    assert "env.sim.forward()" in source
    assert "normal_binary_close_command\": 1.0" in source
    assert "_direct_materialize_and_gate(" in source
    assert "_open_approach_and_gate(" in source
    assert 'role="ik_solve"' in source
    assert 'role="materialization"' in source
    assert "requests.post" not in source
    assert "httpx" not in source
    assert "policy_server" not in source
    assert "model_request_count\": 0" in source


def test_outer_launcher_keeps_exact_successful_r001_e004_launcher_contract() -> None:
    def tuple_value(path: Path, name: str) -> tuple[str, ...]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignment = next(
            row for row in tree.body
            if isinstance(row, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in row.targets)
        )
        return tuple(ast.literal_eval(assignment.value))

    r001 = tuple_value(ROOT / "tools/run_v3e006_r001_state_repair.py", "E004_APP_LAUNCHER_ARGV")
    r002 = tuple_value(ROOT / "tools/run_v3e006_r002_state_repair.py", "E004_APP_LAUNCHER_ARGV")
    assert r002 == r001
    assert r002 == (
        "--headless", "--device", "cuda:0", "--num-envs", "1", "--num-runs", "1",
        "--renderer", "realtime", "--rendering-type", "balanced", "--video-mode", "viewport",
        "--instruction-type", "default", "--disable-subtask",
        "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
    )
