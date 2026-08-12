from __future__ import annotations

import unittest
import json
import copy
from pathlib import Path
import ast

import numpy as np

from experiments.v3.phase_e.canonical_stage_localization_v3e006.state_contract import (
    StateContractError,
    normalized_state_sha256,
    compare_full_reset_to_e004,
    settled_gate,
)


def test_state_gate_retains_failure_before_closing_simulator() -> None:
    source_path = Path(__file__).parents[1] / "experiments/v3/phase_e/canonical_stage_localization_v3e006/state_construction_gate.py"
    source = source_path.read_text(encoding="utf-8")
    retained = source.index("failure_path = _write_failure(exc)")
    close = source.index("simulation_app.close()")
    reraise = source.index("raise construction_failure.with_traceback")
    assert retained < close < reraise
    for field in (
        '"model_request_count": 0',
        '"behavioral_episode_count": 0',
        '"candidate_gate_passed": candidate_gate_passed',
        '"passed_health_preflight"',
        '"traceback": traceback.format_exc()',
    ):
        assert field in source


def test_state_gate_retains_each_partial_gate_before_failure() -> None:
    source_path = Path(__file__).parents[1] / "experiments/v3/phase_e/canonical_stage_localization_v3e006/state_construction_gate.py"
    source = source_path.read_text(encoding="utf-8")
    assert '"partial_stage_evidence": LAST_PARTIAL_STAGES' in source
    for assignment in (
        'state["physics_gate"] =',
        'state["ood_gate"] =',
        'state["camera_evidence"] =',
        'state["companion_pose_gate"] =',
    ):
        assigned_at = source.index(assignment)
        retained_at = source.index("LAST_PARTIAL_STAGES[stage] = state", assigned_at)
        next_gate = source.find('state["', assigned_at + len(assignment))
        assert next_gate == -1 or retained_at < next_gate


def test_health_semantics_are_checked_before_app_launcher() -> None:
    source_path = Path(__file__).parents[1] / "experiments/v3/phase_e/canonical_stage_localization_v3e006/state_construction_gate.py"
    source = source_path.read_text(encoding="utf-8")
    verified = source.index("_verify_passed_health_preflight(health_files)")
    launcher = source.index("simulation_app = AppLauncher(args).app")
    assert verified < launcher
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_verify_passed_health_preflight")
    fn_source = ast.get_source_segment(source, fn) or ""
    for required in (
        "passed_generic_zero_model_health_preflight",
        "passed_generic_zero_model_cuda_vulkan_isaac_physics_render_health_preflight",
        '"model_request_count"',
        '"behavioral_episode_count"',
        '"state_candidate_count"',
        '"child_report"',
    ):
        assert required in fn_source


def test_single_env_prim_regex_materialization_is_narrow_and_fail_closed() -> None:
    source_path = Path(__file__).parents[1] / "experiments/v3/phase_e/canonical_stage_localization_v3e006/state_construction_gate.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_materialize_single_env_prim_path")
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source_path), "exec"), namespace)
    materialize = namespace["_materialize_single_env_prim_path"]
    assert callable(materialize)
    assert materialize("/World/envs/env_.*/scene/bowl", num_envs=1) == "/World/envs/env_0/scene/bowl"
    assert materialize("/World/envs/env_0/scene/bowl", num_envs=1) == "/World/envs/env_0/scene/bowl"
    with unittest.TestCase().assertRaises(RuntimeError):
        materialize("/World/envs/env_.*/scene/bowl", num_envs=2)
    with unittest.TestCase().assertRaises(RuntimeError):
        materialize("/World/.*/bowl", num_envs=1)


def test_bounds_math_tracks_frozen_e004_and_retains_raw_values() -> None:
    repo = Path(__file__).parents[1]
    state_source = (repo / "experiments/v3/phase_e/canonical_stage_localization_v3e006/state_construction_gate.py").read_text(encoding="utf-8")
    frozen_source = (repo / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/model_blind_droid_gate.py").read_text(encoding="utf-8")
    for operation in (
        "ComputeLocalBound(prim)",
        "ComputeAlignedRange()",
        "(minimum[index] + maximum[index]) * 0.5",
        "(maximum[index] - minimum[index]) * 0.5",
    ):
        assert operation in frozen_source
    for evidence_field in (
        '"raw_prim_path"',
        '"resolved_prim_path"',
        '"local_minimum_m"',
        '"local_maximum_m"',
        '"local_center_m"',
        '"half_extents_m"',
    ):
        assert evidence_field in state_source


def test_bounds_boundary_normalizes_only_scalar_types_without_changing_values() -> None:
    source_path = Path(__file__).parents[1] / "experiments/v3/phase_e/canonical_stage_localization_v3e006/state_construction_gate.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_reference_bounds")
    box_return = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "YawOrientedBox"
    )
    center = np.asarray([0.8850742737042376, 0.12666125471719836, 0.1549786306324369], dtype=np.float64)
    half = tuple(np.float64(value) for value in (0.0807566887, 0.0805766839, 0.0275467021))
    yaw = np.float64(4.745085247498086e-07)
    namespace = {"center": center, "half": half, "yaw": yaw}
    normalized = tuple(
        eval(compile(ast.Expression(argument), str(source_path), "eval"), namespace)
        for argument in box_return.value.args
    )
    assert normalized == (tuple(center), half, yaw)
    assert all(type(value) is float for value in (*normalized[0], *normalized[1], normalized[2]))


class StateContractTests(unittest.TestCase):
    def test_hash_ignores_evidence_but_binds_physics(self) -> None:
        state = {
            "robot": {"joint_position_rad": [0.0] * 13},
            "objects": {"rubiks_cube": {"position_world_m": [0.3, 0.0, 0.1]}},
            "eef": {"position_world_m": [0.3, 0.0, 0.2]},
            "camera_evidence": "a",
        }
        first = normalized_state_sha256(state)
        state["camera_evidence"] = "b"
        self.assertEqual(first, normalized_state_sha256(state))
        state["objects"]["rubiks_cube"]["position_world_m"][0] = 0.4
        self.assertNotEqual(first, normalized_state_sha256(state))

    def test_exact_ten_step_window_and_strict_physics(self) -> None:
        sample = {
            "cube_position_world_m": [0.3, 0.0, 0.13],
            "eef_position_world_m": [0.3, 0.0, 0.20],
            "arm_joint_velocity_rad_s": [0.0] * 7,
            "cube_linear_velocity_m_s": [0.0] * 3,
            "cube_angular_velocity_rad_s": [0.0] * 3,
            "contact_force_n": {"gripper__bowl": 0.0, "gripper__rubiks_cube": 2.0},
            "object_grabbed": True,
        }
        self.assertTrue(settled_gate([sample] * 10, unintended_contact_pairs=("gripper__bowl",))["passed"])
        with self.assertRaises(StateContractError):
            settled_gate([sample] * 9, unintended_contact_pairs=("gripper__bowl",))

    def test_retained_e004_reset_reference_round_trip(self) -> None:
        path = Path(
            "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/gates/"
            "e004_full_reset_reference.json"
        )
        reference = json.loads(path.read_text(encoding="utf-8"))
        robot = reference["robot"]
        objects = reference["rigid_objects"]
        state = {
            "robot": {
                "joint_position_rad": copy.deepcopy(robot["joint_position"]["values"]),
                "joint_velocity_rad_s": robot["joint_velocity"]["values"],
                "root_position_world_m": robot["root_position"]["values"],
                "root_quaternion_world_wxyz": robot["root_quaternion_wxyz"]["values"],
                "root_linear_velocity_m_s": robot["root_linear_velocity"]["values"],
                "root_angular_velocity_rad_s": robot["root_angular_velocity"]["values"],
                "gripper": {"joint_position_rad": robot["joint_position"]["values"][7:]},
            },
            "objects": {
                name: {
                    "position_world_m": row["root_position"]["values"],
                    "quaternion_world_wxyz": row["root_quaternion_wxyz"]["values"],
                    "linear_velocity_m_s": copy.deepcopy(row["root_linear_velocity"]["values"]),
                    "angular_velocity_rad_s": copy.deepcopy(row["root_angular_velocity"]["values"]),
                }
                for name, row in objects.items()
            },
        }
        result = compare_full_reset_to_e004(state, reference=reference, reference_file_sha256="0" * 64)
        self.assertTrue(result["passed"])
        state["robot"]["joint_position_rad"][0] = 0.1
        self.assertFalse(compare_full_reset_to_e004(state, reference=reference, reference_file_sha256="0" * 64)["passed"])
        state["robot"]["joint_position_rad"][0] = robot["joint_position"]["values"][0]
        state["objects"]["rubiks_cube"]["linear_velocity_m_s"][0] += 0.1
        self.assertFalse(compare_full_reset_to_e004(state, reference=reference, reference_file_sha256="0" * 64)["passed"])


if __name__ == "__main__":
    unittest.main()
