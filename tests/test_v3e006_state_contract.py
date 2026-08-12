from __future__ import annotations

import unittest
import json
import copy
from pathlib import Path

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
