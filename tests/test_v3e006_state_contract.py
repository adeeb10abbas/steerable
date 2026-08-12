from __future__ import annotations

import unittest

from experiments.v3.phase_e.canonical_stage_localization_v3e006.state_contract import (
    StateContractError,
    normalized_state_sha256,
    settled_gate,
)


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
            "contact_force_n": {"gripper__bowl": 0.0},
            "object_grabbed": True,
        }
        self.assertTrue(settled_gate([sample] * 10, unintended_contact_pairs=("gripper__bowl",))["passed"])
        with self.assertRaises(StateContractError):
            settled_gate([sample] * 9, unintended_contact_pairs=("gripper__bowl",))


if __name__ == "__main__":
    unittest.main()
