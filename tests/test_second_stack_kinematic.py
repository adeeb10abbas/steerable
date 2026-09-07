from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.online_correction_v4.second_stack_kinematic import SecondStackKinematicAdapter


class _Pose:
    def __init__(self, p):
        self.p = list(p)


class _Link:
    def __init__(self, name, p):
        self.name = name
        self.pose = _Pose(p)


class SecondStackKinematicAdapterTests(unittest.TestCase):
    def test_grabbed_coupling_zeros_relative_drift(self) -> None:
        raw = SimpleNamespace(
            episode_source_obj=SimpleNamespace(pose=_Pose([-0.2, 0.0, 0.885])),
            agent=SimpleNamespace(
                robot=SimpleNamespace(pose=_Pose([0.15, 0.03, 0.87])),
                ee_pose=_Pose([-0.21, 0.04, 1.0]),
            ),
        )
        env = SimpleNamespace()

        def fake_unwrap(_env):
            return raw

        def fake_contact(_env):
            return True

        import experiments.online_correction_v4.second_stack_kinematic as mod

        original_unwrap = mod.unwrap_simpler_env
        original_contact = mod._finger_target_contact
        mod.unwrap_simpler_env = fake_unwrap
        mod._finger_target_contact = fake_contact
        try:
            adapter = SecondStackKinematicAdapter(env, control_dt_s=0.2)
            adapter.bind_sim_dt(0.002)
            adapter.on_physics_step()
            adapter.on_control_boundary()
            state = adapter.object_kinematic_state()
        finally:
            mod.unwrap_simpler_env = original_unwrap
            mod._finger_target_contact = original_contact

        self.assertTrue(state.contact)
        self.assertAlmostEqual(state.gripper_x, state.object_x)
        self.assertAlmostEqual(state.gripper_y, state.object_y)
        self.assertAlmostEqual(state.gripper_z, state.object_z_pos)


if __name__ == "__main__":
    unittest.main()
