from __future__ import annotations

import unittest

import numpy as np

from experiments.v3.phase_e.canonical_stage_localization_v3e006.ood_reference import (
    FEATURE_NAMES,
    normalized_distance,
    state_feature,
)


class OODReferenceTests(unittest.TestCase):
    def test_identity_relative_pose_and_zero_distance(self) -> None:
        feature = state_feature(
            arm_joint_positions_rad=[0.0] * 7,
            eef_position_world_m=[1.0, 2.0, 3.0],
            eef_quaternion_world_wxyz=[1.0, 0.0, 0.0, 0.0],
            cube_position_world_m=[1.1, 1.8, 3.3],
            cube_quaternion_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(feature.shape, (len(FEATURE_NAMES),))
        np.testing.assert_allclose(feature[7:10], [0.1, -0.2, 0.3], atol=1e-12)
        np.testing.assert_allclose(feature[10:], [0.0, 0.0, 0.0], atol=1e-12)
        self.assertEqual(normalized_distance(feature, center=feature, scale=[1.0] * len(feature)), 0.0)

    def test_quaternion_sign_is_canonical(self) -> None:
        positive = state_feature(
            arm_joint_positions_rad=[0.0] * 7,
            eef_position_world_m=[0.0, 0.0, 0.0],
            eef_quaternion_world_wxyz=[1.0, 0.0, 0.0, 0.0],
            cube_position_world_m=[0.0, 0.0, 0.0],
            cube_quaternion_world_wxyz=[0.0, 1.0, 0.0, 0.0],
        )
        negative = state_feature(
            arm_joint_positions_rad=[0.0] * 7,
            eef_position_world_m=[0.0, 0.0, 0.0],
            eef_quaternion_world_wxyz=[-1.0, 0.0, 0.0, 0.0],
            cube_position_world_m=[0.0, 0.0, 0.0],
            cube_quaternion_world_wxyz=[0.0, -1.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(positive, negative, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
