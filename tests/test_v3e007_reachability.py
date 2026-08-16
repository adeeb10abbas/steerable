from __future__ import annotations

import unittest

import numpy as np

from experiments.v3.phase_e.zero_model_reachability_v3e007.kinematics import (
    JointFrame,
    KinematicChain,
    deterministic_starts,
    forward,
)


IDENTITY = tuple(tuple(float(row == column) for column in range(4)) for row in range(4))


class ReachabilityKinematicsTest(unittest.TestCase):
    def test_forward_applies_revolute_joint(self) -> None:
        revolute = JointFrame(
            path="joint",
            axis_xyz=(0.0, 0.0, 1.0),
            lower_rad=-np.pi,
            upper_rad=np.pi,
            parent_to_joint=IDENTITY,
            child_to_joint=IDENTITY,
        )
        fixed_matrix = np.eye(4)
        fixed_matrix[0, 3] = 1.0
        fixed = JointFrame(
            path="fixed",
            axis_xyz=(1.0, 0.0, 0.0),
            lower_rad=0.0,
            upper_rad=0.0,
            parent_to_joint=tuple(tuple(float(value) for value in row) for row in fixed_matrix),
            child_to_joint=IDENTITY,
        )
        padding = tuple(
            JointFrame(
                path=f"pad{index}", axis_xyz=(1.0, 0.0, 0.0), lower_rad=-1.0, upper_rad=1.0,
                parent_to_joint=IDENTITY, child_to_joint=IDENTITY,
            )
            for index in range(6)
        )
        chain = KinematicChain((revolute, *padding), (fixed,))
        transform = forward(chain, [np.pi / 2, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(transform[:3, 3], [0.0, 1.0, 0.0], atol=1e-12)

    def test_halton_starts_are_deterministic_and_interior(self) -> None:
        joints = tuple(
            JointFrame(
                path=str(index), axis_xyz=(1.0, 0.0, 0.0), lower_rad=-2.0, upper_rad=2.0,
                parent_to_joint=IDENTITY, child_to_joint=IDENTITY,
            )
            for index in range(7)
        )
        chain = KinematicChain(joints, ())
        first = deterministic_starts(chain, np.zeros(7), 17)
        second = deterministic_starts(chain, np.zeros(7), 17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (17, 7))
        self.assertTrue(np.all(first[1:] > chain.lower))
        self.assertTrue(np.all(first[1:] < chain.upper))


if __name__ == "__main__":
    unittest.main()

