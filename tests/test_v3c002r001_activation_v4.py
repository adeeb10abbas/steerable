"""Focused static checks for the A004 adapter/continuation repair."""

from __future__ import annotations

import unittest

from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v3_lane_replacement_runner import RETRY


class ActivationV4Test(unittest.TestCase):
    def test_retry_scope_remains_exact(self) -> None:
        self.assertEqual(RETRY, {"repair-lane-00": 12060, "repair-lane-01": 12101})

    def test_a004_sources_import(self) -> None:
        from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import activation_v4_continuation_runner
        self.assertTrue(callable(activation_v4_continuation_runner._remaining))


if __name__ == "__main__":
    unittest.main()
