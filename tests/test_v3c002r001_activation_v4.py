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

    def test_exact_adapter_binding(self) -> None:
        from pathlib import Path
        from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import sha256_file
        from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_retry_runner import ADAPTER_SHA
        path = Path(__file__).parents[1] / "experiments/v3/phase_c_semantic_equivalence_v3c002r001/activation_v4_replacement_adapter.py"
        self.assertEqual(sha256_file(path), ADAPTER_SHA)


if __name__ == "__main__":
    unittest.main()
