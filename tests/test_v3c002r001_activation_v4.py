"""Focused static checks for the A004 adapter/continuation repair."""

from __future__ import annotations

import unittest

from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_retry_runner import RETRY


class ActivationV4Test(unittest.TestCase):
    def test_retry_scope_remains_exact(self) -> None:
        self.assertEqual(RETRY, {
            "repair-lane-00": 12060, "repair-lane-01": 12101,
            "repair-lane-02": 12128, "repair-lane-03": 12177,
            "repair-lane-04": 12156, "repair-lane-05": 12107,
            "repair-lane-06": 12176, "repair-lane-07": 12112,
        })

    def test_a004_sources_import(self) -> None:
        from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import activation_v4_continuation_runner
        self.assertTrue(callable(activation_v4_continuation_runner._remaining))

    def test_exact_adapter_binding(self) -> None:
        from pathlib import Path
        from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import sha256_file
        from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_retry_runner import ADAPTER_SHA
        path = Path(__file__).parents[1] / "experiments/v3/phase_c_semantic_equivalence_v3c002r001/activation_v4_replacement_adapter.py"
        self.assertEqual(sha256_file(path), ADAPTER_SHA)

    def test_original_and_a003_release_are_exact(self) -> None:
        from pathlib import Path
        from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_contract import admitted_slots
        root = Path(__file__).parents[1] / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v3"
        self.assertEqual(admitted_slots(root / "release_gate.released.json"), frozenset(f"repair-lane-{i:02d}" for i in range(2, 8)))
        self.assertEqual(admitted_slots(root / "lane_replacement_a003/release_gate.released.json"), frozenset(("repair-lane-00", "repair-lane-01")))


if __name__ == "__main__":
    unittest.main()
