#!/usr/bin/env python3
"""Unit tests for the V3 gap-versus-competence analysis."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_v3_gap_vs_competence.py")
SPEC = importlib.util.spec_from_file_location("gap_vs_competence", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GapVersusCompetenceTest(unittest.TestCase):
    def test_known_checkpoint_counts(self) -> None:
        expected = {
            "pi05_current_stack_droid": (5, 24),
            "groot_n17_droid_vla": (3, 0),
            "cosmos3_edge_policy_droid": (18, 25),
            "cosmos3_nano_policy_droid": (26, 25),
            "dreamzero_droid_action_cfg": (3, 17),
        }
        for source in MODULE.SOURCES:
            row = MODULE.compile_row(*source)
            self.assertEqual(
                (row["left_successes"], row["right_successes"]),
                expected[row["model_id"]],
            )

    def test_gap_envelope(self) -> None:
        for source in MODULE.SOURCES:
            row = MODULE.compile_row(*source)
            self.assertLessEqual(
                row["absolute_directional_gap"],
                row["maximum_observable_gap_magnitude_at_this_success_rate"] + 1e-12,
            )

    def test_rank_correlation_is_not_monotonic(self) -> None:
        rows = [MODULE.compile_row(*source) for source in MODULE.SOURCES]
        result = MODULE.exact_permutation_correlation(
            [row["overall_success_rate"] for row in rows],
            [row["directional_gap_right_minus_left"] for row in rows],
            rank=True,
        )
        self.assertAlmostEqual(result["coefficient"], 0.1)
        self.assertGreater(result["two_sided_exact_permutation_p"], 0.05)
        self.assertEqual(result["permutations_enumerated"], 120)


if __name__ == "__main__":
    unittest.main()
