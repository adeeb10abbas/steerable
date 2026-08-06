#!/usr/bin/env python3
"""Unit tests for the exact V3 failure-mode split analysis."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_v3_failure_mode_split.py")
SPEC = importlib.util.spec_from_file_location("failure_mode_split", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExactContingencyTest(unittest.TestCase):
    def test_pi05_failure_shape(self) -> None:
        result = MODULE.fisher_freeman_halton_two_sided(
            (6, 11, 5, 0), (0, 3, 0, 0)
        )
        self.assertEqual(result["p_value_exact"], "879/2300")
        self.assertAlmostEqual(result["p_value"], 0.38217391304347825)

    def test_dreamzero_failure_shape(self) -> None:
        result = MODULE.fisher_freeman_halton_two_sided(
            (10, 14, 0, 0), (10, 0, 0, 0)
        )
        self.assertEqual(result["p_value_exact"], "1579/916980")
        self.assertAlmostEqual(result["p_value"], 0.0017219568583829528)

    def test_cosmos_edge_failure_shape(self) -> None:
        result = MODULE.fisher_freeman_halton_two_sided(
            (1, 6, 2, 0), (1, 1, 0, 0)
        )
        self.assertEqual(result["p_value_exact"], "34/55")
        self.assertAlmostEqual(result["p_value"], 0.6181818181818182)


if __name__ == "__main__":
    unittest.main()
