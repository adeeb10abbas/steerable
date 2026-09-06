from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.run_v4_second_stack_g3_scripted import (
    _position_at_fraction,
    select_extreme_seeds,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT
    / "artifacts/online_correction_v4/setup"
    / "second_stack_reset_registry.candidate.json"
)


class SecondStackG3ScriptedTests(unittest.TestCase):
    def test_extreme_selection_is_unique_and_starts_with_canonical_seed(self) -> None:
        resets = json.loads(REGISTRY.read_text(encoding="utf-8"))[
            "resets_by_env_seed"
        ]
        selected = select_extreme_seeds(resets)
        self.assertEqual(len(selected), 9)
        self.assertEqual(len({row["environment_seed"] for row in selected}), 9)
        self.assertEqual(
            selected[0]["environment_seed"],
            min(map(int, resets)),
        )

    def test_reference_position_interpolation_has_exact_endpoints(self) -> None:
        row = {
            "initial_reference_scene_xy_m": [-0.1, 0.2],
            "endpoint_reference_scene_xy_m": [0.3, -0.2],
        }
        self.assertEqual(_position_at_fraction(row, 0.0), [-0.1, 0.2])
        for actual, expected in zip(
            _position_at_fraction(row, 0.5),
            [0.1, 0.0],
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            _position_at_fraction(row, 1.0),
            [0.3, -0.2],
        ):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
