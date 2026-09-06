from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.build_v4_second_stack_g3_plan import (
    build_plan,
    canonical_json_bytes,
    goal_area_m2,
    minimum_axis_separation_on_segment,
    point_segment_distance,
)
from tools.run_v4_second_stack_g3_path import minimum_jerk_fraction


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT
    / "artifacts/online_correction_v4/setup"
    / "second_stack_reset_registry.candidate.json"
)
PLAN = (
    ROOT
    / "artifacts/online_correction_v4/setup"
    / "second_stack_g3_plan.candidate.json"
)


class SecondStackG3PlanTests(unittest.TestCase):
    def test_builder_reproduces_committed_plan(self) -> None:
        self.assertEqual(
            PLAN.read_bytes(),
            canonical_json_bytes(build_plan(registry_path=REGISTRY)),
        )

    def test_ladder_selects_largest_analytically_feasible_scale(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.assertEqual(plan["selected_analytical_scale"], 1.5)
        self.assertEqual(
            [row["scale"] for row in plan["scales"] if row["passed"]],
            [1.5, 1.0],
        )
        self.assertEqual(
            {row["check_count"] for row in plan["scales"]},
            {256},
        )
        selected = next(
            row
            for row in plan["scales"]
            if row["scale"] == plan["selected_analytical_scale"]
        )
        shrinking = [
            row for row in selected["checks"] if row["shrinking_direction_case"]
        ]
        self.assertEqual(len(shrinking), 128)
        self.assertTrue(
            all(row["goal_area_removed_fraction"] >= 0.20 for row in shrinking)
        )

    def test_goal_area_shrinks_when_reference_moves_into_relation(self) -> None:
        reference = (-0.11, 0.05)
        self.assertGreater(
            goal_area_m2(reference_xy=reference, relation="left"),
            goal_area_m2(reference_xy=(-0.03, -0.09), relation="left"),
        )

    def test_point_segment_distance_handles_interior_projection(self) -> None:
        self.assertAlmostEqual(
            point_segment_distance((0.5, 1.0), (0.0, 0.0), (1.0, 0.0)),
            1.0,
        )
        self.assertAlmostEqual(
            minimum_axis_separation_on_segment(
                (0.5, 1.0),
                (0.0, 0.0),
                (1.0, 0.0),
            ),
            1.0,
        )

    def test_minimum_jerk_profile_has_frozen_endpoints(self) -> None:
        self.assertEqual(minimum_jerk_fraction(0.0), 0.0)
        self.assertEqual(minimum_jerk_fraction(1.0), 1.0)
        values = [minimum_jerk_fraction(index / 20) for index in range(21)]
        self.assertTrue(
            all(first <= second for first, second in zip(values, values[1:]))
        )


if __name__ == "__main__":
    unittest.main()
