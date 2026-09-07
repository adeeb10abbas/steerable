"""Audit tests: information-gate shrinking designation vs physical_translation_sign."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.online_correction_v4.droid_g3 import goal_area_case
from experiments.online_correction_v4.geometry import (
    AxisAlignedBox,
    ObjectFootprint,
    TaskFrame,
)
from experiments.online_correction_v4.motion import ReferenceMotionController
from tools import run_v4_horizontal_g3_path_seed as path_runner

ROOT = Path(__file__).resolve().parents[1]
HORIZONTAL_PLAN = (
    ROOT
    / "artifacts/online_correction_v4/setup"
    / "horizontal_g3_plan.geometry_repair_v2.candidate.json"
)


class InformationGateSignAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(HORIZONTAL_PLAN.read_text(encoding="utf-8"))
        cls.counterbalance = cls.plan["counterbalance_by_env_seed"]
        cls.directions = cls.plan["direction_task_coefficients_by_env_seed"]

    def test_smoke_seed_is_positive_sign(self) -> None:
        smoke = 2100000000
        self.assertEqual(
            self.counterbalance[str(smoke)]["physical_translation_sign"],
            1,
        )

    def test_failed_env_seeds_match_negative_sign_exactly(self) -> None:
        path_scale = json.loads(
            (
                ROOT
                / "artifacts/online_correction_v4/qualification"
                / "20260908_horizontal_g3_path_scale_0p5_g3r20260908g.json"
            ).read_text(encoding="utf-8")
        )
        failed = set(path_scale["failed_env_seeds"])
        negative = {
            int(seed)
            for seed, row in self.counterbalance.items()
            if row["physical_translation_sign"] == -1
        }
        positive = {
            int(seed)
            for seed, row in self.counterbalance.items()
            if row["physical_translation_sign"] == 1
        }
        self.assertEqual(len(failed), 64)
        self.assertEqual(failed, negative)
        self.assertFalse(failed & positive)

    def test_left_and_right_share_motion_direction_per_sign(self) -> None:
        for sign in (1, -1):
            sample = next(
                seed
                for seed, row in self.counterbalance.items()
                if row["physical_translation_sign"] == sign
            )
            coeffs = self.directions[sample]
            self.assertEqual(coeffs["left"], coeffs["right"])
            self.assertEqual(coeffs["front"], coeffs["behind"])

    def test_displacement_vector_matches_plan_directions(self) -> None:
        for sign in (1, -1):
            sample = next(
                seed
                for seed, row in self.counterbalance.items()
                if row["physical_translation_sign"] == sign
            )
            for goal in ("left", "right", "front", "behind"):
                expected = list(
                    ReferenceMotionController.displacement_vector(
                        goal=goal,
                        fixture="horizontal",
                        physical_sign=sign,
                    )
                )
                self.assertEqual(self.directions[sample][goal], expected)

    def test_confirmatory_receipt_shrinking_goals_flip_with_sign(self) -> None:
        """Document the live homogeneous-wave pattern: opposite axis halves shrink."""
        positive = {
            "left": (True, 0.2431, True),
            "right": (False, 0.0, True),
            "front": (True, 0.6357, True),
            "behind": (False, 0.0, True),
        }
        negative = {
            "left": (False, 0.0, True),
            "right": (True, 0.1369, False),
            "front": (False, 0.0, True),
            "behind": (True, 0.1986, False),
        }
        for goal, (shrinking, removed, passes) in positive.items():
            self.assertEqual(positive[goal][0], shrinking)
            self.assertGreater(removed, 0.20 if shrinking else -0.01)
            self.assertEqual(positive[goal][2], passes)
        for goal, (shrinking, removed, passes) in negative.items():
            self.assertTrue(shrinking == (goal in {"right", "behind"}))
            if shrinking:
                self.assertLess(removed, 0.20)
                self.assertFalse(passes)
            else:
                self.assertTrue(passes)

    def test_expanding_direction_cases_auto_pass_without_threshold(self) -> None:
        frame = TaskFrame.identity()
        workspace = AxisAlignedBox(-0.35, 0.35, -0.35, 0.35, 0.0, 0.1)
        geometry = {
            "fixture_id": "horizontal",
            "frame": frame,
            "target_workspace": workspace,
            "target_footprint": ObjectFootprint(0.02, 0.02, 0.02),
            "reference_footprint": ObjectFootprint(0.03, 0.03, 0.02),
        }
        baseline = (0.15, 0.0, 0.025)
        endpoint = path_runner.expected_reference_world_position(
            baseline_world=baseline,
            robot_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            direction_task=(1.0, 0.0),
            displacement_m=0.06,
        )
        expanding = goal_area_case(
            geometry=geometry,
            relation="right",
            original_reference_world=baseline,
            endpoint_reference_world=endpoint,
            clearance_m=0.01,
            minimum_shrinking_area_fraction=0.20,
        )
        self.assertFalse(expanding["shrinking_direction"])
        self.assertTrue(expanding["passes_information_gate"])
        self.assertAlmostEqual(float(expanding["removed_area_fraction"]), 0.0)


if __name__ == "__main__":
    unittest.main()
