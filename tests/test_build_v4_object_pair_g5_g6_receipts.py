from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.build_v4_object_pair_g5_g6_receipts import (
    build_g5_receipt,
    build_g6_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "artifacts/online_correction_v4/qualification"
SETUP = ROOT / "artifacts/online_correction_v4/setup"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ObjectPairG5G6ReceiptTests(unittest.TestCase):
    def test_g5_selects_only_permitted_c7_fallback(self) -> None:
        g3_path = (
            QUALIFICATION
            / "20260906_object_pair_g3_aggregate_g3s7q20260906l.json"
        )
        g4_path = (
            QUALIFICATION
            / "20260906_object_pair_g4_nano_g4c7q20260906e.json"
        )
        receipt = build_g5_receipt(
            g3=load(g3_path),
            g3_path=g3_path,
            g4=load(g4_path),
            g4_path=g4_path,
        )
        self.assertTrue(receipt["passed"])
        self.assertEqual(
            receipt["selected_prefix_mode"],
            "randomized_event_triggered_evaluation",
        )
        self.assertFalse(receipt["exact_counterfactual_branching_claimed"])
        self.assertIn("not_authorized", receipt["fallback_scope"]["C2"])
        self.assertTrue(all(receipt["checks"].values()))

    def test_g6_passes_bound_geometry_measurement_cases(self) -> None:
        geometry_path = SETUP / "object_pair_scoring_geometry.released.json"
        campaign_path = ROOT / "docs/online_correction_v4/campaign.json"
        receipt = build_g6_receipt(
            geometry_payload=load(geometry_path),
            geometry_path=geometry_path,
            campaign=load(campaign_path),
            campaign_path=campaign_path,
        )
        self.assertTrue(receipt["passed"])
        self.assertTrue(all(receipt["checks"].values()))
        self.assertAlmostEqual(
            receipt["known_motion_fixture"]["signed_left_change_m"],
            -0.06,
        )
        self.assertEqual(
            receipt["failure_case_labels"]["wrong_direction_released"],
            "wrong_goal_region",
        )
        self.assertGreater(receipt["d_cap_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
