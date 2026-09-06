from __future__ import annotations

from pathlib import Path
import unittest

from experiments.online_correction_v4.droid_scorer import load_scoring_context
from experiments.online_correction_v4.geometry import ConvexPolygonPrism
from tools.build_v4_second_stack_g6_receipt import (
    build_g6,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = (
    ROOT
    / "artifacts/online_correction_v4/setup"
    / "second_stack_scoring_geometry.released.json"
)
CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
G5 = (
    ROOT
    / "artifacts/online_correction_v4/qualification"
    / "20260906_second_stack_g5_trigger_branch_g5c8q20260906a.json"
)


class SecondStackG6Tests(unittest.TestCase):
    def test_scoring_loader_preserves_exact_polygon(self) -> None:
        context = load_scoring_context(
            GEOMETRY,
            expected_sha256=sha256_file(GEOMETRY),
            relation="left",
            d_cap_m=1.0,
        )
        self.assertIsInstance(
            context.planar_spec.workspace,
            ConvexPolygonPrism,
        )
        self.assertEqual(
            len(context.planar_spec.workspace.vertices_xy),
            4,
        )

    def test_measurement_receipt_is_a_complete_pass(self) -> None:
        receipt = build_g6(
            geometry_path=GEOMETRY,
            campaign_path=CAMPAIGN,
            g5_path=G5,
        )
        self.assertTrue(receipt["passed"])
        self.assertTrue(all(receipt["checks"].values()))


if __name__ == "__main__":
    unittest.main()
