"""Tests for the horizontal geometry repair core and inventory builder."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


repair = _load(
    "horizontal_geometry_repair",
    ROOT / "experiments/online_correction_v4/horizontal_geometry_repair.py",
)
reset_builder = _load(
    "build_v4_horizontal_reset_registry",
    ROOT / "tools/build_v4_horizontal_reset_registry.py",
)
inventory_builder = _load(
    "build_v4_horizontal_geometry_repair_inventory",
    ROOT / "tools/build_v4_horizontal_geometry_repair_inventory.py",
)
validator = _load(
    "validate_v4_horizontal_geometry_repair",
    ROOT / "tools/validate_v4_horizontal_geometry_repair.py",
)


class HorizontalGeometryRepairTests(unittest.TestCase):
    def test_selects_one_centimeter_cube_offset(self) -> None:
        original = reset_builder.build_registry(
            campaign_path=reset_builder.DEFAULT_CAMPAIGN,
            queue_path=reset_builder.DEFAULT_QUEUE,
            source_report_path=reset_builder.DEFAULT_SOURCE,
        )
        offset, audit = repair.minimum_cube_repair_offset_m(
            base_positions_robot_base_m=original["source_identity"][
                "base_positions_robot_base_m"
            ],
            resets_by_env_seed=original["resets_by_env_seed"],
        )
        self.assertEqual(offset, -0.01)
        self.assertEqual(audit["increments_of_1cm"], 1)
        self.assertGreaterEqual(audit["minimum_swept_separation_m"], 0.005)

    def test_repaired_registry_applies_offset_before_jitter(self) -> None:
        amendment_path = (
            ROOT
            / "artifacts/online_correction_v4/setup/horizontal_geometry_repair_amendment.candidate.json"
        )
        if not amendment_path.is_file():
            self.skipTest("repair amendment not built yet")
        original = reset_builder.build_registry(
            campaign_path=reset_builder.DEFAULT_CAMPAIGN,
            queue_path=reset_builder.DEFAULT_QUEUE,
            source_report_path=reset_builder.DEFAULT_SOURCE,
        )
        repaired = reset_builder.build_registry(
            campaign_path=reset_builder.DEFAULT_CAMPAIGN,
            queue_path=reset_builder.DEFAULT_QUEUE,
            source_report_path=reset_builder.DEFAULT_SOURCE,
            geometry_repair_amendment_path=amendment_path,
        )
        for seed, reset in repaired["resets_by_env_seed"].items():
            orig = original["resets_by_env_seed"][seed]["positions_robot_base_m"]
            fixed = reset["positions_robot_base_m"]
            for name in ("bowl", "banana"):
                self.assertEqual(fixed[name], orig[name])
            self.assertAlmostEqual(
                fixed["rubiks_cube"][0],
                orig["rubiks_cube"][0] - 0.01,
                places=9,
            )


class HorizontalGeometryRepairInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.queue_path = (
            ROOT / "artifacts/online_correction_v4/queue_horizontal_geometry_repair_v1.jsonl"
        )
        if not cls.queue_path.is_file():
            raise unittest.SkipTest("repaired inventory not built yet")
        cls.rows = [
            json.loads(line)
            for line in cls.queue_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_row_count_and_family_allocation(self) -> None:
        self.assertEqual(len(self.rows), 9728)
        self.assertEqual(sum(row["family"] == "C1" for row in self.rows), 6144)
        self.assertEqual(sum(row["family"] == "C3" for row in self.rows), 1536)
        self.assertEqual(sum(row["family"] == "C4" for row in self.rows), 2048)

    def test_validator_passes(self) -> None:
        errors = validator.validate(
            amendment_path=(
                ROOT
                / "artifacts/online_correction_v4/setup/horizontal_geometry_repair_amendment.candidate.json"
            ),
            inventory_manifest_path=(
                ROOT
                / "artifacts/online_correction_v4/setup/horizontal_geometry_repair_inventory_v1.json"
            ),
            queue_path=self.queue_path,
            reset_registry_path=(
                ROOT
                / "artifacts/online_correction_v4/setup/horizontal_reset_registry.geometry_repair_v1.candidate.json"
            ),
            g3_plan_path=(
                ROOT
                / "artifacts/online_correction_v4/setup/horizontal_g3_plan.geometry_repair_v1.candidate.json"
            ),
            historical_queue_path=ROOT / "artifacts/online_correction_v4/queue.jsonl",
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
