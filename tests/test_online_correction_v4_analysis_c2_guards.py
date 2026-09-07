"""C2 primary-response and prefix-verification guard tests."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from experiments.online_correction_v4.analysis import (
    c2_goal_selectivity,
    validate_accepted_ledger,
    _cell_lookup,
    _c2_estimator,
)

from tests.test_online_correction_v4_analysis import (
    GOAL_SET_HASH,
    _fill_c2_block,
    _result,
    _subset_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("online_correction_v4", ROOT / "tools/online_correction_v4.py")
v4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v4)


class C2GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")
        cls.manifest = _subset_manifest(cls.config, cls.config_sha, families=("C2",), blocks=range(2))

    def _lookup(self, results):
        return _cell_lookup(self.manifest, {row["episode_id"]: row for row in results})

    def _valid_block(self, block_id: int = 0):
        return _fill_c2_block(
            self.manifest,
            policy="cosmos3_nano_droid",
            block_id=block_id,
            sham_a=0.20,
            move_a=0.10,
            sham_b=0.30,
            move_b=0.30,
        )

    def _rows_for(self, results, *, block_id: int, goal: str, scenario: str, named_reference: str):
        matched = []
        for row in results:
            manifest_row = next(item for item in self.manifest if item["episode_id"] == row["episode_id"])
            factors = manifest_row["factors"]
            if (
                manifest_row["block_id"] == block_id
                and factors["goal"] == goal
                and factors["scenario"] == scenario
                and factors["named_reference"] == named_reference
            ):
                matched.append(row)
        return matched

    def test_missing_prefix_verification_rejects_ledger_and_makes_c2_unestimable(self):
        results = self._valid_block()
        for row in self._rows_for(results, block_id=0, goal="left", scenario="original_sham", named_reference="A"):
            row["common_prefix_verification_mode"] = "unverified"
        report = validate_accepted_ledger(self.manifest, results, config=self.config)
        self.assertFalse(report["ok"])
        lookup = self._lookup(results)
        self.assertIsNone(
            c2_goal_selectivity(lookup, policy="cosmos3_nano_droid", block_id=0, goal="left", config=self.config)
        )

    def test_mismatched_prefix_identity_between_sham_and_move_rejects_pair(self):
        results = self._valid_block()
        for row in self._rows_for(results, block_id=0, goal="left", scenario="move_A", named_reference="A"):
            row["common_prefix_identity_hash_sha256"] = "f" * 64
        lookup = self._lookup(results)
        self.assertIsNone(
            c2_goal_selectivity(lookup, policy="cosmos3_nano_droid", block_id=0, goal="left", config=self.config)
        )

    def test_mismatched_goal_set_hash_between_sham_and_move_rejects_pair(self):
        results = self._valid_block()
        for row in results:
            manifest_row = next(item for item in self.manifest if item["episode_id"] == row["episode_id"])
            if manifest_row["factors"]["scenario"] == "move_A":
                row["outcome"]["response_goal_set_hash_sha256"] = "e" * 64
        lookup = self._lookup(results)
        self.assertIsNone(
            c2_goal_selectivity(lookup, policy="cosmos3_nano_droid", block_id=0, goal="left", config=self.config)
        )

    def test_wrong_response_horizon_rejects_ledger(self):
        c2_row = next(row for row in self.manifest if row["family"] == "C2")
        broken = _result(c2_row)
        broken["outcome"]["response_horizon_s"] = 4.0
        report = validate_accepted_ledger([c2_row], [broken], config=self.config)
        self.assertFalse(report["ok"])
        self.assertTrue(any("response_horizon_s" in error for error in report["errors"]))

    def test_one_ineligible_branch_rejects_c2_contrast(self):
        results = self._valid_block()
        for row in results:
            manifest_row = next(item for item in self.manifest if item["episode_id"] == row["episode_id"])
            if manifest_row["factors"]["scenario"] == "move_A" and manifest_row["factors"]["named_reference"] == "A":
                row["trigger_eligible"] = False
                row["event_delivered"] = False
        lookup = self._lookup(results)
        self.assertIsNone(
            c2_goal_selectivity(lookup, policy="cosmos3_nano_droid", block_id=0, goal="left", config=self.config)
        )
        aggregate, _ = _c2_estimator(lookup, policy="cosmos3_nano_droid", block_ids=[0], config=self.config)
        self.assertIsNone(aggregate)

    def test_missing_prefix_receipt_hash_rejects_ledger(self):
        c2_row = next(row for row in self.manifest if row["family"] == "C2")
        broken = _result(c2_row)
        broken["common_prefix_verification_receipt_sha256"] = "not-a-hash"
        report = validate_accepted_ledger([c2_row], [broken], config=self.config)
        self.assertFalse(report["ok"])
        self.assertTrue(any("common_prefix_verification_receipt_sha256" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
