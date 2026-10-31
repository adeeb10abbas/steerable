"""Fail-closed tests for the authenticated retained H06 graded symmetry slice."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / "analysis" / "analyze_h06_graded_symmetry.py"
JSON_PATH = PACKAGE / "paper" / "data" / "h06_graded_symmetry.json"
CSV_PATH = PACKAGE / "paper" / "data" / "h06_graded_symmetry.csv"
SPEC = importlib.util.spec_from_file_location("h06_graded_symmetry", MODULE_PATH)
H06 = importlib.util.module_from_spec(SPEC) if MODULE_PATH.exists() else None
if H06 is not None:
    SPEC.loader.exec_module(H06)


class H06GradedSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if H06 is None:
            raise AssertionError("The H06 graded-symmetry analyzer must be implemented")
        cls.documents, cls.provenance = H06.read_pinned_sources(H06.ROOT)
        cls.report = H06.analyze_documents(cls.documents, cls.provenance)

    def test_source_identity_and_full_population(self):
        self.assertEqual(self.provenance, {"commit": H06.PIN, "files": H06.EXPECTED_SOURCES})
        self.assertEqual(self.documents["full_counts"]["episodes"], 4096)
        self.assertEqual(self.documents["full_counts"]["pairs"], 2048)
        self.assertEqual(self.documents["full_counts"]["episodes_by_model"][H06.MODEL], 2246)
        self.assertEqual(self.documents["full_counts"]["pairs_by_model"][H06.MODEL], 1123)

    def test_primary_grid_is_exact_and_inventory_matched(self):
        design = self.report["design"]
        self.assertEqual(design["matched_seeds"], list(range(9400, 9427)))
        self.assertEqual(design["inventory_matched_levels"], [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(design["excluded_level"], 0.0)
        self.assertEqual(len(self.report["matched_seed_levels"]), 108)
        self.assertNotIn(0.0, {row["symmetry_level_s"] for row in self.report["matched_seed_levels"]})
        self.assertEqual(
            {(row["seed"], row["symmetry_level_s"]) for row in self.report["matched_seed_levels"]},
            {(seed, level) for seed in H06.SEEDS for level in H06.LEVELS},
        )

    def test_primary_depth_slope_reproduces_registered_result(self):
        result = self.report["primary_depth_dose_response"]
        self.assertAlmostEqual(result["mean"], 0.02005684366002428, places=15)
        self.assertAlmostEqual(result["median"], 0.01791224813318343, places=15)
        self.assertAlmostEqual(result["ci95"][0], 0.00041030512337149495, places=15)
        self.assertAlmostEqual(result["ci95"][1], 0.040038464864258154, places=15)
        self.assertEqual(result["replicates"], 20_000)
        self.assertEqual(
            result["sign_test"],
            {
                "positive": 20,
                "negative": 7,
                "ties": 0,
                "non_tied_n": 27,
                "two_sided_p": 0.0191572904586792,
                "method": "exact_two_sided_paired_sign_test_zero_ties_excluded",
            },
        )

    def test_binary_slope_is_secondary_and_not_significant(self):
        result = self.report["binary_success_secondary"]
        self.assertAlmostEqual(result["mean"], -0.03909169870630973, places=15)
        self.assertEqual(result["sign_test"]["positive"], 1)
        self.assertEqual(result["sign_test"]["negative"], 6)
        self.assertEqual(result["sign_test"]["ties"], 20)
        self.assertEqual(result["sign_test"]["two_sided_p"], 0.125)

    def test_level_summaries_are_core_only(self):
        expected = {
            0.25: (27, 24, 0.09186856300328616),
            0.50: (27, 25, 0.08871804895224394),
            0.75: (25, 27, 0.0676189372660937),
            1.00: (27, 26, 0.03556595413721408),
        }
        for row in self.report["by_level"]:
            left, right, depth = expected[row["symmetry_level_s"]]
            self.assertEqual((row["left_successes"], row["right_successes"]), (left, right))
            self.assertEqual(row["matched_pairs"], 27)
            self.assertAlmostEqual(row["requested_depth_gap_R_minus_L_mean_m"], depth, places=15)

    def test_zero_current_activity_and_closed_gates(self):
        self.assertEqual(len(self.report["new_activity"]), 13)
        self.assertEqual(sum(self.report["new_activity"].values()), 0)
        self.assertEqual(len(self.report["gate_state"]), 9)
        self.assertFalse(any(self.report["gate_state"].values()))
        boundary = self.report["claim_boundary"]
        self.assertFalse(boundary["forecast_accuracy_evidence"])
        self.assertFalse(boundary["confirmation_test_or_stage_gate"])
        self.assertFalse(boundary["equivalence_claim"])

    def test_outputs_replay_byte_exactly(self):
        self.assertEqual(JSON_PATH.read_text(encoding="utf-8"), H06.render_json(self.report))
        self.assertEqual(CSV_PATH.read_text(encoding="utf-8"), H06.render_csv(self.report))
        self.assertEqual(len(CSV_PATH.read_text(encoding="utf-8").splitlines()), 109)

    def test_parser_rejects_duplicate_keys_nonfinite_and_nonobjects(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[]'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    H06.parse_json(raw, label="test")

    def test_duplicate_episode_fails_closed(self):
        changed = copy.deepcopy(self.documents)
        changed["episodes"][-1] = copy.deepcopy(changed["episodes"][0])
        with self.assertRaisesRegex(ValueError, "duplicate H06 primary episode"):
            H06.analyze_documents(changed, self.provenance)

    def test_duplicate_pair_fails_closed(self):
        changed = copy.deepcopy(self.documents)
        changed["pairs"][-1] = copy.deepcopy(changed["pairs"][0])
        with self.assertRaisesRegex(ValueError, "duplicate H06 compact pair"):
            H06.analyze_documents(changed, self.provenance)

    def test_mismatched_pair_identity_fails_closed(self):
        changed = copy.deepcopy(self.documents)
        changed["pairs"][0]["matched_pair_id"] = "v3e004:nano:seed9400:s000"
        with self.assertRaisesRegex(ValueError, "mismatched H06 compact pair identity"):
            H06.analyze_documents(changed, self.provenance)

    def test_nonfinite_measurement_fails_closed(self):
        changed = copy.deepcopy(self.documents)
        changed["episodes"][0]["requested_side_depth"] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            H06.analyze_documents(changed, self.provenance)

    def test_registered_report_mismatch_fails_closed(self):
        changed = copy.deepcopy(self.documents)
        changed["results"]["checkpoints"][H06.MODEL]["analysis"]["dose_response_on_realised_A"]["depth_gap_m"]["mean_slope"] += 0.1
        with self.assertRaisesRegex(ValueError, "registered depth_gap_m mean"):
            H06.analyze_documents(changed, self.provenance)

    def test_manuscript_and_package_boundaries(self):
        paper = (PACKAGE / "paper" / "main.tex").read_text(encoding="utf-8")
        readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
        qa = (PACKAGE / "docs" / "DELIVERY_QA.md").read_text(encoding="utf-8")
        for text in (paper, readme, qa):
            self.assertIn("H06", text)
            self.assertIn("0.020", text)
            self.assertIn("inventory", text.lower())
            self.assertIn("forecast accuracy", text.lower())

    def test_reproducible_build_includes_h06(self):
        build = (PACKAGE / "analysis" / "build_package.py").read_text(encoding="utf-8")
        for required in (
            'run("H06 retained graded symmetry"',
            "analysis/analyze_h06_graded_symmetry.py",
            "paper/data/h06_graded_symmetry.json",
            '"retained_h06_graded_symmetry": {',
            '"csv_sha256": digest(package / "paper/data/h06_graded_symmetry.csv")',
            '"analyzer_sha256": digest(package / "analysis/analyze_h06_graded_symmetry.py")',
        ):
            self.assertIn(required, build)


if __name__ == "__main__":
    unittest.main()
