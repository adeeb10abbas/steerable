"""Fail-closed tests for the post-snapshot H02 matched-entry supplement."""
from __future__ import annotations

import copy
import csv
import importlib.util
import io
from pathlib import Path
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis/analyze_h02_matched_entry.py"
)
SPEC = importlib.util.spec_from_file_location("h02_matched_entry", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class H02MatchedEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.provenance = ANALYSIS.load_authenticated_base()
        cls.report = ANALYSIS.analyze_base(cls.base, cls.provenance)

    def test_sealed_base_and_source_identities(self):
        provenance = self.report["provenance"]
        self.assertEqual(
            provenance["historical_source_commit"],
            "ce561e66f82e95055e39d3d7711691982f6b2086",
        )
        self.assertEqual(provenance["historical_authenticated_file_count"], 9)
        self.assertEqual(provenance["historical_authenticated_bytes"], 831_259)
        self.assertEqual(provenance["historical_authenticated_lines"], 1_429)
        self.assertEqual(
            provenance["episode_source"],
            {
                "git_blob": "cacce0095ccafa667a0dd4d93f448f3c31e3c55a",
                "sha256": "31f0715c3cbc682f90e5cc6420b9f45f49622f876734d43ba7c8cb6545b80f34",
                "bytes": 281_813,
                "lines": 108,
            },
        )
        self.assertEqual(
            provenance["matched_pair_source"]["git_blob"],
            "dae25a9e2d7bc18f5dd0f41111eb17787c64d1f8",
        )
        self.assertEqual(
            provenance["entry_semantics_source"]["git_blob"],
            "7e610c65d785a3c47674ec41290d91a7afe75d16",
        )
        self.assertEqual(
            {name: value["sha256"] for name, value in provenance["sealed_package_files"].items()},
            {
                "analyzer": "44b941be8d362a9498798051301a52da99b2679d14f1571f049122fded59f43e",
                "json": "9e39905c58cbd674bf41e342cadfd3e97f0c9e618bde8870e1785ed16af6adf1",
                "csv": "8f1a9dd252c80d57fe543707820d3eec0bbd1580bacf61a597d33adc85b9579b",
            },
        )

    def test_analyzer_is_authenticated_before_dynamic_execution(self):
        bad_identity = {
            "sha256": "0" * 64,
            "bytes": 1,
            "lines": 1,
        }
        with mock.patch.object(ANALYSIS, "_identity", return_value=bad_identity), mock.patch.object(
            ANALYSIS.importlib.util, "spec_from_file_location"
        ) as spec_loader:
            with self.assertRaisesRegex(ValueError, "before import"):
                ANALYSIS._load_h02_module()
            spec_loader.assert_not_called()

    def test_population_and_configuration_boundaries(self):
        self.assertEqual(
            self.report["schema_version"],
            "wmf-retained-h02-matched-entry-supplement-v1",
        )
        population = self.report["population"]
        self.assertEqual(population["base_behavioral_episodes"], 108)
        self.assertEqual(population["matched_environment_label_blocks"], 27)
        self.assertEqual(population["seed_labels"], list(range(9400, 9427)))
        self.assertEqual(population["conditions_per_block"], 4)
        self.assertTrue(population["within_layout_reset_identity_authenticated"])
        self.assertFalse(population["cross_layout_reset_hash_equality_claimed"])
        self.assertFalse(population["independent_scene_or_model_noise_replicates_claimed"])
        configuration = self.report["configuration"]
        self.assertEqual(configuration["effective_model_noise_seed"], 1140)
        self.assertFalse(configuration["official_conditional_D1"])

    def test_block_balanced_layout_contrasts_are_exact(self):
        entry = self.report["summaries"]["entry_observed"]
        self.assertEqual(
            entry["block_balanced_position_mirrored_minus_control"],
            {
                "n": 27,
                "sum": 20.0,
                "mean": 20 / 27,
                "median": 1.0,
                "minimum": 0.0,
                "maximum": 1.0,
                "sign_counts": {"positive": 25, "zero": 2, "negative": 0},
            },
        )
        command = entry["command_specific_position_mirrored_minus_control"]
        self.assertAlmostEqual(command["left"]["mean"], 21 / 27)
        self.assertEqual(command["left"]["sign_counts"], {"positive": 22, "zero": 4, "negative": 1})
        self.assertAlmostEqual(command["right"]["mean"], 19 / 27)
        self.assertEqual(command["right"]["sign_counts"], {"positive": 19, "zero": 8, "negative": 0})
        did = entry["command_by_layout_difference_in_differences"]
        self.assertAlmostEqual(did["mean"], -2 / 27)
        self.assertEqual(did["sign_counts"], {"positive": 4, "zero": 16, "negative": 7})

        waiting = self.report["summaries"]["cap_restricted_waiting_action_index"]
        main = waiting["block_balanced_position_mirrored_minus_control"]
        self.assertEqual(main["sum"], -6434.0)
        self.assertAlmostEqual(main["mean"], -6434 / 27)
        self.assertEqual(main["median"], -267.5)
        self.assertEqual((main["minimum"], main["maximum"]), (-328.5, -37.5))
        self.assertEqual(main["sign_counts"], {"positive": 0, "zero": 0, "negative": 27})
        command = waiting["command_specific_position_mirrored_minus_control"]
        self.assertAlmostEqual(command["left"]["mean"], -6765 / 27)
        self.assertEqual(command["left"]["sign_counts"], {"positive": 2, "zero": 0, "negative": 25})
        self.assertAlmostEqual(command["right"]["mean"], -6103 / 27)
        self.assertEqual(command["right"]["sign_counts"], {"positive": 0, "zero": 0, "negative": 27})
        within = waiting["within_layout_right_minus_left"]
        self.assertAlmostEqual(within["control"]["mean"], -827 / 27)
        self.assertEqual(within["control"]["sign_counts"], {"positive": 5, "zero": 15, "negative": 7})
        self.assertAlmostEqual(within["position_mirrored"]["mean"], -165 / 27)
        self.assertEqual(within["position_mirrored"]["sign_counts"], {"positive": 20, "zero": 0, "negative": 7})
        did = waiting["command_by_layout_difference_in_differences"]
        self.assertEqual((did["sum"], did["median"], did["minimum"], did["maximum"]), (662.0, 22.0, -529.0, 309.0))
        self.assertAlmostEqual(did["mean"], 662 / 27)
        self.assertEqual(did["sign_counts"], {"positive": 19, "zero": 0, "negative": 8})

    def test_per_block_identities_and_mutations_fail_closed(self):
        rows = self.report["per_block"]
        self.assertEqual(len(rows), 27)
        self.assertEqual([row["seed"] for row in rows], list(range(9400, 9427)))
        for row in rows:
            for label in ("entry_observed", "cap_restricted_waiting_action_index"):
                left = row[f"{label}_position_mirrored_minus_control_left"]
                right = row[f"{label}_position_mirrored_minus_control_right"]
                self.assertEqual(
                    row[f"{label}_block_balanced_position_mirrored_minus_control"],
                    0.5 * (left + right),
                )
                self.assertEqual(
                    row[f"{label}_difference_in_differences"],
                    right - left,
                )

        changed = copy.deepcopy(self.base)
        changed["matched_entry_censor_blocks"][0]["seed"] = 9999
        with self.assertRaisesRegex(ValueError, "ordered H02 seed-label"):
            ANALYSIS.analyze_base(changed, self.provenance)
        changed = copy.deepcopy(self.base)
        changed["matched_entry_censor_blocks"][0][
            "restricted_waiting_action_index_difference_in_differences"
        ] += 1
        with self.assertRaisesRegex(ValueError, "row-level contrast identity"):
            ANALYSIS.analyze_base(changed, self.provenance)
        changed = copy.deepcopy(self.base)
        changed["matched_entry_censor_blocks"][0][
            "control_left_restricted_waiting_action_index"
        ] = 451
        with self.assertRaisesRegex(ValueError, "integer in"):
            ANALYSIS.analyze_base(changed, self.provenance)
        no_entry = next(
            row for row in copy.deepcopy(self.base["matched_entry_censor_blocks"])
            if not row["control_left_entry_observed"]
        )
        no_entry["control_left_right_censored"] = False
        changed = copy.deepcopy(self.base)
        changed["matched_entry_censor_blocks"] = [
            no_entry if row["seed"] == no_entry["seed"] else row
            for row in changed["matched_entry_censor_blocks"]
        ]
        with self.assertRaisesRegex(ValueError, "censored at the action-450 cap"):
            ANALYSIS.analyze_base(changed, self.provenance)

    def test_zero_activity_closed_gates_and_claim_boundary(self):
        self.assertEqual(len(self.report["new_activity"]), 13)
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertEqual(len(self.report["gate_state"]), 9)
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        boundary = self.report["claim_boundary"]
        self.assertTrue(boundary["retained_post_result_algebra_only"])
        self.assertTrue(boundary["descriptive_capped_index_contrast_reported"])
        for field, value in boundary.items():
            if field not in {
                "retained_post_result_algebra_only",
                "descriptive_capped_index_contrast_reported",
            }:
                self.assertIs(value, False, field)
        definitions = " ".join(self.report["definitions"].values())
        for marker in (
            "no recorded entry by the cap, not an event at 450",
            "mixes entry-by-cap incidence with first-entry index among entrants",
            "model-based or extrapolated survival curve or hazard",
            "no p-value, confidence interval, equivalence or causal claim",
            "registered label",
        ):
            self.assertIn(marker, definitions)

    def test_canonical_outputs_and_public_integration(self):
        self.assertEqual(
            ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"),
            ANALYSIS.render_json(self.report),
        )
        csv_text = ANALYSIS.render_csv(self.report)
        self.assertEqual(ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"), csv_text)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(len(rows), 27)
        self.assertEqual([int(row["seed"]) for row in rows], list(range(9400, 9427)))
        for field in (
            "entry_observed_block_balanced_position_mirrored_minus_control",
            "cap_restricted_waiting_action_index_position_mirrored_minus_control_left",
            "cap_restricted_waiting_action_index_difference_in_differences",
        ):
            self.assertIn(field, rows[0])
        build = (ANALYSIS.PACKAGE / "analysis/build_package.py").read_text(encoding="utf-8")
        readme = (ANALYSIS.PACKAGE / "README.md").read_text(encoding="utf-8")
        paper = (ANALYSIS.PACKAGE / "paper/main.tex").read_text(encoding="utf-8")
        qa = (ANALYSIS.PACKAGE / "docs/DELIVERY_QA.md").read_text(encoding="utf-8")
        self.assertIn('run("H02 matched entry supplement"', build)
        for text in (readme, paper, qa):
            self.assertIn("238.", text)
            self.assertIn("27/27", text)


if __name__ == "__main__":
    unittest.main()
