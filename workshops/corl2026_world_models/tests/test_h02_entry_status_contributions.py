"""Fail-closed tests for the retained H02 entry-status contribution ledger."""
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
    / "analysis/analyze_h02_entry_status_contributions.py"
)
SPEC = importlib.util.spec_from_file_location("h02_entry_status_contributions", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class H02EntryStatusContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.provenance = ANALYSIS.load_authenticated_base()
        cls.report = ANALYSIS.analyze_base(cls.base, cls.provenance)
        cls.by_key = {
            (row["relation"], row["entry_status_pair"]): row
            for row in cls.report["entry_status_contribution_ledger"]
        }

    def test_upstream_package_is_authenticated_and_exactly_regenerated(self):
        self.assertEqual(
            self.report["schema_version"],
            "wmf-retained-h02-entry-status-contributions-v1",
        )
        self.assertEqual(
            self.report["provenance"]["historical_source_commit"],
            "ce561e66f82e95055e39d3d7711691982f6b2086",
        )
        self.assertEqual(
            {
                name: value["sha256"]
                for name, value in self.report["provenance"][
                    "sealed_capped_decomposition_files"
                ].items()
            },
            {
                "analyzer": "bbfe43b7dde0ec4f30e7a7f2fb40c11389c8ced5175d6196dae512f7c363c11a",
                "json": "33d66ed365d2fe330f2a4988a578a6fdbdc83c335184fb9098aeaa0f0c676f09",
                "csv": "11fd4944b7cb5f9d3509bda6d9c0d1764d5d077461a05b027a13f980b16e3cb4",
            },
        )

    def test_analyzer_is_authenticated_before_dynamic_execution(self):
        bad_identity = {"sha256": "0" * 64, "bytes": 1, "lines": 1}
        with mock.patch.object(
            ANALYSIS, "_identity", return_value=bad_identity
        ), mock.patch.object(
            ANALYSIS.importlib.util, "spec_from_file_location"
        ) as spec_loader:
            with self.assertRaisesRegex(ValueError, "before import"):
                ANALYSIS._load_upstream_module()
            spec_loader.assert_not_called()

    def test_population_and_eight_row_ledger_are_exact(self):
        population = self.report["population"]
        self.assertEqual(population["base_behavioral_episodes"], 108)
        self.assertEqual(population["matched_environment_label_blocks"], 27)
        self.assertEqual(population["command_label_comparisons"], 54)
        self.assertEqual(population["ledger_rows"], 8)
        self.assertEqual(population["nonempty_ledger_rows"], 5)
        self.assertEqual(population["seed_labels"], list(range(9400, 9427)))
        self.assertFalse(population["new_cohort_created"])
        self.assertEqual(len(self.by_key), 8)
        self.assertEqual(
            list(self.by_key),
            [
                (relation, status)
                for relation in ANALYSIS.RELATIONS
                for status in ANALYSIS.STATUS_PAIRS
            ],
        )

    def test_exact_status_pair_contribution_values(self):
        expected = {
            ("left", "control_no_entry__reflected_no_entry"): (0, 0.0, 0.0, 0.0),
            ("left", "control_no_entry__reflected_entry"): (
                22,
                -9900.0,
                3008.0,
                -6892.0,
            ),
            ("left", "control_entry__reflected_no_entry"): (
                1,
                450.0,
                -223.0,
                227.0,
            ),
            ("left", "control_entry__reflected_entry"): (4, 0.0, -100.0, -100.0),
            ("right", "control_no_entry__reflected_no_entry"): (0, 0.0, 0.0, 0.0),
            ("right", "control_no_entry__reflected_entry"): (
                19,
                -8550.0,
                2829.0,
                -5721.0,
            ),
            ("right", "control_entry__reflected_no_entry"): (0, 0.0, 0.0, 0.0),
            ("right", "control_entry__reflected_entry"): (8, 0.0, -382.0, -382.0),
        }
        for key, wanted in expected.items():
            row = self.by_key[key]
            observed = (
                row["matched_blocks"],
                row["component_sums"]["entry_incidence_at_cap_term"],
                row["component_sums"]["zero_masked_recorded_entry_index_term"],
                row["component_sums"][
                    "cap_restricted_waiting_action_index_contrast"
                ],
            )
            self.assertEqual(observed, wanted, key)

    def test_empty_strata_are_explicit_empty_sums_not_observed_effects(self):
        empty = [row for row in self.by_key.values() if row["matched_blocks"] == 0]
        self.assertEqual(len(empty), 3)
        for row in empty:
            self.assertTrue(row["empty_stratum"])
            self.assertIsNone(row["conditional_mean"])
            self.assertEqual(row["component_sums"], {name: 0.0 for name in ANALYSIS.COMPONENTS})
            self.assertEqual(
                row["contribution_to_command_mean"],
                {name: 0.0 for name in ANALYSIS.COMPONENTS},
            )
        for row in self.by_key.values():
            if row["matched_blocks"]:
                self.assertFalse(row["empty_stratum"])
                self.assertEqual(
                    row["conditional_mean"],
                    {
                        name: row["component_sums"][name] / row["matched_blocks"]
                        for name in ANALYSIS.COMPONENTS
                    },
                )

    def test_command_balanced_and_interaction_closures_are_exact(self):
        closures = self.report["closures"]
        self.assertEqual(
            closures["command_specific_component_sums"],
            {
                "left": {
                    "entry_incidence_at_cap_term": -9450.0,
                    "zero_masked_recorded_entry_index_term": 2685.0,
                    "cap_restricted_waiting_action_index_contrast": -6765.0,
                },
                "right": {
                    "entry_incidence_at_cap_term": -8550.0,
                    "zero_masked_recorded_entry_index_term": 2447.0,
                    "cap_restricted_waiting_action_index_contrast": -6103.0,
                },
            },
        )
        self.assertEqual(
            closures["block_balanced_component_sums"],
            {
                "entry_incidence_at_cap_term": -9000.0,
                "zero_masked_recorded_entry_index_term": 2566.0,
                "cap_restricted_waiting_action_index_contrast": -6434.0,
            },
        )
        self.assertEqual(
            closures["command_by_layout_interaction_component_sums"],
            {
                "entry_incidence_at_cap_term": 900.0,
                "zero_masked_recorded_entry_index_term": -238.0,
                "cap_restricted_waiting_action_index_contrast": 662.0,
            },
        )
        for relation in ANALYSIS.RELATIONS:
            rows = [row for row in self.by_key.values() if row["relation"] == relation]
            self.assertEqual(sum(row["matched_blocks"] for row in rows), 27)
            for component in ANALYSIS.COMPONENTS:
                self.assertEqual(
                    sum(row["component_sums"][component] for row in rows),
                    closures["command_specific_component_sums"][relation][component],
                )
                self.assertAlmostEqual(
                    sum(row["contribution_to_command_mean"][component] for row in rows),
                    closures["command_specific_component_sums"][relation][component] / 27,
                    places=12,
                )

    def test_mutations_fail_closed(self):
        changed = copy.deepcopy(self.base)
        changed["per_block"][0]["seed"] = 9999
        with self.assertRaisesRegex(ValueError, "ordered H02 seed-label"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["per_block"][0]["left_entry_status_pair"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown cross-layout"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["per_block"][0][
            "left_entry_status_pair"
        ] = "control_entry__reflected_entry"
        with self.assertRaisesRegex(ValueError, "incidence term"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["per_block"][0][
            "left_cap_restricted_waiting_action_index_contrast"
        ] += 1
        with self.assertRaisesRegex(ValueError, "row decomposition"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["cross_layout_entry_status_pair_counts"]["left"][
            "control_no_entry__reflected_entry"
        ] -= 1
        with self.assertRaisesRegex(ValueError, "counts do not close"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["identity"]["contrast_identity"] = "changed"
        with self.assertRaisesRegex(ValueError, "algebra identity"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["new_activity"][next(iter(changed["new_activity"]))] = None
        with self.assertRaisesRegex(ValueError, "13 zero activity"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["gate_state"][next(iter(changed["gate_state"]))] = 0
        with self.assertRaisesRegex(ValueError, "nine closed gates"):
            ANALYSIS.analyze_base(changed, self.provenance)

    def test_zero_activity_closed_gates_and_strict_claim_boundary(self):
        self.assertEqual(len(self.report["new_activity"]), 13)
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertEqual(len(self.report["gate_state"]), 9)
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        boundary = self.report["claim_boundary"]
        true_fields = {
            field for field, value in boundary.items() if value is True
        }
        self.assertEqual(
            true_fields,
            {
                "retained_post_result_algebra_only",
                "exact_status_pair_contribution_ledger_reported",
            },
        )
        self.assertTrue(all(type(value) is bool for value in boundary.values()))
        definitions = " ".join(self.report["definitions"].values())
        for marker in (
            "does not change the estimand",
            "conditional_mean null",
            "not observed zero effects",
            "outcome-defined category",
            "not temporal transitions",
            "principal strata",
            "no p-value, confidence interval",
        ):
            self.assertIn(marker, definitions)

    def test_canonical_outputs_and_csv_shape(self):
        json_text = ANALYSIS.render_json(self.report)
        csv_text = ANALYSIS.render_csv(self.report)
        self.assertEqual(ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"), json_text)
        self.assertEqual(ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"), csv_text)
        self.assertNotIn("-0.0", json_text)
        self.assertNotIn("-0.0", csv_text)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [(row["relation"], row["entry_status_pair"]) for row in rows],
            list(self.by_key),
        )
        empty_rows = [row for row in rows if row["matched_blocks"] == "0"]
        self.assertEqual(len(empty_rows), 3)
        for row in empty_rows:
            self.assertEqual(row["empty_stratum"], "True")
            for component in ANALYSIS.COMPONENTS:
                self.assertEqual(row[f"{component}_conditional_mean"], "")

    def test_public_integration_markers(self):
        build = (ANALYSIS.PACKAGE / "analysis/build_package.py").read_text(
            encoding="utf-8"
        )
        readme = (ANALYSIS.PACKAGE / "README.md").read_text(encoding="utf-8")
        paper = (ANALYSIS.PACKAGE / "paper/main.tex").read_text(encoding="utf-8")
        qa = (ANALYSIS.PACKAGE / "docs/DELIVERY_QA.md").read_text(encoding="utf-8")
        self.assertIn('run("H02 entry-status contributions"', build)
        for text in (readme, paper, qa):
            normalized = text.replace(",", "")
            self.assertIn("6892", normalized)
            self.assertIn("5721", normalized)


if __name__ == "__main__":
    unittest.main()
