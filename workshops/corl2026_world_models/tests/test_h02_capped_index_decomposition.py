"""Fail-closed tests for the retained H02 capped-index decomposition."""
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
    / "analysis/analyze_h02_capped_index_decomposition.py"
)
SPEC = importlib.util.spec_from_file_location("h02_capped_decomposition", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class H02CappedIndexDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.provenance = ANALYSIS.load_authenticated_base()
        cls.report = ANALYSIS.analyze_base(cls.base, cls.provenance)

    def test_upstream_package_is_authenticated_and_exactly_regenerated(self):
        self.assertEqual(
            self.report["schema_version"],
            "wmf-retained-h02-capped-index-decomposition-v1",
        )
        self.assertEqual(
            self.report["provenance"]["historical_source_commit"],
            "ce561e66f82e95055e39d3d7711691982f6b2086",
        )
        self.assertEqual(
            {
                name: value["sha256"]
                for name, value in self.report["provenance"][
                    "sealed_matched_entry_files"
                ].items()
            },
            {
                "analyzer": "25160e55c9f0ba016728e4c1f5b4bc1a32322921eade121f27cf1e0719d47530",
                "json": "95ad90a69aa6e0b5b2a135de6eb93a6a8d6290ec792bb19ba9d05e9864f96cb0",
                "csv": "1af3d5d2276f116bccda0a9ace3cbc657222bd98775be9ba39503a964346270a",
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
                ANALYSIS._load_matched_module()
            spec_loader.assert_not_called()

    def test_population_identity_and_status_pair_counts(self):
        population = self.report["population"]
        self.assertEqual(population["base_behavioral_episodes"], 108)
        self.assertEqual(population["matched_environment_label_blocks"], 27)
        self.assertEqual(population["seed_labels"], list(range(9400, 9427)))
        self.assertFalse(population["new_cohort_created"])
        self.assertEqual(
            self.report["identity"]["cell_identity"],
            "U = C * (1 - E) + Z",
        )
        self.assertEqual(
            self.report["identity"]["contrast_identity"],
            "delta U = -C * delta E + delta Z",
        )
        self.assertEqual(
            self.report["cross_layout_entry_status_pair_counts"],
            {
                "left": {
                    "control_no_entry__reflected_no_entry": 0,
                    "control_no_entry__reflected_entry": 22,
                    "control_entry__reflected_no_entry": 1,
                    "control_entry__reflected_entry": 4,
                },
                "right": {
                    "control_no_entry__reflected_no_entry": 0,
                    "control_no_entry__reflected_entry": 19,
                    "control_entry__reflected_no_entry": 0,
                    "control_entry__reflected_entry": 8,
                },
            },
        )

    def test_command_specific_decomposition_is_exact(self):
        command = self.report["summaries"][
            "command_specific_position_mirrored_minus_control"
        ]
        expected = {
            "left": {
                "entry_incidence_at_cap_term": (
                    -9450.0,
                    -350.0,
                    -450.0,
                    -450.0,
                    450.0,
                    {"positive": 1, "zero": 4, "negative": 22},
                ),
                "zero_masked_recorded_entry_index_term": (
                    2685.0,
                    2685 / 27,
                    128.0,
                    -223.0,
                    208.0,
                    {"positive": 23, "zero": 0, "negative": 4},
                ),
                "cap_restricted_waiting_action_index_contrast": (
                    -6765.0,
                    -6765 / 27,
                    -320.0,
                    -332.0,
                    227.0,
                    {"positive": 2, "zero": 0, "negative": 25},
                ),
            },
            "right": {
                "entry_incidence_at_cap_term": (
                    -8550.0,
                    -8550 / 27,
                    -450.0,
                    -450.0,
                    0.0,
                    {"positive": 0, "zero": 8, "negative": 19},
                ),
                "zero_masked_recorded_entry_index_term": (
                    2447.0,
                    2447 / 27,
                    142.0,
                    -116.0,
                    175.0,
                    {"positive": 19, "zero": 0, "negative": 8},
                ),
                "cap_restricted_waiting_action_index_contrast": (
                    -6103.0,
                    -6103 / 27,
                    -296.0,
                    -333.0,
                    -13.0,
                    {"positive": 0, "zero": 0, "negative": 27},
                ),
            },
        }
        for relation, components in expected.items():
            for component, wanted in components.items():
                observed = command[relation][component]
                self.assertEqual(observed["n"], 27)
                self.assertEqual(
                    (
                        observed["sum"],
                        observed["mean"],
                        observed["median"],
                        observed["minimum"],
                        observed["maximum"],
                        observed["sign_counts"],
                    ),
                    wanted,
                )

    def test_balanced_and_interaction_decompositions_are_exact(self):
        balanced = self.report["summaries"][
            "block_balanced_position_mirrored_minus_control"
        ]
        expected_balanced = {
            "entry_incidence_at_cap_term": (
                -9000.0,
                -9000 / 27,
                -450.0,
                -450.0,
                0.0,
                {"positive": 0, "zero": 2, "negative": 25},
            ),
            "zero_masked_recorded_entry_index_term": (
                2566.0,
                2566 / 27,
                130.0,
                -39.0,
                187.5,
                {"positive": 25, "zero": 0, "negative": 2},
            ),
            "cap_restricted_waiting_action_index_contrast": (
                -6434.0,
                -6434 / 27,
                -267.5,
                -328.5,
                -37.5,
                {"positive": 0, "zero": 0, "negative": 27},
            ),
        }
        for component, wanted in expected_balanced.items():
            observed = balanced[component]
            self.assertEqual(
                (
                    observed["sum"],
                    observed["mean"],
                    observed["median"],
                    observed["minimum"],
                    observed["maximum"],
                    observed["sign_counts"],
                ),
                wanted,
            )
        interaction = self.report["summaries"][
            "command_by_layout_difference_in_differences"
        ]
        self.assertEqual(interaction["entry_incidence_at_cap_term"]["sum"], 900.0)
        self.assertEqual(interaction["zero_masked_recorded_entry_index_term"]["sum"], -238.0)
        self.assertEqual(
            interaction["cap_restricted_waiting_action_index_contrast"]["sum"],
            662.0,
        )
        self.assertEqual(
            interaction["entry_incidence_at_cap_term"]["sign_counts"],
            {"positive": 7, "zero": 16, "negative": 4},
        )
        self.assertEqual(
            interaction["zero_masked_recorded_entry_index_term"]["sign_counts"],
            {"positive": 16, "zero": 0, "negative": 11},
        )

    def test_every_row_closes_and_mutations_fail_closed(self):
        self.assertEqual(len(self.report["per_block"]), 27)
        for row in self.report["per_block"]:
            for group in (*ANALYSIS.RELATIONS, "block_balanced", "difference_in_differences"):
                self.assertEqual(
                    row[f"{group}_entry_incidence_at_cap_term"]
                    + row[f"{group}_zero_masked_recorded_entry_index_term"],
                    row[f"{group}_cap_restricted_waiting_action_index_contrast"],
                )

        changed = copy.deepcopy(self.base)
        changed["per_block"][0]["seed"] = 9999
        with self.assertRaisesRegex(ValueError, "ordered H02 seed-label"):
            ANALYSIS.analyze_base(changed, self.provenance)
        changed = copy.deepcopy(self.base)
        changed["per_block"][0]["control_left_entry_observed"] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            ANALYSIS.analyze_base(changed, self.provenance)
        changed = copy.deepcopy(self.base)
        row = next(
            row for row in changed["per_block"]
            if row["control_left_entry_observed"] == 0.0
        )
        row["control_left_entry_observed"] = 1.0
        with self.assertRaisesRegex(ValueError, "indicator contrast"):
            ANALYSIS.analyze_base(changed, self.provenance)
        changed = copy.deepcopy(self.base)
        row = next(
            row for row in changed["per_block"]
            if row["control_left_entry_observed"] == 0.0
        )
        row["control_left_cap_restricted_waiting_action_index"] = 449.0
        with self.assertRaisesRegex(ValueError, "administrative cap"):
            ANALYSIS.analyze_base(changed, self.provenance)
        changed = copy.deepcopy(self.base)
        changed["per_block"][0][
            "cap_restricted_waiting_action_index_position_mirrored_minus_control_left"
        ] += 1
        with self.assertRaisesRegex(ValueError, "upstream matched-entry contrast"):
            ANALYSIS.analyze_base(changed, self.provenance)
        changed = copy.deepcopy(self.base)
        changed["per_block"][0][
            "control_left_cap_restricted_waiting_action_index"
        ] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            ANALYSIS.analyze_base(changed, self.provenance)

    def test_zero_activity_closed_gates_and_claim_boundary(self):
        self.assertEqual(len(self.report["new_activity"]), 13)
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertEqual(len(self.report["gate_state"]), 9)
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        changed = copy.deepcopy(self.base)
        first_activity = next(iter(changed["new_activity"]))
        changed["new_activity"][first_activity] = None
        with self.assertRaisesRegex(ValueError, "zero activity"):
            ANALYSIS.analyze_base(changed, self.provenance)
        changed = copy.deepcopy(self.base)
        first_gate = next(iter(changed["gate_state"]))
        changed["gate_state"][first_gate] = 0
        with self.assertRaisesRegex(ValueError, "closed gates"):
            ANALYSIS.analyze_base(changed, self.provenance)
        boundary = self.report["claim_boundary"]
        self.assertTrue(boundary["retained_post_result_algebra_only"])
        self.assertTrue(boundary["exact_cap_composite_decomposition_reported"])
        for field, value in boundary.items():
            if field not in {
                "retained_post_result_algebra_only",
                "exact_cap_composite_decomposition_reported",
            }:
                self.assertIs(value, False, field)
        definitions = " ".join(self.report["definitions"].values())
        for marker in (
            "zero is bookkeeping, not an imputed entry index or event time",
            "not a separation into independently identified incidence and entrant-only timing estimands",
            "no p-value, confidence interval, survival curve, hazard",
            "registered environment label",
            "not temporal transitions",
        ):
            self.assertIn(marker, definitions)

    def test_canonical_outputs_and_public_integration(self):
        self.assertEqual(
            ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"),
            ANALYSIS.render_json(self.report),
        )
        csv_text = ANALYSIS.render_csv(self.report)
        self.assertEqual(ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"), csv_text)
        self.assertNotIn("-0.0", ANALYSIS.render_json(self.report))
        self.assertNotIn("-0.0", csv_text)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(len(rows), 27)
        self.assertEqual([int(row["seed"]) for row in rows], list(range(9400, 9427)))
        build = (ANALYSIS.PACKAGE / "analysis/build_package.py").read_text(
            encoding="utf-8"
        )
        readme = (ANALYSIS.PACKAGE / "README.md").read_text(encoding="utf-8")
        paper = (ANALYSIS.PACKAGE / "paper/main.tex").read_text(encoding="utf-8")
        qa = (ANALYSIS.PACKAGE / "docs/DELIVERY_QA.md").read_text(encoding="utf-8")
        self.assertIn('run("H02 capped-index decomposition"', build)
        for text in (readme, paper, qa):
            self.assertIn("333.333", text)
            self.assertIn("95.037", text)


if __name__ == "__main__":
    unittest.main()
