"""Fail-closed tests for retained H02 finite-population deletion sensitivity."""
from __future__ import annotations

import copy
import csv
from fractions import Fraction
import importlib.util
import io
from pathlib import Path
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis/analyze_h02_label_deletion_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location("h02_label_deletion_sensitivity", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


def exact(measure):
    return Fraction(measure["numerator"], measure["denominator"])


class H02LabelDeletionSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.provenance = ANALYSIS.load_authenticated_base()
        cls.report = ANALYSIS.analyze_base(cls.base, cls.provenance)
        cls.by_key = {
            (row["scope"], row["component"]): row
            for row in cls.report["sensitivity_rows"]
        }

    def test_latest_h02_package_is_authenticated_and_exactly_regenerated(self):
        self.assertEqual(
            self.report["schema_version"],
            "wmf-retained-h02-label-deletion-sensitivity-v1",
        )
        self.assertEqual(
            self.report["provenance"]["historical_source_commit"],
            "ce561e66f82e95055e39d3d7711691982f6b2086",
        )
        self.assertEqual(
            {
                name: value["sha256"]
                for name, value in self.report["provenance"][
                    "sealed_entry_status_files"
                ].items()
            },
            {
                "analyzer": "6aa3d0b1f6276ac57b8b3838f2ac22bae5929a5d764f57a5370cb75ba04e8a5e",
                "json": "9bc4c8ade7cdcea065a7eee53d74059c0098ca79a55ddf9f930f29c9d771d9d9",
                "csv": "5a48495fb332ca7ec764df36f69e064899797cea562913f1add5840b3923e1ba",
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

    def test_population_and_ordered_324_deletion_means_are_exact(self):
        population = self.report["population"]
        self.assertEqual(population["base_behavioral_episodes"], 108)
        self.assertEqual(population["matched_environment_label_blocks"], 27)
        self.assertEqual(population["sensitivity_rows"], 12)
        self.assertEqual(population["deletion_estimates_per_row"], 27)
        self.assertEqual(population["total_deletion_means"], 324)
        self.assertEqual(population["seed_labels"], list(range(9400, 9427)))
        self.assertFalse(population["new_cohort_created"])
        self.assertEqual(
            list(self.by_key),
            [
                (scope, component)
                for scope, _ in ANALYSIS.SCOPES
                for component in ANALYSIS.COMPONENTS
            ],
        )
        for row in self.by_key.values():
            self.assertEqual(
                [item["omitted_seed"] for item in row["deletion_estimates"]],
                list(range(9400, 9427)),
            )
            self.assertTrue(
                all(item["remaining_labels"] == 26 for item in row["deletion_estimates"])
            )

    def test_exact_balanced_and_interaction_composite_ranges(self):
        component = "cap_restricted_waiting_action_index_contrast"
        balanced = self.by_key[("block_balanced", component)]
        self.assertEqual(exact(balanced["full_population"]["sum"]), Fraction(-6434))
        self.assertEqual(exact(balanced["full_population"]["mean"]), Fraction(-6434, 27))
        summary = balanced["leave_one_registered_label_out"]
        self.assertEqual(exact(summary["minimum_mean"]), Fraction(-12793, 52))
        self.assertEqual(summary["minimum_mean_omitted_seeds"], [9421])
        self.assertEqual(exact(summary["maximum_mean"]), Fraction(-12211, 52))
        self.assertEqual(summary["maximum_mean_omitted_seeds"], [9409])
        self.assertEqual(
            exact(summary["maximum_absolute_shift_from_full_mean"]),
            Fraction(10843, 1404),
        )
        self.assertEqual(summary["maximum_absolute_shift_omitted_seeds"], [9421])

        interaction = self.by_key[("command_by_layout_interaction", component)]
        self.assertEqual(exact(interaction["full_population"]["sum"]), Fraction(662))
        self.assertEqual(exact(interaction["full_population"]["mean"]), Fraction(662, 27))
        summary = interaction["leave_one_registered_label_out"]
        self.assertEqual(exact(summary["minimum_mean"]), Fraction(353, 26))
        self.assertEqual(summary["minimum_mean_omitted_seeds"], [9400])
        self.assertEqual(exact(summary["maximum_mean"]), Fraction(1191, 26))
        self.assertEqual(summary["maximum_mean_omitted_seeds"], [9421])
        self.assertEqual(
            exact(summary["maximum_absolute_shift_from_full_mean"]),
            Fraction(14945, 702),
        )
        self.assertEqual(summary["maximum_absolute_shift_omitted_seeds"], [9421])

    def test_all_twelve_rows_retain_full_population_sign(self):
        self.assertEqual(len(self.by_key), 12)
        for row in self.by_key.values():
            summary = row["leave_one_registered_label_out"]
            self.assertTrue(summary["all_means_retain_full_population_sign"])
            sign = row["full_population"]["sign"]
            self.assertEqual(summary["mean_sign_counts"][sign], 27)
            self.assertEqual(sum(summary["mean_sign_counts"].values()), 27)

    def test_tied_extrema_are_complete(self):
        component = "cap_restricted_waiting_action_index_contrast"
        left = self.by_key[("left", component)]["leave_one_registered_label_out"]
        self.assertEqual(exact(left["maximum_mean"]), Fraction(-6433, 26))
        self.assertEqual(left["maximum_mean_omitted_seeds"], [9410, 9414])

    def test_all_deletion_component_and_scope_identities_close(self):
        for index, seed in enumerate(range(9400, 9427)):
            for scope in self.report["scope_order"]:
                incidence = exact(
                    self.by_key[(scope, ANALYSIS.COMPONENTS[0])][
                        "deletion_estimates"
                    ][index]["mean"]
                )
                zero_masked = exact(
                    self.by_key[(scope, ANALYSIS.COMPONENTS[1])][
                        "deletion_estimates"
                    ][index]["mean"]
                )
                total = exact(
                    self.by_key[(scope, ANALYSIS.COMPONENTS[2])][
                        "deletion_estimates"
                    ][index]["mean"]
                )
                self.assertEqual(incidence + zero_masked, total, (seed, scope))
            for component in ANALYSIS.COMPONENTS:
                left = exact(self.by_key[("left", component)]["deletion_estimates"][index]["mean"])
                right = exact(self.by_key[("right", component)]["deletion_estimates"][index]["mean"])
                balanced = exact(self.by_key[("block_balanced", component)]["deletion_estimates"][index]["mean"])
                interaction = exact(self.by_key[("command_by_layout_interaction", component)]["deletion_estimates"][index]["mean"])
                self.assertEqual((left + right) / 2, balanced, (seed, component))
                self.assertEqual(right - left, interaction, (seed, component))

    def test_mutations_and_nonfinite_values_fail_closed(self):
        changed = copy.deepcopy(self.base)
        changed["per_block"][0]["seed"] = 9999
        with self.assertRaisesRegex(ValueError, "ordered registered labels"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["per_block"][0][
            "block_balanced_cap_restricted_waiting_action_index_contrast"
        ] += 1
        with self.assertRaisesRegex(ValueError, "component identity|block-balanced identity"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["per_block"][0]["left_entry_incidence_at_cap_term"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            ANALYSIS.analyze_base(changed, self.provenance)

        changed = copy.deepcopy(self.base)
        changed["new_activity"][next(iter(changed["new_activity"]))] = 1
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
        self.assertEqual(
            {field for field, value in boundary.items() if value is True},
            {
                "retained_post_result_finite_population_sensitivity_only",
                "exact_leave_one_registered_label_out_reported",
            },
        )
        self.assertTrue(all(type(value) is bool for value in boundary.values()))
        definitions = " ".join(self.report["definitions"].values())
        for marker in (
            "remaining 26 labels",
            "not a robustness proof",
            "not jackknife inference",
            "not independent physical scenes",
        ):
            self.assertIn(marker, definitions)

    def test_canonical_outputs_and_csv_shape(self):
        json_text = ANALYSIS.render_json(self.report)
        csv_text = ANALYSIS.render_csv(self.report)
        self.assertEqual(ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"), json_text)
        self.assertEqual(ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"), csv_text)
        self.assertNotIn('"decimal": -0.0,', json_text)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(len(rows), 324)
        self.assertEqual(
            [(row["scope"], row["component"], int(row["omitted_seed"])) for row in rows[:27]],
            [("left", ANALYSIS.COMPONENTS[0], seed) for seed in range(9400, 9427)],
        )

    def test_public_integration_markers(self):
        build = (ANALYSIS.PACKAGE / "analysis/build_package.py").read_text(encoding="utf-8")
        readme = (ANALYSIS.PACKAGE / "README.md").read_text(encoding="utf-8")
        paper = (ANALYSIS.PACKAGE / "paper/main.tex").read_text(encoding="utf-8")
        qa = (ANALYSIS.PACKAGE / "docs/DELIVERY_QA.md").read_text(encoding="utf-8")
        self.assertIn('run("H02 label-deletion sensitivity"', build)
        self.assertIn("PDF_REQUIRED_MARKERS", build)
        for marker in (
            "deletion preserves all 12",
            "246.019",
            "234.827",
            "13.577",
            "45.808",
            "not jackknife inference",
            "unobserved",
            "robustness",
        ):
            self.assertIn(f'"{marker}"', build)
        for text in (readme, paper, qa):
            self.assertIn("246.019", text)
            self.assertIn("13.577", text)
            self.assertIn("not jackknife", text)


if __name__ == "__main__":
    unittest.main()
