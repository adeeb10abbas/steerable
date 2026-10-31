"""Fail-closed checks for retained H02 position-reflection packaging."""
from __future__ import annotations

import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "analyze_h02_position_reflection.py"
)
SPEC = importlib.util.spec_from_file_location("h02_position_reflection", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class H02PositionReflectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.provenance = ANALYSIS.read_pinned_sources(ANALYSIS.ROOT)
        cls.report = ANALYSIS.analyze_documents(cls.documents, cls.provenance)

    def test_authenticated_population_and_source_identity(self):
        source = self.report["source_provenance"]
        self.assertEqual(source["commit"], ANALYSIS.PIN)
        self.assertEqual(source["authenticated_file_count"], 9)
        self.assertEqual(source["authenticated_bytes"], 831_259)
        self.assertEqual(source["authenticated_lines"], 1_429)
        self.assertEqual(
            source["files"][ANALYSIS.ROBOT_BRIDGE]["git_blob"],
            "7e610c65d785a3c47674ec41290d91a7afe75d16",
        )
        self.assertEqual(
            self.report["authenticated_population"],
            {
                "registered_cells": 108,
                "behavioral_episodes": 108,
                "matched_left_right_pairs": 54,
                "matched_layout_blocks": 27,
                "seeds": list(range(9400, 9427)),
                "layouts": ["control", "position_mirrored"],
                "relations": ["left", "right"],
                "valid_behavioral_failures_included": True,
                "infrastructure_attempts_included": False,
                "missing_value_imputation": "none",
            },
        )
        configuration = self.report["configuration"]
        self.assertEqual(configuration["effective_model_noise_seed"], 1140)
        self.assertFalse(configuration["official_conditional_D1"])

    def test_outcomes_and_failure_taxonomy_reproduce(self):
        self.assertEqual(
            self.report["layout_totals"],
            {
                "control": {"episodes": 54, "successes": 13, "failures": 41},
                "position_mirrored": {"episodes": 54, "successes": 50, "failures": 4},
            },
        )
        outcomes = self.report["condition_outcomes"]
        self.assertEqual(
            {key: value["successes"] for key, value in outcomes.items()},
            {
                "control:left": 5,
                "control:right": 8,
                "position_mirrored:left": 25,
                "position_mirrored:right": 25,
            },
        )
        self.assertEqual(
            self.report["failure_taxonomy_counts"],
            {
                "correct": 63,
                "pick_failed": 26,
                "release_failed": 1,
                "transport_failed": 17,
                "wrong_side": 1,
            },
        )

    def test_registered_point_estimands_reproduce(self):
        estimates = self.report["registered_point_estimands"]
        d = estimates["endpoint_redirection_D_left_minus_right_m"]
        b = estimates["requested_side_depth_B_right_minus_left_m"]
        self.assertAlmostEqual(d["control"]["mean"], 0.12167757266649494)
        self.assertAlmostEqual(d["position_mirrored"]["mean"], 0.3763758946141159)
        self.assertAlmostEqual(b["control"]["mean"], 0.022953321674355754)
        self.assertAlmostEqual(b["position_mirrored"]["mean"], -0.11796526862653317)
        self.assertAlmostEqual(
            estimates["J_reflected_minus_control_endpoint_interaction_m"]["mean"],
            0.254698321947621,
        )
        self.assertAlmostEqual(
            estimates["I_reflected_minus_control_requested_depth_interaction_m"]["mean"],
            -0.14091859030088894,
        )
        binary = estimates["binary_success_DiD"]
        self.assertEqual(binary["distribution"], {"-2": 0, "-1": 8, "0": 14, "1": 5, "2": 0})
        self.assertEqual(binary["exact_layout_label_permutation"]["extreme_permutations"], 77_987_840)
        self.assertEqual(binary["exact_layout_label_permutation"]["total_permutations"], 134_217_728)
        self.assertAlmostEqual(binary["exact_layout_label_permutation"]["exact_two_sided_p"], 0.5810546875)

    def test_execution_and_declared_artifact_inventory(self):
        self.assertEqual(
            self.report["descriptive_execution"],
            {
                "actions_executed_total": 34_032,
                "minimum_actions": 128,
                "maximum_actions": 450,
                "right_censored_episodes": 45,
                "all_four_cells_successful_seed_count": 1,
                "all_four_cells_successful_seeds": [9403],
            },
        )
        inventory = self.report["declared_raw_artifact_inventory"]
        self.assertEqual(inventory["descriptors"], 540)
        self.assertEqual(inventory["distinct_path_sha_identities"], 540)
        self.assertEqual(inventory["declared_bytes"], 237_163_185)
        self.assertEqual(set(inventory["by_role"]), set(ANALYSIS.ARTIFACT_FIELDS))
        self.assertTrue(all(row["descriptors"] == 108 for row in inventory["by_role"].values()))

    def test_recorded_requested_region_entry_is_censored_without_time_claim(self):
        entry = self.report["recorded_requested_region_entry"]
        self.assertEqual(
            entry["layouts"],
            {
                "control": {
                    "episodes": 54,
                    "recorded_entry_events": 13,
                    "no_entry_cap_censored": 41,
                },
                "position_mirrored": {
                    "episodes": 54,
                    "recorded_entry_events": 53,
                    "no_entry_cap_censored": 1,
                },
            },
        )
        self.assertEqual(
            entry["totals"],
            {
                "episodes": 108,
                "recorded_entry_events": 66,
                "no_entry_cap_censored": 42,
                "observed_sustained_entries": 63,
                "goal_terminated_entry_without_observed_three_sample_sustained_run": 3,
                "successes_with_entry": 63,
                "valid_failures_with_entry": 3,
            },
        )
        conditions = entry["conditions"]
        self.assertEqual(
            {
                key: (
                    value["recorded_entry_events"],
                    value["observed_sustained_entries"],
                    value["valid_failures_with_entry"],
                )
                for key, value in conditions.items()
            },
            {
                "control:left": (5, 4, 0),
                "control:right": (8, 8, 0),
                "position_mirrored:left": (26, 25, 1),
                "position_mirrored:right": (27, 26, 2),
            },
        )
        self.assertAlmostEqual(
            entry["conditions"]["control:left"]["restricted_waiting_action_index"]["mean"],
            405.14814814814815,
        )
        self.assertAlmostEqual(
            entry["conditions"]["control:right"]["restricted_waiting_action_index"]["mean"],
            374.51851851851853,
        )
        self.assertAlmostEqual(
            entry["conditions"]["position_mirrored:left"]["restricted_waiting_action_index"]["mean"],
            154.59259259259258,
        )
        self.assertAlmostEqual(
            entry["conditions"]["position_mirrored:right"]["restricted_waiting_action_index"]["mean"],
            148.4814814814815,
        )
        self.assertIn("not physical time", entry["unit"])
        self.assertIn("not reopened", entry["provenance"])
        self.assertIn("post-result descriptive", entry["provenance"])
        self.assertEqual(entry["event_definition"]["requested_cone_half_angle_degrees"], 45)
        self.assertIn("three consecutive", entry["event_definition"]["observed_sustained_entry"])

    def test_strict_parsing_and_population_mutations_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            ANALYSIS.parse_json(b'{"x":1,"x":2}', label="duplicate")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":NaN}', label="nonfinite")

        malformed = copy.deepcopy(self.documents)
        malformed["cells"][1] = copy.deepcopy(malformed["cells"][0])
        with self.assertRaisesRegex(ValueError, "duplicate registered cell"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["episodes"][0]["requested_success"] = True
        with self.assertRaisesRegex(ValueError, "taxonomy contradiction"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["pairs"][0]["endpoint_redirection_D_m"] += 0.01
        with self.assertRaisesRegex(ValueError, "pair endpoint disagrees"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["summary"]["condition_outcomes"]["control:left"]["successes"] += 1
        with self.assertRaisesRegex(ValueError, "condition outcome differs"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        missing_entry = next(row for row in malformed["episodes"] if row["cone_entry_step"] is None)
        missing_entry["right_censored"] = False
        with self.assertRaisesRegex(ValueError, "right-censoring contract"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        entered_failure = next(
            row for row in malformed["episodes"]
            if row["cone_entry_step"] is not None and not row["requested_success"]
        )
        entered_failure["right_censored"] = False
        with self.assertRaisesRegex(ValueError, "right-censoring contract"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["episodes"][0]["episode_length_steps"] -= 1
        with self.assertRaisesRegex(ValueError, "length must equal"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["episodes"][0]["schema_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "schema version differs"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        mid_episode_entry = next(
            row for row in malformed["episodes"]
            if row["cone_entry_sustained"]
            and row["cone_entry_step"] != row["episode_length_steps"]
        )
        mid_episode_entry["cone_entry_sustained"] = False
        with self.assertRaisesRegex(ValueError, "goal-terminated endpoints"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

    def test_outputs_and_claim_boundaries_are_exact(self):
        self.assertEqual(
            ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"),
            ANALYSIS.render_json(self.report),
        )
        self.assertEqual(
            ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"),
            ANALYSIS.render_csv(self.report),
        )
        rows = list(csv.DictReader(io.StringIO(ANALYSIS.render_csv(self.report))))
        self.assertEqual(len(rows), 27)
        self.assertEqual([int(row["seed"]) for row in rows], list(range(9400, 9427)))
        for field in (
            "control_left_entry_observed",
            "position_mirrored_right_entry_action_index",
            "control_right_minus_left_entry_observed",
            "position_mirrored_minus_control_left_restricted_waiting_action_index",
            "restricted_waiting_action_index_difference_in_differences",
        ):
            self.assertIn(field, rows[0])
        self.assertEqual(
            set(self.report["new_activity"]),
            {
                "annotation_media", "behavioral_actions", "behavioral_cells",
                "confirmation_inference_cells", "forecast_accuracy_estimates",
                "human_responses", "labels", "model_requests", "queue_descriptors",
                "request_rows", "request_selections", "retrieval_jobs", "runnable_manifests",
            },
        )
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertEqual(
            set(self.report["gate_state"]),
            {
                "annotation_media_ready", "camera_alignment_ready",
                "confirmation_release_ready", "current_GM_PVC_worker_state_reconciled",
                "development_annotation_gate_ready", "human_annotation_ready",
                "live_quiescence_inferred", "resource_freeze_ready",
                "safe_to_release_new_work",
            },
        )
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        boundary = self.report["claim_boundary"]
        self.assertFalse(boundary["forecast_accuracy_evidence"])
        self.assertFalse(boundary["official_D1_evidence"])
        self.assertFalse(boundary["changes_232_cell_core_bound"])

        paper = (ANALYSIS.PACKAGE / "paper" / "main.tex").read_text(encoding="utf-8")
        readme = (ANALYSIS.PACKAGE / "README.md").read_text(encoding="utf-8")
        qa = (ANALYSIS.PACKAGE / "docs" / "DELIVERY_QA.md").read_text(encoding="utf-8")
        for text in (paper, readme, qa):
            self.assertIn("H02", text)
            self.assertIn("540", text)
            self.assertIn("custom", text.lower())
            self.assertIn("forecast", text.lower())
            self.assertIn("post-result descriptive", text.lower())
        build = (ANALYSIS.PACKAGE / "analysis" / "build_package.py").read_text(encoding="utf-8")
        for required in (
            'run("H02 retained position reflection"',
            "analysis/analyze_h02_position_reflection.py",
            "paper/data/h02_position_reflection.json",
            '"retained_h02_position_reflection": {',
            '"csv_sha256": digest(package / "paper/data/h02_position_reflection.csv")',
            '"analyzer_sha256": digest(package / "analysis/analyze_h02_position_reflection.py")',
        ):
            self.assertIn(required, build)


if __name__ == "__main__":
    unittest.main()
