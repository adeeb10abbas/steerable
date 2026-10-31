"""Fail-closed tests for the pinned retained H01 requested-entry analysis."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / "analysis" / "analyze_h01_requested_region_entry.py"
JSON_PATH = PACKAGE / "paper" / "data" / "h01_requested_region_entry.json"
CSV_PATH = PACKAGE / "paper" / "data" / "h01_requested_region_entry.csv"
SPEC = importlib.util.spec_from_file_location("h01_requested_region_entry", MODULE_PATH)
H01 = importlib.util.module_from_spec(SPEC) if MODULE_PATH.exists() else None
if H01 is not None:
    SPEC.loader.exec_module(H01)


class H01RequestedRegionEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if H01 is None:
            raise AssertionError("The H01 requested-region entry analyzer must be implemented")
        cls.rows, cls.provenance = H01.read_pinned_sources(H01.ROOT)
        cls.report = H01.analyze_rows(cls.rows, cls.provenance)

    def test_exact_pinned_source_identities_and_cohort(self):
        self.assertEqual(self.provenance["commit"], H01.PIN)
        self.assertEqual(self.provenance["files"], H01.EXPECTED_SOURCES)
        self.assertEqual(len(self.rows), 108)
        self.assertEqual(
            self.report["cohort"],
            {
                "episodes": 108,
                "matched_seed_labels": 27,
                "seed_range": [9400, 9426],
                "episodes_per_condition": 27,
                "condition_order": [
                    "control_left",
                    "control_right",
                    "position_mirrored_left",
                    "position_mirrored_right",
                ],
                "retained_action_samples": 22080,
                "behavioral_successes": 102,
                "behavioral_failures": 6,
                "right_censored_episodes": 6,
                "first_entry_events": 104,
                "cap_right_censored_without_entry": 4,
                "sustained_entry_events": 91,
                "transient_entries": 13,
                "unique_initial_states_by_layout": {
                    "control": 1,
                    "position_mirrored": 1,
                },
                "prescribed_base_scenes": 1,
                "prescribed_layout_states": 2,
            },
        )

    def test_primary_condition_event_counts_and_restricted_waiting(self):
        expected = {
            "control_left": (27, 0, 168.1851851851852, 136, 90, 415),
            "control_right": (27, 0, 176.85185185185185, 123, 94, 410),
            "position_mirrored_left": (27, 0, 119.55555555555556, 107, 83, 288),
            "position_mirrored_right": (23, 4, 209.59259259259258, 139, 95, 450),
        }
        for condition, values in expected.items():
            with self.subTest(condition=condition):
                summary = self.report["conditions"][condition]
                events, censored, mean, median, minimum, maximum = values
                self.assertEqual(summary["entry_events"], events)
                self.assertEqual(summary["cap_right_censored_without_entry"], censored)
                self.assertEqual(summary["restricted_waiting_actions"]["n"], 27)
                self.assertAlmostEqual(summary["restricted_waiting_actions"]["mean"], mean)
                self.assertEqual(summary["restricted_waiting_actions"]["median"], median)
                self.assertEqual(summary["restricted_waiting_actions"]["min"], minimum)
                self.assertEqual(summary["restricted_waiting_actions"]["max"], maximum)
        self.assertEqual(
            self.report["conditions"]["position_mirrored_right"]["entry_kind_counts"],
            {"sustained": 21, "transient": 2, "none": 4},
        )

    def test_ecdf_keeps_cap_censoring_out_of_event_count(self):
        reflected_right = self.report["conditions"]["position_mirrored_right"]
        final = reflected_right["event_time_ecdf"][-1]
        self.assertEqual(final["action_step"], 450)
        self.assertEqual(final["at_risk_before_step"], 4)
        self.assertEqual(final["events_at_step"], 0)
        self.assertEqual(final["cap_censored_at_step"], 4)
        self.assertEqual(final["cumulative_event_count"], 23)
        self.assertAlmostEqual(final["event_cdf"], 23 / 27)
        for condition in ("control_left", "control_right", "position_mirrored_left"):
            with self.subTest(condition=condition):
                final = self.report["conditions"][condition]["event_time_ecdf"][-1]
                self.assertEqual(final["cumulative_event_count"], 27)
                self.assertEqual(final["cap_censored_at_step"], 0)
                self.assertAlmostEqual(final["event_cdf"], 1.0)

    def test_exact_27_block_matched_interaction(self):
        expected = [
            -195, -255, 123, -146, -4, 280, -218, 110, 1,
            127, 121, 399, 397, 306, -110, 65, 15, 157,
            123, 63, 187, 110, 216, -4, 217, 17, 95,
        ]
        blocks = self.report["matched_blocks"]
        self.assertEqual([block["seed"] for block in blocks], list(range(9400, 9427)))
        self.assertEqual(
            [block["contrasts"]["difference_in_differences_actions"] for block in blocks],
            expected,
        )
        interaction = self.report["matched_analysis"]["difference_in_differences_actions"]
        self.assertEqual(interaction["summary"]["n"], 27)
        self.assertAlmostEqual(interaction["summary"]["mean"], 81.37037037037037)
        self.assertEqual(interaction["summary"]["median"], 110)
        self.assertEqual(interaction["summary"]["min"], -255)
        self.assertEqual(interaction["summary"]["max"], 399)
        self.assertEqual(interaction["sign_counts"], {"positive": 20, "negative": 7, "ties": 0})
        self.assertEqual(
            self.report["matched_analysis"]["paired_entry_attainment"],
            {
                "left": {
                    "both_observed": 27,
                    "control_observed_mirrored_censored": 0,
                    "mirrored_observed_control_censored": 0,
                    "both_censored": 0,
                },
                "right": {
                    "both_observed": 23,
                    "control_observed_mirrored_censored": 4,
                    "mirrored_observed_control_censored": 0,
                    "both_censored": 0,
                },
            },
        )

    def test_sustained_entry_is_trace_audit_not_cap_imputation(self):
        expected = {
            "control_left": (23, {"goal_terminated_after_transient_entry_without_three_sample_run": 4,
                                  "observed_sustained_entry": 23}),
            "control_right": (22, {"goal_terminated_after_transient_entry_without_three_sample_run": 5,
                                   "observed_sustained_entry": 22}),
            "position_mirrored_left": (25, {"goal_terminated_after_transient_entry_without_three_sample_run": 2,
                                            "observed_sustained_entry": 25}),
            "position_mirrored_right": (21, {
                "administratively_right_censored_at_cap_without_entry": 4,
                "goal_terminated_after_transient_entry_without_three_sample_run": 2,
                "observed_sustained_entry": 21,
            }),
        }
        for condition, (events, statuses) in expected.items():
            with self.subTest(condition=condition):
                audit = self.report["conditions"][condition]["sustained_entry_audit"]
                self.assertEqual(audit["observed_sustained_entries"], events)
                self.assertEqual(audit["status_counts"], statuses)
                self.assertIn("not assigned an action-450 event time", audit["scope"])

    def test_claim_boundary_forbids_success_scene_and_time_overclaims(self):
        boundary = self.report["claim_boundary"]
        self.assertFalse(boundary["new_model_inference"])
        self.assertFalse(boundary["new_robot_episodes"])
        self.assertFalse(boundary["new_human_labels"])
        self.assertFalse(boundary["forecast_accuracy_evidence"])
        self.assertFalse(boundary["matched_seed_labels_are_independent_scenes"])
        self.assertFalse(boundary["action_index_is_physical_time"])
        self.assertFalse(boundary["first_entry_is_behavioral_success"])
        self.assertFalse(boundary["confirmation_test_or_stage_gate"])
        self.assertEqual(boundary["independent_scene_replications"], 1)
        entered_failures = sum(
            condition["entry_events_among_behavioral_failures"]
            for condition in self.report["conditions"].values()
        )
        self.assertEqual(entered_failures, 2)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for raw in (b'{"seed": 1, "seed": 2}', b'{"value": NaN}', b'[]'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    H01.parse_json_line(raw)

    def test_xyz_measurement_frame_and_stored_derived_fields_fail_closed(self):
        cases = []
        wrong_frame = copy.deepcopy(self.rows[0])
        wrong_frame["measurement_frame"] = "camera_xy"
        cases.append(wrong_frame)

        forged_derived = copy.deepcopy(self.rows[0])
        forged_derived["measurements"]["first_requested_entry_step"] = 0
        cases.append(forged_derived)

        forged_xyz = copy.deepcopy(self.rows[0])
        reference = forged_xyz["steps"][0]["reference_xyz"]
        forged_xyz["steps"][0]["object_xyz"] = [reference[0], reference[1] - 10.0, reference[2]]
        cases.append(forged_xyz)

        nonfinite_xyz = copy.deepcopy(self.rows[0])
        nonfinite_xyz["steps"][0]["object_xyz"][0] = float("nan")
        cases.append(nonfinite_xyz)

        for changed in cases:
            with self.subTest(case=len(cases)):
                with self.assertRaises(ValueError):
                    H01.normalize_episode(changed)

    def test_noncap_no_entry_duplicate_cell_and_state_mismatch_fail_closed(self):
        no_entry = next(
            row for row in self.rows
            if row["measurements"]["first_requested_entry_step"] is None
        )
        early = copy.deepcopy(no_entry)
        early["steps"] = early["steps"][:11]
        early["actions_executed"] = 10
        early["right_censored"] = False
        early["measurements"].update({
            "actions_executed": 10,
            "episode_length_steps": 10,
            "executed_action_count": 10,
            "scored_state_sample_count": 11,
            "right_censored": False,
        })
        early["event_timeline"] = [
            {"event": "episode_start", "action_step": 0},
            {"event": "episode_end", "action_step": 10},
        ]
        with self.assertRaisesRegex(ValueError, "administrative censoring"):
            H01.normalize_episode(early)

        duplicate = list(self.rows)
        duplicate[-1] = copy.deepcopy(self.rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate H01 cohort cell"):
            H01.analyze_rows(duplicate, self.provenance)

        mismatched_state = list(self.rows)
        mismatched_state[0] = copy.deepcopy(self.rows[0])
        mismatched_state[0]["initial_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "share the retained layout state"):
            H01.analyze_rows(mismatched_state, self.provenance)

    def test_checked_in_json_and_csv_are_exact_reproductions(self):
        self.assertEqual(JSON_PATH.read_text(encoding="utf-8"), H01.render_json(self.report))
        self.assertEqual(CSV_PATH.read_text(encoding="utf-8"), H01.render_csv(self.report))
        self.assertEqual(len(CSV_PATH.read_text(encoding="utf-8").splitlines()), 28)
        self.assertEqual(
            hashlib.sha256(JSON_PATH.read_bytes()).hexdigest(),
            "ff35ff9888ccd86e7bc70970b2f56279fbe90d621f56ea91cfe6069892466152",
        )
        self.assertEqual(
            hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
            "df715dd4b696a70d93ff0bb7d4520da1d146e8f74c868dcb7fa5dba1759c430f",
        )


if __name__ == "__main__":
    unittest.main()
