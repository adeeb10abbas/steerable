"""Fail-closed tests for retained H01 request-window motion provenance."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / "analysis/analyze_h01_request_window_motion.py"
JSON_PATH = PACKAGE / "paper/data/h01_request_window_motion.json"
CSV_PATH = PACKAGE / "paper/data/h01_request_window_motion.csv"
BUILD_PATH = PACKAGE / "analysis/build_package.py"
PAPER_PATH = PACKAGE / "paper/main.tex"
SPEC = importlib.util.spec_from_file_location("h01_request_window_motion", MODULE_PATH)
H01 = importlib.util.module_from_spec(SPEC) if MODULE_PATH.exists() else None
if H01 is not None:
    assert SPEC.loader is not None
    SPEC.loader.exec_module(H01)


class H01RequestWindowMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if H01 is None:
            raise AssertionError("H01 request-window analyzer must exist")
        cls.rows, cls.provenance = H01.read_pinned_sources(H01.ROOT)
        cls.report = H01.analyze_rows(cls.rows, cls.provenance)

    def test_exact_sources_population_and_descriptor_identity(self):
        self.assertEqual(self.provenance, {"commit": H01.PIN, "files": H01.EXPECTED_SOURCES})
        self.assertEqual(len(self.rows), 108)
        cohort = self.report["cohort"]
        self.assertEqual(cohort["episodes"], 108)
        self.assertEqual(cohort["matched_seed_labels"], 27)
        self.assertEqual(cohort["retained_requests"], 738)
        self.assertEqual(cohort["requested_successes"], 102)
        self.assertEqual(cohort["administrative_cap_right_censored_failures"], 6)
        self.assertEqual(cohort["request_window_actions"], 32)
        for key in (
            "decoded_future_descriptors", "returned_action_descriptors",
            "unique_decoded_future_path_identities", "unique_decoded_future_hashes",
            "unique_returned_action_path_identities", "unique_returned_action_hashes",
        ):
            self.assertEqual(cohort[key], 738)
        self.assertEqual(cohort["decoded_future_manifest_bytes_not_retrieved"], 24_689_205_504)
        self.assertEqual(cohort["returned_action_manifest_bytes_not_retrieved"], 850_176)

    def test_exact_condition_coverage_and_censoring(self):
        coverage = self.report["coverage"]
        self.assertEqual(
            (coverage["requests"], coverage["complete_32_action_windows"], coverage["censored_windows"]),
            (738, 632, 106),
        )
        expected = {
            "control_left": (189, 164, 25),
            "control_right": (192, 165, 27),
            "position_mirrored_left": (145, 118, 27),
            "position_mirrored_right": (212, 185, 27),
        }
        for condition, counts in expected.items():
            with self.subTest(condition=condition):
                value = coverage["by_condition"][condition]
                self.assertEqual(
                    (value["requests"], value["complete_32_action_windows"], value["censored_windows"]),
                    counts,
                )
                self.assertEqual(value["episodes"], 27)
                self.assertEqual(
                    value["episode_balanced_complete_window_metrics"]["cube_displacement_m"]["n"],
                    27,
                )
        self.assertEqual(
            [
                (row["action_step_start"], row["requests"], row["complete_32_action_windows"], row["censored_windows"])
                for row in coverage["by_action_step_start"]
            ],
            [
                (0, 108, 108, 0), (32, 108, 108, 0), (64, 108, 107, 1),
                (96, 107, 88, 19), (128, 86, 52, 34), (160, 52, 42, 10),
                (192, 42, 30, 12), (224, 30, 26, 4), (256, 26, 18, 8),
                (288, 18, 16, 2), (320, 16, 14, 2), (352, 14, 10, 4),
                (384, 10, 7, 3), (416, 7, 6, 1), (448, 6, 0, 6),
            ],
        )
        self.assertEqual(
            coverage["censored_windows_by_episode_outcome"],
            {"goal_stopped_success": 100, "administrative_cap_censored_failure": 6},
        )
        self.assertEqual(
            coverage["censored_observed_prefix_actions"],
            {"n": 106, "mean": 16.49056603773585, "median": 19.0, "min": 1,
             "max": 31, "sum": 1748},
        )

    def test_episode_balanced_execution_motion_is_descriptive(self):
        expected_cube_means = {
            "control_left": 0.045015273205,
            "control_right": 0.078566208264,
            "position_mirrored_left": 0.076515030522,
            "position_mirrored_right": 0.068001709153,
        }
        for condition, expected in expected_cube_means.items():
            observed = self.report["coverage"]["by_condition"][condition][
                "episode_balanced_complete_window_metrics"
            ]["cube_displacement_m"]
            self.assertEqual(observed["n"], 27)
            self.assertAlmostEqual(observed["mean"], expected, places=12)
        request_level = self.report["coverage"]["by_condition"]["control_left"][
            "request_level_complete_window_metrics"
        ]
        self.assertEqual(request_level["cube_displacement_m"]["n"], 164)
        self.assertAlmostEqual(request_level["cube_displacement_m"]["mean"], 0.04190115262348475)

    def test_matched_complete_windows_expose_selection_and_signed_motion(self):
        self.assertEqual(self.report["schema_version"], "wmf-retained-h01-request-window-motion-v2")
        matched = self.report["matched_complete_window_analysis"]
        self.assertEqual(
            matched["status"],
            "retrospective matched-window selection and execution-motion audit",
        )
        expected = {
            "control": {
                "pairs": 114,
                "excluded": {"left": 50, "right": 51, "total": 101},
                "excluded_timing": {
                    "counterpart_request_terminated_before_32_action_endpoint": {
                        "left": 11, "right": 11, "total": 22,
                    },
                    "owner_request_started_at_or_after_counterpart_episode_end": {
                        "left": 39, "right": 40, "total": 79,
                    },
                },
                "window_signs": {"positive": 50, "negative": 18, "exact_zero": 46},
                "seed_signs": {"positive": 26, "negative": 0, "exact_zero": 1},
                "seed_mean": 0.05754318106229658,
                "seed_median": 0.04749090472857157,
                "pair_outcomes": {
                    "cap_failure_left__success_right": 1,
                    "success_left__cap_failure_right": 1,
                    "success_left__success_right": 25,
                },
                "outcome_windows": {
                    "cap_failure_left__success_right": 4,
                    "success_left__cap_failure_right": 4,
                    "success_left__success_right": 106,
                },
            },
            "position_mirrored": {
                "pairs": 113,
                "excluded": {"left": 5, "right": 72, "total": 77},
                "excluded_timing": {
                    "counterpart_request_terminated_before_32_action_endpoint": {
                        "left": 2, "right": 19, "total": 21,
                    },
                    "owner_request_started_at_or_after_counterpart_episode_end": {
                        "left": 3, "right": 53, "total": 56,
                    },
                },
                "window_signs": {"positive": 52, "negative": 17, "exact_zero": 44},
                "seed_signs": {"positive": 27, "negative": 0, "exact_zero": 0},
                "seed_mean": 0.06396968954323251,
                "seed_median": 0.0648304121568799,
                "pair_outcomes": {
                    "success_left__cap_failure_right": 4,
                    "success_left__success_right": 23,
                },
                "outcome_windows": {
                    "success_left__cap_failure_right": 22,
                    "success_left__success_right": 91,
                },
            },
        }
        for layout, wanted in expected.items():
            with self.subTest(layout=layout):
                observed = matched["by_layout"][layout]
                self.assertEqual(observed["shared_complete_window_pairs"], wanted["pairs"])
                self.assertEqual(
                    observed["complete_window_records_excluded_without_exact_counterpart_endpoint"],
                    wanted["excluded"],
                )
                self.assertEqual(observed["excluded_record_timing"], wanted["excluded_timing"])
                self.assertEqual(
                    observed["request_window_signed_lateral_contrast_m"]["sign_counts"],
                    wanted["window_signs"],
                )
                seed_summary = observed[
                    "equal_seed_mean_signed_lateral_contrast_m_per_32_action_window"
                ]
                self.assertEqual(seed_summary["sign_counts"], wanted["seed_signs"])
                self.assertAlmostEqual(seed_summary["mean"], wanted["seed_mean"], places=15)
                self.assertAlmostEqual(seed_summary["median"], wanted["seed_median"], places=15)
                self.assertEqual(observed["pair_outcome_counts"], wanted["pair_outcomes"])
                self.assertEqual(
                    observed["shared_window_pairs_by_pair_outcome"], wanted["outcome_windows"]
                )
                first = observed["fixed_first_0_to_32_action_window_contrast_m"]
                self.assertEqual(first["sign_counts"], {
                    "positive": 0, "negative": 0, "exact_zero": 27,
                })
                self.assertEqual(first["min"], first["max"])
                self.assertEqual(first["max"], 0.0)
                self.assertTrue(all(
                    abs(row["telescoping_residual_m"]) <= 1e-15
                    for row in observed["per_seed"]
                ))

        same = matched["four_condition_same_request_index_support"]
        self.assertEqual(same["matched_seed_labels"], 27)
        self.assertEqual(same["shared_complete_window_pairs_per_layout"], 96)
        self.assertEqual(
            same["shared_window_pair_count_per_seed"],
            {"n": 27, "mean": 3.5555555555555554, "median": 3, "min": 2, "max": 6},
        )
        interaction = same[
            "position_mirrored_minus_control_mean_contrast_m_per_32_action_window"
        ]
        self.assertAlmostEqual(interaction["mean"], 0.006533360597390084, places=15)
        self.assertAlmostEqual(interaction["median"], 0.004710602884491287, places=15)
        self.assertEqual(
            interaction["sign_counts"],
            {"positive": 15, "negative": 11, "exact_zero": 1},
        )

    def test_every_request_has_exact_lineage_and_censoring_is_null(self):
        for request in self.report["request_windows"]:
            with self.subTest(request=request["request_id"]):
                self.assertEqual(request["action_step_start"], 32 * request["request_index"])
                self.assertEqual(request["decoded_future"]["shape"], [33, 528, 640, 3])
                self.assertEqual(request["returned_action"]["shape"], [32, 8])
                self.assertNotIn("path", request["decoded_future"])
                self.assertNotIn("path", request["returned_action"])
                if request["complete_32_action_window"]:
                    self.assertEqual(request["action_step_target"], request["action_step_start"] + 32)
                    self.assertIsNone(request["censor_reason"])
                    self.assertIsNone(request["censor_outcome"])
                    self.assertEqual(request["observed_prefix_actions"], 32)
                    self.assertTrue(all(request[name] is not None for name in H01.METRICS))
                else:
                    self.assertIsNone(request["action_step_target"])
                    self.assertEqual(request["censor_reason"], "episode_terminated_before_action_start_plus_32")
                    self.assertIn(
                        request["censor_outcome"],
                        ("goal_stopped_success", "administrative_cap_censored_failure"),
                    )
                    self.assertLess(request["observed_prefix_actions"], 32)
                    self.assertTrue(all(request[name] is None for name in H01.METRICS))

    def test_motion_metric_translation_invariance_and_decomposition(self):
        base = H01.motion_metrics((1, 2, 3), (4, 5, 6), (2, 5, 7), (4.5, 6, 8))
        shifted = H01.motion_metrics((11, 12, 13), (14, 15, 16), (12, 15, 17), (14.5, 16, 18))
        self.assertEqual(base, shifted)
        self.assertAlmostEqual(base["cube_displacement_m"], (26) ** 0.5)
        self.assertAlmostEqual(base["bowl_displacement_m"], (5.25) ** 0.5)
        self.assertAlmostEqual(base["signed_lateral_relative_change_m"], 2.0)
        cube_y_delta = 5 - 2
        bowl_y_delta = 6 - 5
        self.assertAlmostEqual(base["signed_lateral_relative_change_m"], cube_y_delta - bowl_y_delta)
        with self.assertRaises(ValueError):
            H01.motion_metrics((0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        with self.assertRaises(ValueError):
            H01.motion_metrics((0, 0, float("nan")), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    def test_strict_parser_rejects_duplicate_nonfinite_and_nonobject(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    H01.parse_json_object(raw)

    def test_descriptor_request_fingerprint_and_step_tampering_fail_closed(self):
        mutations = []
        wrong_shape = copy.deepcopy(self.rows[0])
        wrong_shape["future_requests"][0]["decoded_future_shape"] = [32, 528, 640, 3]
        mutations.append((wrong_shape, "shape"))
        bad_hash = copy.deepcopy(self.rows[0])
        bad_hash["future_requests"][0]["decoded_future"]["sha256"] = "not-a-digest"
        mutations.append((bad_hash, "digest"))
        bad_bytes = copy.deepcopy(self.rows[0])
        bad_bytes["future_requests"][0]["decoded_future"]["bytes"] -= 1
        mutations.append((bad_bytes, "byte count"))
        bad_fingerprint = copy.deepcopy(self.rows[0])
        bad_fingerprint["future_requests"][0]["reset_fingerprint_sha256"] = "0" * 64
        mutations.append((bad_fingerprint, "fingerprint"))
        bad_sampling_seed = copy.deepcopy(self.rows[0])
        bad_sampling_seed["future_requests"][0]["sampling_seed"] += 1
        mutations.append((bad_sampling_seed, "sampling seed"))
        missing_final_request = copy.deepcopy(self.rows[0])
        missing_final_request["future_requests"].pop()
        mutations.append((missing_final_request, "per-episode request count"))
        bad_index = copy.deepcopy(self.rows[0])
        bad_index["future_requests"][0]["request_index"] = 1
        mutations.append((bad_index, "lineage"))
        missing_step = copy.deepcopy(self.rows[0])
        del missing_step["steps"][1]
        mutations.append((missing_step, "contiguous"))
        duplicate_path = copy.deepcopy(self.rows[0])
        duplicate_path["future_requests"][1]["decoded_future"]["path"] = duplicate_path["future_requests"][0]["decoded_future"]["path"]
        mutations.append((duplicate_path, "duplicate"))
        mismatched_state = copy.deepcopy(self.rows[0])
        mismatched_state["initial_state_sha256"] = "0" * 64
        mutations.append((mismatched_state, "initial state"))
        for changed, marker in mutations:
            candidate = list(self.rows)
            candidate[0] = changed
            with self.subTest(marker=marker):
                with self.assertRaises(ValueError):
                    H01.analyze_rows(candidate, self.provenance)

    def test_claim_boundary_zero_activity_closed_gates_and_core_bound(self):
        boundary = self.report["claim_boundary"]
        self.assertTrue(boundary["retrospective_execution_motion_evidence"])
        self.assertTrue(boundary["forecast_descriptor_provenance_evidence"])
        for name in (
            "forecast_arrays_retrieved", "forecast_payload_bytes_reopened_or_reverified",
            "forecast_accuracy_or_skill_estimated",
            "generated_versus_executed_comparison", "A3_A9_completed",
            "physical_time_established", "A6_movement_threshold_established",
            "independent_scene_or_layout_generalization_established",
            "language_understanding_established", "new_episodes_or_requests_created",
            "new_labels_or_retrieval_created", "models_or_cohorts_pooled",
            "changes_232_cell_core_bound",
        ):
            self.assertIs(boundary[name], False)
        self.assertEqual(boundary["maximum_core_behavioral_cells"], 232)
        self.assertEqual(len(self.report["new_activity"]), 13)
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertEqual(len(self.report["gate_state"]), 9)
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))

    def test_checked_in_outputs_are_deterministic_and_public(self):
        json_text = H01.render_json(self.report)
        csv_text = H01.render_csv(self.report)
        self.assertEqual(JSON_PATH.read_text(encoding="utf-8"), json_text)
        self.assertEqual(CSV_PATH.read_text(encoding="utf-8"), csv_text)
        self.assertEqual(len(csv_text.splitlines()), 739)
        source_paths = {
            descriptor["path"]
            for row in self.rows
            for request in row["future_requests"]
            for descriptor in (request["decoded_future"], request["returned_action"])
        }
        for forbidden in (*source_paths, "/data/users/", "/home/ali/", "attempt01/decoded_futures"):
            self.assertNotIn(forbidden, json_text)
            self.assertNotIn(forbidden, csv_text)
        self.assertEqual(
            hashlib.sha256(json_text.encode()).hexdigest(),
            "2e19a9ac3ef16786f304bea834019a49f1ba50537e02af0ad068e6799386bd5f",
        )
        self.assertEqual(
            hashlib.sha256(csv_text.encode()).hexdigest(),
            "10dbc70122b36493398622b9e5882e1508468a762fb4d5d6d9b5ee7e2ba3eb29",
        )

    def test_builder_and_manuscript_preserve_scope(self):
        build = BUILD_PATH.read_text(encoding="utf-8")
        paper = PAPER_PATH.read_text(encoding="utf-8")
        for marker in (
            'run("H01 request-window execution motion"',
            '"retained_h01_request_window_motion"',
            '"complete_32_action_windows"',
            '"censored_windows"',
            '"censored_windows_by_episode_outcome"',
            '"matched_complete_window_analysis"',
            '"shared_complete_window_pairs"',
            '"complete_window_records_excluded_without_exact_counterpart_endpoint"',
            '"four_condition_same_request_index_support"',
            '"forecast_payload_bytes_reopened_or_reverified"',
            '"A6_movement_threshold_established"',
            '"maximum_core_behavioral_cells"',
        ):
            self.assertIn(marker, build)
        for marker in (
            "Retained H01 request-window audit", "632/738", "106 $+32$ targets",
            "100 goal-stopped successes", "six cap-censored failures",
            "length- and outcome-dependent census", "cube and relative motion remain separate",
            "No forecast payload was reopened, byte-reverified or scored", "not physical time",
            "completion of A3--A9",
        ):
            self.assertIn(marker, paper)


if __name__ == "__main__":
    unittest.main()
