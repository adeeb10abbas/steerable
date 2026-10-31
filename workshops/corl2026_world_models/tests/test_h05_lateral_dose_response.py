"""Fail-closed tests for the authenticated retained H05 dose response."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / "analysis" / "analyze_h05_lateral_dose_response.py"
JSON_PATH = PACKAGE / "paper" / "data" / "h05_lateral_dose_response.json"
CSV_PATH = PACKAGE / "paper" / "data" / "h05_lateral_dose_response.csv"
SPEC = importlib.util.spec_from_file_location("h05_lateral_dose_response", MODULE_PATH)
H05 = importlib.util.module_from_spec(SPEC) if MODULE_PATH.exists() else None
if H05 is not None:
    SPEC.loader.exec_module(H05)


class H05LateralDoseResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if H05 is None:
            raise AssertionError("The H05 retained-dose-response analyzer must be implemented")
        cls.documents, cls.provenance = H05.read_pinned_sources(H05.ROOT)
        cls.report = H05.analyze_documents(cls.documents, cls.provenance)

    def test_exact_pinned_source_identities_and_complete_design(self):
        self.assertEqual(self.provenance, {"commit": H05.PIN, "files": H05.EXPECTED_SOURCES})
        self.assertEqual(len(self.documents["cells"]), 210)
        self.assertEqual(
            self.report["design"],
            {
                "study_id": "vla_wam_language_steerability_v3",
                "amendment_id": "V3-B005",
                "model_id": "cosmos3_nano_policy_droid",
                "matched_seed_count": 15,
                "controlled_level_count": 7,
                "relations": ["left", "right"],
                "registered_levels_m": list(H05.LEVELS),
                "registered_behavioral_episode_ceiling": 210,
                "inferential_unit": "matched_seed",
            },
        )
        self.assertEqual(len(self.report["matched_seed_levels"]), 105)
        self.assertEqual(
            {(row["seed"], row["level_index"]) for row in self.report["matched_seed_levels"]},
            {(seed, level) for seed in H05.SEEDS for level in range(7)},
        )

    def test_primary_registered_slope_bootstrap_signs_and_support(self):
        primary = self.report["primary_depth_dose_response"]
        self.assertEqual(primary["mean"], 1.1247301174859916)
        self.assertEqual(primary["median"], 1.2092121477637972)
        self.assertEqual(primary["ci95"], [0.7189605736750224, 1.562330189504705])
        self.assertEqual(
            primary["sign_test"],
            {
                "positive": 13,
                "negative": 2,
                "ties": 0,
                "non_tied_n": 15,
                "two_sided_p": 0.00738525390625,
                "method": "exact_two_sided_paired_sign_test_zero_ties_excluded",
            },
        )
        crossing = primary["linear_zero_crossing"]
        self.assertAlmostEqual(crossing["unclipped_m"], 0.01878784442645)
        self.assertFalse(crossing["in_registered_support"])
        self.assertIsNone(crossing["reported_value_m"])
        self.assertEqual(crossing["registered_support_m"], [H05.LEVELS[0], H05.LEVELS[-1]])

    def test_secondary_success_result_remains_secondary(self):
        population = self.report["authenticated_historical_population"]
        self.assertEqual(population["left_successes"], 93)
        self.assertEqual(population["right_successes"], 99)
        self.assertEqual(population["infrastructure_invalid_cells"], 0)
        secondary = self.report["binary_success_secondary"]
        self.assertEqual(secondary["mean"], 1.8253968253968256)
        self.assertEqual(secondary["ci95"], [0.5555555555555556, 3.253968253968254])
        self.assertEqual(secondary["median"], 0.0)
        self.assertEqual(
            {key: secondary["sign_test"][key] for key in ("positive", "negative", "ties", "two_sided_p")},
            {"positive": 7, "negative": 1, "ties": 7, "two_sided_p": 0.0703125},
        )

    def test_level_means_are_not_promoted_to_monotonicity_or_zero_imputation(self):
        levels = self.report["by_level"]
        means = [row["depth_contrast_mean_m"] for row in levels]
        self.assertEqual(
            means,
            [
                0.013066606099406878,
                0.09222178235650062,
                0.10299765734622876,
                0.09795385127266248,
                0.09205955316623052,
                0.17177735889951387,
                0.2786000226934751,
            ],
        )
        self.assertFalse(all(left <= right for left, right in zip(means, means[1:])))
        conditional_n = [row["both_success_conditional_pair_n"] for row in levels]
        self.assertEqual(conditional_n, [13, 14, 14, 14, 13, 11, 10])
        self.assertEqual(sum(conditional_n), 89)
        self.assertIn("never encodes a failed direction as zero", " ".join(self.report["claim_boundary"]["boundaries"]))

    def test_source_manifest_and_historical_audit_boundaries_are_explicit(self):
        self.assertEqual(
            self.report["source_manifest"],
            {
                "pair_source_identities": 105,
                "episode_source_identities": 210,
                "unique_source_sha256": 315,
                "raw_source_bytes_present_in_checkout": False,
                "raw_source_bytes_rehashed_by_this_reproduction": False,
            },
        )
        self.assertEqual(self.report["historical_final_audit"]["mismatch_count"], 0)
        boundary = self.report["claim_boundary"]
        for key in (
            "new_model_inference",
            "new_robot_episodes",
            "new_human_labels",
            "forecast_accuracy_evidence",
            "raw_pair_and_episode_bytes_revalidated_now",
            "matched_seeds_are_independent_scenes",
            "controlled_positions_are_independent_scenes",
            "confirmation_test_or_stage_gate",
            "changes_232_cell_core_bound",
        ):
            with self.subTest(key=key):
                self.assertFalse(boundary[key])

    def test_historical_counts_are_disjoint_from_current_activity_and_gates(self):
        self.assertEqual(
            self.report["new_activity"],
            {
                "model_requests": 0,
                "behavioral_actions": 0,
                "behavioral_cells": 0,
                "request_rows": 0,
                "request_selections": 0,
                "annotation_media": 0,
                "human_responses": 0,
                "labels": 0,
                "forecast_accuracy_estimates": 0,
                "retrieval_jobs": 0,
                "queue_descriptors": 0,
                "runnable_manifests": 0,
                "confirmation_inference_cells": 0,
            },
        )
        self.assertEqual(
            self.report["gate_state"],
            {
                "current_GM_PVC_worker_state_reconciled": False,
                "live_quiescence_inferred": False,
                "camera_alignment_ready": False,
                "development_annotation_gate_ready": False,
                "annotation_media_ready": False,
                "human_annotation_ready": False,
                "resource_freeze_ready": False,
                "confirmation_release_ready": False,
                "safe_to_release_new_work": False,
            },
        )
        self.assertIn(
            "authenticated historical-population counts, not current-turn activity",
            " ".join(self.report["claim_boundary"]["boundaries"]),
        )

    def test_strict_json_and_semantic_mutations_fail_closed(self):
        for raw in (b'{"seed": 1, "seed": 2}', b'{"value": NaN}', b'[]'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    H05.parse_json(raw, label="test")

        cases = []
        duplicate_cell = copy.deepcopy(self.documents)
        duplicate_cell["cells"][-1] = copy.deepcopy(duplicate_cell["cells"][0])
        cases.append(duplicate_cell)

        missing_seed = copy.deepcopy(self.documents)
        missing_seed["report"]["seed_level"].pop()
        cases.append(missing_seed)

        forged_slope = copy.deepcopy(self.documents)
        forged_slope["report"]["seed_level"][0]["depth_slope_m_per_m"] = 0.0
        cases.append(forged_slope)

        bad_source = copy.deepcopy(self.documents)
        bad_source["source_manifest"]["pair_sources"][0]["path"] = "/unregistered/pair.json"
        cases.append(bad_source)

        nonfinite_depth = copy.deepcopy(self.documents)
        nonfinite_depth["report"]["seed_level"][0]["depth_contrast_by_level_m"][0] = float("nan")
        cases.append(nonfinite_depth)

        for index, changed in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    H05.analyze_documents(changed, self.provenance)

    def test_checked_in_outputs_are_exact_reproductions(self):
        self.assertEqual(JSON_PATH.read_text(encoding="utf-8"), H05.render_json(self.report))
        self.assertEqual(CSV_PATH.read_text(encoding="utf-8"), H05.render_csv(self.report))
        self.assertEqual(len(CSV_PATH.read_text(encoding="utf-8").splitlines()), 106)
        self.assertEqual(
            hashlib.sha256(JSON_PATH.read_bytes()).hexdigest(),
            "754391cf49021912171d6341237d20aa4523a94db22927df6d26d0b0493a2b6c",
        )
        self.assertEqual(
            hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
            "767b8cc102496407191584894a516f59e6b2baed9ea0ca01aef41dc9890364fd",
        )

    def test_manuscript_readme_and_build_keep_h05_boundaries(self):
        paper = (PACKAGE / "paper/main.tex").read_text(encoding="utf-8")
        for required in (
            "H05 dose response",
            "seven fixed bowl-$y$ levels span 18\\,cm",
            "1.125\\,m/m",
            "13/15 positive",
            "Level means were not monotone",
            "outside the registered 0.0366--0.2166\\,m support",
            "not new-scene generalization or forecast accuracy",
            "not seven independent scenes",
        ):
            with self.subTest(required=required):
                self.assertIn(required, paper)

        readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
        for required in (
            "retained H05 position sweep",
            "paper/data/h05_lateral_dose_response.json",
            "paper/data/h05_lateral_dose_response.csv",
            "this is not a fresh raw-trace audit",
            "no forecast-accuracy or annotation\ngate evidence",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)

        build = (PACKAGE / "analysis/build_package.py").read_text(encoding="utf-8")
        for required in (
            'run("H05 retained lateral dose response"',
            "analysis/analyze_h05_lateral_dose_response.py",
            "paper/data/h05_lateral_dose_response.json",
            '"retained_h05_lateral_dose_response": {',
            '"new_activity": h05["new_activity"]',
            '"gate_state": h05["gate_state"]',
            '"csv_sha256": digest(package / "paper/data/h05_lateral_dose_response.csv")',
            '"analyzer_sha256": digest(package / "analysis/analyze_h05_lateral_dose_response.py")',
        ):
            with self.subTest(required=required):
                self.assertIn(required, build)


if __name__ == "__main__":
    unittest.main()
