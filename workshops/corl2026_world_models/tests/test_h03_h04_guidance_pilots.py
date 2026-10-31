"""Fail-closed checks for retained H03/H04 guidance-pilot packaging."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / "analysis" / "analyze_h03_h04_guidance_pilots.py"
JSON_PATH = PACKAGE / "paper" / "data" / "h03_h04_guidance_pilots.json"
CSV_PATH = PACKAGE / "paper" / "data" / "h03_h04_guidance_pilots.csv"
SPEC = importlib.util.spec_from_file_location("h03_h04_guidance_pilots", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class H03H04GuidancePilotTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.documents, cls.provenance = ANALYSIS.read_pinned_sources(ANALYSIS.ROOT)
        cls.report = ANALYSIS.analyze_documents(cls.documents, cls.provenance)

    def test_seven_exact_sources_recorded_heads_and_blob_commits_are_authenticated(self):
        self.assertEqual(self.provenance["commit"], ANALYSIS.PIN)
        self.assertEqual(self.provenance["files"], ANALYSIS.EXPECTED_SOURCES)
        self.assertEqual(self.provenance["recorded_heads"], ANALYSIS.RECORDED_HEADS)
        self.assertEqual(self.provenance["source_blob_commits"], ANALYSIS.SOURCE_BLOB_COMMITS)
        self.assertTrue(self.provenance["all_recorded_heads_are_ancestors_of_pin"])
        self.assertTrue(self.provenance["all_source_blob_commits_are_ancestors_of_pin"])
        self.assertEqual(
            self.provenance["post_result_freeze"],
            {
                "recorded_at_git_head": ANALYSIS.FREEZE_HEAD,
                "frozen_study_commit": ANALYSIS.FROZEN_STUDY_COMMIT,
                "recorded_head_is_direct_parent_of_frozen_study_commit": True,
            },
        )
        self.assertEqual(
            self.report["source_manifest"],
            {
                "authenticated_bytes": 810_860,
                "authenticated_file_count": 7,
                "authenticated_lines": 18_939,
                "compact_git_files_byte_rehashed_now": True,
                "exported_source_paths_are_git_object_paths_only": True,
                "raw_pvc_files_revalidated_now": False,
                "raw_source_paths_exported": False,
            },
        )

        current = ANALYSIS._git(ANALYSIS.ROOT, "rev-parse", "HEAD^{commit}").decode().strip()
        self.assertNotEqual(current, ANALYSIS.PIN)
        with mock.patch.dict(
            ANALYSIS.RECORDED_HEADS,
            {"v2a015_result_compilation": current},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "historical producer ancestry check failed"):
                ANALYSIS.read_pinned_sources(ANALYSIS.ROOT)
        with mock.patch.dict(
            ANALYSIS.SOURCE_BLOB_COMMITS,
            {ANALYSIS.COMPARISON: current},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "historical producer ancestry check failed"):
                ANALYSIS.read_pinned_sources(ANALYSIS.ROOT)

    def test_complete_population_stable_cell_trace_and_reset_identities(self):
        design = self.report["design"]
        self.assertEqual(design["historical_episode_rows_authenticated"], 24)
        self.assertEqual(design["retained_baseline_behavioral_episodes"], 12)
        self.assertEqual(design["new_v2a015_behavioral_episodes"], 12)
        self.assertEqual(design["historical_matched_baseline_intervention_cells"], 12)
        self.assertEqual(design["stable_family_configuration_seed_relation_identities"], 24)
        self.assertEqual(design["distinct_executed_action_trace_paths"], 24)
        self.assertEqual(design["distinct_executed_action_trace_sha256"], 24)
        self.assertEqual(design["physical_reset_fingerprints"], 1)
        self.assertEqual(design["physical_reset_fingerprint_sha256"], ANALYSIS.PHYSICAL_RESET_FINGERPRINT)
        self.assertFalse(design["environment_seed_labels_are_independent_physical_scenes"])
        self.assertEqual(len(self.report["matched_cell_contrasts"]), 12)
        self.assertEqual(len(self.report["matched_seed_diagnostics"]), 6)

        rows = self.report["matched_cell_contrasts"]
        cell_ids = {
            value
            for row in rows
            for value in (row["baseline_cell_id"], row["intervention_cell_id"])
        }
        path_ids = {
            value
            for row in rows
            for value in (
                row["baseline_action_trace_path_sha256"],
                row["intervention_action_trace_path_sha256"],
            )
        }
        trace_ids = {
            value
            for row in rows
            for value in (
                row["baseline_action_trace_sha256"],
                row["intervention_action_trace_sha256"],
            )
        }
        self.assertEqual(len(cell_ids), 24)
        self.assertEqual(len(path_ids), 24)
        self.assertEqual(len(trace_ids), 24)
        self.assertEqual({row["physical_initial_state_sha256"] for row in rows}, {ANALYSIS.PHYSICAL_RESET_FINGERPRINT})
        self.assertTrue(all("::seed" in cell_id and cell_id.count("::") == 3 for cell_id in cell_ids))
        self.assertEqual(
            {(row["cohort"], row["environment_seed_label"]) for row in self.report["matched_seed_diagnostics"]},
            {(cohort, seed) for cohort in ANALYSIS.COHORTS for seed in ANALYSIS.SEEDS},
        )

    def test_h03_counts_margins_transitions_and_accounting_reproduce(self):
        cohort = self.report["cohorts"]["H03"]
        self.assertEqual(cohort["model_id"], "cosmos3_nano_policy_droid")
        self.assertEqual(
            cohort["authenticated_population"],
            {
                "baseline_behavioral_episodes": 6,
                "intervention_behavioral_episodes": 6,
                "matched_configuration_cells": 6,
                "matched_seed_labels": 3,
                "missing_value_imputation": "none",
                "physical_reset_fingerprints": 1,
            },
        )
        self.assertEqual(cohort["successes"], {"baseline": 6, "intervention": 4})
        self.assertEqual(cohort["behavioral_request_action_accounting"], ANALYSIS.EXPECTED_BEHAVIORAL_ACCOUNTING["H03"])
        self.assertEqual(cohort["baseline"]["by_relation"]["left"]["successes"], 3)
        self.assertEqual(cohort["intervention"]["by_relation"]["left"]["successes"], 1)
        self.assertEqual(cohort["intervention"]["by_relation"]["right"]["successes"], 3)
        self.assertEqual(cohort["success_transitions"], {"regressed_success_to_failure": 2, "unchanged_success": 4})
        self.assertEqual(cohort["net_success_change"], -2)
        self.assertEqual(cohort["baseline"]["by_relation"]["left"]["mean_requested_margin_m"], 0.10322198768456776)
        self.assertEqual(cohort["intervention"]["by_relation"]["right"]["mean_requested_margin_m"], 0.215896712616086)

    def test_h04_counts_margins_transitions_and_accounting_reproduce(self):
        cohort = self.report["cohorts"]["H04"]
        self.assertEqual(cohort["model_id"], "dreamzero_droid_action_cfg")
        self.assertEqual(cohort["baseline_model_id"], "dreamzero_droid")
        self.assertEqual(cohort["successes"], {"baseline": 3, "intervention": 4})
        self.assertEqual(cohort["behavioral_request_action_accounting"], ANALYSIS.EXPECTED_BEHAVIORAL_ACCOUNTING["H04"])
        self.assertEqual(cohort["baseline"]["by_relation"]["left"]["successes"], 2)
        self.assertEqual(cohort["baseline"]["by_relation"]["right"]["successes"], 1)
        self.assertEqual(cohort["intervention"]["by_relation"]["left"]["successes"], 1)
        self.assertEqual(cohort["intervention"]["by_relation"]["right"]["successes"], 3)
        self.assertEqual(
            cohort["success_transitions"],
            {
                "improved_failure_to_success": 2,
                "regressed_success_to_failure": 1,
                "unchanged_failure": 1,
                "unchanged_success": 2,
            },
        )
        self.assertEqual(cohort["net_success_change"], 1)
        self.assertEqual(cohort["baseline"]["by_relation"]["right"]["mean_requested_margin_m"], 0.040088834861914315)
        self.assertEqual(cohort["intervention"]["by_relation"]["right"]["mean_requested_margin_m"], 0.21683129481971264)

    def test_requests_actions_censoring_and_historical_probe_accounting(self):
        self.assertEqual(
            self.report["historical_v2a015_request_accounting"],
            {
                "behavioral_model_action_requests": 309,
                "fixed_observation_action_requests": 9,
                "total_model_action_requests": 318,
                "setup_invalid_attempts": 16,
                "setup_invalid_attempts_in_behavioral_denominator": 0,
                "runtime_interventions": 0,
            },
        )
        for row in self.report["matched_cell_contrasts"]:
            for prefix in ("baseline", "intervention"):
                if row[f"{prefix}_success"]:
                    self.assertFalse(row[f"{prefix}_completion_right_censored"])
                    self.assertEqual(row[f"{prefix}_completion_time_actions"], row[f"{prefix}_actions_executed"])
                else:
                    self.assertTrue(row[f"{prefix}_completion_right_censored"])
                    self.assertEqual(row[f"{prefix}_actions_executed"], 450)
                    self.assertIsNone(row[f"{prefix}_completion_time_actions"])
        self.assertEqual(self.report["design"]["dreamzero_official_model_noise_seed"], 1140)
        self.assertFalse(self.report["design"]["dreamzero_noise_seed_varies_with_environment_seed_label"])

    def test_post_result_freeze_and_fixed_input_qualification_are_bounded(self):
        self.assertEqual(
            self.report["post_result_disclosure"],
            {
                "status": "frozen_after_known_baselines_before_v2a015_requests",
                "post_result_disclosed": True,
                "preregistered": False,
                "baseline_results_known_at_selection": True,
                "optional_followup_behavioral_episodes_authorized": 0,
            },
        )
        qualification = self.report["historical_fixed_input_qualification"]
        self.assertEqual(qualification["record_scope"], "authenticated_compact_historical_qualification_record")
        self.assertFalse(qualification["raw_fixed_observation_probe_bytes_present_in_package"])
        self.assertFalse(qualification["raw_fixed_observation_probe_bytes_replayed_now"])
        self.assertEqual(qualification["fixed_observation_action_requests_historically_reported"], 9)
        self.assertEqual(qualification["dreamzero_official_noise_seed"], 1140)
        self.assertEqual(qualification["dreamzero_s1_overlay_equivalence_status"], "passed_historically_not_replayed_now")
        self.assertEqual(qualification["dreamzero_s2_fixed_input_gate_status"], "passed_historically_not_replayed_now")
        reproduction = self.report["reproduction"]
        self.assertEqual(reproduction["authenticated_compact_git_files_rehashed"], 7)
        self.assertEqual(reproduction["authenticated_compact_git_bytes_rehashed"], 810_860)
        self.assertEqual(reproduction["authenticated_compact_git_lines_counted"], 18_939)
        self.assertFalse(reproduction["raw_fixed_observation_probe_bytes_replayed"])
        self.assertFalse(reproduction["external_model_source_commits_locally_authenticated"])
        self.assertEqual(reproduction["external_model_source_commit_literals"], ANALYSIS.EXTERNAL_MODEL_SOURCE_COMMITS)

    def test_strict_json_duplicate_nonfinite_and_root_type_rejection(self):
        for raw in (b'{"seed":1,"seed":2}', b'{"value":NaN}', b'{"value":1e999}', b'[]'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    ANALYSIS.parse_json(raw, label="test")
        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            ANALYSIS.render_json({"value": float("nan")})

    def test_freeze_qualification_schema_configuration_and_join_mutations_fail_closed(self):
        cases: list[tuple[dict, str]] = []

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.AMENDMENT]["status"] = "retroactive"
        cases.append((changed, "amendment freeze identity"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.AMENDMENT]["known_result_disclosure"]["statement"] = "preregistered"
        cases.append((changed, "known-result disclosure"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.AMENDMENT]["fixed_observation_release_gates"]["dreamzero_action_cfg_s2"][2] = "different noise"
        cases.append((changed, "fixed-input qualification contract"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.PREFLIGHT]["dreamzero_fixed_observation_gates"]["action_s1_overlay_equivalence"]["status"] = "unknown"
        cases.append((changed, "s1-overlay equivalence"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.PREFLIGHT]["inference_accounting"]["model_action_request_count"] = 317
        cases.append((changed, "inference accounting"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.NANO_BASELINE]["schema_version"] = "wrong"
        cases.append((changed, "H03 baseline identity"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.DREAMZERO_INTERVENTION]["configuration"]["action_cfg_style_scale"] = 1.0
        cases.append((changed, "H04 s2 intervention identity"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.NANO_INTERVENTION]["episodes"][0]["pair_id"] = "wrong"
        cases.append((changed, "static prompt/pair identity"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.DREAMZERO_INTERVENTION]["episodes"][0]["physical_initial_state_sha256"] = "0" * 64
        cases.append((changed, "physical-reset fingerprint boundary"))

        changed = copy.deepcopy(self.documents)
        failure = changed[ANALYSIS.NANO_INTERVENTION]["episodes"][0]
        failure["completion_actions_observed"] = 450
        cases.append((changed, "censored completion semantics"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.NANO_INTERVENTION]["episodes"][1]["executed_action_trace"] = copy.deepcopy(
            changed[ANALYSIS.NANO_INTERVENTION]["episodes"][0]["executed_action_trace"]
        )
        cases.append((changed, "trace path identity is not distinct"))

        changed = copy.deepcopy(self.documents)
        changed[ANALYSIS.COMPARISON]["comparisons"]["dreamzero"]["cells"][0]["physical_initial_state_sha256"] = "0" * 64
        cases.append((changed, "comparison prompt/state changed"))

        for changed, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ANALYSIS.analyze_documents(changed, self.provenance)

    def test_closed_activity_gates_and_scientific_boundaries(self):
        self.assertEqual(len(self.report["new_activity"]), 13)
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertEqual(len(self.report["gate_state"]), 9)
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        boundary = self.report["claim_boundary"]
        self.assertTrue(boundary["retained_historical_evidence_only"])
        self.assertTrue(boundary["exploratory_post_result_only"])
        self.assertTrue(boundary["post_result_selection_disclosed"])
        for key in (
            "powered_inference",
            "powered_or_general_performance_claim",
            "new_model_inference",
            "new_robot_episodes",
            "new_human_labels",
            "forecast_accuracy_evidence",
            "forecast_accuracy_claim",
            "fixed_video_guidance_intervention",
            "official_dreamzero_action_cfg_claim",
            "core_D1_qualification",
            "H04_s1_may_fill_core_D1",
            "d2_extension_selected",
            "independent_scene_generalization",
            "environment_seed_labels_claimed_as_independent_scenes",
            "fixed_noise_labels_claimed_as_independent_noise_replicates",
            "models_pooled",
            "confirmation_test_or_stage_gate",
            "changes_232_cell_core_bound",
            "raw_fixed_observation_probe_bytes_replayed_now",
            "historical_fixed_input_gate_is_current_requalification",
        ):
            with self.subTest(key=key):
                self.assertFalse(boundary[key])
        self.assertEqual(boundary["maximum_core_behavioral_cells"], 232)
        rendered = ANALYSIS.render_json(self.report)
        self.assertNotIn("/data/users/", rendered)
        self.assertNotIn("private half", rendered)

    def test_checked_in_outputs_and_cli_check_are_exact(self):
        rendered_json = ANALYSIS.render_json(self.report)
        rendered_csv = ANALYSIS.render_csv(self.report)
        self.assertEqual(JSON_PATH.read_text(encoding="utf-8"), rendered_json)
        self.assertEqual(CSV_PATH.read_text(encoding="utf-8"), rendered_csv)
        rows = list(csv.DictReader(io.StringIO(rendered_csv)))
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["cohort"] for row in rows}, {"H03", "H04"})
        self.assertEqual(
            hashlib.sha256(JSON_PATH.read_bytes()).hexdigest(),
            "24a28a49b4fd2dd2a0d79ce83f1506266e20702a71b997356a9902c62b2ac7b8",
        )
        self.assertEqual(
            hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
            "6d4225e4466b23511763da887629114e3e7ac4293b5bc5722fc8cc22ca8f845f",
        )
        result = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--check"],
            cwd=ANALYSIS.ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("12 paired cells", result.stdout)
        self.assertIn("7 authenticated Git objects", result.stdout)


if __name__ == "__main__":
    unittest.main()
