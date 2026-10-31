"""Fail-closed checks for retained H11/H12 Phase-A direct packaging."""
from __future__ import annotations

import copy
import csv
import importlib.util
import io
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "analyze_h11_h12_phase_a_direct.py"
)
SPEC = importlib.util.spec_from_file_location("h11_h12_phase_a_direct", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class H11H12PhaseADirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.provenance = ANALYSIS.read_pinned_sources(ANALYSIS.ROOT)
        cls.report = ANALYSIS.analyze_documents(cls.documents, cls.provenance)

    def test_authenticated_population_accounting_and_exact_effects(self):
        source = self.report["source_manifest"]
        self.assertEqual(source["authenticated_file_count"], 10)
        self.assertEqual(source["authenticated_bytes"], 5_654_967)
        self.assertEqual(source["authenticated_lines"], 101_058)
        self.assertEqual(source["h11_h12_payload_file_count"], 8)
        self.assertEqual(source["h11_h12_payload_bytes"], 5_636_174)
        self.assertEqual(source["h11_h12_payload_lines"], 100_675)
        self.assertEqual(source["design_and_overlap_helper_file_count"], 2)
        self.assertEqual(source["design_and_overlap_helper_bytes"], 18_793)
        self.assertEqual(source["design_and_overlap_helper_lines"], 383)
        self.assertEqual(source["h11_h12_payload_file_count"], 8)
        self.assertEqual(source["h11_h12_payload_bytes"], 5_636_174)
        self.assertEqual(source["h11_h12_payload_lines"], 100_675)
        self.assertEqual(source["design_and_overlap_helper_file_count"], 2)
        self.assertEqual(source["design_and_overlap_helper_bytes"], 18_793)
        self.assertEqual(source["design_and_overlap_helper_lines"], 383)
        self.assertTrue(source["prospective_freeze_ancestry_verified"])
        self.assertEqual(self.report["provenance"]["commit"], ANALYSIS.PIN)
        self.assertEqual(
            self.report["provenance"]["prospective_freeze_commit"],
            ANALYSIS.PROSPECTIVE_FREEZE_COMMIT,
        )
        self.assertEqual(
            self.report["provenance"]["result_commits"],
            {
                "H11": "150a362701656a3992c9738f497f1fd09605dc54",
                "H12": "8651aef704f6369b5b514d17ba7fd4bd0913b8d3",
            },
        )
        self.assertTrue(self.report["provenance"]["result_commits_are_ancestors_of_pinned_evidence"])
        self.assertTrue(self.report["provenance"]["result_blobs_unchanged_at_pin"])
        self.assertEqual(
            self.report["design"]["h06_h07_h08_canonical_core_seed_labels"],
            list(range(9400, 9427)),
        )
        self.assertEqual(self.report["design"]["h11_h12_overlap_with_h06_h07_h08_canonical_core"], [])
        self.assertEqual(
            self.report["reproduction"]["h11_h12_identity_overlap_with_h06_h07_h08_canonical_core"],
            0,
        )
        self.assertEqual(len(self.report["matched_seed_pairs"]), 54)

        expected = {
            "H11": {
                "model": "cosmos3_nano_policy_droid",
                "successes": {"left": 26, "right": 25},
                "requests": {"left": 179, "right": 170},
                "failures": 3,
                "taxonomy": {"correct": 51, "release_failed": 2, "transport_failed": 1},
                "success_effect": -1 / 27,
                "discordance": (24, 2, 1, 0),
                "mcnemar": 1.0,
                "endpoint": -0.4090210889914521,
                "endpoint_median": -0.4119686782360077,
                "ordering": (27, 0),
                "sign_p": 1.4901161193847656e-08,
                "margin": 0.12360139639565239,
            },
            "H12": {
                "model": "dreamzero_droid_action_cfg",
                "successes": {"left": 3, "right": 17},
                "requests": {"left": 1_470, "right": 1_084},
                "failures": 34,
                "taxonomy": {"correct": 20, "pick_failed": 20, "transport_failed": 14},
                "success_effect": 14 / 27,
                "discordance": (1, 2, 16, 8),
                "mcnemar": 0.001312255859375,
                "endpoint": -0.14558204704964603,
                "endpoint_median": -0.15802326053380966,
                "ordering": (25, 2),
                "sign_p": 5.647540092468262e-06,
                "margin": 0.0766849465392254,
            },
        }
        for cohort, values in expected.items():
            actual = self.report["cohorts"][cohort]
            population = actual["authenticated_population"]
            self.assertEqual(actual["model_id"], values["model"])
            self.assertEqual(population["behavioral_episodes"], 54)
            self.assertEqual(population["matched_pairs"], 27)
            self.assertEqual(population["valid_behavioral_failures"], values["failures"])
            self.assertEqual(population["preserved_v2_candidate_episodes_excluded"], 6)
            self.assertEqual(population["historical_v2_baseline_episodes_counted"], 0)
            self.assertEqual(actual["overall_failure_taxonomy"], values["taxonomy"])
            for relation in ANALYSIS.RELATIONS:
                self.assertEqual(actual["directional"][relation]["behavioral_episodes"], 27)
                self.assertEqual(actual["directional"][relation]["successes"], values["successes"][relation])
                self.assertEqual(actual["directional"][relation]["model_requests"], values["requests"][relation])
                directional = actual["directional"][relation]
                self.assertNotIn("official_or_per_request_decodes", directional)
                if cohort == "H11":
                    self.assertEqual(directional["decoded_rgb_future_records"], values["requests"][relation])
                    self.assertEqual(directional["official_full_reset_decodes"], 0)
                else:
                    self.assertEqual(directional["decoded_rgb_future_records"], 0)
                    self.assertEqual(directional["official_full_reset_decodes"], 27)
            self.assertEqual(population["model_requests"], sum(values["requests"].values()))
            success = actual["paired_estimands"]["success_right_minus_left"]
            self.assertAlmostEqual(success["mean"], values["success_effect"])
            discordance = success["matched_discordance"]
            self.assertEqual(
                (
                    discordance["both_success"], discordance["left_only_success"],
                    discordance["right_only_success"], discordance["neither_success"],
                ),
                values["discordance"],
            )
            self.assertAlmostEqual(success["exact_two_sided_mcnemar_p"], values["mcnemar"])
            endpoint = actual["paired_estimands"]["endpoint_right_minus_left_m"]
            self.assertAlmostEqual(endpoint["mean"], values["endpoint"])
            self.assertAlmostEqual(endpoint["median"], values["endpoint_median"])
            self.assertEqual(
                (endpoint["aligned_negative"], endpoint["anti_aligned_positive"]),
                values["ordering"],
            )
            self.assertAlmostEqual(endpoint["exact_two_sided_sign_p"], values["sign_p"])
            margin = actual["paired_estimands"]["requested_margin_right_minus_left_m"]
            self.assertAlmostEqual(margin["mean"], values["margin"])
            self.assertTrue(margin["algebraically_distinct_from_endpoint_shift"])

    def test_request_latent_decode_semantics_are_separate_and_exact(self):
        nano = self.report["cohorts"]["H11"]["future_accounting"]
        self.assertEqual(nano["model_requests"], 349)
        self.assertEqual(nano["decoded_rgb_future_records"], 349)
        self.assertEqual(nano["returned_action_chunks"], 349)
        self.assertEqual(nano["retained_latent_future_records"], 0)
        self.assertEqual(nano["official_full_reset_decodes"], 0)
        self.assertFalse(nano["forecast_accuracy_scored"])

        dream = self.report["cohorts"]["H12"]["future_accounting"]
        self.assertEqual(dream["model_requests"], 2_554)
        self.assertEqual(dream["decoded_rgb_future_records"], 0)
        self.assertEqual(dream["retained_latent_future_records"], 2_554)
        self.assertEqual(dream["official_full_reset_decodes"], 54)
        self.assertEqual(dream["actual_rollout_videos"], 54)
        self.assertFalse(dream["latent_records_are_per_request_decoded_rgb"])
        self.assertFalse(dream["forecast_accuracy_scored"])
        self.assertIn("custom DreamZero action-guidance s2", self.report["cohorts"]["H12"]["configuration_boundary"])
        self.assertIn("not default DreamZero", self.report["cohorts"]["H12"]["configuration_boundary"])

    def test_strict_json_duplicate_nonfinite_and_canonical_rendering(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            ANALYSIS.parse_json(b'{"x":1,"x":2}', label="duplicate")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":NaN}', label="nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":1e999}', label="overflow")
        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            ANALYSIS.render_json({"x": float("nan")})

    def test_registry_scope_grid_and_no_baseline_double_count_fail_closed(self):
        malformed = copy.deepcopy(self.documents)
        malformed["ablation_spec"] = malformed["ablation_spec"].replace(
            "54 new episodes, excluding the six historical V2 baseline episodes. 51/54 successes.",
            "60 episodes",
        )
        with self.assertRaisesRegex(ValueError, "scope binding"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["cells"] = malformed["cells"][:-1]
        with self.assertRaisesRegex(ValueError, "exactly 780"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["cells"][1]["cell_id"] = malformed["cells"][0]["cell_id"]
        with self.assertRaisesRegex(ValueError, "duplicate Phase-A registry cell"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        selected = next(
            row for row in malformed["cells"]
            if row["model_id"] == "cosmos3_nano_policy_droid" and row["status"] == "authorized_new"
        )
        selected["status"] = "preserved_candidate"
        with self.assertRaisesRegex(ValueError, "registry status counts|preserved/new split"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"]["cosmos3_nano_policy_droid"]["summary"]["cells"][0]["seed"] = 8300
        with self.assertRaisesRegex(ValueError, "invalid or duplicate compact episode"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

    def test_episode_schema_join_nonfinite_and_request_mutations_fail_closed(self):
        nano = "cosmos3_nano_policy_droid"
        dream = "dreamzero_droid_action_cfg"
        malformed = copy.deepcopy(self.documents)
        del malformed["models"][nano]["summary"]["cells"][0]["measurements"]["signed_final_lateral_offset_m"]
        with self.assertRaisesRegex(ValueError, "measurements.*schema changed"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][nano]["summary"]["cells"][0]["measurements"]["final_requested_signed_margin_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite number"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][nano]["summary"]["cells"][-1] = copy.deepcopy(
            malformed["models"][nano]["summary"]["cells"][0]
        )
        with self.assertRaisesRegex(ValueError, "invalid or duplicate compact episode"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][nano]["summary"]["cells"][0]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "episode/registry mismatch"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][nano]["summary"]["cells"][0]["future_request_count"] = True
        with self.assertRaisesRegex(ValueError, "future_request_count must be a positive integer"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][dream]["summary"]["cells"][0]["policy_request_count"] = 0
        with self.assertRaisesRegex(ValueError, "policy_request_count must be a positive integer"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][dream]["summary"]["cells"][0]["latent_future_count"] += 1
        with self.assertRaisesRegex(ValueError, "request/latent/reset-decode accounting"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

    def test_evidence_and_infrastructure_accounting_mutations_fail_closed(self):
        nano = "cosmos3_nano_policy_droid"
        dream = "dreamzero_droid_action_cfg"
        malformed = copy.deepcopy(self.documents)
        malformed["models"][nano]["evidence"]["pairs"][0]["cells"]["left"]["future_requests"].pop()
        with self.assertRaisesRegex(ValueError, "evidence request accounting"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][nano]["evidence"]["verification"]["decoded_future_count"] += 1
        with self.assertRaisesRegex(ValueError, "decoded-future/action-chunk totals"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][dream]["evidence"]["pairs"][0]["cells"][0]["policy_requests"].pop()
        with self.assertRaisesRegex(ValueError, "evidence request accounting"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][dream]["evidence"]["pairs"][0]["cells"][0]["official_reset_decodes"].clear()
        with self.assertRaisesRegex(ValueError, "reset-decode accounting"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][dream]["infrastructure"]["excluded_attempts"][0]["model_request_count"] = 1
        with self.assertRaisesRegex(ValueError, "excluded infrastructure attempt entered behavior"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

    def test_exact_regeneration_separation_closed_gates_and_no_raw_paths(self):
        rendered_json = ANALYSIS.render_json(self.report)
        rendered_csv = ANALYSIS.render_csv(self.report)
        self.assertEqual(ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"), rendered_json)
        self.assertEqual(ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"), rendered_csv)
        rows = list(csv.DictReader(io.StringIO(rendered_csv)))
        self.assertEqual(len(rows), 54)
        self.assertEqual({row["cohort"] for row in rows}, {"H11", "H12"})
        self.assertEqual({row["model_id"] for row in rows}, set(ANALYSIS.MODELS))
        self.assertEqual(
            {(row["cohort"], int(row["seed"])) for row in rows},
            {(cohort, seed) for cohort in ("H11", "H12") for seed in ANALYSIS.SEEDS},
        )
        self.assertNotIn("left_decode_count", rows[0])
        self.assertNotIn("right_decode_count", rows[0])
        for row in rows:
            if row["cohort"] == "H11":
                self.assertGreater(int(row["left_decoded_rgb_future_records"]), 0)
                self.assertGreater(int(row["right_decoded_rgb_future_records"]), 0)
                self.assertEqual(int(row["left_official_full_reset_decodes"]), 0)
                self.assertEqual(int(row["right_official_full_reset_decodes"]), 0)
            else:
                self.assertEqual(int(row["left_decoded_rgb_future_records"]), 0)
                self.assertEqual(int(row["right_decoded_rgb_future_records"]), 0)
                self.assertEqual(int(row["left_official_full_reset_decodes"]), 1)
                self.assertEqual(int(row["right_official_full_reset_decodes"]), 1)
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        boundary = self.report["claim_boundary"]
        self.assertTrue(boundary["historical_behavior_and_recovery_source_only"])
        self.assertFalse(boundary["new_model_inference"])
        self.assertFalse(boundary["new_robot_episodes"])
        self.assertFalse(boundary["forecast_accuracy_evidence"])
        self.assertFalse(boundary["independent_scene_claim"])
        self.assertFalse(boundary["annotation_evidence"])
        self.assertFalse(boundary["equivalence_claim"])
        self.assertFalse(boundary["confirmation_test_or_stage_gate"])
        self.assertFalse(boundary["models_pooled"])
        self.assertFalse(boundary["historical_v2_baselines_double_counted"])
        self.assertFalse(boundary["changes_232_cell_core_bound"])
        self.assertEqual(boundary["maximum_core_behavioral_cells"], 232)
        self.assertFalse(self.report["source_manifest"]["raw_pvc_files_revalidated_now"])
        self.assertFalse(self.report["source_manifest"]["raw_source_paths_exported"])
        for forbidden in ("/data/users/", "kubernetes_context", "namespace", "pod", "checkpoint_path"):
            self.assertNotIn(forbidden, rendered_json)

    def test_manuscript_documentation_and_builder_integration(self):
        package = ANALYSIS.PACKAGE
        paper = (package / "paper" / "main.tex").read_text(encoding="utf-8")
        readme = (package / "README.md").read_text(encoding="utf-8")
        delivery = (package / "docs" / "DELIVERY_QA.md").read_text(encoding="utf-8")
        build = (package / "analysis" / "build_package.py").read_text(encoding="utf-8")
        for required in (
            "H06--H12 retained controls",
            "H11/H12 are separate fixed-scene Phase-A direct-command cohorts",
            "Nano 51/54; DreamZero-s2 3/27 LEFT and 17/27 RIGHT",
            "2,554 retained futures are latents, not per-request RGB decodes",
            "H12's totals match H07 numerically but use distinct seeds and are not a replication",
            "H05--H12 reuse fixed scenes or matched seeds",
        ):
            with self.subTest(document="paper", required=required):
                self.assertIn(required, paper)
        for required in (
            "The retained H11/H12 Phase-A direct-command package",
            "h11_h12_phase_a_direct.json",
            "h11_h12_phase_a_direct.csv",
            "eight H11/H12 payload objects total 5,636,174 bytes",
            "ten objects and 5,654,967 bytes total",
            "349 historical requests each\nretain a decoded RGB future and returned action chunk",
            "2,554 latent futures under fixed model-noise seed 1140",
            "Six preserved V2 cells per model\nat seeds 8300--8302 are evidence-only and excluded",
            "numerical coincidence across\nidentity-distinct cohorts",
            "the 232-cell core bound unchanged",
            "H05--H12 replays",
        ):
            with self.subTest(document="README", required=required):
                self.assertIn(required, readme)
        for required in (
            "The retained H11/H12 Phase-A package",
            "authenticated boundary to ten objects and 5,654,967 bytes",
            "54 official full-reset decodes and 54 rollout videos",
            "No raw recovery, forecast accuracy",
            "H11/H12 each reuse 27 neutral-scene reset labels",
        ):
            with self.subTest(document="DELIVERY_QA", required=required):
                self.assertIn(required, delivery)
        self.assertLess(
            build.index('run("H09 H10 retained wording control"'),
            build.index('run("H11 H12 retained Phase A direct commands"'),
        )
        self.assertLess(
            build.index('run("H11 H12 retained Phase A direct commands"'),
            build.index('run("representative H01 media"'),
        )
        for required in (
            "analysis/analyze_h11_h12_phase_a_direct.py",
            "paper/data/h11_h12_phase_a_direct.json",
            '"retained_h11_h12_phase_a": {',
            '"matched_seed_pair_rows": len(h11_h12["matched_seed_pairs"])',
            '"paper_cohorts": ["nano_reflection", "dreamzero_reflection"]',
            '"actual_rollout_videos"] != 54',
            '"maximum_core_behavioral_cells"] != 232',
        ):
            with self.subTest(document="build", required=required):
                self.assertIn(required, build)


if __name__ == "__main__":
    unittest.main()
