"""Fail-closed checks for retained H09/H10 wording-control packaging."""
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
    / "analyze_h09_h10_wording_control.py"
)
SPEC = importlib.util.spec_from_file_location("h09_h10_wording_control", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class H09H10WordingControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.provenance = ANALYSIS.read_pinned_sources(ANALYSIS.ROOT)
        cls.report = ANALYSIS.analyze_documents(cls.documents, cls.provenance)

    def test_authenticated_population_and_exact_descriptive_contrasts(self):
        self.assertEqual(self.report["source_manifest"]["authenticated_file_count"], 11)
        self.assertEqual(self.report["source_manifest"]["authenticated_bytes"], 1_814_647)
        self.assertEqual(self.report["design"]["full_prospective_registration"]["behavioral_cells"], 480)
        self.assertTrue(self.report["design"]["full_prospective_registration"]["includes_out_of_scope_groot"])
        self.assertEqual(self.report["design"]["ablation_spec_h09_h10_slice"]["behavioral_episodes"], 320)
        self.assertEqual(len(self.report["matched_seed_contrasts"]), 120)

        expected = {
            "H09": {
                "model": "cosmos3_nano_policy_droid",
                "direct": 39,
                "requests": 1234,
                "failures": 26,
                "taxonomy": {"correct": 134, "pick_failed": 2, "release_failed": 0, "transport_failed": 18, "wrong_side": 6},
                "request_grid": {
                    "direct_command": {"left": 108, "right": 125},
                    "short_command": {"left": 111, "right": 125},
                    "goal_as_outcome": {"left": 140, "right": 166},
                    "desired_plus_negated_opposite": {"left": 232, "right": 227},
                },
                "short_command": (0.0, -0.019251487543806435, 0.03850297508761287),
                "goal_as_outcome": (-0.1, -0.06020642803050578, 0.12041285606101156),
                "desired_plus_negated_opposite": (-0.45, -0.19390022521838546, 0.3878004504367709),
            },
            "H10": {
                "model": "cosmos3_edge_policy_droid",
                "direct": 38,
                "requests": 1365,
                "failures": 37,
                "taxonomy": {"correct": 123, "pick_failed": 11, "release_failed": 0, "transport_failed": 14, "wrong_side": 12},
                "request_grid": {
                    "direct_command": {"left": 175, "right": 114},
                    "short_command": {"left": 274, "right": 131},
                    "goal_as_outcome": {"left": 162, "right": 136},
                    "desired_plus_negated_opposite": {"left": 243, "right": 130},
                },
                "short_command": (-0.35, -0.13170764790847897, 0.26341529581695794),
                "goal_as_outcome": (-0.125, -0.021546873403713106, 0.04309374680742621),
                "desired_plus_negated_opposite": (-0.25, -0.10842464661691338, 0.21684929323382676),
            },
        }
        for cohort, values in expected.items():
            actual = self.report["cohorts"][cohort]
            self.assertEqual(actual["model_id"], values["model"])
            self.assertEqual(actual["authenticated_population"]["behavioral_episodes"], 160)
            self.assertEqual(actual["authenticated_population"]["model_requests"], values["requests"])
            self.assertEqual(actual["authenticated_population"]["valid_behavioral_failures"], values["failures"])
            self.assertEqual(actual["overall_failure_taxonomy"], values["taxonomy"])
            self.assertEqual(actual["condition_summaries"]["direct_command"]["both_directions"]["successes"], values["direct"])
            for prompt, by_relation in values["request_grid"].items():
                for relation, requests in by_relation.items():
                    self.assertEqual(actual["condition_summaries"][prompt][relation]["model_requests"], requests)
                self.assertEqual(
                    actual["condition_summaries"][prompt]["both_directions"]["model_requests"],
                    sum(by_relation.values()),
                )
            for prompt in ANALYSIS.ALTERNATIVES:
                contrast = actual["same_seed_direct_control_contrasts"][prompt]
                success, margin, endpoint = values[prompt]
                self.assertAlmostEqual(contrast["success"]["mean"], success)
                self.assertAlmostEqual(contrast["requested_margin_m"]["mean"], margin)
                self.assertAlmostEqual(contrast["right_minus_left_endpoint_shift_m"]["mean"], endpoint)
                self.assertAlmostEqual(margin, -0.5 * endpoint)
                self.assertTrue(contrast["algebraic_dependency"]["validated_for_all_20_seed_blocks"])
                self.assertTrue(contrast["algebraic_dependency"]["metrics_are_not_independent_estimands"])
                self.assertEqual(contrast["inference"], "descriptive_only_not_registered")

    def test_strict_json_duplicate_and_nonfinite_rejection(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            ANALYSIS.parse_json(b'{"x":1,"x":2}', label="duplicate")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":NaN}', label="nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":1e999}', label="overflow")
        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            ANALYSIS.render_json({"x": float("nan")})

    def test_registration_full_grid_and_ablation_scope_mutations_fail_closed(self):
        malformed = copy.deepcopy(self.documents)
        malformed["registration"]["analysis_status"]["confirmatory_inference"] = "registered"
        with self.assertRaisesRegex(ValueError, "exploratory analysis status"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["ablation_spec"] = malformed["ablation_spec"].replace(
            "160 episodes: 20 blocks × four phrasings × two commands; 134/160 successes.",
            "159 episodes",
        )
        with self.assertRaisesRegex(ValueError, "scope binding"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["cells"] = malformed["cells"][:-1]
        with self.assertRaisesRegex(ValueError, "exactly 480 cells"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["cells"][0]["direct_command_within_seed_control"] = True
        with self.assertRaisesRegex(ValueError, "direct-command control flag"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

    def test_episode_duplicate_nonfinite_schema_and_join_mutations_fail_closed(self):
        model = "cosmos3_nano_policy_droid"
        malformed = copy.deepcopy(self.documents)
        malformed["models"][model]["episodes"][-1] = copy.deepcopy(
            malformed["models"][model]["episodes"][0]
        )
        with self.assertRaisesRegex(ValueError, "duplicate compact episode"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][model]["episodes"][0]["measurements"]["final_requested_signed_margin_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite number"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        del malformed["models"][model]["episodes"][0]["measurements"]["signed_final_lateral_offset_m"]
        with self.assertRaisesRegex(ValueError, "measurements.*schema changed"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][model]["episodes"][0]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "episode/cell mismatch"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][model]["episodes"][0]["model_request_count"] = True
        with self.assertRaisesRegex(ValueError, "model_request_count must be a positive integer"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

    def test_summary_and_evidence_identity_mutations_fail_closed(self):
        model = "cosmos3_edge_policy_droid"
        malformed = copy.deepcopy(self.documents)
        condition = malformed["models"][model]["summary"]["success_by_condition"]["short_command:left"]
        condition["successes"] += 1
        with self.assertRaisesRegex(ValueError, "success count changed"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][model]["summary"]["counts"]["model_requests"] += 1
        with self.assertRaisesRegex(ValueError, "source summary count changed: model_requests"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["models"][model]["summary"]["failure_taxonomy_overall"]["wrong_side"] += 1
        with self.assertRaisesRegex(ValueError, "failure taxonomy changed"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        name = Path(ANALYSIS.MODELS[model]["episodes"]).name
        malformed["models"][model]["evidence"]["outputs"][name]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "evidence output identity"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

    def test_exact_regeneration_no_pooling_and_closed_gates(self):
        rendered_json = ANALYSIS.render_json(self.report)
        rendered_csv = ANALYSIS.render_csv(self.report)
        self.assertEqual(ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"), rendered_json)
        self.assertEqual(ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"), rendered_csv)
        rows = list(csv.DictReader(io.StringIO(rendered_csv)))
        self.assertEqual(len(rows), 120)
        self.assertEqual({row["cohort"] for row in rows}, {"H09", "H10"})
        self.assertEqual({row["model_id"] for row in rows}, set(ANALYSIS.MODELS))
        self.assertEqual({row["alternative_prompt_family"] for row in rows}, set(ANALYSIS.ALTERNATIVES))
        for row in self.report["matched_seed_contrasts"]:
            self.assertAlmostEqual(
                row["requested_margin_alternative_minus_direct_m"],
                -0.5 * row["endpoint_shift_alternative_minus_direct_m"],
            )
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        boundary = self.report["claim_boundary"]
        self.assertEqual(boundary["wording_inference"], "not_registered")
        self.assertTrue(boundary["exploratory_only"])
        self.assertFalse(boundary["confirmatory_wording_claim"])
        self.assertFalse(boundary["equivalence_claim"])
        self.assertFalse(boundary["models_pooled"])
        self.assertFalse(boundary["changes_232_cell_core_bound"])
        self.assertEqual(boundary["maximum_core_behavioral_cells"], 232)
        self.assertFalse(self.report["source_manifest"]["raw_pvc_files_revalidated_now"])
        self.assertFalse(self.report["source_manifest"]["raw_source_paths_exported"])
        self.assertNotIn("/data/users/", rendered_json)
        self.assertNotIn("gate_root", rendered_json)

    def test_manuscript_documentation_and_builder_integration(self):
        package = ANALYSIS.PACKAGE
        main = (package / "paper" / "main.tex").read_text(encoding="utf-8")
        readme = (package / "README.md").read_text(encoding="utf-8")
        delivery = (package / "docs" / "DELIVERY_QA.md").read_text(encoding="utf-8")
        build = (package / "analysis" / "build_package.py").read_text(encoding="utf-8")
        for required in (
            "H09/H10",
            "134/160",
            "123/160",
            "Confirmatory wording inference was not registered",
        ):
            self.assertIn(required, main)
        for required in (
            "320 Nano/Edge episode rows",
            "120-row CSV",
            "third GR00T cohort",
            "wording inference was not registered",
        ):
            self.assertIn(required, readme)
        for required in (
            "H09/H10",
            "320",
            "480-row",
            "not registered",
        ):
            self.assertIn(required, delivery)
        for required in (
            'run("H09 H10 retained wording control"',
            "analysis/analyze_h09_h10_wording_control.py",
            "paper/data/h09_h10_wording_control.json",
            '"retained_h09_h10_wording_control": {',
            '"wording inference was not registered"',
            'len(reader.pages) != 4',
            'if "/data/users/" in json.dumps(h09_h10)',
        ):
            self.assertIn(required, build)


if __name__ == "__main__":
    unittest.main()
