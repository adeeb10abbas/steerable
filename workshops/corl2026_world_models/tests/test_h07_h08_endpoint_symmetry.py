"""Fail-closed checks for retained H07/H08 endpoint-symmetry packaging."""
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
    / "analyze_h07_h08_endpoint_symmetry.py"
)
SPEC = importlib.util.spec_from_file_location("h07_h08_endpoint_symmetry", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class H07H08EndpointSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.provenance = ANALYSIS.read_pinned_sources(ANALYSIS.ROOT)
        cls.report = ANALYSIS.analyze_documents(cls.documents, cls.provenance)

    def test_authenticated_population_and_registered_interactions(self):
        self.assertEqual(
            self.report["reproduction"]["full_compact_episode_rows_strictly_parsed"],
            4096,
        )
        self.assertEqual(
            self.report["reproduction"]["full_compact_pair_rows_strictly_parsed"],
            2048,
        )
        expected = {
            "H07": {
                "model": "dreamzero_droid_action_cfg",
                "episode_rows": (2355, 2462),
                "pair_rows": (1, 54),
                "binary": (-0.5555555555555556, 0.0008544921875),
                "depth": (-0.12434867601324287, 6.014108657836914e-05),
                "endpoint": (0.103770262622102, 0.0034134238958358765),
            },
            "H08": {
                "model": "cosmos3_edge_policy_droid",
                "episode_rows": (1, 108),
                "pair_rows": (55, 108),
                "binary": (-0.14814814814814814, 0.3984375),
                "depth": (-0.19177691574225686, 8.893013000488281e-05),
                "endpoint": (-0.027263488987012318, 0.48384031653404236),
            },
        }
        metric_names = {
            "binary": "binary_gap",
            "depth": "depth_gap_m",
            "endpoint": "endpoint_redirection_m",
        }
        for cohort, values in expected.items():
            with self.subTest(cohort=cohort):
                actual = self.report["cohorts"][cohort]
                self.assertEqual(actual["model_id"], values["model"])
                self.assertEqual(actual["authenticated_population"]["behavioral_episodes"], 108)
                self.assertEqual(actual["authenticated_population"]["matched_pairs"], 54)
                self.assertEqual(
                    (min(actual["source_rows"]["episode_rows"]), max(actual["source_rows"]["episode_rows"])),
                    values["episode_rows"],
                )
                self.assertEqual(
                    (min(actual["source_rows"]["pair_rows"]), max(actual["source_rows"]["pair_rows"])),
                    values["pair_rows"],
                )
                for short_name, metric in metric_names.items():
                    interaction = actual["interactions_s1_minus_s0"][metric]
                    mean, p_value = values[short_name]
                    self.assertAlmostEqual(interaction["mean"], mean)
                    self.assertAlmostEqual(
                        interaction["exact_layout_label_permutation"]["exact_two_sided_p"],
                        p_value,
                    )
                    self.assertEqual(len(interaction["seed_values"]), 27)

    def test_strict_json_and_exact_sign_flip_helpers(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            ANALYSIS.parse_json(b'{"x":1,"x":2}', label="duplicate")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":NaN}', label="nonfinite")
        self.assertEqual(
            ANALYSIS._exact_sign_flip([1.0, 1.0, 1.0, 0.0, 0.0]),
            {
                "n": 5,
                "permutations": 32,
                "extreme": 8,
                "exact_two_sided_p": 0.25,
            },
        )

    def test_registration_and_population_mutations_fail_closed(self):
        malformed = copy.deepcopy(self.documents)
        malformed["registration"]["model_request_count_before_registration"] = 1
        with self.assertRaisesRegex(ValueError, "before model activity"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["episodes"] = malformed["episodes"][:-1]
        with self.assertRaisesRegex(ValueError, "exactly 216 episodes"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        malformed["pairs"] = malformed["pairs"][:-1]
        with self.assertRaisesRegex(ValueError, "exactly 216 episodes and 108 pairs"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

    def test_pair_join_and_registered_report_mutations_fail_closed(self):
        malformed = copy.deepcopy(self.documents)
        malformed["pairs"][0]["endpoint_redirection_left_minus_right_m"] += 0.01
        with self.assertRaisesRegex(ValueError, "endpoint disagrees"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

        malformed = copy.deepcopy(self.documents)
        source = malformed["results"]["checkpoints"]["dreamzero_droid_action_cfg"]
        source["analysis"]["interaction_s1_minus_s0_core"]["depth_gap_m"]["mean"] += 0.01
        with self.assertRaisesRegex(ValueError, "interaction mean"):
            ANALYSIS.analyze_documents(malformed, self.provenance)

    def test_generated_outputs_are_exact_and_boundaries_remain_closed(self):
        self.assertEqual(ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"), ANALYSIS.render_json(self.report))
        self.assertEqual(ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"), ANALYSIS.render_csv(self.report))
        rows = list(csv.DictReader(io.StringIO(ANALYSIS.render_csv(self.report))))
        self.assertEqual(len(rows), 108)
        self.assertEqual({row["cohort"] for row in rows}, {"H07", "H08"})
        self.assertEqual({row["symmetry_level_s"] for row in rows}, {"0.0", "1.0"})
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        boundary = self.report["claim_boundary"]
        self.assertFalse(boundary["forecast_accuracy_evidence"])
        self.assertFalse(boundary["equivalence_claim"])
        self.assertFalse(boundary["changes_232_cell_core_bound"])
        self.assertFalse(boundary["position_only_reflection_replication"])

        paper = (ANALYSIS.PACKAGE / "paper" / "main.tex").read_text(encoding="utf-8")
        readme = (ANALYSIS.PACKAGE / "README.md").read_text(encoding="utf-8")
        qa = (ANALYSIS.PACKAGE / "docs" / "DELIVERY_QA.md").read_text(encoding="utf-8")
        for text in (paper, readme, qa):
            self.assertIn("H07", text)
            self.assertIn("H08", text)
            self.assertIn("inventory", text.lower())
            self.assertIn("forecast accuracy", text.lower())
        build = (ANALYSIS.PACKAGE / "analysis" / "build_package.py").read_text(encoding="utf-8")
        for required in (
            'run("H07 H08 retained endpoint symmetry"',
            "analysis/analyze_h07_h08_endpoint_symmetry.py",
            "paper/data/h07_h08_endpoint_symmetry.json",
            '"retained_h07_h08_endpoint_symmetry": {',
            '"H07/H08", "Edge", "12.4", "19.2", "endpoint interactions"',
            '"csv_sha256": digest(package / "paper/data/h07_h08_endpoint_symmetry.csv")',
            '"analyzer_sha256": digest(package / "analysis/analyze_h07_h08_endpoint_symmetry.py")',
        ):
            self.assertIn(required, build)


if __name__ == "__main__":
    unittest.main()
