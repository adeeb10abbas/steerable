"""Fail-closed checks for retained H13 V1 evaluator-development packaging."""
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
    / "analyze_h13_v1_forecast_corpus.py"
)
SPEC = importlib.util.spec_from_file_location("h13_v1_forecast_corpus", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class H13V1ForecastCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.bundle = ANALYSIS._read_bundle(ANALYSIS.ROOT)
        cls.rows = ANALYSIS._parse_rows(cls.raw[ANALYSIS.ROWS_PATH])
        cls.report = ANALYSIS.analyze(ANALYSIS.ROOT)

    def test_exact_authenticated_source_and_population_accounting(self):
        provenance = self.report["provenance"]
        self.assertEqual(provenance["upstream_commit"], ANALYSIS.UPSTREAM_PIN)
        self.assertEqual(provenance["audit_commit"], ANALYSIS.AUDIT_COMMIT)
        self.assertEqual(provenance["bundle_files"], 5)
        self.assertEqual(provenance["bundle_bytes"], 998_249)
        source = provenance["upstream_sources"]
        self.assertEqual(source["files"], 761)
        self.assertEqual(source["bytes"], 7_230_812)
        self.assertEqual(source["lines"], 222_796)
        self.assertEqual(source["manifest_sha256"], ANALYSIS.EXPECTED_SOURCE["manifest_sha256"])
        self.assertEqual(source["cache_files"], 752)
        self.assertEqual(source["cache_bytes"], 5_269_817)
        self.assertEqual(source["cache_lines"], 210_592)

        cohort = self.report["population"]["cohort"]
        self.assertEqual(cohort["episodes"], 80)
        self.assertEqual(cohort["chunks"], 752)
        self.assertEqual(cohort["frames"], 3_008)
        self.assertEqual(cohort["reliable_frames"], 1_460)
        self.assertEqual(cohort["unreliable_frames"], 1_548)
        self.assertEqual(self.report["population"]["matched_seed_blocks"], 20)
        self.assertEqual(self.report["population"]["episodes_per_block"], 4)

    def test_legacy_arithmetic_missingness_and_cell_table(self):
        legacy = self.report["legacy_replay"]
        self.assertEqual((legacy["certain"], legacy["abstained"]), (421, 331))
        self.assertEqual(legacy["executed_positive"], 97)
        self.assertEqual(legacy["covered_executed_positive"], 25)
        self.assertEqual(
            legacy["quadrants"],
            {
                "imagines_requested_executes_requested": 22,
                "imagines_requested_executes_not_requested": 5,
                "does_not_imagine_requested_executes_requested": 3,
                "neither_imagines_nor_executes_requested": 391,
                "uncertain_future": 331,
            },
        )
        arithmetic = legacy["same_covered_subset_arithmetic"]
        self.assertEqual(arithmetic["observed_agreement_numerator"], 413)
        self.assertEqual(arithmetic["constant_negative_agreement_numerator"], 396)
        self.assertEqual(arithmetic["difference_numerator"], 17)
        self.assertAlmostEqual(arithmetic["difference_percentage_points"], 4.038004750593824)
        bounds = legacy["unresolved_case_full_corpus_agreement_bounds"]
        self.assertEqual((bounds["lower_numerator"], bounds["upper_numerator"]), (413, 744))
        self.assertAlmostEqual(bounds["lower"], 413 / 752)
        self.assertAlmostEqual(bounds["upper"], 744 / 752)

        cells = self.report["population"]["cells"]
        self.assertEqual(len(cells), 8)
        observed = {
            (row["wording"], row["requested_relation"]): (row["chunks"], row["legacy_certain"])
            for row in cells
        }
        self.assertEqual(
            observed,
            {
                key: (ANALYSIS.EXPECTED_CELL_CHUNKS[key], ANALYSIS.EXPECTED_CELL_CERTAIN[key])
                for key in ANALYSIS.EXPECTED_CELL_CHUNKS
            },
        )

    def test_endpoint_thresholds_and_block_uncertainty_are_descriptive_only(self):
        endpoint = self.report["endpoint_sensitivity"]
        self.assertEqual(endpoint["metrics"]["certain"], 282)
        self.assertEqual(endpoint["metrics"]["covered_executed_positive"], 21)
        self.assertIn("NOT corrected fidelity", endpoint["status"])
        self.assertEqual(
            {key: value["certain"] for key, value in self.report["threshold_replay"].items()},
            {"0.1": 41, "0.15": 262, "0.2": 421},
        )
        uncertainty = self.report["legacy_coverage_uncertainty"]
        self.assertEqual(uncertainty["independent_seed_blocks"], 20)
        self.assertEqual(uncertainty["episodes_per_block"], 4)
        self.assertEqual(uncertainty["replicates"], 2_000)
        self.assertEqual(uncertainty["seed"], 20260912)
        self.assertEqual(
            uncertainty["intervals"]["coverage"],
            [0.5361216730038023, 0.583011583011583],
        )

    def test_frozen_annotation_draw_is_exact_and_advances_no_gate(self):
        sample = self.report["annotation_sampling"]
        self.assertEqual(sample["population_size"], 752)
        self.assertEqual(sample["sample_size"], 160)
        self.assertEqual(sample["seed"], 20260912)
        self.assertEqual(
            sample["draw_sha256"],
            "ab780ea4e1b91918ca72801ef0641b563725d0a7d935f6de15e1d135b0ed6d47",
        )
        self.assertEqual(
            {row["stratum"]: (row["population"], row["sample"]) for row in sample["strata"]},
            ANALYSIS.STRATA,
        )
        self.assertEqual(sample["media_recovered_and_verified"], 0)
        self.assertEqual(sample["alignment_verified"], 0)
        self.assertEqual(sample["human_labels_collected"], 0)
        self.assertFalse(sample["gate_advanced"])
        readiness = self.report["corrected_fidelity_readiness"]
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["required_paths"], 840)
        self.assertEqual(readiness["supporting_generated_media_and_action_receipts_requiring_recovery"], 2_256)
        self.assertFalse(readiness["actual_execution_imagery_verified"])

    def test_claim_boundary_keeps_all_scientific_gates_closed(self):
        boundary = self.report["claim_boundary"]
        self.assertFalse(boundary["new_model_inference"])
        self.assertEqual(boundary["new_behavioral_episodes"], 0)
        self.assertEqual(boundary["new_human_labels"], 0)
        self.assertFalse(boundary["forecast_fidelity_or_accuracy_estimated"])
        self.assertFalse(boundary["annotation_or_release_gate_advanced"])
        self.assertFalse(boundary["core_232_cell_accounting_changed"])
        self.assertEqual(boundary["maximum_core_behavioral_cells"], 232)
        self.assertEqual(len(self.report["new_activity"]), 13)
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertEqual(len(self.report["gate_state"]), 9)
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        self.assertIn("validated forecast accuracy or fidelity", boundary["prohibited_interpretations"])

    def test_strict_json_and_typed_identifiers_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            ANALYSIS.parse_json(b'{"x":1,"x":2}', label="duplicate")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":NaN}', label="nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":1e999}', label="overflow")
        with self.assertRaisesRegex(ValueError, "integer"):
            ANALYSIS._typed_int(True, "boolean")
        with self.assertRaisesRegex(ValueError, "boolean or null"):
            ANALYSIS._typed_label(1, "numeric")

    def test_row_mutations_fail_closed(self):
        rows = [json.loads(line) for line in self.raw[ANALYSIS.ROWS_PATH].splitlines()]
        malformed = copy.deepcopy(rows)
        malformed[1]["id"] = malformed[0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate reconciled ID"):
            ANALYSIS._parse_rows("\n".join(json.dumps(row) for row in malformed).encode())

        malformed = copy.deepcopy(rows)
        malformed[0]["sampling_seed"] += 1
        with self.assertRaisesRegex(ValueError, "sampling-seed identity mismatch"):
            ANALYSIS._parse_rows("\n".join(json.dumps(row) for row in malformed).encode())

        malformed = copy.deepcopy(rows)
        malformed[0]["threshold_predictions"]["0.2"] = None
        with self.assertRaisesRegex(ValueError, "legacy label differs"):
            ANALYSIS._parse_rows("\n".join(json.dumps(row) for row in malformed).encode())

        malformed = copy.deepcopy(rows)
        malformed[0]["actual"] = 1
        with self.assertRaisesRegex(ValueError, "execution label must be boolean"):
            ANALYSIS._parse_rows("\n".join(json.dumps(row) for row in malformed).encode())

    def test_exact_regeneration_compact_csv_and_no_private_paths(self):
        rendered_json = ANALYSIS.render_json(self.report)
        rendered_csv = ANALYSIS.render_csv(self.report)
        self.assertEqual(ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"), rendered_json)
        self.assertEqual(ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"), rendered_csv)
        rows = list(csv.DictReader(io.StringIO(rendered_csv)))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["wording"] for row in rows}, set(ANALYSIS.WORDING_TIER))
        self.assertEqual({row["requested_relation"] for row in rows}, set(ANALYSIS.RELATION_TASK))
        self.assertNotIn("/home/", rendered_json)
        self.assertNotIn("/home/", rendered_csv)
        self.assertNotIn("source_id", rendered_csv)
        self.assertEqual(ANALYSIS.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
