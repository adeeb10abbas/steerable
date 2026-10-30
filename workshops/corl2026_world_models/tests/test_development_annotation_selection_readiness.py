import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    Path(__file__).resolve().parents[1]
    / "analysis/audit_development_annotation_selection_readiness.py"
)
REPORT = (
    Path(__file__).resolve().parents[1]
    / "results/development_annotation_selection_readiness.json"
)
spec = importlib.util.spec_from_file_location("annotation_selection_readiness", MODULE)
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


class DevelopmentAnnotationSelectionReadinessTests(unittest.TestCase):
    def test_saved_report_is_canonical_and_fail_closed(self):
        report = readiness.build_audit()
        self.assertEqual(REPORT.read_bytes(), readiness.canonical_bytes(report))
        self.assertEqual(
            report["authenticated_source_blobs"]["source_bytes"], 740501
        )
        self.assertEqual(report["global_study_snapshot"]["planned_core_cells"], 232)
        self.assertEqual(report["global_study_snapshot"]["launched_core_cells"], 44)
        self.assertEqual(report["global_study_snapshot"]["terminal_valid_cells"], 40)
        self.assertEqual(
            report["global_study_snapshot"]["terminal_technical_invalid_cells"], 4
        )
        self.assertEqual(report["global_study_snapshot"]["terminal_requests"], 1460)
        self.assertEqual(
            report["global_study_snapshot"]["valid_behavioral_actions"], 18000
        )
        self.assertEqual(
            report["global_study_snapshot"]["valid_behavioral_observations"], 18040
        )
        self.assertEqual(
            report["global_study_snapshot"]["confirmation_inference_cells"], 0
        )
        self.assertTrue(all(value == 0 for value in report["new_activity"].values()))
        self.assertTrue(all(value == 0 for value in report["audit_activity"].values()))
        self.assertIs(
            report["annotation_and_release_boundary"][
                "production_eligibility_determined"
            ],
            False,
        )
        self.assertIs(
            report["annotation_and_release_boundary"][
                "production_request_selection_manifest_created"
            ],
            False,
        )
        self.assertEqual(
            report["annotation_and_release_boundary"]["selected_request_count"], 0
        )
        self.assertIs(
            report["annotation_and_release_boundary"]["safe_to_release_confirmation"],
            False,
        )

    def test_exact_population_and_schema_gap_are_preserved(self):
        report = readiness.build_audit()
        population = report["retained_population"]
        self.assertEqual(population["rows"], 1152)
        self.assertEqual(population["cells"], 32)
        self.assertEqual(population["rows_by_model"], {"D1": 912, "N3": 240})
        self.assertEqual(population["cells_by_model"], {"D1": 16, "N3": 16})
        self.assertEqual(population["rows_per_cell"], {"D1": 57, "N3": 15})
        self.assertEqual(population["annotation_request_schema_field_count"], 30)
        self.assertEqual(
            population["exact_annotation_schema_fields_available"],
            ["cell_id", "layout_pair_id", "model_id", "request_index"],
        )
        self.assertEqual(population["exact_annotation_schema_fields_available_count"], 4)
        self.assertEqual(population["exact_annotation_schema_fields_missing_count"], 26)
        self.assertIs(population["exact_annotation_schema_coverage_complete"], False)
        selection = report["selection_readiness"]
        self.assertEqual(selection["sampling_key_feasible_rows"], 0)
        self.assertEqual(selection["eligibility_decision_feasible_rows"], 0)
        self.assertEqual(
            selection["eligibility_decision_status"], "undetermined_not_ineligible"
        )
        self.assertEqual(selection["sampling_ranks_emitted"], 0)
        self.assertEqual(selection["eligibility_rows_emitted"], 0)
        self.assertIs(selection["selection_manifest_created"], False)

    def test_workflow_contract_and_ceiling_arithmetic_are_exact(self):
        report = readiness.build_audit()
        workflow = report["workflow_contract"]
        self.assertEqual(workflow["request_sampling_seed"], 2026091302)
        self.assertEqual(workflow["request_sample_cap_per_episode"], 4)
        self.assertEqual(
            workflow["eligibility_metadata_fields"],
            [
                "camera_identity_match",
                "target_within_executed_prefix",
                "technical_valid",
                "timestamp_error_s",
                "timestamp_tolerance_s",
            ],
        )
        self.assertEqual(
            workflow["eligibility_function"]["sha256"],
            readiness.EXPECTED_ELIGIBILITY_FUNCTION_SHA256,
        )
        self.assertEqual(
            workflow["selection_function"]["sha256"],
            readiness.EXPECTED_SELECTION_FUNCTION_SHA256,
        )
        self.assertIs(
            workflow["selection_uses_object_visibility_or_forecast_quality"], False
        )
        ceilings = report["annotation_workload_ceilings"]
        self.assertEqual(ceilings["maximum_selected_requests"], 128)
        self.assertEqual(
            ceilings["primary_current_predicted_executed"],
            {"images": 384, "two_rater_judgments": 768},
        )
        self.assertEqual(
            ceilings["with_A4_preceding"],
            {"images": 512, "two_rater_judgments": 1024},
        )
        self.assertEqual(
            ceilings["with_A4_and_A9_if_supported"],
            {"images": 768, "two_rater_judgments": 1536},
        )

    def test_timing_scope_is_not_promoted(self):
        timing = readiness.build_audit()["timing_boundary"]
        self.assertEqual(timing["D1_retained_request_identities"], 912)
        self.assertEqual(timing["D1_source_proven_full_decode_requests"], 240)
        self.assertEqual(timing["D1_native_clock_matched_requests"], 224)
        self.assertEqual(timing["D1_action_prefix_truncated_requests"], 16)
        self.assertEqual(
            timing["D1_source_unmapped_incremental_decode_requests"], 672
        )
        self.assertIs(timing["D1_physical_time_coverage_complete"], False)
        self.assertEqual(timing["N3_retained_request_identities"], 240)
        self.assertIs(timing["all_requests_timing_complete"], False)

    def test_workflow_semantic_drift_is_rejected(self):
        payload = readiness.git_blob(readiness.WORKFLOW_COMMIT, readiness.WORKFLOW_PATH)
        cases = {
            "seed": (
                b"REQUEST_SAMPLING_SEED = 2026091302",
                b"REQUEST_SAMPLING_SEED = 2026091303",
                "sampling seed",
            ),
            "cap": (
                b"REQUEST_SAMPLE_CAP = 4",
                b"REQUEST_SAMPLE_CAP = 5",
                "sample cap",
            ),
            "algorithm": (
                b"select the first min(4, eligible_count).",
                b"select the first min(5, eligible_count).",
                "sampling algorithm",
            ),
            "identity_fallback": (
                b"row['source_request_id']}\".encode(\"utf-8\")",
                b"row['request_index']}\".encode(\"utf-8\")",
                "Workflow function changed: select_requests",
            ),
            "eligibility_outcome": (
                b"if not row[\"camera_identity_match\"]:",
                b"if not row[\"forecast_quality_match\"]:",
                "Workflow function changed: _eligibility_reasons",
            ),
        }
        for label, (old, new, message) in cases.items():
            with self.subTest(label=label):
                self.assertIn(old, payload)
                mutated = payload.replace(old, new, 1)
                with self.assertRaisesRegex(ValueError, message):
                    readiness.parse_workflow_contract(mutated)

    def test_recovery_accounting_gate_and_mapping_drift_are_rejected(self):
        payload = readiness.git_blob(readiness.RECOVERY_COMMIT, readiness.RECOVERY_PATH)
        original = readiness.load_json(payload, "recovery")
        cases = []

        changed = copy.deepcopy(original)
        changed["global_study_snapshot"]["planned_core_cells"] = 233
        cases.append(("core", changed, "Global study snapshot changed"))

        changed = copy.deepcopy(original)
        changed["timing_boundary"]["all_requests_timing_complete"] = True
        cases.append(("timing", changed, "Timing boundary changed"))

        changed = copy.deepcopy(original)
        changed["annotation_and_release_boundary"]["production_eligibility_determined"] = True
        cases.append(("eligibility", changed, "Annotation boundary changed"))

        changed = copy.deepcopy(original)
        changed["new_activity"]["request_selections"] = 1
        cases.append(("activity", changed, "New activity changed"))

        changed = copy.deepcopy(original)
        changed["request_lineage"]["rows"][1] = copy.deepcopy(
            changed["request_lineage"]["rows"][0]
        )
        cases.append(("duplicate", changed, "Duplicate lineage mapping key"))

        for label, report, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    readiness.validate_recovery(report)

    def test_output_contract_rejects_any_scientific_promotion(self):
        original = readiness.build_audit()
        cases = []
        changed = copy.deepcopy(original)
        changed["selection_readiness"]["sampling_key_feasible_rows"] = 1
        cases.append(("rank", changed, "Sampling feasibility promoted"))
        changed = copy.deepcopy(original)
        changed["selection_readiness"]["eligibility_decision_feasible_rows"] = 1
        cases.append(("eligibility", changed, "Eligibility feasibility promoted"))
        changed = copy.deepcopy(original)
        changed["selection_readiness"]["selection_manifest_created"] = True
        cases.append(("selection", changed, "Selection manifest promoted"))
        changed = copy.deepcopy(original)
        changed["audit_activity"]["raw_paths_opened"] = 1
        cases.append(("raw", changed, "prohibited activity"))
        changed = copy.deepcopy(original)
        changed["annotation_and_release_boundary"]["labels"] = 1
        cases.append(("label", changed, "Output annotation boundary changed"))
        for label, report, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    readiness.validate_output_contract(report)

    def test_cli_check_accepts_only_the_canonical_report(self):
        subprocess.run(
            [sys.executable, str(MODULE), "--check", str(REPORT)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "altered.json"
            document = json.loads(REPORT.read_text())
            document["selection_readiness"]["selected_request_count"] = 1
            altered.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            process = subprocess.run(
                [sys.executable, str(MODULE), "--check", str(altered)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Saved report is not canonical or current", process.stderr)


if __name__ == "__main__":
    unittest.main()
