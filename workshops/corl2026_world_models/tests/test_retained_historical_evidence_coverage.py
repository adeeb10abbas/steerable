"""Fail-closed tests for retained historical H01--H13 package coverage."""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / "analysis" / "audit_retained_historical_evidence_coverage.py"
SPEC = importlib.util.spec_from_file_location("retained_historical_evidence_coverage", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class RetainedHistoricalEvidenceCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.receipts = ANALYSIS.authenticate_sources(ANALYSIS.ROOT)
        cls.report = ANALYSIS.analyze(ANALYSIS.ROOT)

    def test_exact_pinned_snapshot_and_complete_grouped_crosswalk(self):
        provenance = self.report["provenance"]
        self.assertEqual(provenance["base_commit"], ANALYSIS.BASE_COMMIT)
        self.assertEqual(provenance["base_tree"], ANALYSIS.BASE_TREE)
        self.assertEqual(
            provenance["authenticated_source_aggregate"],
            {
                "files": 50,
                "bytes": 2_054_896,
                "lines": 45_726,
                "manifest_sha256": "26dac013bb654e28145e0f2bdfbfe2d8622080c9af84b3773efe5c217a337b38",
            },
        )
        self.assertEqual(len(provenance["authenticated_sources"]), 50)
        registry = self.report["registry"]
        self.assertEqual(registry["registered_historical_ids"], list(ANALYSIS.EXPECTED_IDS))
        self.assertEqual(registry["registered_historical_cohorts"], 13)
        self.assertEqual(registry["grouped_analysis_packages"], 9)
        self.assertTrue(registry["all_registered_ids_mapped_exactly_once_to_grouped_analysis"])
        self.assertFalse(registry["episode_counts_summed_across_cohorts"])
        self.assertEqual(len(self.report["cohort_crosswalk"]), 13)
        self.assertEqual(
            [row["historical_id"] for row in self.report["cohort_crosswalk"]],
            list(ANALYSIS.EXPECTED_IDS),
        )

    def test_multiple_roles_are_explicit_and_never_inflate_cohorts(self):
        rows = {row["historical_id"]: row for row in self.report["cohort_crosswalk"]}
        self.assertEqual(
            rows["H01"]["package_roles"],
            [
                "primary_behavioral_paper:nano_reflection",
                "stopping_sensitivity_supplement",
                "requested_region_entry_supplement",
                "representative_media:presentation_only",
            ],
        )
        self.assertEqual(
            rows["H02"]["package_roles"],
            ["primary_behavioral_paper:dreamzero_reflection", "compact_position_reflection_audit"],
        )
        self.assertEqual(
            rows["H07"]["package_roles"],
            ["primary_behavioral_paper:dreamzero_symmetry", "endpoint_symmetry_audit"],
        )
        self.assertTrue(all(row["multiple_roles_are_not_additional_cohorts"] for row in rows.values()))
        primary = self.report["integration_coverage"]["primary_behavioral_paper"]
        self.assertEqual(
            primary["historical_id_roles"],
            {"H01": "nano_reflection", "H02": "dreamzero_reflection", "H07": "dreamzero_symmetry"},
        )
        self.assertTrue(primary["role_is_additional_view_not_new_cohort"])
        self.assertEqual(
            self.report["coverage_summary"][
                "additional_primary_or_presentation_roles_counted_as_new_cohorts"
            ],
            0,
        )

    def test_all_package_tests_builder_stages_and_manuscript_markers_are_authenticated(self):
        integration = self.report["integration_coverage"]
        self.assertTrue(integration["manuscript"]["all_13_historical_ids_covered"])
        self.assertEqual(
            set(integration["manuscript"]["historical_id_markers"]),
            set(ANALYSIS.EXPECTED_IDS),
        )
        self.assertTrue(integration["build"]["all_9_grouped_analysis_packages_executed"])
        self.assertTrue(integration["build"]["H01_stopping_supplement_executed"])
        self.assertTrue(integration["build"]["primary_behavioral_paper_extractor_executed"])
        self.assertEqual(len(integration["focused_tests"]), 9)
        self.assertEqual(sum(map(len, integration["focused_tests"].values())), 10)
        for package in self.report["packages"]:
            self.assertTrue(package["claim_boundary_closed"])
            self.assertTrue(package["focused_tests"])
            self.assertTrue(package["base_build_stage_markers"])

    def test_boundaries_keep_activity_gates_and_missing_evidence_open(self):
        summary = self.report["coverage_summary"]
        self.assertEqual(summary["registered_cohorts"], 13)
        self.assertEqual(summary["mapped_cohorts"], 13)
        self.assertEqual(summary["grouped_analysis_packages"], 9)
        self.assertEqual(summary["unmapped_registered_cohorts"], 0)
        self.assertEqual(summary["duplicate_mappings"], 0)
        self.assertEqual(summary["packages_with_closed_claim_boundaries"], 9)
        self.assertEqual(summary["scientific_confirmation_claims_created_by_this_audit"], 0)
        self.assertEqual(
            self.report["historical_custom_s2_ids_not_official_D1"],
            ["H02", "H04", "H07", "H12"],
        )
        self.assertTrue(all(value == 0 for value in self.report["new_activity"].values()))
        self.assertTrue(all(value is False for value in self.report["gate_state"].values()))
        self.assertTrue(
            all(value is False for value in self.report["preserved_evidence_gaps"].values())
        )
        boundary = self.report["claim_boundary"]
        for field in (
            "packaging_coverage_is_scientific_confirmation", "scientific_outcomes_reestimated",
            "models_or_cohorts_pooled", "episode_counts_summed",
            "equivalence_established", "independent_scene_or_layout_generalization_established",
            "language_understanding_established",
            "forecast_accuracy_or_fidelity_estimated", "raw_or_replay_evidence_recovered",
            "new_scientific_evidence_created", "current_worker_or_queue_state_inferred",
            "changes_232_cell_core_bound",
        ):
            self.assertIs(boundary[field], False)
        self.assertEqual(boundary["maximum_core_behavioral_cells"], 232)

    def test_stale_inventory_and_build_report_are_exposed_not_rewritten(self):
        completion = self.report["packaging_completion"]
        self.assertEqual(
            completion["pinned_spec_and_inventory_review_status"],
            "standalone reproducible script not yet packaged",
        )
        self.assertTrue(completion["current_authenticated_H01_stopping_package"])
        self.assertFalse(completion["historical_inventory_rewritten"])
        self.assertFalse(completion["checked_in_build_report_used_as_current_coverage_evidence"])
        self.assertIn("predates", completion["build_report_boundary"])

    def test_absent_protected_results_are_authenticated_from_git(self):
        original = Path.is_file

        def hide_results(path: Path) -> bool:
            if "/workshops/corl2026_world_models/results/" in str(path):
                return False
            return original(path)

        with mock.patch.object(Path, "is_file", hide_results):
            raw, receipts = ANALYSIS.authenticate_sources(ANALYSIS.ROOT)
        self.assertEqual(len(raw), 50)
        self.assertEqual(len(receipts), 50)
        self.assertIn(ANALYSIS.INVENTORY_PATH, raw)
        self.assertIn(ANALYSIS.STOPPING_JSON, raw)
        self.assertIn(ANALYSIS.PAPER_RESULTS_PATH, raw)

    def test_source_identity_and_checkout_tampering_fail_closed(self):
        target = "workshops/corl2026_world_models/analysis/analyze_h01_requested_region_entry.py"
        with mock.patch.dict(ANALYSIS.EXPECTED_SOURCES[target], {"git_blob": "0" * 40}):
            with self.assertRaisesRegex(ValueError, "pinned Git blob changed"):
                ANALYSIS.authenticate_sources(ANALYSIS.ROOT)

        target_path = ANALYSIS.ROOT / target
        original = Path.read_bytes

        def tamper_one(path: Path) -> bytes:
            if path == target_path:
                return b"tampered"
            return original(path)

        with mock.patch.object(Path, "read_bytes", tamper_one):
            with self.assertRaisesRegex(ValueError, "checkout diverges"):
                ANALYSIS.authenticate_sources(ANALYSIS.ROOT)

    def test_regenerated_paper_results_checkout_keeps_pinned_object_authoritative(self):
        paper_results_path = ANALYSIS.ROOT / ANALYSIS.PAPER_RESULTS_PATH
        original = Path.read_bytes

        def regenerate_paper_results(path: Path) -> bytes:
            if path == paper_results_path:
                return b'{"generated_checkout":"newer"}\n'
            return original(path)

        with mock.patch.object(Path, "read_bytes", regenerate_paper_results):
            raw, _ = ANALYSIS.authenticate_sources(ANALYSIS.ROOT)
        pinned = raw[ANALYSIS.PAPER_RESULTS_PATH]
        self.assertEqual(
            ANALYSIS._sha256(pinned),
            ANALYSIS.EXPECTED_SOURCES[ANALYSIS.PAPER_RESULTS_PATH]["sha256"],
        )
        self.assertNotIn(b"generated_checkout", pinned)

    def test_strict_json_and_csv_parsers_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            ANALYSIS.parse_json(b'{"x":1,"x":2}', label="duplicate")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":NaN}', label="nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ANALYSIS.parse_json(b'{"x":1e999}', label="overflow")
        with self.assertRaisesRegex(ValueError, "duplicate CSV field"):
            ANALYSIS.parse_csv(b"x,x\n1,2\n", label="duplicate header")
        with self.assertRaisesRegex(ValueError, "malformed CSV row"):
            ANALYSIS.parse_csv(b"x,y\n1\n", label="short row")

    def test_registry_and_integration_mutations_fail_closed(self):
        malformed_spec = self.raw[ANALYSIS.SPEC_PATH].replace(b"| H13 |", b"| H12 |", 1)
        with self.assertRaisesRegex(ValueError, "registered historical IDs changed"):
            ANALYSIS.parse_spec_table(malformed_spec)

        inventory = json.loads(self.raw[ANALYSIS.INVENTORY_PATH])
        inventory["entries"][0]["confirmation_eligible"] = True
        with self.assertRaisesRegex(ValueError, "became confirmation eligible"):
            ANALYSIS.parse_inventory(json.dumps(inventory).encode())

        altered = dict(self.raw)
        altered[ANALYSIS.BUILD_PATH] = altered[ANALYSIS.BUILD_PATH].replace(
            b'run("H02 retained position reflection"', b'run("H02 missing"', 1
        )
        with self.assertRaisesRegex(ValueError, "build-stage marker"):
            ANALYSIS.validate_package_integration(altered)

        altered = dict(self.raw)
        altered[ANALYSIS.PAPER_PATH] = altered[ANALYSIS.PAPER_PATH].replace(
            b"Retained H02 compact audit", b"Removed H02 marker", 1
        )
        with self.assertRaisesRegex(ValueError, "manuscript coverage marker"):
            ANALYSIS.validate_package_integration(altered)

    def test_exact_deterministic_regeneration_and_stale_output_rejection(self):
        rendered_json = ANALYSIS.render_json(self.report)
        rendered_csv = ANALYSIS.render_csv(self.report)
        self.assertEqual(rendered_json, ANALYSIS.render_json(ANALYSIS.analyze(ANALYSIS.ROOT)))
        self.assertEqual(rendered_csv, ANALYSIS.render_csv(ANALYSIS.analyze(ANALYSIS.ROOT)))
        self.assertEqual(ANALYSIS.DEFAULT_JSON.read_text(encoding="utf-8"), rendered_json)
        self.assertEqual(ANALYSIS.DEFAULT_CSV.read_text(encoding="utf-8"), rendered_csv)
        rows = list(csv.DictReader(io.StringIO(rendered_csv)))
        self.assertEqual(len(rows), 13)
        self.assertEqual([row["historical_id"] for row in rows], list(ANALYSIS.EXPECTED_IDS))
        self.assertNotIn("/home/", rendered_json)
        self.assertNotIn("/data/users/", rendered_json)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ANALYSIS.main(["--check"]), 0)
        with tempfile.TemporaryDirectory() as directory:
            stale_json = Path(directory) / "stale.json"
            stale_csv = Path(directory) / "stale.csv"
            stale_json.write_text("{}\n", encoding="utf-8")
            stale_csv.write_text("stale\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "canonical JSON is absent or stale"):
                    ANALYSIS.main(
                        ["--check", "--json", str(stale_json), "--csv", str(stale_csv)]
                    )

    def test_current_readme_manuscript_and_builder_integrate_the_audit(self):
        build = (PACKAGE / "analysis/build_package.py").read_text(encoding="utf-8")
        marker = 'run("retained historical evidence coverage"'
        self.assertIn(marker, build)
        self.assertGreater(build.index(marker), build.index('run("paper experiment results"'))
        self.assertLess(build.index(marker), build.index('run("historical evidence replay"'))
        for required in (
            "retained_historical_evidence_coverage.json",
            '"retained_historical_evidence_coverage": {',
            '"preserved_evidence_gaps": coverage["preserved_evidence_gaps"]',
            '"auditor_sha256": digest(package / "analysis/audit_retained_historical_evidence_coverage.py")',
        ):
            self.assertIn(required, build)

        readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
        for required in (
            "retained historical evidence coverage audit",
            "maps all 13 historical inventory IDs to\nnine grouped analysis packages",
            "without counting those views as additional cohorts or summing episode counts",
            "H02 compact position-reflection audit",
            "Package coverage is neither pooled\ninference nor confirmation",
            "raw/PVC recovery, physical-time pairing, independent\nlabels",
        ):
            self.assertIn(required, readme)

        paper = (PACKAGE / "paper/main.tex").read_text(encoding="utf-8")
        self.assertIn("All 13 registered historical IDs have authenticated packages", paper)
        self.assertIn("not pooled inference or confirmation", paper)
        self.assertIn("physical-time, labels and forecast accuracy remain unestablished", paper)


if __name__ == "__main__":
    unittest.main()
