"""The manuscript and reproducible build must retain the H01 stopping boundary."""
import hashlib
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1]


class PaperStoppingIntegrationTests(unittest.TestCase):
    def test_manuscript_reports_retrospective_control_without_fixed_time_overclaim(self):
        paper = (PACKAGE / "paper/main.tex").read_text()
        for required in (
            "Both archived cohorts stop a success",
            "unequal goal-stopped endpoints",
            "latest shared strictly pretermination action",
            "31.1 and 35.8",
            "89--290 (median 133)",
            "96--320 (median 149)",
            "18/27 and 17/27",
            "17/27--26/27 and 16/27--26/27",
            "pair-specific action remains outcome-dependent",
            "At action 64 all pairs remain",
            "20/26 (six reversed)",
            "25/27 (one reversed, one tie)",
            "At action 450 no matched pair remains",
            "not A5 fixed-duration evidence",
            "not 27 independent scenes",
            "not established physical time",
            "DreamZero-s2",
            "baseline scale 1 is not used",
            "27/27 original LEFT",
            "23/27 reflected RIGHT",
            "all four absent entries are cap-censored reflected RIGHT trials",
            "168.2, 176.9, 119.6 and 209.6",
            "mean 81.4 actions",
            "91 sustained and 13 transient entries",
            "104 entries but 102 successes",
            "neither physical-time latency, language understanding, A5--A7, nor forecast accuracy",
        ):
            with self.subTest(required=required):
                self.assertIn(required, paper)
        self.assertNotIn("Nano's mean response $R$ is 44.8", paper)

    def test_readme_marks_stopping_revision_incorporated(self):
        readme = (PACKAGE / "README.md").read_text()
        self.assertIn("stopping-sensitivity analysis is now incorporated", readme)
        self.assertIn("DreamZero-s2", readme)
        self.assertIn("neither prospective\nfixed-duration evidence nor forecast accuracy", readme)
        self.assertNotIn("revisions are planned,\nnot yet incorporated", readme)
        self.assertIn("first action-index entry into the\nrequested 45-degree cone", readme)
        self.assertIn("without assigning goal-terminated transient cases a\ncommon-horizon time", readme)

    def test_delivery_qa_matches_current_pdf_and_does_not_cite_absent_outputs(self):
        delivery = (PACKAGE / "docs/DELIVERY_QA.md").read_text()
        pdf_sha256 = hashlib.sha256((PACKAGE / "paper/main.pdf").read_bytes()).hexdigest()
        self.assertIn(pdf_sha256, delivery)
        self.assertIn("1029/1029 tests", delivery)
        self.assertIn("1014/1014 tests", delivery)
        self.assertIn(
            "seven of those methods belong to the protected untracked\nN3 test",
            delivery,
        )
        self.assertIn("generated build\nproducts, not retained checkout evidence", delivery)
        self.assertNotIn("23 tests", delivery)
        self.assertNotIn("paper/qa_pages", delivery)

    def test_dreamzero_s2_label_reaches_figure_and_clean_build(self):
        figures = (PACKAGE / "analysis/make_paper_figures.py").read_text()
        requirements = (PACKAGE / "requirements-build.txt").read_text()
        readme = (PACKAGE / "README.md").read_text()
        self.assertIn('["Nano", "DreamZero-s2"]', figures)
        self.assertIn("jsonschema==4.26.0", requirements)
        self.assertIn("matplotlib==3.10.9", requirements)
        self.assertIn("--system-site-packages", readme)
        self.assertIn("WMF_TORCH_PYTHON=/home/ali/miniforge3/envs/any6d/bin/python", readme)

    def test_build_fails_fast_without_and_records_all_required_dependencies(self):
        build = (PACKAGE / "analysis/build_package.py").read_text()
        self.assertIn(
            'required_build_dependencies = ["matplotlib", "numpy", "pypdf", "jsonschema", "torch"]',
            build,
        )
        self.assertIn('"build_dependencies": build_dependency_versions', build)
        self.assertIn("Build environment is incomplete", build)
        self.assertIn('"SOURCE_DATE_EPOCH": "1786718341"', build)
        self.assertIn('"FORCE_SOURCE_DATE": "1"', build)
        self.assertIn('"reproducible_output_environment": {', build)

    def test_build_reproduces_and_hashes_stopping_outputs(self):
        build = (PACKAGE / "analysis/build_package.py").read_text()
        self.assertLess(build.index('run("stopping-time controls"'),
                        build.index('run("paper experiment results"'))
        for required in (
            "analysis/analyze_stopping_controls.py",
            '"source_commit": stopping["provenance"]["commit"]',
            '"source_jsonl": {"path": stopping_source_path, **stopping_source}',
            '"json_sha256": digest(stopping_path)',
            '"csv_sha256": digest(results / "stopping_controls.csv")',
            '"analyzer_sha256": digest(package / "analysis/analyze_stopping_controls.py")',
        ):
            with self.subTest(required=required):
                self.assertIn(required, build)
        self.assertIn('expected_stopping_source_sha256 = "ce67aecb2705394e8161fa4b65df7130e62648e2f6d8b6a93408c8da685a7207"',
                      build)

    def test_build_reproduces_and_hashes_requested_region_entry(self):
        build = (PACKAGE / "analysis/build_package.py").read_text()
        self.assertIn('run("H01 requested-region entry"', build)
        for required in (
            "analysis/analyze_h01_requested_region_entry.py",
            "paper/data/h01_requested_region_entry.json",
            "paper/data/h01_requested_region_entry.csv",
            'entry["provenance"]["commit"]',
            'entry["matched_analysis"]["difference_in_differences_actions"]',
            '"json_sha256": digest(entry_path)',
            '"csv_sha256": digest(package / "paper/data/h01_requested_region_entry.csv")',
            '"analyzer_sha256": digest(package / "analysis/analyze_h01_requested_region_entry.py")',
        ):
            with self.subTest(required=required):
                self.assertIn(required, build)


if __name__ == "__main__":
    unittest.main()
