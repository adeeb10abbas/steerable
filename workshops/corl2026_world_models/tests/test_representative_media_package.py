"""The H01 media package must remain outcome-blind and presentation-only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / "analysis/package_representative_media.py"
SPEC = importlib.util.spec_from_file_location("representative_media", MODULE_PATH)
MEDIA = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MEDIA)


class RepresentativeMediaPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = MEDIA.build_package()

    def test_authenticates_complete_lowest_seed_block(self):
        package = self.package
        self.assertEqual(package["source"]["commit"], MEDIA.PIN)
        self.assertEqual(package["selection"]["selected_seed"], 9400)
        self.assertEqual(package["selection"]["selected_cell_count"], 4)
        self.assertTrue(package["selection"]["all_four_registered_conditions_included"])
        self.assertFalse(package["selection"]["outcome_used_for_selection"])
        self.assertFalse(package["selection"]["request_used_for_selection"])
        self.assertEqual(
            {row["registered_cell_id"] for row in package["videos"]},
            set(MEDIA.EXPECTED_VIDEOS),
        )
        self.assertEqual(package["totals"]["video_bytes"], 12_475_015)
        self.assertEqual(package["totals"]["source_manifest_and_video_file_count"], 5)
        self.assertEqual(package["totals"]["source_manifest_and_video_bytes"], 12_522_030)
        audit = package["selection"]["no_outcome_hiding_audit"]
        self.assertEqual((audit["selected_successes"], audit["selected_failures"]), (4, 0))
        self.assertEqual(audit["full_cohort_failure_taxonomy_counts"]["correct"], 102)

    def test_selected_cells_are_authenticated_historical_episodes(self):
        for row in self.package["videos"]:
            expected = MEDIA.EXPECTED_VIDEOS[row["registered_cell_id"]]
            with self.subTest(cell=row["registered_cell_id"]):
                self.assertEqual(row["git_object"]["git_blob"], expected["git_blob"])
                self.assertEqual(row["git_object"]["sha256"], expected["sha256"])
                self.assertEqual(row["actions_executed"], expected["actions_executed"])
                self.assertTrue(row["behavioral_result_valid"])
                self.assertTrue(row["requested_success"])
                self.assertTrue(row["final_detached_release"])
                self.assertEqual(row["historical_full_stream_qualification"]["codec_name"], "h264")

    def test_claim_boundary_and_zero_new_science_are_explicit(self):
        totals = self.package["totals"]
        self.assertEqual(
            (totals["new_model_requests"], totals["new_behavioral_episodes"], totals["new_human_labels"]),
            (0, 0, 0),
        )
        boundary = self.package["claim_boundary"]
        for key in (
            "not_scientific_timing_maps",
            "not_physical_time_alignment",
            "not_forecast_accuracy_evidence",
            "not_annotation_evidence",
            "does_not_complete_a5_or_any_forecast_gate",
        ):
            self.assertTrue(boundary[key])

    def test_checked_in_manifest_is_exact_reproduction(self):
        path = PACKAGE / "paper/media/representative_h01_videos.json"
        self.assertEqual(json.loads(path.read_text()), self.package)

    def test_paper_and_readme_preserve_media_and_a7_boundaries(self):
        paper = (PACKAGE / "paper/main.tex").read_text()
        readme = (PACKAGE / "README.md").read_text()
        for required in (
            "31.121/0.017",
            "35.865/0.020",
            "define $\\Delta y_o=y_{o,\\mathrm{LEFT}}(t_{\\mathrm{pre}})-y_{o,\\mathrm{RIGHT}}(t_{\\mathrm{pre}})$",
            "only the mean response is cube-dominated",
            "not completed prospective A7",
            "lowest released H01 seed (9400)",
            "selected without outcomes",
            "selected block happened to contain four successes while full H01 has six failures",
            "not timing maps or outcome coverage",
        ):
            self.assertIn(required, paper)
        for required in (
            "all four original/reflected x LEFT/RIGHT cells",
            "minimum released\nseed 9400",
            "not timing maps, physical-time alignment, annotations or\nforecast-accuracy evidence",
        ):
            self.assertIn(required, readme)

    def test_build_records_representative_package_hashes(self):
        build = (PACKAGE / "analysis/build_package.py").read_text()
        self.assertIn('run("representative H01 media"', build)
        self.assertIn('run("representative H01 media verification"', build)
        self.assertIn('"manifest_sha256": digest(representative_media_path)', build)
        self.assertIn('"packager_sha256": digest(package / "analysis/package_representative_media.py")', build)
        self.assertIn("paper/media/representative_h01_video_verification.json", build)
        self.assertIn('"receipt_sha256": digest(representative_verification_path)', build)
        self.assertIn('"verifier_sha256": digest(package / "analysis/verify_representative_media.py")', build)


if __name__ == "__main__":
    unittest.main()
