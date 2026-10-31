"""Independent cached-byte and full-decode checks for H01 publication media."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / "analysis/verify_representative_media.py"
SPEC = importlib.util.spec_from_file_location("verify_representative_media", MODULE_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


EXPECTED_VIDEOS = {
    "v3b001:nano:seed9400:control:left": {
        "git_blob": "0f12b9e60c0697929b90b5303b77e78567095056",
        "bytes": 1_662_892,
        "sha256": "78e9c2111faf3f4c180724314e8d0fe8a61dbdfb78e0fefa44d34434318a0b62",
        "frames": 164,
        "duration_microseconds": 10_933_333,
    },
    "v3b001:nano:seed9400:control:right": {
        "git_blob": "59d744da63e3bb8ff0f6ba041631cf6b5fb33e6e",
        "bytes": 4_944_505,
        "sha256": "67422e7d8d52dcdc74325d3d08e37cbd5b6c45bdef3721f4323625166065d84b",
        "frames": 492,
        "duration_microseconds": 32_800_000,
    },
    "v3b001:nano:seed9400:position_mirrored:left": {
        "git_blob": "a2a51663dbc8741a77d9b352c30e06a10032b85f",
        "bytes": 2_903_051,
        "sha256": "db73e5c504f6aa802008b4679165861b575b371604b84be0a6bdb12cd9e38df2",
        "frames": 287,
        "duration_microseconds": 19_133_333,
    },
    "v3b001:nano:seed9400:position_mirrored:right": {
        "git_blob": "1406493af421f8fdd10321fc2ebc092eef6a8a7b",
        "bytes": 2_964_567,
        "sha256": "296c8128b98674c94b402a56206314e48dc28e11ff62eedec55c0ea1ad5a46df",
        "frames": 287,
        "duration_microseconds": 19_133_333,
    },
}


class RepresentativeMediaVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = VERIFY.build_receipt()

    def test_pins_exact_media_evidence_and_canonical_indices(self):
        receipt = self.receipt
        self.assertEqual(
            receipt["basis"]["media_evidence_commit"],
            "34239ca102cc7b1a819e4c5c7c32b24e3bdc43f0",
        )
        self.assertEqual(
            receipt["basis"]["evidence_manifest"]["git_blob"],
            "cf16c8176ff66a13fc3dd21df8c7e9380d7eddcf",
        )
        self.assertEqual(
            receipt["basis"]["evidence_manifest"]["sha256"],
            "34f626e91c44f51bdbfd73cfa4ffc48d68fe645d1e85f94e863c56b3451b12ba",
        )
        self.assertEqual(
            receipt["canonical_indices"]["videos"]["sha256"],
            "c22cf0539603aac259d99006b4fc1fa240b277e6423dbb10e9e81586d253e87a",
        )
        self.assertEqual(
            receipt["canonical_indices"]["source_manifest_and_videos"]["sha256"],
            "e80f7eed465b8df4b479a08caaef1412c0030c2a31e2ff17e29eb6fbe9c12cfd",
        )
        self.assertEqual(
            (
                receipt["canonical_indices"]["videos"]["canonical_bytes"],
                receipt["canonical_indices"]["source_manifest_and_videos"][
                    "canonical_bytes"
                ],
            ),
            (1224, 1303),
        )

    def test_all_four_git_blobs_are_independently_probed_and_decoded(self):
        self.assertEqual(
            {row["registered_cell_id"] for row in self.receipt["videos"]},
            set(EXPECTED_VIDEOS),
        )
        for row in self.receipt["videos"]:
            expected = EXPECTED_VIDEOS[row["registered_cell_id"]]
            with self.subTest(cell=row["registered_cell_id"]):
                self.assertEqual(row["git_object"]["git_blob"], expected["git_blob"])
                self.assertEqual(row["git_object"]["bytes"], expected["bytes"])
                self.assertEqual(row["git_object"]["sha256"], expected["sha256"])
                self.assertEqual(row["ffprobe"]["codec_name"], "h264")
                self.assertEqual(row["ffprobe"]["pixel_format"], "yuv420p")
                self.assertEqual(
                    (row["ffprobe"]["width"], row["ffprobe"]["height"]),
                    (1280, 720),
                )
                self.assertEqual(
                    (
                        row["ffprobe"]["fps_numerator"],
                        row["ffprobe"]["fps_denominator"],
                    ),
                    (15, 1),
                )
                self.assertFalse(row["ffprobe"]["has_audio"])
                self.assertEqual(row["ffprobe"]["frame_count"], expected["frames"])
                self.assertEqual(
                    row["ffprobe"]["duration_microseconds"],
                    expected["duration_microseconds"],
                )
                self.assertTrue(row["ffmpeg"]["full_stream_decode_passed"])
                self.assertEqual(row["ffmpeg"]["decoded_frame_count"], expected["frames"])
                self.assertEqual(
                    row["ffmpeg"]["decoded_duration_microseconds"],
                    expected["duration_microseconds"],
                )
                self.assertEqual(row["ffmpeg"]["duplicate_frames"], 0)
                self.assertEqual(row["ffmpeg"]["dropped_frames"], 0)

    def test_totals_selection_and_zero_activity_are_fail_closed(self):
        receipt = self.receipt
        totals = receipt["totals"]
        self.assertEqual(
            (
                totals["video_count"],
                totals["video_bytes"],
                totals["source_manifest_and_video_file_count"],
                totals["source_manifest_and_video_bytes"],
            ),
            (4, 12_475_015, 5, 12_522_030),
        )
        self.assertEqual(totals["video_full_stream_decodes_passed"], 4)
        self.assertEqual(totals["materialized_video_files"], 0)
        selection = receipt["selection_disclosure"]
        self.assertEqual(selection["selected_seed"], 9400)
        self.assertEqual(
            (
                selection["selected_successes"],
                selection["selected_failures"],
                selection["full_cohort_successes"],
                selection["full_cohort_failures"],
            ),
            (4, 0, 102, 6),
        )
        self.assertFalse(selection["outcome_used_for_selection"])
        self.assertFalse(selection["request_used_for_selection"])
        self.assertFalse(selection["substitution_for_missing_media"])
        self.assertTrue(all(value == 0 for value in receipt["zero_activity"].values()))

    def test_annotation_timing_and_gate_boundaries_match_spec(self):
        boundary = self.receipt["claim_boundary"]
        self.assertFalse(boundary["human_visual_review_performed"])
        for key in (
            "not_annotation_media",
            "not_annotation_evidence",
            "not_human_label_evidence",
            "not_scientific_timing_maps",
            "not_physical_time_alignment",
            "not_forecast_accuracy_evidence",
            "not_outcome_coverage_evidence",
            "does_not_advance_any_science_or_release_gate",
            "does_not_verify_remote_raw_or_pvc_availability",
            "cache_presence_is_point_in_time",
        ):
            self.assertTrue(boundary[key])
        spec = (PACKAGE / "docs/ABLATION_SPEC.md").read_text()
        self.assertIn("They must not be used as scientific timing maps.", spec)
        self.assertIn(
            "without counterpart images, model labels, instructions or success outcomes",
            spec,
        )
        self.assertIn("representative videos chosen by a declared rule", spec)

    def test_checked_in_receipt_is_exact_deterministic_reproduction(self):
        path = PACKAGE / "paper/media/representative_h01_video_verification.json"
        self.assertEqual(json.loads(path.read_text()), self.receipt)


if __name__ == "__main__":
    unittest.main()
