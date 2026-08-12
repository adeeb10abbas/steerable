from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, file_binding
from tools import validate_v3c002r001_activation_v3_publication_bundle as publication


class ActivationV3PublicationBundleTests(unittest.TestCase):
    def _write_all_final_files(self, root: Path) -> None:
        for path in publication.final_paths(root).values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

    def test_no_final_files_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(publication.result_bundle_present(Path(directory)))

    def test_any_final_file_requires_the_complete_activation_v3_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = publication.final_paths(root)["results"]
            results.parent.mkdir(parents=True)
            results.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "partial activation-v3 result bundle"):
                publication.result_bundle_present(root)

    def test_streaming_raw_file_is_not_mistaken_for_a_compiled_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = publication.final_paths(root)["raw_episodes"]
            raw.parent.mkdir(parents=True)
            raw.write_text("\n", encoding="utf-8")
            self.assertFalse(publication.result_bundle_present(root))

    def test_complete_file_set_is_recognized_before_content_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_all_final_files(root)
            self.assertTrue(publication.result_bundle_present(root))

    def test_manifest_binding_cannot_point_outside_activation_v3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "activation_v3"
            root.mkdir()
            expected = root / "raw/episodes.jsonl"
            expected.parent.mkdir()
            expected.write_text("{}\n", encoding="utf-8")
            outside = Path(directory) / "elsewhere.jsonl"
            outside.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "not rooted in activation_v3"):
                publication._binding_at(file_binding(outside), expected, "test binding")

    def test_progress_mapping_requires_each_released_lane_once(self) -> None:
        values = [f"{slot}=/tmp/{slot}" for slot in publication.LANE_SLOTS]
        parsed = publication._slot_path_arguments(values, flag="--progress-lane-root")
        self.assertEqual(tuple(sorted(parsed)), publication.LANE_SLOTS)
        with self.assertRaisesRegex(ContractError, "all eight repair lanes"):
            publication._slot_path_arguments(values[:-1], flag="--progress-lane-root")


if __name__ == "__main__":
    unittest.main()
