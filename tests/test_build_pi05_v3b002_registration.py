from __future__ import annotations

import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002"
SOURCE = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001"
    / "nano_mirror_v3b001_cells.jsonl"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class Pi05V3B002RegistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = jsonl(OUT / "pi05_mirror_v3b002_cells.jsonl")
        cls.source_rows = jsonl(SOURCE)
        cls.amendment = json.loads(
            (OUT / "post_result_pi05_mirror_v3b002_amendment.json").read_text()
        )
        cls.manifest = json.loads(
            (OUT / "pi05_mirror_v3b002_manifest.json").read_text()
        )

    def test_exact_seed_blocks_and_conditions(self) -> None:
        self.assertEqual(len(self.rows), 108)
        by_seed: dict[int, list[dict]] = defaultdict(list)
        for row in self.rows:
            by_seed[row["environment_seed"]].append(row)
        self.assertEqual(sorted(by_seed), list(range(9400, 9427)))
        expected = {
            ("control", "left"),
            ("control", "right"),
            ("position_mirrored", "left"),
            ("position_mirrored", "right"),
        }
        for seed_rows in by_seed.values():
            self.assertEqual({(r["arm"], r["relation"]) for r in seed_rows}, expected)
            self.assertEqual(
                sorted(r["execution_order_index_within_seed"] for r in seed_rows),
                [1, 2, 3, 4],
            )

    def test_exact_v3b001_order_and_fixture_reuse(self) -> None:
        source = {r["cell_id"]: r for r in self.source_rows}
        source_sha = sha256(SOURCE)
        for row in self.rows:
            original = source[row["source_v3b001_cell_id"]]
            self.assertEqual(row["source_v3b001_queue_sha256"], source_sha)
            self.assertEqual(row["environment_seed"], original["environment_seed"])
            self.assertEqual(row["arm"], original["arm"])
            self.assertEqual(row["relation"], original["relation"])
            self.assertEqual(row["fixture_sha256"], original["fixture_sha256"])
            self.assertEqual(
                row["execution_order_index_within_seed"],
                original["execution_order_index_within_seed"],
            )

    def test_prompts_predictions_and_zero_preinference_counts(self) -> None:
        prompts = self.amendment["design"]["exact_prompts"]
        self.assertEqual(
            prompts["left"], "Put the Rubik's cube to the left of the bowl."
        )
        self.assertEqual(
            prompts["right"], "Put the Rubik's cube to the right of the bowl."
        )
        self.assertEqual(set(self.amendment["registered_predictions"]), {"H1_endpoint_redirection", "H2_requested_side_depth", "H3_binary_success"})
        continuous = self.amendment["analysis_plan"]["continuous_reporting"]
        self.assertEqual(continuous["bootstrap_replicates"], 20_000)
        self.assertEqual(continuous["bootstrap_master_seed"], 3_104_159)
        boundary = self.amendment["release_boundary"]
        self.assertEqual(boundary["model_requests_before_registration"], 0)
        self.assertEqual(boundary["behavioral_episodes_before_registration"], 0)
        self.assertFalse(boundary["behavioral_release"])

    def test_manifest_hashes_and_counts(self) -> None:
        files = self.manifest["files"]
        amendment_path = OUT / files["amendment"]["path"]
        cells_path = OUT / files["cells"]["path"]
        self.assertEqual(files["amendment"]["sha256"], sha256(amendment_path))
        self.assertEqual(files["amendment"]["bytes"], amendment_path.stat().st_size)
        self.assertEqual(files["cells"]["sha256"], sha256(cells_path))
        self.assertEqual(files["cells"]["bytes"], cells_path.stat().st_size)
        self.assertEqual(files["cells"]["row_count"], 108)
        self.assertEqual(self.manifest["counts"]["matched_seeds"], 27)


if __name__ == "__main__":
    unittest.main()
