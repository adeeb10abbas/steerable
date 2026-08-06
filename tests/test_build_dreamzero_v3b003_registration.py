from __future__ import annotations

import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003"
SOURCE = ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class DreamZeroV3B003RegistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = jsonl(OUT / "dreamzero_mirror_v3b003_cells.jsonl")
        cls.source = {row["cell_id"]: row for row in jsonl(SOURCE)}
        cls.amendment = json.loads((OUT / "post_result_dreamzero_mirror_v3b003_amendment.json").read_text())
        cls.manifest = json.loads((OUT / "dreamzero_mirror_v3b003_manifest.json").read_text())

    def test_exact_108_cell_inheritance(self) -> None:
        self.assertEqual(len(self.rows), 108)
        by_seed = defaultdict(list)
        for row in self.rows:
            by_seed[row["environment_seed"]].append(row)
        self.assertEqual(sorted(by_seed), list(range(9400, 9427)))
        for rows in by_seed.values():
            self.assertEqual({(row["arm"], row["relation"]) for row in rows}, {("control", "left"), ("control", "right"), ("position_mirrored", "left"), ("position_mirrored", "right")})
            self.assertEqual(sorted(row["execution_order_index_within_seed"] for row in rows), [1, 2, 3, 4])

    def test_source_order_fixture_and_prompt_bytes(self) -> None:
        for row in self.rows:
            source = self.source[row["source_v3b001_cell_id"]]
            self.assertEqual(row["fixture_sha256"], source["fixture_sha256"])
            self.assertEqual(row["execution_order_index_within_seed"], source["execution_order_index_within_seed"])
            self.assertEqual(row["prompt"], "Put the Rubik's cube to the left of the bowl." if row["relation"] == "left" else "Put the Rubik's cube to the right of the bowl.")

    def test_s2_identity_and_effective_seed_disclosure(self) -> None:
        self.assertEqual(self.amendment["design"]["identity_binding"], "V2-A015:dreamzero_action_cfg_s2")
        self.assertFalse(self.amendment["runtime_contract_source"]["baseline_s1_used"])
        self.assertTrue(all(row["effective_model_noise_seed"] == 1140 for row in self.rows))
        self.assertTrue(all(row["runtime_identity_requirement"]["action_cfg_style_scale"] == 2 for row in self.rows))
        self.assertTrue(all(row["runtime_identity_requirement"]["open_loop_horizon"] == 8 for row in self.rows))

    def test_zero_preinference_and_hash_bound_outputs(self) -> None:
        boundary = self.amendment["release_boundary"]
        self.assertEqual(boundary["model_requests_before_registration"], 0)
        self.assertEqual(boundary["behavioral_episodes_before_registration"], 0)
        self.assertFalse(boundary["behavioral_release"])
        for key in ("amendment", "cells"):
            record = self.manifest["files"][key]
            path = OUT / record["path"]
            self.assertEqual(record["sha256"], sha256(path))
            self.assertEqual(record["bytes"], path.stat().st_size)


if __name__ == "__main__":
    unittest.main()
