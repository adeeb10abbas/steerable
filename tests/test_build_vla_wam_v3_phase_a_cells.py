#!/usr/bin/env python3
"""Tests for the fail-closed v3 Phase-A queue builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "build_vla_wam_v3_phase_a_cells.py"
SPEC = importlib.util.spec_from_file_location("build_v3_phase_a", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PhaseACellsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.manifest, cls.payload = MODULE.build_phase_a(ROOT)

    def test_exact_target_and_status_counts(self) -> None:
        self.assertEqual(len(self.rows), 780)
        self.assertEqual(Counter(row["arena"] for row in self.rows), {"droid_robolab": 360, "robotwin": 420})
        self.assertEqual(
            Counter(row["status"] for row in self.rows),
            {"authorized_new": 648, "preserved_candidate": 50, "blocked_pi0": 40, "preserved_r0": 42},
        )
        self.assertEqual(self.manifest["queue_sha256"], hashlib.sha256(self.payload).hexdigest())

    def test_every_pair_preserves_reset_and_seed(self) -> None:
        grouped: dict[str, list[dict]] = {}
        for row in self.rows:
            grouped.setdefault(row["pair_id"], []).append(row)
        self.assertEqual(len(grouped), 390)
        for pair_id, rows in grouped.items():
            self.assertEqual({row["relation"] for row in rows}, {"left", "right"}, pair_id)
            self.assertEqual(len(rows), 2, pair_id)
            for field in ("environment_seed", "sampling_seed", "reset_identity", "runtime_identity_requirement"):
                self.assertEqual(rows[0][field], rows[1][field], f"{pair_id}: {field}")

    def test_exact_prompts_and_pi0_blocker(self) -> None:
        droid = [row for row in self.rows if row["arena"] == "droid_robolab"]
        self.assertEqual({row["prompt"] for row in droid if row["relation"] == "left"}, {"Put the Rubik's cube to the left of the bowl."})
        self.assertEqual({row["prompt"] for row in droid if row["relation"] == "right"}, {"Put the Rubik's cube to the right of the bowl."})
        pi0_blocked = [row for row in droid if row["model_id"] == "pi0_fast_droid_vla" and row["status"] == "blocked_pi0"]
        self.assertEqual(len(pi0_blocked), 40)
        self.assertTrue(all(row["execution_status"].startswith("blocked_") for row in pi0_blocked))
        pair03 = [row for row in self.rows if row.get("scene_pair") == 3 and row["replicate"] == 1]
        self.assertEqual({row["prompt"] for row in pair03 if row["relation"] == "left"}, {"Put the small woodenblock to the left of the red playingcards box."})
        self.assertEqual({row["prompt"] for row in pair03 if row["relation"] == "right"}, {"Put the small woodenblock to the right of the red playingcards box."})

    def test_seed_schedules_are_shared_and_replicates_follow_rule(self) -> None:
        droid_models = {row["model_id"] for row in self.rows if row["arena"] == "droid_robolab"}
        for model in droid_models:
            self.assertEqual({row["environment_seed"] for row in self.rows if row["model_id"] == model}, set(range(8300, 8330)))
        for row in self.rows:
            if row["arena"] == "robotwin":
                self.assertEqual(row["environment_seed"], 4_300_000 + row["scene_pair"])
                self.assertEqual(row["sampling_seed"], 8_400 + row["scene_pair"] + 100 * row["replicate"])

    def test_deterministic_rebuild_matches_committed_artifacts(self) -> None:
        queue = ROOT / "artifacts" / "vla_wam_shared_v3" / "phase_a_cells.jsonl"
        manifest = ROOT / "artifacts" / "vla_wam_shared_v3" / "phase_a_cells_manifest.json"
        self.assertEqual(queue.read_bytes(), self.payload)
        self.assertEqual(json.loads(manifest.read_text()), self.manifest)
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.jsonl"
            manifest_path = Path(tmpdir) / "manifest.json"
            MODULE.write_outputs(queue_path, manifest_path, self.payload, self.manifest)
            self.assertEqual(queue_path.read_bytes(), self.payload)
            self.assertEqual(json.loads(manifest_path.read_text()), self.manifest)

    def test_validation_fails_when_pairing_is_broken(self) -> None:
        broken = [dict(row) for row in self.rows]
        broken[1]["sampling_seed"] += 1
        with self.assertRaises(MODULE.RegistryError):
            MODULE.validate_rows(broken)


if __name__ == "__main__":
    unittest.main()
