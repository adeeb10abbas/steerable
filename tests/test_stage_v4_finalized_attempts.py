from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.stage_v4_finalized_attempts import load_manifest_ids, stage_attempts


class StageV4FinalizedAttemptsTests(unittest.TestCase):
    def test_indexes_nested_attempts_for_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"episode_id": "episode-a"}) + "\n",
                encoding="utf-8",
            )
            attempt = root / "raw" / "run" / "nested" / "attempt-1"
            attempt.mkdir(parents=True)
            (attempt / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "episode_id": "episode-a",
                        "attempt_id": "attempt-1",
                        "status": "valid",
                    }
                ),
                encoding="utf-8",
            )
            ignored = root / "raw" / "other" / "attempt-2"
            ignored.mkdir(parents=True)
            (ignored / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "episode_id": "episode-b",
                        "attempt_id": "attempt-2",
                        "status": "valid",
                    }
                ),
                encoding="utf-8",
            )
            output = root / "indexed"
            inventory = stage_attempts(
                source_roots=[root / "raw"],
                manifest_ids=load_manifest_ids(manifest),
                output_root=output,
            )
            self.assertEqual(inventory["indexed_attempt_count"], 1)
            staged = output / "episode-a" / "attempt-1"
            self.assertTrue(staged.is_symlink())
            self.assertEqual(staged.resolve(), attempt.resolve())

    def test_loads_hash_bound_queue_from_release_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "pilot.jsonl"
            queue_bytes = (
                json.dumps({"episode_id": "episode-a"}, sort_keys=True) + "\n"
            ).encode("utf-8")
            queue.write_bytes(queue_bytes)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "queue_path": str(queue),
                        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_manifest_ids(manifest), {"episode-a"})

    def test_rejects_release_manifest_queue_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "pilot.jsonl"
            queue.write_text(
                json.dumps({"episode_id": "episode-a"}) + "\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "queue_path": str(queue),
                        "queue_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "queue hash differs"):
                load_manifest_ids(manifest)


if __name__ == "__main__":
    unittest.main()
