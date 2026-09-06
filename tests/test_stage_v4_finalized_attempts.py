from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
