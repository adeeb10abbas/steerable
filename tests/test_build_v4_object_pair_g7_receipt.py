from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.build_v4_object_pair_g7_receipt import build_receipt


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


class ObjectPairG7ReceiptTests(unittest.TestCase):
    def test_complete_valid_pilot_passes_without_success_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = []
            ledger = []
            records = []
            scenarios = (
                ["original_sham"] * 8
                + ["destination_static"] * 8
                + ["move_stop"] * 8
            )
            for index, scenario in enumerate(scenarios):
                episode_id = f"pilot-{index:02d}"
                queue.append(
                    {
                        "episode_id": episode_id,
                        "family": "C7",
                        "cohort": "engineering_pilot",
                        "reuse_episode_ids": [],
                        "factors": {"scenario": scenario},
                    }
                )
                ledger.append(
                    {
                        "episode_id": episode_id,
                        "status": "valid",
                        "outcome": {"failure_label": "no_grasp"},
                    }
                )
                records.append({"episode_id": episode_id})
            queue_path = root / "queue.jsonl"
            lock_path = root / "lock.json"
            inventory_path = root / "inventory.json"
            review_path = root / "review.json"
            ledger_path = root / "ledger.jsonl"
            ledger_manifest_path = root / "ledger-manifest.json"
            montage_path = root / "montage.png"
            montage_path.write_bytes(b"montage")
            montage_sha = __import__("hashlib").sha256(b"montage").hexdigest()
            _write_jsonl(queue_path, queue)
            _write_json(
                lock_path,
                {
                    "release_status": "PILOT_RELEASED",
                    "released_families": ["C7"],
                },
            )
            _write_json(
                inventory_path,
                {
                    "videos_all_hash_verified_and_decoded": True,
                    "valid_episode_count": 24,
                    "montage": {"sha256": montage_sha},
                    "records": records,
                },
            )
            _write_json(
                review_path,
                {
                    "passed": True,
                    "montage_sha256": montage_sha,
                    "reviewed_episode_ids": [row["episode_id"] for row in queue],
                    "assertions": {
                        "all_videos_decode_without_visible_corruption": True,
                        "robot_scene_and_task_objects_visible": True,
                    },
                },
            )
            _write_jsonl(ledger_path, ledger)
            _write_json(
                ledger_manifest_path,
                {
                    "validation_preview": {"ok": True, "error_count": 0},
                    "reconciliation": {"missing_episode_ids": []},
                    "outputs": {"accepted_count": 24},
                },
            )
            receipt = build_receipt(
                queue_path=queue_path,
                runtime_lock_path=lock_path,
                inventory_path=inventory_path,
                review_path=review_path,
                accepted_ledger_path=ledger_path,
                ledger_manifest_path=ledger_manifest_path,
            )
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["behavioral_outcomes_preserved"], {"no_grasp": 24})


if __name__ == "__main__":
    unittest.main()
