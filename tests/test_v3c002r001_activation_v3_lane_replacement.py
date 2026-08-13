"""Regression checks for the prospective activation-v3 replacement contract."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, file_binding
from tools.compile_v3c002r001_activation_v3_lane_replacement import (
    COMPLETION_INDEX_SCHEMA,
    TERMINATION_SCHEMA,
    _require_completion_index,
    _require_termination,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v3_lane_replacement_runner import _replacement_attempt_root
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v3_lane_replacement_runner import (
    _replacement_attempt_root,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


class ActivationV3LaneReplacementTest(unittest.TestCase):
    def _completion_index(self, root: Path, *, unstarted: list[int]) -> tuple[dict, dict, dict]:
        old_path = _write(root / "old_lane.json", {"lane": "00"})
        old_binding = file_binding(old_path)
        raw_paths = [_write(root / f"raw-{n}.jsonl", {"cell": n}) for n in range(4)]
        marker_path = _write(root / "completed.json", {
            "schema_version": "vla-wam-shared-v3c002-completed-block-v1",
            "status": "completed_behavioral_block",
            "authorization_mode": "behavioral",
            "episode_seed": 12061,
            "raw_episodes": [file_binding(path) for path in raw_paths],
        })
        index = {
            "schema_version": COMPLETION_INDEX_SCHEMA,
            "status": "captured_complete_blocks_before_replacement",
            "passed": True,
            "lane_slot": "repair-lane-00",
            "old_lane_manifest": old_binding,
            "completed_seed_blocks": [12061],
            "incomplete_seed_blocks": [12060],
            "unstarted_seed_blocks": unstarted,
            "completed_block_markers": [{"episode_seed": 12061, "marker": file_binding(marker_path)}],
        }
        return index, old_binding, {"assigned_seed_blocks": [12060, 12061, 12062]}

    def test_completion_partition_allows_only_the_one_known_incomplete_retry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            index, old_binding, old = self._completion_index(Path(raw), unstarted=[12062])
            self.assertEqual(
                _require_completion_index(index, old_binding, old, "repair-lane-00", 12060, [12060, 12061, 12062]),
                [12061],
            )

    def test_completion_partition_rejects_retry_seed_as_unstarted_or_completed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            index, old_binding, old = self._completion_index(Path(raw), unstarted=[12060, 12062])
            with self.assertRaises(ContractError):
                _require_completion_index(index, old_binding, old, "repair-lane-00", 12060, [12060, 12061, 12062])

    def test_combined_deleted_pod_capture_is_correlated_by_old_pod_uid(self) -> None:
        old = {
            "policy_server_pod_uid": "old-pod-00",
            "policy_server_gpu_uuid": "GPU-old",
            "server_port": 8100,
            "server_process_identity": "old-process",
            "server_lock_identity": "old-lock",
        }
        capture = {
            "schema_version": TERMINATION_SCHEMA,
            "captured_at_utc": "2026-08-12T12:00:00Z",
            "model_request_count": 0,
            "behavioral_action_count": 0,
            "behavioral_episode_count": 0,
            "records": [{
                "policy_server_pod_uid": "old-pod-00",
                "current_pod_lookup": "NotFound",
                "old_pod_uid_absent_from_current_cluster": True,
                "events": [{"reason": "Killing", "involvedObject": {"uid": "old-pod-00"}}],
            }],
        }
        _require_termination(capture, {"sha256": "present"}, old, "repair-lane-00")
        capture["records"][0]["events"][0]["involvedObject"]["uid"] = "wrong"
        with self.assertRaises(ContractError):
            _require_termination(capture, {"sha256": "present"}, old, "repair-lane-00")

    def test_replacement_attempt_never_reuses_partial_attempt001_or_existing_attempt002(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "behavioral"
            (root / "seed12060" / "attempt001").mkdir(parents=True)
            expected = root / "seed12060" / "attempt002"
            self.assertEqual(_replacement_attempt_root(root, 12060), expected)
            expected.mkdir()
            with self.assertRaises(ContractError):
                _replacement_attempt_root(root, 12060)

    def test_replacement_attempt_is_exactly_fresh_attempt002(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            behavioral = Path(raw)
            retained = behavioral / "seed12060" / "attempt001"
            retained.mkdir(parents=True)
            expected = behavioral / "seed12060" / "attempt002"
            self.assertEqual(_replacement_attempt_root(behavioral, 12060), expected)
            expected.mkdir()
            with self.assertRaises(ContractError):
                _replacement_attempt_root(behavioral, 12060)

    def test_replacement_attempt_rejects_missing_retained_partial(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ContractError):
                _replacement_attempt_root(Path(raw), 12060)


if __name__ == "__main__":
    unittest.main()
