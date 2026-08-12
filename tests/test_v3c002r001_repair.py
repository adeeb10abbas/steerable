from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import validate_assignment
from tools.build_v3c002r001_registration import assignment_rows
from tools.compile_v3c002r001_repeat_gate import validate_repeat_evidence


class RepairContractTests(unittest.TestCase):
    def test_assignment_is_exact_balanced_and_complete(self) -> None:
        rows = assignment_rows()
        self.assertEqual(len(rows), 341)
        counts = {f"repair-lane-{index:02d}": 0 for index in range(8)}
        for row in rows:
            counts[row["lane_slot"]] += 1
            self.assertEqual(len(row["conditions"]), 4)
            self.assertTrue(row["block_indivisible"])
        self.assertEqual(set(counts.values()), {42, 43})

    def test_assignment_validator_rejects_lane_mutation(self) -> None:
        rows = assignment_rows()
        rows[0]["lane_slot"] = "repair-lane-07" if rows[0]["lane_slot"] != "repair-lane-07" else "repair-lane-06"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignment.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            record = {"path": str(path), "bytes": path.stat().st_size, "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, "SHA-rank"):
                validate_assignment(record)

    def test_assignment_validator_rejects_incomplete_block(self) -> None:
        rows = assignment_rows()
        rows[0]["conditions"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignment.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            record = {"path": str(path), "bytes": path.stat().st_size, "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, "omits a condition"):
                validate_assignment(record)

    def _repeat_evidence(self, directory: Path) -> tuple[dict, dict]:
        first = np.zeros((15, 8), dtype=np.float32); middle = np.ones((15, 8), dtype=np.float32)
        records = []
        for ordinal, (condition, array) in enumerate((("canonical_left", first), ("canonical_right", middle), ("canonical_left", first))):
            path = directory / f"a{ordinal}.npy"; np.save(path, array, allow_pickle=False)
            records.append({"ordinal": ordinal, "condition": condition, "actions": file_binding(path), "seed_echo": 13_000_000})
        fixture = directory / "fixture.npz"; np.savez(fixture, x=np.zeros(1))
        server = {"policy_server_pod_uid": "pod", "policy_server_gpu_uuid": "GPU-x", "server_port": 8000, "server_process_identity": "pid:1", "server_lock_identity": "lock:x"}
        physical = {"schema_version": "vla-wam-shared-v3c002r001-model-blind-physical-gate-v1", "status": "passed_repair_same_process_zero_request_preflight", "passed": True, "lane_slot": "repair-lane-00", **server}
        response = {"schema_version": "vla-wam-shared-v3c002r001-single-server-repeat-response-v1", "status": "completed_excluded_single_server_interleaved_repeat", "passed": True, "lane_slot": "repair-lane-00", "model_request_count": 3, "successful_response_count": 3, "behavioral_episode_count": 0, "probe_seed": 13_000_000, "sequence": ["canonical_left", "canonical_right", "canonical_left"], "records": records, "first_final_repeat_exact": True, "prompt_sensitivity_distinct": True, "fixture": file_binding(fixture), "fixture_sha256": file_binding(fixture)["sha256"], **server}
        return response, physical

    def test_repeat_gate_recomputes_arrays_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            response, physical = self._repeat_evidence(Path(directory))
            self.assertEqual(len(validate_repeat_evidence(response, physical, "repair-lane-00")), 3)
            response["policy_server_gpu_uuid"] = "GPU-mutated"
            with self.assertRaisesRegex(ValueError, "different policy server"):
                validate_repeat_evidence(response, physical, "repair-lane-00")

    def test_repeat_gate_rejects_nonidentical_final_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            response, physical = self._repeat_evidence(Path(directory))
            final = Path(response["records"][2]["actions"]["path"])
            np.save(final, np.full((15, 8), 2, dtype=np.float32), allow_pickle=False)
            response["records"][2]["actions"] = file_binding(final)
            with self.assertRaisesRegex(ValueError, "recomputation"):
                validate_repeat_evidence(response, physical, "repair-lane-00")


if __name__ == "__main__":
    unittest.main()
