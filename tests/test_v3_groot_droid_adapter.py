from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/v3/groot_droid/adapter.py"
SPEC = importlib.util.spec_from_file_location("v3_groot_adapter", MODULE_PATH)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def _runtime(tmp_path: Path) -> Path:
    payload = {
        "schema_version": adapter.RUNTIME_SCHEMA,
        "study_id": adapter.STUDY_ID,
        "model_id": adapter.MODEL_ID,
        "runtime_id": "test-runtime",
        "isaac_groot_commit": adapter.FROZEN_ISAAC_GROOT_COMMIT,
        "robolab_commit": adapter.FROZEN_ROBOLAB_COMMIT,
        "checkpoint_identifier": adapter.FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": adapter.FROZEN_CHECKPOINT["revision"],
        "checkpoint_sha256": adapter.checkpoint_contract_sha256(),
        "checkpoint_files": adapter.FROZEN_CHECKPOINT_FILES,
        "external_repository_diff_hash": "2" * 64,
        "isaac_groot_dir_status_sha256": "2" * 64,
        "robolab_dir_status_sha256": "8" * 64,
        "environment_lock_hash": "3" * 64,
        "adapter_contract_hash": adapter.adapter_contract_sha256(ROOT),
        "open_loop_horizon": 8,
        "embodiment_tag": "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        "simulator_version": "test",
        "renderer_backend": "test",
        "phase_a_queue_sha256": adapter.sha256_file(ROOT / adapter.QUEUE_RELATIVE),
        "frozen_v2_source_sha256": {
            relative: adapter.sha256_file(ROOT / relative)
            for relative in adapter.FROZEN_SOURCES
        },
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _gate(tmp_path: Path, runtime: Path) -> Path:
    payload = {
        "schema_version": adapter.GATE_SCHEMA,
        "study_id": adapter.STUDY_ID,
        "model_id": adapter.MODEL_ID,
        "phase": adapter.PHASE,
        "phase_a_queue_sha256": adapter.sha256_file(ROOT / adapter.QUEUE_RELATIVE),
        "runtime_identity_sha256": adapter.sha256_file(runtime),
        "left_prompt": adapter.PROMPTS["left"],
        "right_prompt": adapter.PROMPTS["right"],
        "model_blind_neutral_reset_fixture_passed": True,
        "raw_video_action_jsonl_write_passed": True,
        "fixed_observation_exact_repeat_passed": True,
        "fixed_observation_left_right_prompt_sensitivity_passed": True,
        "behavioral_release": True,
        "left_exact_repeat_rms": 0.0,
        "left_right_action_rms": 0.1,
        "left_action_sha256": "5" * 64,
        "right_action_sha256": "6" * 64,
        "gate_artifact_sha256": "7" * 64,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


class GrootAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selects_only_authorized_new_groot_pair(self) -> None:
        cells = adapter.load_authorized_pair(ROOT, 8303)
        self.assertEqual([cell["relation"] for cell in cells], ["left", "right"])
        self.assertTrue(all(cell["status"] == "authorized_new" for cell in cells))
        with self.assertRaisesRegex(adapter.AdapterError, "8303-8329"):
            adapter.load_authorized_pair(ROOT, 8302)

    def test_preflight_binds_queue_runtime_and_gate(self) -> None:
        runtime = _runtime(self.tmp_path)
        gate = _gate(self.tmp_path, runtime)
        result = adapter.preflight(ROOT, 8329, runtime, gate)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(result["cell_ids"]), 2)

    def test_preflight_fails_on_gate_or_source_mismatch(self) -> None:
        runtime = _runtime(self.tmp_path)
        gate = _gate(self.tmp_path, runtime)
        value = json.loads(gate.read_text())
        value["fixed_observation_left_right_prompt_sensitivity_passed"] = False
        gate.write_text(json.dumps(value))
        with self.assertRaisesRegex(adapter.AdapterError, "prompt_sensitivity"):
            adapter.preflight(ROOT, 8303, runtime, gate)

        runtime_value = json.loads(runtime.read_text())
        runtime_value["frozen_v2_source_sha256"][next(iter(adapter.FROZEN_SOURCES))] = "0" * 64
        runtime.write_text(json.dumps(runtime_value))
        with self.assertRaisesRegex(adapter.AdapterError, "source_sha256"):
            adapter.validate_runtime_identity(ROOT, runtime)

    def test_bridge_command_is_complete_matched_pair(self) -> None:
        command = adapter.bridge_command(
            ROOT, 8303, self.tmp_path / "runtime.json", self.tmp_path / "gate.json",
            self.tmp_path / "sim", self.tmp_path / "actions", "127.0.0.1", 5555,
        )
        self.assertEqual(command[command.index("--condition") + 1], "both")
        self.assertEqual(command[command.index("--sampling-seed-base") + 1], "8303")
        self.assertIn("--disable-subtask", command)
        self.assertEqual(command[command.index("--video-mode") + 1], "viewport")

    def test_behavioral_compile_uses_shared_schema(self) -> None:
        runtime = _runtime(self.tmp_path)
        cell = adapter.load_authorized_pair(ROOT, 8303)[0]
        video = self.tmp_path / "video.mp4"
        actions = self.tmp_path / "actions.npy"
        video.write_bytes(b"video")
        actions.write_bytes(b"actions")
        samples = []
        for step in range(7):
            samples.append({
                "action_step": step,
                "object_xyz": [0.1, 0.0 if step < 4 else 0.2, 0.1 if step == 0 else 0.14],
                "reference_xyz": [0.1, 0.0, 0.1],
                "grippers_open": step == 6,
                "contact_detected": step >= 1,
            })
        capture = {
            "schema_version": adapter.CAPTURE_SCHEMA,
            "registered_cell_id": cell["cell_id"],
            "attempt_id": "attempt-1",
            "environment_seed": 8303,
            "policy_seed": 8303,
            "requested_relation": "left",
            "prompt": adapter.PROMPTS["left"],
            "requested_success": True,
            "frozen_failure_stage": "success",
            "actions_executed": 6,
            "action_cap": 450,
            "right_censored": False,
            "final_detached_release": True,
            "first_contact_step": 1,
            "first_contact_unavailable_reason": None,
            "wall_time_s": 1.0,
            "operational_wall_time_valid": True,
            "behavioral_result_valid_candidate": True,
            "samples": samples,
        }
        record = adapter.build_behavioral_record(
            ROOT, cell, capture, runtime, video, actions, self.tmp_path / "raw.jsonl"
        )
        self.assertEqual(record["failure_taxonomy"], "correct")
        self.assertEqual(record["measurements"]["first_contact_step"], 1)
        self.assertGreater(record["measurements"]["object_path_length_m"], 0)

    def test_gate_digest_is_bound_to_runtime_bytes(self) -> None:
        runtime = _runtime(self.tmp_path)
        gate = _gate(self.tmp_path, runtime)
        runtime.write_text(runtime.read_text() + "\n")
        with self.assertRaisesRegex(adapter.AdapterError, "runtime_identity_sha256"):
            adapter.preflight(ROOT, 8303, runtime, gate)

    def test_infrastructure_compile_stays_outside_behavioral_schema(self) -> None:
        runtime = _runtime(self.tmp_path)
        cell = adapter.load_authorized_pair(ROOT, 8303)[1]
        capture = {
            "schema_version": adapter.INFRA_CAPTURE_SCHEMA,
            "registered_cell_id": cell["cell_id"],
            "attempt_id": "attempt-infra-1",
            "environment_seed": 8303,
            "policy_seed": 8303,
            "requested_relation": "right",
            "prompt": adapter.PROMPTS["right"],
            "classification": "partial",
            "stage": "simulator_step",
            "error": "renderer transport closed after retained partial outputs",
            "log_hash": "9" * 64,
            "runtime_intervention": False,
            "repair_attempt_id": None,
            "event_timeline": [
                {"sequence": 0, "stage": "episode_start"},
                {"sequence": 1, "stage": "renderer_transport_closed"},
            ],
        }
        record = adapter.build_infrastructure_record(
            ROOT, cell, capture, runtime, None, None, self.tmp_path / "infra.jsonl"
        )
        self.assertEqual(record["record_type"], "infrastructure_attempt")
        self.assertFalse(record["behavioral_result_valid"])
        self.assertNotIn("requested_success", record)
        self.assertNotIn("failure_taxonomy", record)
        self.assertNotIn("viewport_video", record["artifacts"])
        self.assertNotIn("executed_action_trace", record["artifacts"])

    def test_behavioral_compile_rejects_failure_early_stop(self) -> None:
        runtime = _runtime(self.tmp_path)
        cell = adapter.load_authorized_pair(ROOT, 8303)[0]
        video = self.tmp_path / "video.mp4"
        actions = self.tmp_path / "actions.npy"
        video.write_bytes(b"video")
        actions.write_bytes(b"actions")
        capture = {
            "schema_version": adapter.CAPTURE_SCHEMA,
            "registered_cell_id": cell["cell_id"],
            "attempt_id": "attempt-early-stop",
            "environment_seed": 8303,
            "policy_seed": 8303,
            "requested_relation": "left",
            "prompt": adapter.PROMPTS["left"],
            "requested_success": False,
            "frozen_failure_stage": "partial",
            "actions_executed": 1,
            "action_cap": 450,
            "right_censored": False,
            "final_detached_release": False,
            "first_contact_step": None,
            "first_contact_unavailable_reason": "no verified contact stream",
            "wall_time_s": 1.0,
            "operational_wall_time_valid": True,
            "behavioral_result_valid_candidate": True,
            "samples": [
                {
                    "action_step": index,
                    "object_xyz": [0.1, 0.0, 0.1],
                    "reference_xyz": [0.2, 0.0, 0.1],
                    "grippers_open": False,
                }
                for index in range(2)
            ],
        }
        with self.assertRaisesRegex(adapter.AdapterError, "failure must run"):
            adapter.build_behavioral_record(
                ROOT, cell, capture, runtime, video, actions, self.tmp_path / "raw.jsonl"
            )


if __name__ == "__main__":
    unittest.main()
