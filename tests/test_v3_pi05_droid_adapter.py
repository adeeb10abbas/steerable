from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/v3/pi05_droid/adapter.py"
SPEC = importlib.util.spec_from_file_location("v3_pi05_adapter", MODULE_PATH)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def _runtime(directory: Path) -> Path:
    openpi_status = "2" * 64
    robolab_status = "3" * 64
    combined = hashlib.sha256(json.dumps(
        {"openpi": openpi_status, "robolab": robolab_status},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    payload = {
        "schema_version": adapter.RUNTIME_SCHEMA,
        "study_id": adapter.STUDY_ID,
        "model_id": adapter.MODEL_ID,
        "runtime_id": "pi05-test-runtime",
        "openpi_commit": adapter.FROZEN_OPENPI_COMMIT,
        "robolab_commit": adapter.FROZEN_ROBOLAB_COMMIT,
        "openpi_config": adapter.FROZEN_CONFIG,
        "checkpoint_identifier": adapter.FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": adapter.FROZEN_CHECKPOINT["revision"],
        "checkpoint_manifest_sha256": adapter.FROZEN_CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_sha256": adapter.checkpoint_contract_sha256(ROOT),
        "environment_lock_hash": "4" * 64,
        "adapter_contract_hash": adapter.adapter_contract_sha256(ROOT),
        "external_repository_diff_hash": combined,
        "openpi_dir_status_sha256": openpi_status,
        "robolab_dir_status_sha256": robolab_status,
        "simulator_version": "test",
        "renderer_backend": "test",
        "open_loop_horizon": 15,
        "action_chunk_shape": [15, 8],
        "phase_a_queue_sha256": adapter.sha256_file(ROOT / adapter.QUEUE_RELATIVE),
        "frozen_v2_source_sha256": adapter._observed_source_hashes(ROOT),
        "adapter_source_sha256": adapter.adapter_source_sha256(ROOT),
    }
    path = directory / "runtime.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _gate(directory: Path, runtime: Path) -> Path:
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
        "open_loop_horizon": 15,
        "action_shape": [15, 8],
        "left_exact_repeat_bit_identical": True,
        "left_right_action_rms": 0.01,
        "left_action_sha256": "5" * 64,
        "right_action_sha256": "6" * 64,
        "gate_artifact_sha256": "7" * 64,
    }
    path = directory / "release.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _action_trace(directory: Path, cell: dict, count: int) -> Path:
    actions_path = directory / "actions.npy"
    chunks_path = directory / "chunks.npy"
    actions_path.write_bytes(b"retained-float32-action-array")
    chunks_path.write_bytes(b"retained-float32-returned-chunk-array")
    metadata = {
        "schema_version": "vla-wam-shared-v3-pi05-action-trace-v1",
        "study_id": adapter.STUDY_ID,
        "model_id": adapter.MODEL_ID,
        "environment_seed": cell["environment_seed"],
        "sampling_seed_base": cell["sampling_seed"],
        "prompt": cell["prompt"],
        "requested_relation": cell["relation"],
        "prompt_controller": "episode_static",
        "open_loop_execution_horizon": 15,
        "request_sampling_seeds": [cell["sampling_seed"] * 1000],
        "executed_actions": {
            "path": str(actions_path),
            "sha256": adapter.sha256_file(actions_path),
            "bytes": actions_path.stat().st_size,
            "count": count,
            "shape": [count, 8],
            "dtype": "float32",
        },
        "returned_action_chunks": {
            "path": str(chunks_path),
            "sha256": adapter.sha256_file(chunks_path),
            "bytes": chunks_path.stat().st_size,
            "count": 1,
            "shape": [1, 15, 8],
            "dtype": "float32",
        },
    }
    path = directory / "trace.json"
    path.write_text(json.dumps(metadata, sort_keys=True))
    return path


class Pi05AdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selects_only_new_complete_pair(self) -> None:
        pair = adapter.load_authorized_pair(ROOT, 8303)
        self.assertEqual([row["relation"] for row in pair], ["left", "right"])
        self.assertTrue(all(row["status"] == "authorized_new" for row in pair))
        with self.assertRaisesRegex(adapter.AdapterError, "8303-8329"):
            adapter.load_authorized_pair(ROOT, 8302)

    def test_preflight_binds_runtime_and_fresh_release(self) -> None:
        runtime = _runtime(self.directory)
        gate = _gate(self.directory, runtime)
        result = adapter.preflight(ROOT, 8329, runtime, gate)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(result["cell_ids"]), 2)
        value = json.loads(gate.read_text())
        value["fixed_observation_left_right_prompt_sensitivity_passed"] = False
        gate.write_text(json.dumps(value))
        with self.assertRaisesRegex(adapter.AdapterError, "prompt_sensitivity"):
            adapter.preflight(ROOT, 8329, runtime, gate)

    def test_plan_is_full_matched_viewport_pair(self) -> None:
        command = adapter.bridge_command(
            ROOT, 8303, self.directory / "runtime.json", self.directory / "gate.json",
            self.directory / "sim", self.directory / "actions", "127.0.0.1", 8001,
        )
        self.assertEqual(command[command.index("--condition") + 1], "both")
        self.assertEqual(command[command.index("--sampling-seed-base") + 1], "8303")
        self.assertEqual(command[command.index("--video-mode") + 1], "viewport")
        self.assertIn("--disable-subtask", command)

    def test_compiles_contact_unavailable_success_with_raw_state(self) -> None:
        runtime = _runtime(self.directory)
        cell = adapter.load_authorized_pair(ROOT, 8303)[0]
        trace = _action_trace(self.directory, cell, 6)
        video = self.directory / "video.mp4"
        video.write_bytes(b"video")
        samples = []
        for step in range(7):
            samples.append({
                "action_step": step,
                "object_xyz": [0.1, 0.2 if step >= 4 else 0.0, 0.14 if step >= 1 else 0.1],
                "reference_xyz": [0.1, 0.0, 0.1],
                "grippers_open": step == 6,
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
            "first_contact_step": None,
            "first_contact_unavailable_reason": "no verified physical contact stream",
            "wall_time_s": 1.0,
            "operational_wall_time_valid": True,
            "behavioral_result_valid_candidate": True,
            "samples": samples,
        }
        record = adapter.build_behavioral_record(
            ROOT, cell, capture, runtime, video, trace, self.directory / "raw.jsonl"
        )
        self.assertEqual(record["failure_taxonomy"], "correct")
        self.assertEqual(
            record["measurements"]["first_contact_status"],
            "instrumentation_unavailable",
        )
        self.assertEqual(record["measurements"]["scored_state_sample_count"], 7)

    def test_rejects_behavioral_failure_early_stop(self) -> None:
        runtime = _runtime(self.directory)
        cell = adapter.load_authorized_pair(ROOT, 8303)[0]
        trace = _action_trace(self.directory, cell, 1)
        video = self.directory / "video.mp4"
        video.write_bytes(b"video")
        capture = {
            "schema_version": adapter.CAPTURE_SCHEMA,
            "registered_cell_id": cell["cell_id"],
            "attempt_id": "attempt-early",
            "environment_seed": 8303,
            "policy_seed": 8303,
            "requested_relation": "left",
            "prompt": cell["prompt"],
            "requested_success": False,
            "frozen_failure_stage": "no_object_interaction",
            "actions_executed": 1,
            "action_cap": 450,
            "right_censored": False,
            "final_detached_release": True,
            "first_contact_step": None,
            "first_contact_unavailable_reason": "no contact stream",
            "wall_time_s": 1.0,
            "operational_wall_time_valid": True,
            "behavioral_result_valid_candidate": True,
            "samples": [
                {"action_step": i, "object_xyz": [0.1, 0, 0.1], "reference_xyz": [0.2, 0, 0.1], "grippers_open": True}
                for i in range(2)
            ],
        }
        with self.assertRaisesRegex(adapter.AdapterError, "450-action cap"):
            adapter.build_behavioral_record(
                ROOT, cell, capture, runtime, video, trace, self.directory / "raw.jsonl"
            )

    def test_partial_attempt_compiles_only_to_infrastructure_stream(self) -> None:
        runtime = _runtime(self.directory)
        cell = adapter.load_authorized_pair(ROOT, 8303)[1]
        capture = {
            "schema_version": adapter.INFRA_CAPTURE_SCHEMA,
            "registered_cell_id": cell["cell_id"],
            "attempt_id": "attempt-technical-1",
            "environment_seed": cell["environment_seed"],
            "policy_seed": cell["sampling_seed"],
            "prompt": cell["prompt"],
            "requested_relation": cell["relation"],
            "classification": "partial",
            "stage": "policy_request",
            "error": "policy server disconnected after partial state capture",
            "log_hash": "a" * 64,
            "runtime_intervention": False,
            "repair_attempt_id": None,
            "event_timeline": [
                {"sequence": 0, "stage": "attempt_started"},
                {"sequence": 1, "stage": "policy_request_failed"},
            ],
        }
        record = adapter.build_infrastructure_record(
            ROOT, cell, capture, runtime, self.directory / "infra.jsonl"
        )
        self.assertEqual(record["classification"], "partial")
        self.assertFalse(record["behavioral_result_valid"])
        self.assertNotIn("requested_success", record)
        self.assertNotIn("failure_taxonomy", record)


if __name__ == "__main__":
    unittest.main()
