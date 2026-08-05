from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/v3/dreamzero_droid/adapter.py"
SPEC = importlib.util.spec_from_file_location("v3_dreamzero_adapter", MODULE_PATH)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def _server_contract(directory: Path) -> Path:
    future_root = directory / "futures"
    future_root.mkdir(exist_ok=True)
    payload = {
        "schema_version": "vla-wam-shared-v2-dreamzero-v2a015-server-contract-v1",
        "amendment_id": "V2-A015",
        "official_repository_commit": adapter.FROZEN_SOURCE_COMMIT,
        "port": 18022,
        "world_size": 2,
        "official_noise_seed": adapter.OFFICIAL_NOISE_SEED,
        "enable_dit_cache": True,
        "runtime_num_inference_steps": adapter.RUNTIME_NUM_INFERENCE_STEPS,
        "evaluated_dit_steps_with_cache": adapter.EVALUATED_DIT_STEPS,
        "action_cfg_style_scale": adapter.ACTION_CFG_STYLE_SCALE,
        "video_cfg_scale": adapter.VIDEO_CFG_SCALE,
        "future_root": str(future_root),
    }
    path = directory / "server_contract.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _runtime(directory: Path) -> Path:
    dreamzero_status = "2" * 64
    robolab_status = "3" * 64
    combined = hashlib.sha256(json.dumps(
        {"dreamzero": dreamzero_status, "robolab": robolab_status},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    payload = {
        "schema_version": adapter.RUNTIME_SCHEMA,
        "study_id": adapter.STUDY_ID,
        "model_id": adapter.MODEL_ID,
        "identity_binding": adapter.IDENTITY_BINDING,
        "baseline_model_id_rejected": adapter.BASELINE_MODEL_ID,
        "runtime_id": "dreamzero-s2-test-runtime",
        "source_commit": adapter.FROZEN_SOURCE_COMMIT,
        "robolab_commit": adapter.FROZEN_ROBOLAB_COMMIT,
        "checkpoint_identifier": adapter.FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": adapter.FROZEN_CHECKPOINT["revision"],
        "checkpoint_manifest_sha256": adapter.FROZEN_CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_payload_aggregate_sha256": adapter.FROZEN_CHECKPOINT_AGGREGATE_SHA256,
        "tokenizer_payload_aggregate_sha256": adapter.FROZEN_TOKENIZER_AGGREGATE_SHA256,
        "checkpoint_sha256": adapter.checkpoint_contract_sha256(ROOT),
        "overlay_target_sha256": adapter.OVERLAY_TARGET_SHA256,
        "overlay_patch_sha256": adapter.OVERLAY_PATCH_SHA256,
        "action_cfg_style_scale": adapter.ACTION_CFG_STYLE_SCALE,
        "baseline_action_cfg_equivalent": adapter.BASELINE_ACTION_CFG_EQUIVALENT,
        "video_cfg_scale": adapter.VIDEO_CFG_SCALE,
        "official_noise_seed": adapter.OFFICIAL_NOISE_SEED,
        "runtime_num_inference_steps": adapter.RUNTIME_NUM_INFERENCE_STEPS,
        "evaluated_dit_steps": adapter.EVALUATED_DIT_STEPS,
        "dit_cache": True,
        "open_loop_horizon": adapter.OPEN_LOOP_HORIZON,
        "action_chunk_shape": adapter.ACTION_CHUNK_SHAPE,
        "environment_lock_hash": "4" * 64,
        "adapter_contract_hash": adapter.adapter_contract_sha256(ROOT),
        "binding_contract_sha256": adapter.binding_contract_sha256(ROOT),
        "external_repository_diff_hash": combined,
        "dreamzero_dir_status_sha256": dreamzero_status,
        "robolab_dir_status_sha256": robolab_status,
        "simulator_version": "test",
        "renderer_backend": "test",
        "phase_a_queue_sha256": adapter.sha256_file(ROOT / adapter.QUEUE_RELATIVE),
        "frozen_v2_source_sha256": adapter._observed_source_hashes(ROOT),
        "adapter_source_sha256": adapter.adapter_source_sha256(ROOT),
    }
    path = directory / "runtime.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _gate(directory: Path, runtime: Path, server: Path) -> Path:
    future_root = json.loads(server.read_text())["future_root"]
    payload = {
        "schema_version": adapter.GATE_SCHEMA,
        "study_id": adapter.STUDY_ID,
        "model_id": adapter.MODEL_ID,
        "identity_binding": adapter.IDENTITY_BINDING,
        "phase": adapter.PHASE,
        "phase_a_queue_sha256": adapter.sha256_file(ROOT / adapter.QUEUE_RELATIVE),
        "runtime_identity_sha256": adapter.sha256_file(runtime),
        "left_prompt": adapter.PROMPTS["left"],
        "right_prompt": adapter.PROMPTS["right"],
        "action_cfg_style_scale": adapter.ACTION_CFG_STYLE_SCALE,
        "video_cfg_scale": adapter.VIDEO_CFG_SCALE,
        "effective_official_model_noise_seed": adapter.OFFICIAL_NOISE_SEED,
        "model_blind_neutral_reset_fixture_passed": True,
        "raw_video_action_jsonl_write_passed": True,
        "fixed_observation_exact_repeat_passed": True,
        "fixed_observation_left_right_prompt_sensitivity_passed": True,
        "all_exposed_futures_retained": True,
        "official_full_reset_decode_passed": True,
        "behavioral_release": True,
        "action_shape": adapter.ACTION_CHUNK_SHAPE,
        "left_exact_repeat_action_bit_identical": True,
        "left_exact_repeat_latent_bit_identical": True,
        "left_right_action_rms": 0.03,
        "left_right_latent_rms": 0.17,
        "left_action_sha256": "5" * 64,
        "right_action_sha256": "6" * 64,
        "left_latent_sha256": "7" * 64,
        "right_latent_sha256": "8" * 64,
        "gate_artifact_sha256": "9" * 64,
        "server_contract_path": str(server),
        "server_contract_sha256": adapter.sha256_file(server),
        "future_root": future_root,
    }
    path = directory / "release.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _trace_and_futures(directory: Path, cell: dict, count: int) -> Path:
    actions = directory / "actions.npy"
    chunks = directory / "chunks.npy"
    executable = directory / "executable.npy"
    returned = directory / "returned.npy"
    latent = directory / "latent.pt"
    decoded = directory / "official_full_decode.mp4"
    for path, payload in (
        (actions, b"actions"), (chunks, b"chunks"),
        (executable, b"executable"), (returned, b"returned"),
        (latent, b"latent"), (decoded, b"decoded-full-imagination"),
    ):
        path.write_bytes(payload)
    future = {
        "schema_version": "vla-wam-shared-v2-dreamzero-v2a015-future-retention-v1",
        "amendment_id": "V2-A015",
        "action_cfg_style_scale": adapter.ACTION_CFG_STYLE_SCALE,
        "video_cfg_scale": adapter.VIDEO_CFG_SCALE,
        "requests": [{
            "prompt": cell["prompt"],
            "action_cfg_style_scale": adapter.ACTION_CFG_STYLE_SCALE,
            "returned_action": {"path": str(returned), "sha256": adapter.sha256_file(returned)},
            "latent_video": {"path": str(latent), "sha256": adapter.sha256_file(latent)},
        }],
        "official_reset_decode": [{
            "path": str(decoded), "sha256": adapter.sha256_file(decoded)
        }],
    }
    future_path = directory / "future_manifest.json"
    future_path.write_text(json.dumps(future, sort_keys=True))
    trace = {
        "schema_version": "vla-wam-shared-v3-dreamzero-s2-action-trace-v1",
        "study_id": adapter.STUDY_ID,
        "model_id": adapter.MODEL_ID,
        "identity_binding": adapter.IDENTITY_BINDING,
        "environment_seed": cell["environment_seed"],
        "sampling_seed_label": cell["sampling_seed"],
        "effective_official_model_noise_seed": adapter.OFFICIAL_NOISE_SEED,
        "sampling_seed_semantics": (
            "registered matched-pair label; released checkpoint noise remains fixed at 1140"
        ),
        "prompt": cell["prompt"],
        "requested_relation": cell["relation"],
        "prompt_controller": "episode_static",
        "checkpoint": adapter.FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": adapter.FROZEN_CHECKPOINT["revision"],
        "official_repository_commit": adapter.FROZEN_SOURCE_COMMIT,
        "action_cfg_style_scale": adapter.ACTION_CFG_STYLE_SCALE,
        "video_cfg_scale": adapter.VIDEO_CFG_SCALE,
        "open_loop_execution_horizon": adapter.OPEN_LOOP_HORIZON,
        "request_count": 1,
        "executed_actions": {
            "path": str(actions), "sha256": adapter.sha256_file(actions),
            "bytes": actions.stat().st_size, "count": count,
            "shape": [count, 8], "dtype": "float32",
        },
        "returned_raw_chunks": {
            "path": str(chunks), "sha256": adapter.sha256_file(chunks),
            "bytes": chunks.stat().st_size, "count": 1,
            "shape": [1, 24, 8], "dtype": "float32",
        },
        "returned_executable_chunks": {
            "path": str(executable), "sha256": adapter.sha256_file(executable),
            "bytes": executable.stat().st_size, "count": 1,
            "shape": [1, 24, 8], "dtype": "float32",
        },
        "future_manifest": {
            "path": str(future_path), "sha256": adapter.sha256_file(future_path),
            "request_count": 1, "official_decode_count": 1,
        },
    }
    path = directory / "trace.json"
    path.write_text(json.dumps(trace, sort_keys=True))
    return path


class DreamZeroAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_v3_model_id_binds_only_to_v2a015_s2(self) -> None:
        arm = adapter._binding_arm(ROOT)
        self.assertEqual(arm["model_id"], "dreamzero_droid_action_cfg")
        self.assertEqual(arm["action_guidance"], 2.0)
        runtime = _runtime(self.directory)
        value = json.loads(runtime.read_text())
        value["model_id"] = adapter.BASELINE_MODEL_ID
        value["action_cfg_style_scale"] = 1.0
        runtime.write_text(json.dumps(value))
        with self.assertRaisesRegex(adapter.AdapterError, "model_id"):
            adapter.validate_runtime_identity(ROOT, runtime)

    def test_selects_only_new_complete_pair(self) -> None:
        pair = adapter.load_authorized_pair(ROOT, 8303)
        self.assertEqual([row["relation"] for row in pair], ["left", "right"])
        with self.assertRaisesRegex(adapter.AdapterError, "8303-8329"):
            adapter.load_authorized_pair(ROOT, 8302)

    def test_preflight_requires_future_aware_release(self) -> None:
        server = _server_contract(self.directory)
        runtime = _runtime(self.directory)
        gate = _gate(self.directory, runtime, server)
        result = adapter.preflight(ROOT, 8329, runtime, gate)
        self.assertTrue(result["baseline_s1_rejected"])
        value = json.loads(gate.read_text())
        value["official_full_reset_decode_passed"] = False
        gate.write_text(json.dumps(value))
        with self.assertRaisesRegex(adapter.AdapterError, "official_full_reset_decode"):
            adapter.preflight(ROOT, 8329, runtime, gate)

    def test_plan_is_full_rtx_pair_and_rejects_port_5000(self) -> None:
        command = adapter.bridge_command(
            ROOT, 8303, self.directory / "runtime.json", self.directory / "gate.json",
            self.directory / "sim", self.directory / "actions", "server", 18022,
        )
        self.assertEqual(command[command.index("--condition") + 1], "both")
        self.assertEqual(
            command[command.index("--simulator-lane") + 1],
            "raytrace-rtxpro6000-ali",
        )
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        with self.assertRaisesRegex(adapter.AdapterError, "5000"):
            adapter.bridge_command(
                ROOT, 8303, Path("r"), Path("g"), Path("o"), Path("a"), "x", 5000
            )

    def test_compiles_full_imagination_and_execution_separately(self) -> None:
        runtime = _runtime(self.directory)
        cell = adapter.load_authorized_pair(ROOT, 8303)[0]
        trace = _trace_and_futures(self.directory, cell, 6)
        video = self.directory / "rollout.mp4"
        video.write_bytes(b"actual-simulator-rollout")
        samples = [
            {
                "action_step": step,
                "object_xyz": [0.1, 0.2 if step >= 4 else 0.0, 0.14 if step >= 1 else 0.1],
                "reference_xyz": [0.1, 0.0, 0.1],
                "grippers_open": step == 6,
            }
            for step in range(7)
        ]
        capture = {
            "schema_version": adapter.CAPTURE_SCHEMA,
            "registered_cell_id": cell["cell_id"],
            "attempt_id": "attempt-1",
            "identity_binding": adapter.IDENTITY_BINDING,
            "environment_seed": 8303,
            "policy_seed": 8303,
            "effective_official_model_noise_seed": adapter.OFFICIAL_NOISE_SEED,
            "requested_relation": "left",
            "prompt": cell["prompt"],
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
        self.assertFalse(record["dreamzero_identity"]["baseline_s1_used"])
        retained = record["future_evidence"]["retained_artifacts"]
        self.assertTrue(any("official_reset_decode" in row for row in retained))
        self.assertNotEqual(
            record["artifacts"]["viewport_video"]["sha256"],
            record["artifacts"]["exposed_future_manifest"]["sha256"],
        )

    def test_missing_official_decode_cannot_compile(self) -> None:
        runtime = _runtime(self.directory)
        cell = adapter.load_authorized_pair(ROOT, 8303)[0]
        trace_path = _trace_and_futures(self.directory, cell, 6)
        trace = json.loads(trace_path.read_text())
        future_path = Path(trace["future_manifest"]["path"])
        future = json.loads(future_path.read_text())
        future["official_reset_decode"] = []
        future_path.write_text(json.dumps(future))
        trace["future_manifest"]["sha256"] = adapter.sha256_file(future_path)
        trace_path.write_text(json.dumps(trace))
        with self.assertRaisesRegex(adapter.AdapterError, "official full reset decode"):
            adapter._validate_action_and_future_trace(trace_path, cell, 6)

    def test_partial_attempt_compiles_only_to_infrastructure_stream(self) -> None:
        runtime = _runtime(self.directory)
        cell = adapter.load_authorized_pair(ROOT, 8303)[1]
        capture = {
            "schema_version": adapter.INFRA_CAPTURE_SCHEMA,
            "registered_cell_id": cell["cell_id"],
            "attempt_id": "attempt-technical-1",
            "identity_binding": adapter.IDENTITY_BINDING,
            "environment_seed": cell["environment_seed"],
            "policy_seed": cell["sampling_seed"],
            "prompt": cell["prompt"],
            "requested_relation": cell["relation"],
            "classification": "technical_invalid",
            "stage": "official_future_finalize",
            "error": "official reset decode did not finalize",
            "log_hash": "b" * 64,
            "runtime_intervention": False,
            "repair_attempt_id": None,
            "event_timeline": [
                {"sequence": 0, "stage": "attempt_started"},
                {"sequence": 1, "stage": "official_future_finalize_failed"},
            ],
        }
        record = adapter.build_infrastructure_record(
            ROOT, cell, capture, runtime, self.directory / "infra.jsonl"
        )
        self.assertEqual(record["classification"], "technical_invalid")
        self.assertFalse(record["behavioral_result_valid"])
        self.assertNotIn("requested_success", record)
        self.assertNotIn("failure_taxonomy", record)


if __name__ == "__main__":
    unittest.main()
