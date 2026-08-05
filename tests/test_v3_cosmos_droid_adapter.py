from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.v3.cosmos_droid.compile_pair import compile_pair
from experiments.v3.cosmos_droid.contract import (
    MODEL_CONTRACTS,
    FROZEN_GROUNDED_OBSERVATION_SHA256,
    STUDY_ID,
    ContractError,
    compute_adapter_contract_hash,
    load_authorized_pair,
    verify_repository_pins,
    verify_runtime_identity,
)
from experiments.v3.cosmos_droid.fixed_observation_gate import evaluate_responses
from experiments.v3.cosmos_droid.record_infrastructure import build_infrastructure_record
from tools.vla_wam_v3_episode_schema import (
    MEASUREMENT_FRAME_ID,
    derive_initial_state_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _runtime(tmp_path: Path, model_id: str) -> tuple[dict, Path]:
    spec = MODEL_CONTRACTS[model_id]
    payload = {
        "schema_version": "vla-wam-shared-v3-cosmos-runtime-identity-v1",
        "study_id": STUDY_ID,
        "model_id": model_id,
        "checkpoint_identifier": spec["checkpoint_id"],
        "checkpoint_revision": spec["checkpoint_revision"],
        "checkpoint_sha256": "1" * 64,
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": spec["server_repository_commit"],
        "external_repository_diff_hash": hashlib.sha256(b"").hexdigest(),
        "simulator_repository_commit": spec["robolab_repository_commit"],
        "simulator_repository_diff_hash": hashlib.sha256(b"").hexdigest(),
        "environment_lock_hash": "2" * 64,
        "adapter_contract_hash": compute_adapter_contract_hash(ROOT),
        "simulator_version": "test-simulator",
        "renderer_backend": "test-vulkan",
        "repository_pins": verify_repository_pins(ROOT, model_id),
    }
    payload["runtime_identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / f"{model_id}_runtime.json"
    path.write_text(json.dumps(payload))
    return payload, path


def _responses(seed: int, model_id: str) -> dict:
    left_action = np.zeros((32, 8), dtype=np.float32)
    right_action = np.ones((32, 8), dtype=np.float32)
    left_future = np.zeros((33, 2, 2, 3), dtype=np.uint8)
    right_future = np.ones((33, 2, 2, 3), dtype=np.uint8)
    echo = seed if MODEL_CONTRACTS[model_id]["sampling_seed_echo_required"] else None
    responses = {
        "left": {"action": left_action, "video": left_future, "sampling_seed": echo},
        "left_exact_repeat": {"action": left_action.copy(), "video": left_future.copy(), "sampling_seed": echo},
        "right": {"action": right_action, "video": right_future, "sampling_seed": echo},
    }
    for response in responses.values():
        response["observation_hashes"] = {"image": "a" * 64, "joint": "b" * 64, "gripper": "c" * 64}
    return responses


def _make_export(tmp_path: Path, *, pair, relation: str, runtime: dict) -> Path:
    action_path = tmp_path / f"{relation}_executed.npy"
    returned_path = tmp_path / f"{relation}_returned.npy"
    future_path = tmp_path / f"{relation}_future.npy"
    video_path = tmp_path / f"{relation}.mp4"
    np.save(action_path, np.zeros((4, 8), dtype=np.float32), allow_pickle=False)
    np.save(returned_path, np.zeros((32, 8), dtype=np.float32), allow_pickle=False)
    np.save(future_path, np.zeros((33, 2, 2, 3), dtype=np.uint8), allow_pickle=False)
    video_path.write_bytes(b"bounded-test-video")
    sign = 1.0 if relation == "left" else -1.0
    steps = []
    for index in range(5):
        steps.append({
            "action_step": index,
            "object_xyz": [0.05, 0.0 if index < 2 else sign * 0.20, 0.0 if index == 0 else 0.04],
            "reference_xyz": [0.0, 0.0, 0.0],
            "grippers_open": index == 4,
            "contact_detected": False,
        })
    cell = pair.cell(relation)
    initial_hash = derive_initial_state_sha256(
        {"measurement_frame": MEASUREMENT_FRAME_ID, "steps": steps}
    )
    export = {
        "schema_version": "vla-wam-shared-v3-cosmos-simulator-export-v1",
        "study_id": STUDY_ID,
        "registered_cell_id": cell["cell_id"],
        "attempt_id": f"attempt-{relation}",
        "model_id": pair.model_id,
        "pair_id": pair.pair_id,
        "requested_relation": relation,
        "prompt": cell["prompt"],
        "environment_seed": pair.seed,
        "sampling_seed": pair.seed,
        "reset_id": cell["reset_identity"],
        "predicate_id": cell["success_predicate_id"],
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "steps": steps,
        "actions_executed": 4,
        "executed_action_trace_path": str(action_path),
        "viewport_video_path": str(video_path),
        "policy_requests": [{
            "request_index": 0,
            "sampling_seed": pair.seed,
            "returned_action_path": str(returned_path),
            "decoded_future_path": str(future_path),
        }],
        "source_artifacts": {},
        "requested_success": True,
        "final_detached_release": True,
        "frozen_failure_stage": "success",
        "action_cap": 400,
        "right_censored": False,
        "wall_time_s": 1.0,
        "operational_wall_time_valid": True,
        "first_contact_step": None,
        "first_contact_unavailable_reason": None,
        "initial_state_sha256": initial_hash,
    }
    path = tmp_path / f"{relation}_export.json"
    path.write_text(json.dumps(export))
    return path


class CosmosV3AdapterTests(unittest.TestCase):
    def test_only_authorized_new_cosmos_pairs_resolve(self) -> None:
        for model_id in sorted(MODEL_CONTRACTS):
            with self.subTest(model_id=model_id):
                pair = load_authorized_pair(ROOT, model_id, 8303)
                self.assertEqual(pair.seed, 8303)
                self.assertEqual(pair.left["status"], "authorized_new")
                self.assertEqual(pair.right["status"], "authorized_new")
                self.assertEqual(pair.left["reset_identity"], pair.right["reset_identity"])
                with self.assertRaisesRegex(ContractError, "8303..8329"):
                    load_authorized_pair(ROOT, model_id, 8302)

    def test_runtime_identity_and_fixed_observation_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            for model_id in sorted(MODEL_CONTRACTS):
                with self.subTest(model_id=model_id):
                    pair = load_authorized_pair(ROOT, model_id, 8303)
                    runtime, runtime_path = _runtime(tmp_path, model_id)
                    self.assertEqual(verify_runtime_identity(ROOT, model_id, runtime_path), runtime)
                    gate = evaluate_responses(
                        pair=pair, runtime=runtime, responses=_responses(pair.seed, model_id),
                        conditioning_image_sha256=FROZEN_GROUNDED_OBSERVATION_SHA256,
                    )
                    self.assertEqual(gate["status"], "passed")
                    self.assertEqual(gate["metrics"]["left_repeat_action_rms"], 0.0)
                    self.assertGreater(gate["metrics"]["left_right_action_rms"], 0.0)

    def test_missing_future_is_never_zero_filled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_id = "cosmos3_edge_policy_droid"
            pair = load_authorized_pair(ROOT, model_id, 8303)
            runtime, _ = _runtime(Path(directory), model_id)
            responses = _responses(pair.seed, model_id)
            responses["right"].pop("video")
            with self.assertRaisesRegex(ContractError, "decoded future is missing"):
                evaluate_responses(
                    pair=pair, runtime=runtime, responses=responses,
                    conditioning_image_sha256=FROZEN_GROUNDED_OBSERVATION_SHA256,
                )

    def test_compile_pair_emits_two_valid_behavioral_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            model_id = "cosmos3_edge_policy_droid"
            pair = load_authorized_pair(ROOT, model_id, 8303)
            runtime, runtime_path = _runtime(tmp_path, model_id)
            gate = evaluate_responses(
                pair=pair, runtime=runtime, responses=_responses(pair.seed, model_id),
                conditioning_image_sha256=FROZEN_GROUNDED_OBSERVATION_SHA256,
            )
            gate_path = tmp_path / "gate.json"
            gate_path.write_text(json.dumps(gate))
            left = _make_export(tmp_path, pair=pair, relation="left", runtime=runtime)
            right = _make_export(tmp_path, pair=pair, relation="right", runtime=runtime)
            output = tmp_path / "pair.jsonl"
            manifest = compile_pair(
                study_root=ROOT, model_id=model_id, seed=pair.seed,
                runtime_manifest=runtime_path, release_manifest=gate_path,
                left_export=left, right_export=right, output_jsonl=output,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(manifest["row_count"], 2)
            self.assertEqual({row["requested_relation"] for row in rows}, {"left", "right"})
            self.assertTrue(all(row["failure_taxonomy"] == "correct" for row in rows))
            self.assertTrue(all(
                row["future_requests"][0]["future_evidence_status"] == "exposed_and_retained"
                for row in rows
            ))

    def test_infrastructure_attempt_stays_outside_behavioral_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            model_id = "cosmos3_nano_policy_droid"
            pair = load_authorized_pair(ROOT, model_id, 8303)
            runtime, _ = _runtime(tmp_path, model_id)
            log = tmp_path / "attempt.log"
            log.write_text("missing decoded future\n")
            cell = pair.cell("left")
            attempt = {
                "schema_version": "vla-wam-shared-v3-cosmos-infrastructure-export-v1",
                "study_id": STUDY_ID,
                "registered_cell_id": cell["cell_id"],
                "model_id": model_id,
                "pair_id": pair.pair_id,
                "requested_relation": "left",
                "environment_seed": pair.seed,
                "sampling_seed": pair.seed,
                "runtime_identity_sha256": runtime["runtime_identity_sha256"],
                "classification": "technical_invalid",
                "attempt_id": "attempt-left-infra-01",
                "stage": "decoded_future_contract",
                "error": "server response omitted decoded future",
                "runtime_intervention": False,
                "repair_attempt_id": None,
                "log_path": str(log),
                "event_timeline": [{"sequence": 0, "stage": "decoded_future_contract"}],
            }
            attempt_path = tmp_path / "attempt.json"
            attempt_path.write_text(json.dumps(attempt))
            record = build_infrastructure_record(
                pair=pair, relation="left", attempt_path=attempt_path,
                runtime=runtime, output_jsonl=tmp_path / "infra.jsonl",
            )
            self.assertFalse(record["behavioral_result_valid"])
            self.assertNotIn("failure_taxonomy", record)
            self.assertNotIn("viewport_video", record["artifacts"])
            self.assertNotIn("executed_action_trace", record["artifacts"])
            self.assertEqual(record["denominator_policy"], "excluded_from_behavioral_denominator")


if __name__ == "__main__":
    unittest.main()
