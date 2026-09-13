from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


WORKSHOP = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSHOP.parents[1]
FORECAST = WORKSHOP / "experiments/forecast_layout"
sys.path.insert(0, str(FORECAST))

import fixture_layouts  # noqa: E402
import model_blind_fixture_gate as fixture_gate  # noqa: E402
import n3_behavioral_pilot_job as pilot  # noqa: E402
import n3_development_block_job as development  # noqa: E402
import recorder_qualification_job as recorder_job  # noqa: E402


STUDY_COMMIT = "a" * 40


def _write_json(path: Path, value: dict) -> dict:
    pilot.immutable_json(path, value)
    return pilot.file_identity(path)


def _make_passed_cell(
    raw_root: Path,
    block: development.DevelopmentBlock,
    attempt_name: str,
    index: int,
    *,
    candidate_id: str | None = None,
    accepted_gate_record_sha256: str = "d" * 64,
    pose_manifest_sha256: str = "c" * 64,
) -> Path:
    cell_root = raw_root / attempt_name / "cells" / f"{index:02d}-cell"
    artifacts = cell_root / "artifacts"
    artifacts.mkdir(parents=True)
    candidate_id = candidate_id or f"{block.layout_pair_id}__candidate_00"
    arm, command, _task = block.conditions[index]
    completion = _write_json(
        artifacts / "adapter_completion.json",
        {
            "schema_version": "wmf-forecast-recording-attempt-v1",
            "study_id": pilot.STUDY_ID,
            "identity": {
                "attempt_id": f"{attempt_name}:{index:02d}",
                "cell_id": block.cell_ids[index],
                "stage": "development",
                "layout_pair_id": block.layout_pair_id,
                "layout_arm": arm,
                "command": command,
                "prompt": pilot.PROMPTS[command],
                "model_config": "N3",
                "effective_seed": block.effective_seed,
                "source_identity": (
                    f"study:{STUDY_COMMIT};robolab:{pilot.ROBOLAB_COMMIT};"
                    f"cosmos:{pilot.COSMOS_COMMIT};pose:{pose_manifest_sha256}"
                ),
                "checkpoint_identity": (
                    f"revision:{pilot.CHECKPOINT_REVISION};"
                    f"aggregate:{pilot.CHECKPOINT_AGGREGATE_SHA256}"
                ),
                "study_id": pilot.STUDY_ID,
            },
            "stop_reason": "action_cap",
            "behavioral_result_valid": True,
            "recording_qualification_valid": False,
            "model_attached": True,
            "right_censored": False,
            "technical_invalid": False,
            "actions_executed": 450,
            "action_cap": 450,
            "observation_count": 451,
            "request_count": 15,
            "success_configured_as_termination": False,
            "runner_reset_calls": 2,
            "physical_reset_calls": 1,
            "final_two_action_truncation_recorded": True,
            "validation_errors": [],
        },
    )
    descriptors = {
        name: _write_json(artifacts / f"{name}.json", {"name": name, "index": index})
        for name in ("adapter_journal", "native_timing_support")
    }
    video = artifacts / "viewport.mp4"
    video.write_bytes(f"development-video-{index}".encode())
    context_id = f"development-context-{index}"
    receipt = {
        "schema_version": development.CELL_RECEIPT_SCHEMA,
        "status": "passed",
        "study_id": pilot.STUDY_ID,
        "block_id": block.block_id,
        "cell_id": block.cell_ids[index],
        "condition_index": index,
        "layout_pair_id": block.layout_pair_id,
        "layout_arm": arm,
        "command": command,
        "prompt": pilot.PROMPTS[command],
        "model_config": "N3",
        "effective_seed": block.effective_seed,
        "actions_executed": 450,
        "observation_count": 451,
        "behavioral_model_request_count": 15,
        "behavioral_episode_count": 1,
        "generation_qualification_request_count": 0,
        "final_chunk_executed_actions": 2,
        "transport_contract": dict(development.NO_REPLAY_TRANSPORT_CONTRACT),
        "candidate_id": candidate_id,
        "accepted_gate_record_sha256": accepted_gate_record_sha256,
        "source_pins": {
            "study_commit": STUDY_COMMIT,
            "robolab_commit": pilot.ROBOLAB_COMMIT,
            "cosmos_commit": pilot.COSMOS_COMMIT,
        },
        "checkpoint_pin": {
            "revision": pilot.CHECKPOINT_REVISION,
            "aggregate_sha256": pilot.CHECKPOINT_AGGREGATE_SHA256,
        },
        "adapter_completion": completion,
        **descriptors,
        "viewport_video": pilot.file_identity(video),
        "server_begin_receipt": {
            "passed": True,
            "server_context_id": context_id,
            "cache_reset_evidence": {
                "passed": True,
                "episode_context_id": context_id,
                "unresolved_mutable_temporal_fields": [],
            },
        },
        "server_end_receipt": {
            "passed": True,
            "status": "completed",
            "cell_id": block.cell_ids[index],
            "condition_index": index,
            "server_context_id": context_id,
            "server_request_count": 15,
            "client_request_count": 15,
            "actions_executed": 450,
        },
    }
    path = cell_root / "cell_receipt.json"
    pilot.immutable_json(path, receipt)
    return path


def _make_fixture_release(
    root: Path, block: development.DevelopmentBlock
) -> tuple[Path, str, Path, str]:
    pool_path = FORECAST / "layout_candidate_pool.json"
    pool = json.loads(pool_path.read_text())
    candidate = next(
        row
        for row in pool["candidates"]
        if row["layout_pair_id"] == block.layout_pair_id and row["candidate_rank"] == 0
    )
    attempt = {
        "schema_version": recorder_job.GATE_ATTEMPT_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "candidate_pool_sha256": recorder_job.CANDIDATE_POOL_SHA256,
        "attempt_number": 0,
        "decision": "accepted",
        "passed": True,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "evaluation": {"passed": True, "decision": "accepted", "failures": []},
        "error": None,
    }
    attempt_path = root / "gate_attempt_receipt.json"
    attempt_identity = _write_json(attempt_path, attempt)
    record = {
        "layout_pair_id": block.layout_pair_id,
        "candidate_id": candidate["candidate_id"],
        "candidate_rank": 0,
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "candidate_pool_sha256": recorder_job.CANDIDATE_POOL_SHA256,
        "decision": "accepted",
        "passed": True,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "attempt_number": 0,
        "attempt_receipt": attempt_identity,
        "failure_count": 0,
        "failures": [],
        "release_boundary": "test accepted gate",
        "schema_version": recorder_job.GATE_RECORD_SCHEMA,
        "study_namespace": pilot.NAMESPACE,
        "sequence": 0,
        "previous_record_sha256": None,
        "recorded_at_utc": "2026-09-13T00:00:00Z",
    }
    record["record_sha256"] = recorder_job.sha256_bytes(
        recorder_job.canonical_bytes(record)
    )
    ledger_path = root / "gate_ledger.jsonl"
    ledger_path.write_text(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    )
    ledger_identity = pilot.file_identity(ledger_path)
    pose = fixture_gate.build_frozen_pose_manifest(
        pool=pool,
        candidate_pool_sha256=recorder_job.CANDIDATE_POOL_SHA256,
        records=[record],
        gate_ledger_sha256=ledger_identity["sha256"],
        layout_pair_ids=[block.layout_pair_id],
    )
    pose_path = root / "pose_manifest.json"
    pilot.immutable_json(pose_path, pose)
    gate = {
        "schema_version": recorder_job.GATE_RECEIPT_SCHEMA,
        "study_namespace": pilot.NAMESPACE,
        "status": "finished",
        "job_id": f"fixture-{block.layout_pair_id.lower()}-candidate-00",
        "layout_pair_id": block.layout_pair_id,
        "candidate_id": candidate["candidate_id"],
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "decision": "accepted",
        "exit_code": 0,
        "source_contract_sha256": recorder_job.SOURCE_CONTRACT_SHA256,
        "candidate_pool_sha256": recorder_job.CANDIDATE_POOL_SHA256,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "gate_ledger": ledger_identity,
        "gate_evidence": {
            "gate_record_sha256": record["record_sha256"],
            "gate_attempt_receipt": attempt_identity,
        },
    }
    gate_path = root / "gate_receipt.json"
    pilot.immutable_json(gate_path, gate)
    return (
        gate_path,
        pilot.sha256_file(gate_path),
        pose_path,
        pilot.sha256_file(pose_path),
    )


def _make_passed_p00_pilot(root: Path) -> tuple[Path, str]:
    prerequisites_root = root / "pilot_prerequisites"
    prerequisites_root.mkdir()
    simple = {
        name: _write_json(prerequisites_root / f"{name}.json", {"kind": name})
        for name in (
            "gate_receipt",
            "pose_manifest",
            "recorder_child_receipt",
            "native_timing_support",
        )
    }
    p00_candidate_id = "P00__candidate_00"
    p00_candidate_sha256 = "a" * 64
    p00_gate_record_sha256 = "b" * 64
    p00_pose_sha256 = simple["pose_manifest"]["sha256"]
    capture = _write_json(
        prerequisites_root / "capture_receipt.json",
        {
            "schema_version": pilot.CAPTURE_RECEIPT_SCHEMA,
            "status": "passed",
            "layout_pair_id": "P00",
            "environment_seed": development.P00_EFFECTIVE_SEED,
            "model_request_count": 0,
            "behavioral_action_count": 0,
        },
    )
    recorder = _write_json(
        prerequisites_root / "recorder_receipt.json",
        {
            "schema_version": pilot.RECORDER_RECEIPT_SCHEMA,
            "status": "passed",
            "exit_code": 0,
            "layout_pair_id": "P00",
            "environment_seed": development.P00_EFFECTIVE_SEED,
            "actions_executed": 450,
            "observation_count": 451,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
        },
    )
    qualification = _write_json(
        prerequisites_root / "n3_qualification.json",
        {
            "schema_version": pilot.N3_QUALIFICATION_SCHEMA,
            "status": "passed",
            "qualified": True,
            "model_config": "N3",
            "effective_seed": development.P00_EFFECTIVE_SEED,
            "generation_request_count": 6,
            "robot_episode_count": 0,
            "source": {
                "commit": pilot.COSMOS_COMMIT,
                "git_tree": pilot.COSMOS_GIT_TREE,
            },
            "checkpoint": {
                "revision": pilot.CHECKPOINT_REVISION,
                "payload_aggregate_sha256": pilot.CHECKPOINT_AGGREGATE_SHA256,
            },
        },
    )

    conditions = tuple(
        development._condition_tuple(label)
        for label in development.P00_CONDITION_ORDER
    )
    cell_descriptors = []
    for index, (arm, command, _task) in enumerate(conditions):
        cell_root = root / "pilot_attempt" / "cells" / f"{index:02d}-cell"
        artifacts = cell_root / "artifacts"
        artifacts.mkdir(parents=True)
        completion = _write_json(
            artifacts / "adapter_completion.json",
            {
                "schema_version": "wmf-forecast-recording-attempt-v1",
                "study_id": pilot.STUDY_ID,
                "identity": {
                    "attempt_id": f"pilot-attempt:{index:02d}",
                    "cell_id": development.P00_CELL_IDS[index],
                    "stage": "pilot",
                    "layout_pair_id": "P00",
                    "layout_arm": arm,
                    "command": command,
                    "prompt": pilot.PROMPTS[command],
                    "model_config": "N3",
                    "effective_seed": development.P00_EFFECTIVE_SEED,
                    "source_identity": (
                        f"study:{STUDY_COMMIT};robolab:{pilot.ROBOLAB_COMMIT};"
                        f"cosmos:{pilot.COSMOS_COMMIT};pose:{p00_pose_sha256}"
                    ),
                    "checkpoint_identity": (
                        f"revision:{pilot.CHECKPOINT_REVISION};"
                        f"aggregate:{pilot.CHECKPOINT_AGGREGATE_SHA256}"
                    ),
                    "study_id": pilot.STUDY_ID,
                },
                "stop_reason": "action_cap",
                "behavioral_result_valid": True,
                "recording_qualification_valid": False,
                "model_attached": True,
                "right_censored": False,
                "technical_invalid": False,
                "actions_executed": 450,
                "action_cap": 450,
                "observation_count": 451,
                "request_count": 15,
                "success_configured_as_termination": False,
                "runner_reset_calls": 2,
                "physical_reset_calls": 1,
                "final_two_action_truncation_recorded": True,
                "validation_errors": [],
            },
        )
        descriptors = {
            name: _write_json(
                artifacts / f"{name}.json", {"name": name, "index": index}
            )
            for name in (
                "adapter_journal",
                "native_timing_support",
            )
        }
        video = artifacts / "viewport.mp4"
        video.write_bytes(f"pilot-video-{index}".encode())
        context = f"pilot-context-{index}"
        cell = {
            "schema_version": development.PILOT_CELL_RECEIPT_SCHEMA,
            "status": "passed",
            "study_id": pilot.STUDY_ID,
            "block_id": development.P00_BLOCK_ID,
            "cell_id": development.P00_CELL_IDS[index],
            "condition_index": index,
            "layout_pair_id": "P00",
            "layout_arm": arm,
            "command": command,
            "prompt": pilot.PROMPTS[command],
            "model_config": "N3",
            "effective_seed": development.P00_EFFECTIVE_SEED,
            "actions_executed": 450,
            "observation_count": 451,
            "behavioral_model_request_count": 15,
            "behavioral_episode_count": 1,
            "generation_qualification_request_count": 0,
            "final_chunk_executed_actions": 2,
            "candidate_id": p00_candidate_id,
            "accepted_gate_record_sha256": p00_gate_record_sha256,
            "source_pins": {
                "study_commit": STUDY_COMMIT,
                "robolab_commit": pilot.ROBOLAB_COMMIT,
                "cosmos_commit": pilot.COSMOS_COMMIT,
            },
            "checkpoint_pin": {
                "revision": pilot.CHECKPOINT_REVISION,
                "aggregate_sha256": pilot.CHECKPOINT_AGGREGATE_SHA256,
            },
            "adapter_completion": completion,
            **descriptors,
            "viewport_video": pilot.file_identity(video),
            "server_begin_receipt": {
                "passed": True,
                "server_context_id": context,
                "cache_reset_evidence": {
                    "passed": True,
                    "episode_context_id": context,
                    "unresolved_mutable_temporal_fields": [],
                },
            },
            "server_end_receipt": {
                "passed": True,
                "status": "completed",
                "cell_id": development.P00_CELL_IDS[index],
                "condition_index": index,
                "server_context_id": context,
                "server_request_count": 15,
                "client_request_count": 15,
                "actions_executed": 450,
            },
        }
        cell_path = cell_root / "cell_receipt.json"
        pilot.immutable_json(cell_path, cell)
        cell_descriptors.append(pilot.file_identity(cell_path))

    topology = _write_json(
        root / "pilot_topology.json",
        {
            "schema_version": pilot.TOPOLOGY_SCHEMA,
            "status": "passed",
            "devices": [
                {"index": "0", "uuid": "GPU-a", "name": "NVIDIA B200"},
                {"index": "1", "uuid": "GPU-b", "name": "NVIDIA B200"},
            ],
            "preexisting_compute_process_count": 0,
        },
    )
    ready = _write_json(
        root / "pilot_server_ready.json",
        {
            "schema_version": pilot.SERVER_READY_SCHEMA,
            "status": "ready",
            "block_id": development.P00_BLOCK_ID,
            "model_config": "N3",
            "expected_cell_order": list(development.P00_CELL_IDS),
            "planned_block_behavioral_request_count": 60,
            "generation_qualification_requests_rerun": 0,
        },
    )
    receipt = {
        "schema_version": development.PILOT_QUEUE_RECEIPT_SCHEMA,
        "status": "passed",
        "exit_code": 0,
        "study_id": pilot.STUDY_ID,
        "namespace": pilot.NAMESPACE,
        "block_id": development.P00_BLOCK_ID,
        "phase": "pilot",
        "layout_pair_id": "P00",
        "model_config": "N3",
        "effective_seed": development.P00_EFFECTIVE_SEED,
        "condition_order": list(development.P00_CONDITION_ORDER),
        "cell_ids": list(development.P00_CELL_IDS),
        "counts": {
            "planned_behavioral_cells": 4,
            "launched_behavioral_cells": 4,
            "completed_valid_behavioral_cells": 4,
            "technically_invalid_behavioral_cells": 0,
            "right_censored_behavioral_cells": 0,
            "unrun_behavioral_cells": 0,
            "actual_behavioral_actions": 1800,
            "actual_behavioral_model_requests": 60,
            "new_generation_qualification_requests": 0,
            "reused_prerequisite_generation_qualification_requests": 6,
            "recorder_only_episodes_counted_as_behavioral": 0,
        },
        "source_commit": STUDY_COMMIT,
        "cell_receipts": cell_descriptors,
        "prerequisites": {
            **simple,
            "candidate_id": p00_candidate_id,
            "candidate_payload_sha256": p00_candidate_sha256,
            "accepted_gate_record_sha256": p00_gate_record_sha256,
            "capture_receipt": capture,
            "recorder_receipt": recorder,
            "n3_qualification_receipt": qualification,
            "generation_qualification_requests_reused_not_rerun": 6,
        },
        "topology": topology,
        "server_ready": ready,
        "server_exit": {
            "schema_version": pilot.SERVER_EXIT_SCHEMA,
            "child_reaped": True,
            "terminated_by_queue_supervisor": True,
        },
    }
    receipt_path = root / "n3_behavioral_pilot_receipt.json"
    pilot.immutable_json(receipt_path, receipt)
    return receipt_path, pilot.sha256_file(receipt_path)


class FrozenScheduleTests(unittest.TestCase):
    def test_all_four_rows_bind_exact_order_seed_and_inventory(self) -> None:
        for layout_id, (seed, order) in development.DEVELOPMENT_SCHEDULE.items():
            with self.subTest(layout_id=layout_id):
                block = development.load_development_block(SOURCE_ROOT, layout_id)
                self.assertEqual(block.effective_seed, seed)
                self.assertEqual(block.condition_order, order)
                self.assertEqual(len(block.conditions), 4)
                self.assertEqual(len(set(block.conditions)), 4)
                self.assertEqual(
                    block.cell_ids,
                    tuple(
                        f"wmf1__development__{layout_id}__N3__{label.replace('-', '__')}"
                        for label in order
                    ),
                )
                self.assertEqual(block.schedule_row["ordered_cell_ids"], list(block.cell_ids))

    def test_confirmation_and_pilot_layouts_are_unsupported(self) -> None:
        for layout_id in ("P00", "C01", "D05", "d01"):
            with self.subTest(layout_id=layout_id):
                with self.assertRaisesRegex(
                    pilot.N3BehavioralPilotError, "development_layout_unsupported"
                ):
                    development.load_development_block(SOURCE_ROOT, layout_id)

    def test_configuration_is_scoped_and_protocol_uses_development_seed(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D03")
        original = (pilot.PHASE, pilot.LAYOUT_PAIR_ID, pilot.BLOCK_ID, pilot.EFFECTIVE_SEED)
        with development.configured_pilot(block):
            self.assertEqual(pilot.PHASE, "development")
            self.assertEqual(pilot.LAYOUT_PAIR_ID, "D03")
            protocol = pilot.ServerProtocol()
            arm, command, _task = block.conditions[0]
            begin = protocol.begin(
                {
                    "study_id": pilot.STUDY_ID,
                    "block_id": block.block_id,
                    "cell_id": block.cell_ids[0],
                    "condition_index": 0,
                    "layout_arm": arm,
                    "command": command,
                    "prompt": pilot.PROMPTS[command],
                    "effective_seed": block.effective_seed,
                    "expected_actions": 450,
                    "expected_requests": 15,
                },
                server_context_id="development-context",
                temporal_reset_evidence={
                    "passed": True,
                    "unresolved_mutable_temporal_fields": [],
                },
            )
            self.assertEqual(begin["effective_seed"], 2026091103)
        self.assertEqual(
            (pilot.PHASE, pilot.LAYOUT_PAIR_ID, pilot.BLOCK_ID, pilot.EFFECTIVE_SEED),
            original,
        )

    def test_child_commands_reenter_the_development_runner(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D02")
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "attempt"
            server = development.build_server_command(
                source_root=SOURCE_ROOT,
                attempt_root=attempt,
                port=18011,
                study_commit=STUDY_COMMIT,
                start_cell_index=2,
                block=block,
            )
            cell = development.build_cell_command(
                source_root=SOURCE_ROOT,
                attempt_root=attempt,
                study_commit=STUDY_COMMIT,
                gate_receipt=Path("/evidence/gate.json"),
                gate_receipt_sha256="b" * 64,
                pose_manifest=Path("/evidence/pose.json"),
                pose_manifest_sha256="c" * 64,
                port=18011,
                condition_index=2,
                block=block,
            )
        self.assertTrue(server[1].endswith("/n3_development_block_job.py"))
        self.assertEqual(server[server.index("--layout-pair-id") + 1], "D02")
        self.assertEqual(server[server.index("--start-cell-index") + 1], "2")
        self.assertTrue(cell[1].endswith("/n3_development_block_job.py"))
        self.assertEqual(cell[cell.index("--layout-arm") + 1], "reflected")
        self.assertEqual(cell[cell.index("--command") + 1], "left")


class NoReplayTransportTests(unittest.TestCase):
    def test_transport_disables_pings_and_query_is_exactly_once(self) -> None:
        connect_calls = []
        infer_calls = []

        class Connection:
            def recv(self):
                return b"metadata"

            def close(self):
                raise AssertionError("successful metadata receive must keep connection open")

        def connect(uri, **kwargs):
            connect_calls.append((uri, kwargs))
            return Connection()

        class Policy:
            def __init__(self, host, port):
                self._uri = f"ws://{host}:{port}"

        class OriginalClient:
            _remote_host = "127.0.0.1"
            _remote_port = 18011

        transport_type, client_type = development.build_no_replay_transport_types(
            original_client_class=OriginalClient,
            websocket_policy_class=Policy,
            connect=connect,
            unpackb=lambda payload: {"decoded": payload.decode()},
        )
        client = client_type()
        transport = client._connect()
        self.assertIsInstance(transport, transport_type)
        connection, metadata = transport._wait_for_server()
        self.assertIsInstance(connection, Connection)
        self.assertEqual(metadata, {"decoded": "metadata"})
        self.assertEqual(
            connect_calls,
            [
                (
                    "ws://127.0.0.1:18011",
                    {
                        "compression": None,
                        "max_size": None,
                        "ping_interval": None,
                        "ping_timeout": None,
                    },
                )
            ],
        )

        class FailingInference:
            def infer(self, request):
                infer_calls.append(request)
                raise OSError("lost response")

        client.client = FailingInference()
        request = {"request_index": 3}
        with self.assertRaisesRegex(OSError, "lost response"):
            client._query_server(request)
        self.assertEqual(infer_calls, [request])

    def test_development_receipt_requires_no_replay_attestation(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D01")
        with tempfile.TemporaryDirectory() as temporary:
            path = _make_passed_cell(Path(temporary), block, "attempt-a", 0)
            value = json.loads(path.read_text())
            value.pop("transport_contract")
            replacement = path.with_name("without_transport.json")
            pilot.immutable_json(replacement, value)
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError,
                "resume_cell_transport_contract_mismatch",
            ):
                development.validate_passed_development_cell(
                    replacement, condition_index=0, block=block
                )


class FixtureReleaseTests(unittest.TestCase):
    def test_exact_development_gate_and_pose_chain_passes_for_both_arms(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D01")
        with tempfile.TemporaryDirectory() as temporary:
            gate, gate_sha, pose, pose_sha = _make_fixture_release(Path(temporary), block)
            original = development.verify_development_fixture_release(
                gate_receipt_path=gate,
                gate_receipt_sha256=gate_sha,
                pose_manifest_path=pose,
                pose_manifest_sha256=pose_sha,
                layout_arm="original",
                block=block,
            )
            reflected = development.verify_development_fixture_release(
                gate_receipt_path=gate,
                gate_receipt_sha256=gate_sha,
                pose_manifest_path=pose,
                pose_manifest_sha256=pose_sha,
                layout_arm="reflected",
                block=block,
            )
        self.assertEqual(original["candidate_id"], "D01__candidate_00")
        self.assertEqual(original["candidate_id"], reflected["candidate_id"])

    def test_gate_or_pose_hash_mismatch_fails_closed(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D04")
        with tempfile.TemporaryDirectory() as temporary:
            gate, gate_sha, pose, pose_sha = _make_fixture_release(Path(temporary), block)
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "evidence_sha256_mismatch"
            ):
                development.verify_development_fixture_release(
                    gate_receipt_path=gate,
                    gate_receipt_sha256="0" * 64,
                    pose_manifest_path=pose,
                    pose_manifest_sha256=pose_sha,
                    layout_arm="original",
                    block=block,
                )
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "evidence_sha256_mismatch"
            ):
                development.verify_development_fixture_release(
                    gate_receipt_path=gate,
                    gate_receipt_sha256=gate_sha,
                    pose_manifest_path=pose,
                    pose_manifest_sha256="1" * 64,
                    layout_arm="original",
                    block=block,
                )


class PilotPrerequisiteTests(unittest.TestCase):
    def test_only_a_complete_hash_bound_p00_pilot_releases_development(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt, digest = _make_passed_p00_pilot(Path(temporary))
            verified = development.verify_passed_p00_pilot(receipt, digest)
            self.assertEqual(verified["pilot_behavioral_cells"], 4)
            self.assertEqual(verified["pilot_behavioral_actions"], 1800)
            self.assertEqual(verified["pilot_behavioral_model_requests"], 60)
            self.assertEqual(verified["new_generation_qualification_requests"], 0)
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "evidence_sha256_mismatch"
            ):
                development.verify_passed_p00_pilot(receipt, "0" * 64)


class ResumeTests(unittest.TestCase):
    def test_contiguous_prefix_is_reused_without_relaunch(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D02")
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            first = _make_passed_cell(raw, block, "attempt-a", 0)
            second = _make_passed_cell(raw, block, "attempt-b", 1)
            receipts, identities, provenance = development.discover_completed_prefix(
                raw, block=block
            )
        self.assertEqual([row["cell_id"] for row in receipts], list(block.cell_ids[:2]))
        self.assertEqual([row["path"] for row in identities], [str(first), str(second)])
        self.assertEqual(provenance["start_cell_index"], 2)
        self.assertEqual(provenance["completed_prefix_cell_ids"], list(block.cell_ids[:2]))

    def test_noncontiguous_or_duplicate_passed_cells_fail_closed(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D03")
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            _make_passed_cell(raw, block, "attempt-a", 1)
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "resume_passed_cells_not_contiguous"
            ):
                development.discover_completed_prefix(raw, block=block)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            _make_passed_cell(raw, block, "attempt-a", 0)
            _make_passed_cell(raw, block, "attempt-b", 0)
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "resume_duplicate_passed_cell"
            ):
                development.discover_completed_prefix(raw, block=block)

    def test_reusable_cell_must_bind_the_selected_gate_and_pose(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D04")
        with tempfile.TemporaryDirectory() as temporary:
            path = _make_passed_cell(Path(temporary), block, "attempt-a", 0)
            receipt, _identity = development.validate_passed_development_cell(
                path, condition_index=0, block=block
            )
            development.validate_cells_bind_fixture(
                [receipt],
                candidate_id="D04__candidate_00",
                accepted_gate_record_sha256="d" * 64,
                pose_manifest_sha256="c" * 64,
            )
            for field, values, error in (
                (
                    "candidate",
                    {
                        "candidate_id": "D04__candidate_01",
                        "accepted_gate_record_sha256": "d" * 64,
                        "pose_manifest_sha256": "c" * 64,
                    },
                    "cell_fixture_candidate_mismatch",
                ),
                (
                    "gate",
                    {
                        "candidate_id": "D04__candidate_00",
                        "accepted_gate_record_sha256": "e" * 64,
                        "pose_manifest_sha256": "c" * 64,
                    },
                    "cell_fixture_gate_record_mismatch",
                ),
                (
                    "pose",
                    {
                        "candidate_id": "D04__candidate_00",
                        "accepted_gate_record_sha256": "d" * 64,
                        "pose_manifest_sha256": "e" * 64,
                    },
                    "cell_fixture_pose_manifest_mismatch",
                ),
            ):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(
                        pilot.N3BehavioralPilotError, error
                    ):
                        development.validate_cells_bind_fixture([receipt], **values)

    def test_receipt_counts_do_not_count_pilot_or_qualification_as_development(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D01")
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            receipt_path = _make_passed_cell(raw, block, "attempt-a", 0)
            receipt, _identity = development.validate_passed_development_cell(
                receipt_path, condition_index=0, block=block
            )
            counts = development._receipt_counts(
                block=block,
                start_cell_index=1,
                launched=0,
                completed=[receipt],
                attempt_root=raw / "new-attempt",
            )
        self.assertEqual(counts["completed_valid_behavioral_cells"], 1)
        self.assertEqual(counts["actual_behavioral_actions"], 450)
        self.assertEqual(counts["actual_behavioral_model_requests"], 15)
        self.assertEqual(counts["new_generation_qualification_requests"], 0)
        self.assertEqual(counts["recorder_only_episodes_counted_as_behavioral"], 0)

    def test_attempt_id_cannot_escape_the_block_raw_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                development.resolve_attempt_root(root, "n3-development-d01-001"),
                root / "n3-development-d01-001",
            )
            for unsafe in ("../escape", "nested/escape", ".", ""):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaisesRegex(
                        pilot.N3BehavioralPilotError, "invalid_job_id"
                    ):
                        development.resolve_attempt_root(root, unsafe)


if __name__ == "__main__":
    unittest.main()
